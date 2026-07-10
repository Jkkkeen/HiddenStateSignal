#!/usr/bin/env python3
"""Answer-only offline evaluation for Qwen3-VL VERL LoRA checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch


VALID_OPTIONS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/data2/hjk/projects/AI-HiddenState-ER/rl_data/mathverse_qwen3vl_stage3_pilot200_128/mathverse_grpo_val.parquet",
        help="VERL-style MathVerse parquet validation file.",
    )
    parser.add_argument("--model-dir", required=True, help="Merged HF model directory.")
    parser.add_argument(
        "--adapter-dir",
        default="",
        help="Optional PEFT LoRA adapter directory. Defaults to <model-dir>/lora_adapter if present.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=-1, help="Use -1 for all rows.")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k filtering for HF generate.")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--merge-adapter", action="store_true", help="Merge LoRA into the model after loading.")
    parser.add_argument("--hf-home", default="/data2/hjk/cache/huggingface")
    parser.add_argument("--write-response-chars", type=int, default=12000)
    return parser.parse_args()


def _to_plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value if isinstance(value, dict) else {}


def normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in _to_plain_list(prompt):
        if isinstance(item, dict):
            messages.append(
                {
                    "role": str(item.get("role", "user")),
                    "content": str(item.get("content", "")),
                }
            )
    return messages or [{"role": "user", "content": ""}]


def normalize_images(images: Any) -> list[str]:
    paths: list[str] = []
    for item in _to_plain_list(images):
        if item is None:
            continue
        path = str(item)
        if path:
            paths.append(path)
    return paths


def build_messages(prompt: Any, images: Any) -> tuple[list[dict[str, Any]], str, list[str]]:
    source_messages = normalize_prompt(prompt)
    image_paths = normalize_images(images)
    messages: list[dict[str, Any]] = []
    prompt_text_parts: list[str] = []
    image_index = 0

    for message in source_messages:
        raw_text = message["content"]
        chunks = raw_text.split("<image>")
        content: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks):
            if chunk_index > 0 and image_index < len(image_paths):
                content.append({"type": "image", "image": image_paths[image_index]})
                image_index += 1
            text = chunk.strip()
            if text:
                content.append({"type": "text", "text": text})
                prompt_text_parts.append(text)
        if not content:
            text = raw_text.strip()
            content.append({"type": "text", "text": text})
            prompt_text_parts.append(text)
        messages.append({"role": message["role"], "content": content})

    while image_index < len(image_paths):
        messages[0]["content"].insert(0, {"type": "image", "image": image_paths[image_index]})
        image_index += 1

    return messages, "\n".join(part for part in prompt_text_parts if part), image_paths


def normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in VALID_OPTIONS else None


def extract_final_choice(text: str | None) -> str | None:
    if not text:
        return None
    value = str(text)
    close_idx = value.rfind("</think>")
    candidates = [value[close_idx + len("</think>") :], value] if close_idx >= 0 else [value]
    patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\**\s*([ABCD])\b",
        r"\b([ABCD])\s*(?:is\s+the\s+answer|is\s+correct)\b",
        r"(?:option|choice)\s*\**\s*([ABCD])\b",
        r"\*\*([ABCD])\*\*",
        r"\(([ABCD])\)",
        r"\b([ABCD])\s*[:\.\)]",
    ]
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        for pattern in patterns:
            matches = re.findall(pattern, candidate, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        tail = candidate[-500:]
        matches = re.findall(r"\b([ABCD])\b", tail)
        if matches:
            return matches[-1].upper()
    return None


def get_answer(row: dict[str, Any]) -> str | None:
    extra_info = _to_plain_dict(row.get("extra_info"))
    reward_model = _to_plain_dict(row.get("reward_model"))
    for value in (
        extra_info.get("answer"),
        reward_model.get("ground_truth"),
        extra_info.get("ground_truth"),
        row.get("answer"),
    ):
        choice = normalize_choice(value)
        if choice is not None:
            return choice
    return None


def get_question_id(row: dict[str, Any], index: int) -> str:
    extra_info = _to_plain_dict(row.get("extra_info"))
    for key in ("question_id", "id", "sample_index", "source_index", "index"):
        value = extra_info.get(key, row.get(key))
        if value is not None and value != "":
            return str(value)
    return f"row_{index}"


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_rows(input_path: Path, limit: int) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(input_path)
    if limit >= 0:
        df = df.head(limit)
    return df.to_dict(orient="records")


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    lengths = [int(row.get("output_token_count") or 0) for row in records]
    correct_values = [bool(row.get("correct")) for row in records if row.get("answer") in VALID_OPTIONS]
    by_answer: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        answer = row.get("answer")
        if answer in VALID_OPTIONS:
            by_answer[str(answer)].append(bool(row.get("correct")))

    answer_acc = {
        answer: float(sum(values) / len(values)) if values else 0.0
        for answer, values in sorted(by_answer.items())
    }
    balanced = float(mean(answer_acc.values())) if answer_acc else 0.0
    return {
        "run_name": args.run_name,
        "model_dir": args.model_dir,
        "adapter_dir": args.adapter_dir,
        "num_records": len(records),
        "accuracy": float(sum(correct_values) / len(correct_values)) if correct_values else 0.0,
        "balanced_accuracy": balanced,
        "answer_accuracy": answer_acc,
        "answer_counts": dict(sorted(Counter(row.get("answer") for row in records).items())),
        "prediction_counts": dict(sorted(Counter(row.get("prediction") or "INVALID" for row in records).items())),
        "invalid_predictions": sum(1 for row in records if row.get("prediction") not in VALID_OPTIONS),
        "output_tokens_mean": float(mean(lengths)) if lengths else 0.0,
        "output_tokens_min": min(lengths) if lengths else 0,
        "output_tokens_p50": percentile(lengths, 0.50),
        "output_tokens_p90": percentile(lengths, 0.90),
        "output_tokens_max": max(lengths) if lengths else 0,
        "truncated_count": sum(1 for row in records if row.get("truncated_by_length")),
        "has_think_open_count": sum(1 for row in records if row.get("has_think_open")),
        "has_think_close_count": sum(1 for row in records if row.get("has_think_close")),
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }


def preview(text: str, max_chars: int = 1800) -> str:
    text = str(text).replace("\r\n", "\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated preview]..."


def write_markdown(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = [
        f"# Offline Answer-Only Eval: {summary['run_name']}",
        "",
        "## Summary",
        "",
        f"- Records: {summary['num_records']}",
        f"- Accuracy: {summary['accuracy']:.4f}",
        f"- Balanced accuracy: {summary['balanced_accuracy']:.4f}",
        f"- Answer counts: `{summary['answer_counts']}`",
        f"- Prediction counts: `{summary['prediction_counts']}`",
        f"- Invalid predictions: {summary['invalid_predictions']}",
        f"- Output tokens mean/min/p50/p90/max: {summary['output_tokens_mean']:.1f} / {summary['output_tokens_min']} / {summary['output_tokens_p50']:.1f} / {summary['output_tokens_p90']:.1f} / {summary['output_tokens_max']}",
        f"- Truncated by max_new_tokens: {summary['truncated_count']}",
        f"- Has `<think>` / `</think>`: {summary['has_think_open_count']} / {summary['has_think_close_count']}",
        f"- Decoding: do_sample={summary['do_sample']}, temperature={summary['temperature']}, top_p={summary['top_p']}, top_k={summary['top_k']}, max_new_tokens={summary['max_new_tokens']}",
        "",
        "## Samples",
        "",
    ]
    for row in records[:20]:
        lines.extend(
            [
                f"### index={row['index']} question_id={row['question_id']} answer={row['answer']} pred={row['prediction']} correct={row['correct']} tokens={row['output_token_count']}",
                "",
                "```text",
                preview(row.get("response", "")),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dir = Path(args.model_dir)
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else model_dir / "lora_adapter"
    args.adapter_dir = str(adapter_dir) if adapter_dir.exists() else ""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading processor from {model_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Loading model from {model_dir}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=dtype_from_name(args.dtype),
        device_map="auto",
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if args.adapter_dir:
        print(f"Loading adapter from {args.adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        if args.merge_adapter:
            print("Merging adapter into base model", flush=True)
            model = model.merge_and_unload()
    model.eval()

    rows = load_rows(Path(args.input), args.limit)
    records: list[dict[str, Any]] = []
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
        if args.top_k and args.top_k > 0:
            generation_kwargs["top_k"] = args.top_k

    for index, row in enumerate(rows):
        messages, prompt_text, image_paths = build_messages(row.get("prompt"), row.get("images"))
        try:
            rendered_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            rendered_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[rendered_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[:, prompt_len:]
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        output_token_count = int(generated_ids.shape[1])
        answer = get_answer(row)
        prediction = extract_final_choice(response)
        correct = bool(answer is not None and prediction == answer)
        full_response = response
        stored_response = (
            full_response
            if args.write_response_chars < 0 or len(full_response) <= args.write_response_chars
            else full_response[: args.write_response_chars] + "\n...[truncated stored response]..."
        )
        record = {
            "run_name": args.run_name,
            "index": index,
            "question_id": get_question_id(row, index),
            "answer": answer,
            "prediction": prediction,
            "correct": correct,
            "output_token_count": output_token_count,
            "truncated_by_length": output_token_count >= args.max_new_tokens,
            "has_think_open": "<think>" in full_response.lower(),
            "has_think_close": "</think>" in full_response.lower(),
            "image_paths": image_paths,
            "prompt_text": prompt_text,
            "response": stored_response,
        }
        records.append(record)
        print(
            "EVAL_ROW",
            args.run_name,
            index,
            record["question_id"],
            "answer",
            answer,
            "pred",
            prediction,
            "correct",
            int(correct),
            "tokens",
            output_token_count,
            flush=True,
        )

    summary = summarize(records, args)
    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(output_dir / "summary.md", summary, records)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
