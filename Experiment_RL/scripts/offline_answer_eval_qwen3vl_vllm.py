#!/usr/bin/env python3
"""vLLM answer-only eval for standalone merged Qwen3-VL checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


VALID_OPTIONS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/data2/hjk/projects/AI-HiddenState-ER/rl_data/mathverse_qwen3vl_stage3_pilot200_128/mathverse_grpo_val.parquet",
    )
    parser.add_argument("--model", required=True, help="Standalone merged HF model directory.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trust-remote-code", action="store_true")
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


def to_vllm_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from PIL import Image

    converted: list[dict[str, Any]] = []
    for message in messages:
        content: list[dict[str, Any]] = []
        for item in message.get("content", []):
            if item.get("type") == "image":
                image = Image.open(item["image"]).convert("RGB")
                content.append({"type": "image_pil", "image_pil": image})
            else:
                content.append(item)
        converted.append({"role": message.get("role", "user"), "content": content})
    return converted


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
    return {
        "run_name": args.run_name,
        "model": args.model,
        "num_records": len(records),
        "accuracy": float(sum(correct_values) / len(correct_values)) if correct_values else 0.0,
        "balanced_accuracy": float(mean(answer_acc.values())) if answer_acc else 0.0,
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
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }


def preview(text: str, max_chars: int = 1800) -> str:
    text = str(text).replace("\r\n", "\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated preview]..."


def write_markdown(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = [
        f"# vLLM Offline Answer-Only Eval: {summary['run_name']}",
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
        f"- Truncated by max_tokens: {summary['truncated_count']}",
        f"- Has `<think>` / `</think>`: {summary['has_think_open_count']} / {summary['has_think_close_count']}",
        f"- Decoding: temperature={summary['temperature']}, top_p={summary['top_p']}, top_k={summary['top_k']}, max_tokens={summary['max_tokens']}",
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


def batched(values: list[Any], batch_size: int) -> list[list[Any]]:
    batch_size = max(1, int(batch_size))
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def main() -> None:
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.input), args.limit)
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages, prompt_text, image_paths = build_messages(row.get("prompt"), row.get("images"))
        prepared.append(
            {
                "index": index,
                "row": row,
                "messages": to_vllm_messages(messages),
                "prompt_text": prompt_text,
                "image_paths": image_paths,
                "answer": get_answer(row),
                "question_id": get_question_id(row, index),
            }
        )

    print(f"Loading vLLM model from {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1, "video": 0},
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    records: list[dict[str, Any]] = []
    for batch in batched(prepared, args.batch_size):
        outputs = llm.chat(
            [item["messages"] for item in batch],
            sampling_params=sampling_params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": True},
        )
        for item, output in zip(batch, outputs, strict=True):
            completion = output.outputs[0]
            response = completion.text
            token_ids = getattr(completion, "token_ids", None)
            output_token_count = len(token_ids) if token_ids is not None else 0
            finish_reason = getattr(completion, "finish_reason", None)
            prediction = extract_final_choice(response)
            answer = item["answer"]
            correct = bool(answer is not None and prediction == answer)
            stored_response = (
                response
                if args.write_response_chars < 0 or len(response) <= args.write_response_chars
                else response[: args.write_response_chars] + "\n...[truncated stored response]..."
            )
            record = {
                "run_name": args.run_name,
                "index": item["index"],
                "question_id": item["question_id"],
                "answer": answer,
                "prediction": prediction,
                "correct": correct,
                "output_token_count": int(output_token_count),
                "finish_reason": finish_reason,
                "truncated_by_length": bool(
                    str(finish_reason).lower() == "length" or output_token_count >= args.max_tokens
                ),
                "has_think_open": "<think>" in response.lower(),
                "has_think_close": "</think>" in response.lower(),
                "image_paths": item["image_paths"],
                "prompt_text": item["prompt_text"],
                "response": stored_response,
            }
            records.append(record)
            print(
                "EVAL_ROW",
                args.run_name,
                record["index"],
                record["question_id"],
                "answer",
                answer,
                "pred",
                prediction,
                "correct",
                int(correct),
                "tokens",
                output_token_count,
                "finish",
                finish_reason,
                flush=True,
            )

    records.sort(key=lambda row: int(row["index"]))
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
