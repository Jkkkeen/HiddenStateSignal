#!/usr/bin/env python3
"""Audit Qwen3-VL-Thinking generations on VERL MathVerse parquet data."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any


EARLY_FINAL_RE = re.compile(
    r"(final\s+answer|therefore[^.\n]{0,80}answer|answer\s+is|option\s+[ABCD])",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/data2/hjk/projects/AI-HiddenState-ER/rl_data/mathverse_verl_smoke16/mathverse_grpo_val.parquet",
        help="VERL-style parquet containing prompt/images/reward_model/extra_info.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data2/hjk/projects/AI-HiddenState-ER/rl_audit/qwen3vl_cot_generation_goal2",
    )
    parser.add_argument(
        "--model",
        default="/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
    )
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=18432)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
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
    if isinstance(value, dict):
        return value
    return {}


def normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in _to_plain_list(prompt):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        messages.append({"role": role, "content": content})
    return messages


def normalize_images(images: Any) -> list[str]:
    paths: list[str] = []
    for item in _to_plain_list(images):
        if item is None:
            continue
        path = str(item)
        if path:
            paths.append(path)
    return paths


def build_messages(prompt: Any, images: Any) -> tuple[list[dict[str, Any]], str]:
    source_messages = normalize_prompt(prompt)
    image_paths = normalize_images(images)
    if not source_messages:
        source_messages = [{"role": "user", "content": ""}]

    messages: list[dict[str, Any]] = []
    image_index = 0
    prompt_text_parts: list[str] = []

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
            content.append({"type": "text", "text": raw_text.strip()})
            prompt_text_parts.append(raw_text.strip())
        messages.append({"role": message["role"], "content": content})

    while image_index < len(image_paths):
        messages[0]["content"].insert(
            0, {"type": "image", "image": image_paths[image_index]}
        )
        image_index += 1

    return messages, "\n".join(part for part in prompt_text_parts if part)


def to_vllm_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from PIL import Image

    converted: list[dict[str, Any]] = []
    for message in messages:
        content: list[dict[str, Any]] = []
        for item in message.get("content", []):
            item_type = item.get("type")
            if item_type == "image":
                image = Image.open(item["image"]).convert("RGB")
                content.append({"type": "image_pil", "image_pil": image})
            else:
                content.append(item)
        converted.append({"role": message.get("role", "user"), "content": content})
    return converted


def analyze_generation(
    response: str,
    output_token_count: int | None,
    finish_reason: str | None,
    rendered_prompt_tail: str = "",
) -> dict[str, Any]:
    text = response or ""
    prompt_tail = rendered_prompt_tail or ""
    combined = (prompt_tail + text).lstrip()
    stripped = text.lstrip()
    combined_lower = combined.lower()
    template_prefilled_think = prompt_tail.rstrip().lower().endswith("<think>")
    think_close_pos = combined_lower.find("</think>")
    final_match = EARLY_FINAL_RE.search(combined)
    final_pos = final_match.start() if final_match else -1
    has_think_close = think_close_pos >= 0

    return {
        "output_token_count": int(output_token_count or 0),
        "finish_reason": finish_reason,
        "starts_with_think": stripped.lower().startswith("<think>"),
        "template_prefilled_think": template_prefilled_think,
        "combined_starts_with_think": combined_lower.startswith("<think>")
        or template_prefilled_think,
        "has_think_open": "<think>" in combined_lower,
        "has_think_close": has_think_close,
        "early_final_answer": final_match is not None,
        "final_answer_pos": final_pos,
        "think_close_pos": think_close_pos,
        "final_answer_before_think_close": bool(
            final_match is not None and has_think_close and final_pos < think_close_pos
        ),
        "truncated_by_length": str(finish_reason).lower() == "length",
    }


def analyze_response(
    response: str, output_token_count: int | None, finish_reason: str | None
) -> dict[str, Any]:
    return analyze_generation(response, output_token_count, finish_reason)


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


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(row.get("output_token_count") or 0) for row in records]
    finish_counts: dict[str, int] = {}
    for row in records:
        key = str(row.get("finish_reason"))
        finish_counts[key] = finish_counts.get(key, 0) + 1

    def count_flag(name: str) -> int:
        return sum(1 for row in records if row.get(name))

    return {
        "num_records": len(records),
        "output_tokens_mean": float(mean(lengths)) if lengths else 0.0,
        "output_tokens_min": min(lengths) if lengths else 0,
        "output_tokens_p50": percentile(lengths, 0.50),
        "output_tokens_p90": percentile(lengths, 0.90),
        "output_tokens_max": max(lengths) if lengths else 0,
        "starts_with_think_count": count_flag("starts_with_think"),
        "template_prefilled_think_count": count_flag("template_prefilled_think"),
        "combined_starts_with_think_count": count_flag("combined_starts_with_think"),
        "has_think_open_count": count_flag("has_think_open"),
        "has_think_close_count": count_flag("has_think_close"),
        "early_final_answer_count": count_flag("early_final_answer"),
        "final_answer_before_think_close_count": count_flag(
            "final_answer_before_think_close"
        ),
        "truncated_by_length_count": count_flag("truncated_by_length"),
        "finish_reason_counts": finish_counts,
    }


def preview_text(text: str, max_chars: int = 1600) -> str:
    text = text.replace("\r\n", "\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated preview]..."


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_markdown(
    path: Path, summary: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    lines = [
        "# Qwen3-VL CoT Generation Audit",
        "",
        "## Summary",
        "",
        f"- Records: {summary['num_records']}",
        f"- Output tokens mean/min/p50/p90/max: {summary['output_tokens_mean']:.1f} / {summary['output_tokens_min']} / {summary['output_tokens_p50']:.1f} / {summary['output_tokens_p90']:.1f} / {summary['output_tokens_max']}",
        f"- Starts with `<think>`: {summary['starts_with_think_count']}",
        f"- Template-prefilled `<think>`: {summary.get('template_prefilled_think_count', 0)}",
        f"- Combined starts with `<think>`: {summary.get('combined_starts_with_think_count', 0)}",
        f"- Has `<think>`: {summary['has_think_open_count']}",
        f"- Has `</think>`: {summary['has_think_close_count']}",
        f"- Early final-answer phrase: {summary['early_final_answer_count']}",
        f"- Final-answer phrase before `</think>`: {summary['final_answer_before_think_close_count']}",
        f"- Truncated by length: {summary['truncated_by_length_count']}",
        f"- Finish reasons: `{summary['finish_reason_counts']}`",
        "",
        "## Samples",
        "",
    ]
    for row in records[:20]:
        lines.extend(
            [
                f"### index={row.get('index')} question_id={row.get('question_id')} answer={row.get('answer')} tokens={row.get('output_token_count')}",
                "",
                "```text",
                preview_text(str(row.get("response", ""))),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_rows(input_path: Path, limit: int) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(input_path)
    if limit >= 0:
        df = df.head(limit)
    return df.to_dict(orient="records")


def run_generation(args: argparse.Namespace) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    rows = load_rows(Path(args.input), args.limit)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1, "video": 0},
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages, prompt_text = build_messages(row.get("prompt"), row.get("images"))
        rendered_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        rendered_prompt_tail = rendered_prompt[-200:]
        extra_info = _to_plain_dict(row.get("extra_info"))
        reward_model = _to_plain_dict(row.get("reward_model"))
        answer = (
            extra_info.get("answer")
            or reward_model.get("ground_truth")
            or extra_info.get("ground_truth")
        )
        image_paths = normalize_images(row.get("images"))

        outputs = llm.chat(
            to_vllm_messages(messages),
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        output = outputs[0].outputs[0]
        response = output.text
        token_ids = getattr(output, "token_ids", None)
        output_token_count = len(token_ids) if token_ids is not None else None
        finish_reason = getattr(output, "finish_reason", None)

        record = {
            "index": index,
            "question_id": str(extra_info.get("question_id", extra_info.get("source_index", index))),
            "answer": answer,
            "image_paths": image_paths,
            "prompt_text": prompt_text,
            "rendered_prompt_tail": rendered_prompt_tail,
            "response": response,
        }
        record.update(
            analyze_generation(
                response, output_token_count, finish_reason, rendered_prompt_tail
            )
        )
        records.append(record)
        print(
            "AUDIT_ROW",
            index,
            record["question_id"],
            "tokens",
            record["output_token_count"],
            "finish",
            record["finish_reason"],
            "think",
            record["starts_with_think"],
            "close",
            record["has_think_close"],
            flush=True,
        )

    return records


def main() -> None:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = run_generation(args)
    summary = summarize_records(records)

    write_jsonl(output_dir / "generations.jsonl", records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_markdown(output_dir / "summary.md", summary, records)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
