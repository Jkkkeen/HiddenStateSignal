#!/usr/bin/env python3
"""Generate multiple Qwen3-VL rollouts for MathVerse queries with vLLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams


QUERY_FIELDS = (
    "query_wo",
    "query",
    "question",
    "problem",
    "prompt",
)

ANSWER_FIELDS = (
    "answer",
    "gt_answer",
    "ground_truth",
    "solution",
    "target",
)

IMAGE_FIELDS = (
    "image",
    "image_path",
    "image_file",
    "image_name",
    "img",
    "figure_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out Qwen3-VL on MathVerse.")
    parser.add_argument("--data-dir", default="data/mathverse")
    parser.add_argument("--split", default="testmini")
    parser.add_argument(
        "--output",
        default="data/rollouts_mathverse_smoke.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--rollouts", default=8, type=int)
    parser.add_argument("--limit", default=5, type=int, help="Use -1 for all records.")
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--max-tokens", default=1024, type=int)
    parser.add_argument("--max-model-len", default=32768, type=int)
    parser.add_argument("--gpu-memory-utilization", default=0.85, type=float)
    parser.add_argument(
        "--query-field",
        default="query_wo",
        help="Preferred query field; falls back to common alternatives.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records whose question_id already appears in the output file.",
    )
    parser.add_argument(
        "--question-ids",
        default="",
        help="Optional newline-separated question_id file. If set, only these records are rolled out.",
    )
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    return parser.parse_args()


def read_records(data_dir: Path, split: str) -> list[dict[str, Any]]:
    json_path = data_dir / f"{split}.json"
    parquet_path = data_dir / f"{split}.parquet"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("data", "examples", "records"):
                if isinstance(data.get(key), list):
                    return data[key]
            return list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported JSON structure in {json_path}")

    if parquet_path.exists():
        return pd.read_parquet(parquet_path).to_dict(orient="records")

    raise FileNotFoundError(f"Cannot find {json_path} or {parquet_path}")


def first_present(record: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = record.get(field)
        if value is not None and value != "":
            return value
    return None


def get_query(record: dict[str, Any], preferred_field: str) -> str:
    preferred = record.get(preferred_field)
    if preferred:
        return str(preferred)
    value = first_present(record, QUERY_FIELDS)
    if value is None:
        raise KeyError(f"No query field found. Available keys: {sorted(record.keys())}")
    return str(value)


def get_question_id(record: dict[str, Any], index: int) -> str:
    for field in ("question_id", "sample_index", "id", "uid", "problem_id"):
        value = record.get(field)
        if value is not None and value != "":
            return str(value)
    return f"mathverse_{index:06d}"


def normalize_image_value(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "file_name", "filename", "name"):
            if value.get(key):
                return str(value[key])
    return None


def find_image_path(record: dict[str, Any], data_dir: Path) -> Path | None:
    images_dir = data_dir / "images"
    candidates: list[Path] = []

    for field in IMAGE_FIELDS:
        image_value = normalize_image_value(record.get(field))
        if image_value:
            raw = Path(image_value)
            candidates.extend(
                [
                    raw,
                    data_dir / raw,
                    images_dir / raw,
                    images_dir / raw.name,
                ]
            )

    for field in ("sample_index", "question_id", "id"):
        value = record.get(field)
        if value is None or value == "":
            continue
        stem = str(value)
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            candidates.append(images_dir / f"{stem}{suffix}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def already_done(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("question_id") is not None:
                done.add(str(row["question_id"]))
    return done


def read_question_ids(path: str) -> set[str]:
    if not path:
        return set()
    with Path(path).open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def build_messages(query: str, image_path: Path | None) -> list[dict[str, Any]]:
    text = (
        "Solve the following MathVerse problem. Give concise reasoning and the final answer.\n\n"
        f"{query}"
    )
    if image_path is None:
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    image = Image.open(image_path).convert("RGB")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": text},
            ],
        }
    ]


def main() -> None:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_records(data_dir, args.split)
    keep_ids = read_question_ids(args.question_ids)
    if keep_ids:
        records = [
            record
            for index, record in enumerate(records)
            if get_question_id(record, index) in keep_ids
        ]
    if args.limit is not None and args.limit >= 0:
        records = records[: args.limit]

    skip_ids = already_done(output_path) if args.resume else set()
    print(f"Loaded {len(records)} records from {data_dir}")
    if skip_ids:
        print(f"Resume enabled: skipping {len(skip_ids)} completed question ids")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    sampling_params = SamplingParams(
        n=args.rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    with output_path.open("a", encoding="utf-8") as out:
        for index, record in enumerate(records):
            question_id = get_question_id(record, index)
            if question_id in skip_ids:
                continue

            query = get_query(record, args.query_field)
            image_path = find_image_path(record, data_dir)
            messages = build_messages(query, image_path)
            outputs = llm.chat(messages, sampling_params=sampling_params)
            completions = outputs[0].outputs

            answer = first_present(record, ANSWER_FIELDS)
            for rollout_id, completion in enumerate(completions):
                token_ids = getattr(completion, "token_ids", None)
                output_token_count = len(token_ids) if token_ids is not None else None
                finish_reason = getattr(completion, "finish_reason", None)
                stop_reason = getattr(completion, "stop_reason", None)
                row = {
                    "dataset": "MathVerse",
                    "split": args.split,
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "prompt": query,
                    "response": completion.text,
                    "answer": answer,
                    "image_path": str(image_path) if image_path else None,
                    "model": args.model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                    "output_token_count": output_token_count,
                    "truncated_by_length": bool(
                        finish_reason == "length"
                        or (
                            output_token_count is not None
                            and output_token_count >= args.max_tokens
                        )
                    ),
                    "source_index": index,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

            print(
                f"Wrote {len(completions)} rollouts for {question_id} "
                f"(image={'yes' if image_path else 'no'})"
            )


if __name__ == "__main__":
    main()
