#!/usr/bin/env python3
"""Smoke test for local effective-rank metrics on response hidden states."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute chunk-level effective rank on response hidden states."
    )
    parser.add_argument("--input", default="data/smoke.jsonl", help="Input JSONL file.")
    parser.add_argument(
        "--output",
        default="outputs/local_er_smoke.parquet",
        help="Output parquet file.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or path.")
    parser.add_argument(
        "--layer",
        default=-1,
        type=int,
        help="Hidden-state layer index. -1 means last layer.",
    )
    parser.add_argument(
        "--chunk-size",
        default=8,
        type=int,
        help="Number of response tokens per ER chunk.",
    )
    parser.add_argument(
        "--hf-cache",
        default="/data2/hjk/models/huggingface",
        help="Hugging Face cache directory on the data disk.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint or mirror.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {line}") from exc
    return rows


def effective_rank(hidden: torch.Tensor) -> float:
    """Entropy effective rank from singular values of [tokens, hidden_dim]."""
    if hidden.ndim != 2:
        raise ValueError(f"Expected 2D hidden tensor, got shape {tuple(hidden.shape)}")
    if hidden.shape[0] == 0:
        return float("nan")
    if hidden.shape[0] == 1:
        return 1.0

    centered = hidden.float()
    centered = centered - centered.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    total = singular_values.sum()
    if total <= 0:
        return 0.0

    probs = singular_values / total
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).item())


def response_token_span(tokenizer: Any, prompt: str, full_text: str) -> tuple[int, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    start = min(len(prompt_ids), len(full_ids))
    return start, len(full_ids)


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.hf_cache)
    Path(args.hf_cache).mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    metric_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            prompt = record["prompt"]
            response = record["response"]
            full_text = prompt + response

            inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

            hidden_states = outputs.hidden_states
            layer_index = args.layer if args.layer >= 0 else len(hidden_states) + args.layer
            if layer_index < 0 or layer_index >= len(hidden_states):
                raise ValueError(
                    f"Layer index {args.layer} is out of range for "
                    f"{len(hidden_states)} hidden-state tensors."
                )

            layer_hidden = hidden_states[layer_index][0]
            response_start, response_end = response_token_span(tokenizer, prompt, full_text)
            response_hidden = layer_hidden[response_start:response_end]
            response_length = int(response_hidden.shape[0])

            num_chunks = max(1, math.ceil(response_length / args.chunk_size))
            for chunk_id in range(num_chunks):
                chunk_start = chunk_id * args.chunk_size
                chunk_end = min((chunk_id + 1) * args.chunk_size, response_length)
                chunk_hidden = response_hidden[chunk_start:chunk_end]
                er_local = effective_rank(chunk_hidden)
                metric_rows.append(
                    {
                        "question_id": record.get("question_id"),
                        "rollout_id": record.get("rollout_id"),
                        "chunk_id": chunk_id,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "layer": layer_index,
                        "er_local": er_local,
                        "is_correct": record.get("is_correct"),
                        "response_length": response_length,
                        "model": args.model,
                    }
                )

            del outputs, hidden_states, layer_hidden, response_hidden
            torch.cuda.empty_cache()

    df = pd.DataFrame(metric_rows)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
