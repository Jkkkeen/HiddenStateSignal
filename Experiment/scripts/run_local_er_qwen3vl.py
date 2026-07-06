#!/usr/bin/env python3
"""Experiment A: per-rollout local ER from token-level Qwen3-VL hidden states.

This script forwards each rollout, extracts token-level hidden states for
selected layers, computes Effective Rank for each response chunk online, and
saves only scalar metrics.  It does not save full token hidden states.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local ER for Qwen3-VL rollouts.")
    parser.add_argument(
        "--input",
        default="data/rollouts_mathverse_full_cot_labeled_mcq_mixed.jsonl",
    )
    parser.add_argument("--output", default="er_local_results/local_er.parquet")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")

    end = len(rows) if args.end < 0 else min(args.end, len(rows))
    selected = rows[max(args.start, 0) : end]
    if args.num_shards > 1:
        selected = [
            row for pos, row in enumerate(selected) if pos % args.num_shards == args.shard_index
        ]
    if args.limit is not None and args.limit >= 0:
        selected = selected[: args.limit]
    return selected


def image_part(image_path: str | None) -> dict[str, Any] | None:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(path).convert("RGB")
    return {"type": "image", "image": image}


def build_prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = (
        "Solve the following MathVerse problem. Give concise reasoning and the final answer.\n\n"
        f"{row.get('prompt', '')}"
    )
    content: list[dict[str, Any]] = []
    part = image_part(row.get("image_path"))
    if part is not None:
        content.append(part)
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def build_full_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = build_prompt_messages(row)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(row.get("response", ""))}],
        }
    )
    return messages


def encode_messages(processor: Any, messages: list[dict[str, Any]], add_generation_prompt: bool) -> Any:
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def to_device(batch: Any, device: torch.device | str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    a_ids = a[0].detach().cpu().tolist()
    b_ids = b[0].detach().cpu().tolist()
    limit = min(len(a_ids), len(b_ids))
    idx = 0
    while idx < limit and a_ids[idx] == b_ids[idx]:
        idx += 1
    return idx


def effective_rank(hidden: torch.Tensor, center: bool) -> float:
    """Entropy effective rank from token-level hidden matrix [tokens, dim]."""
    if hidden.ndim != 2:
        raise ValueError(f"Expected [tokens, hidden_dim], got {tuple(hidden.shape)}")
    n_tokens = int(hidden.shape[0])
    if n_tokens <= 0:
        return float("nan")
    if n_tokens == 1:
        return 0.0 if center else 1.0

    x = hidden.float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)

    gram = x @ x.transpose(0, 1)
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0)
    singular_values = torch.sqrt(eigvals)
    total = singular_values.sum()
    if float(total.item()) <= EPS:
        return 0.0

    probs = singular_values / total
    entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum()
    return float(torch.exp(entropy).item())


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    if not layers:
        raise ValueError("--layers must not be empty.")

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = select_rows(load_jsonl(input_path), args)
    if not rows:
        raise ValueError("No rows selected.")

    print("Experiment A: Local ER")
    print(f"input: {input_path}")
    print(f"selected rollouts: {len(rows)}")
    print(f"output: {output_path}")
    print(f"model: {args.model}")
    print(f"layers: {layers}")
    print(f"chunk_size: {args.chunk_size}")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    device = next(model.parameters()).device

    metric_rows: list[dict[str, Any]] = []
    skipped = 0

    for row in tqdm(rows, desc="local ER"):
        try:
            prompt_inputs = encode_messages(
                processor, build_prompt_messages(row), add_generation_prompt=True
            )
            full_inputs = encode_messages(
                processor, build_full_messages(row), add_generation_prompt=False
            )
            response_start = common_prefix_len(prompt_inputs["input_ids"], full_inputs["input_ids"])
            response_end = int(full_inputs["input_ids"].shape[-1])
            response_length = response_end - response_start
            if response_length <= 0:
                raise ValueError(
                    f"No response tokens for question={row.get('question_id')} "
                    f"rollout={row.get('rollout_id')}"
                )

            full_inputs = to_device(full_inputs, device)
            with torch.inference_mode():
                outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)

            hidden_states = outputs.hidden_states
            n_layers = len(hidden_states)
            num_chunks = (response_length + args.chunk_size - 1) // args.chunk_size

            for layer in layers:
                if layer < 0 or layer >= n_layers:
                    raise ValueError(f"Layer {layer} out of range; n_layers={n_layers}")
                layer_hidden = hidden_states[layer][0]
                for chunk_id in range(num_chunks):
                    chunk_start = chunk_id * args.chunk_size
                    chunk_end = min((chunk_id + 1) * args.chunk_size, response_length)
                    full_start = response_start + chunk_start
                    full_end = response_start + chunk_end
                    chunk_hidden = layer_hidden[full_start:full_end]
                    er_raw = effective_rank(chunk_hidden, center=False)
                    er_centered = effective_rank(chunk_hidden, center=True)
                    metric_rows.append(
                        {
                            "question_id": str(row.get("question_id", "")),
                            "rollout_id": int(row.get("rollout_id", -1)),
                            "source_line": int(row.get("_source_line", -1)),
                            "chunk_id": chunk_id,
                            "chunk_start": chunk_start,
                            "chunk_end": chunk_end,
                            "relative_pos": chunk_start / max(response_length, 1),
                            "layer": layer,
                            "er_raw": er_raw,
                            "er_centered": er_centered,
                            "is_correct": bool(row.get("is_correct", False)),
                            "response_length": response_length,
                            "num_chunks": num_chunks,
                            "answer": str(row.get("answer", "")),
                            "pred_answer": str(row.get("pred_answer", "")),
                            "model": args.model,
                            "chunk_size": args.chunk_size,
                        }
                    )

            del outputs, hidden_states, full_inputs, prompt_inputs
        except Exception as exc:
            skipped += 1
            print(
                f"[skip] question={row.get('question_id')} rollout={row.get('rollout_id')} "
                f"reason={exc}",
                flush=True,
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(metric_rows)
    df.to_parquet(output_path, index=False)
    print(f"saved: {output_path}")
    print(f"rows: {len(df)}")
    print(f"skipped_rollouts: {skipped}")
    if not df.empty:
        print(df.head())
        print(df[["er_raw", "er_centered"]].describe())


if __name__ == "__main__":
    main()
