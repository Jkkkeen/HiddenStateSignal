#!/usr/bin/env python3
"""Experiment G: cross-chunk angular dynamics from Qwen3-VL hidden states.

This script forwards each labeled rollout once, pools the response hidden states
into medium-grain trajectory points, and saves rollout-level angular metrics.
It does not save token-level hidden states.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment G cross-chunk angular dynamics.")
    parser.add_argument("--input", default="data/rollouts_mathverse_full_cot_labeled_mcq_mixed.jsonl")
    parser.add_argument("--output", default="cross_chunk_angular_results/cross_chunk_angular.parquet")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--window-sizes", default="32,64,128")
    parser.add_argument("--pools", default="mean,last")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model and processor from local cache/path without any HuggingFace network calls.",
    )
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
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
    from PIL import Image

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


def to_device(batch: Any, device: Any) -> Any:
    import torch

    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def common_prefix_len(a: Any, b: Any) -> int:
    a_ids = a[0].detach().cpu().tolist()
    b_ids = b[0].detach().cpu().tolist()
    limit = min(len(a_ids), len(b_ids))
    idx = 0
    while idx < limit and a_ids[idx] == b_ids[idx]:
        idx += 1
    return idx


def pool_points(hidden: np.ndarray, window_size: int, pool: str) -> np.ndarray:
    """Pool [tokens, dim] hidden into medium-grain trajectory points."""
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be [tokens, dim], got {hidden.shape}")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if pool not in {"mean", "last"}:
        raise ValueError(f"pool must be mean or last, got {pool}")
    if hidden.shape[0] < window_size:
        return np.empty((0, hidden.shape[-1]), dtype=np.float32)
    n_windows = hidden.shape[0] // window_size
    trimmed = hidden[: n_windows * window_size].reshape(n_windows, window_size, hidden.shape[-1])
    if pool == "mean":
        return trimmed.mean(axis=1).astype(np.float32, copy=False)
    return trimmed[:, -1, :].astype(np.float32, copy=False)


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def late_values(values: np.ndarray) -> np.ndarray:
    values = finite(values)
    if values.size == 0:
        return values
    start = int(math.floor(0.75 * values.size))
    start = min(start, values.size - 1)
    return values[start:]


def cross_chunk_angular_metrics(points: np.ndarray) -> dict[str, float | int]:
    """Compute angular metrics over medium-grain trajectory points."""
    points = np.asarray(points, dtype=np.float32)
    empty: dict[str, float | int] = {
        "n_points": int(points.shape[0]) if points.ndim >= 1 else 0,
        "n_displacements": 0,
        "n_angles": 0,
        "gcos_mean": np.nan,
        "gcos_std": np.nan,
        "gcos_p10": np.nan,
        "gcos_min": np.nan,
        "gcos_late_mean": np.nan,
        "gcos_late_min": np.nan,
        "gtheta_mean": np.nan,
        "gtheta_std": np.nan,
        "gtheta_late_mean": np.nan,
        "gspike_rate_90": np.nan,
        "gstep_mean": np.nan,
        "gstep_late_mean": np.nan,
    }
    if points.ndim != 2 or points.shape[0] < 3:
        return empty

    disp = points[1:] - points[:-1]
    step_norm = np.linalg.norm(disp, axis=1)
    v0 = disp[:-1]
    v1 = disp[1:]
    denom = np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1)
    valid = denom > EPS
    if not np.any(valid):
        metrics = dict(empty)
        metrics["n_displacements"] = int(disp.shape[0])
        return metrics

    cos = np.sum(v0[valid] * v1[valid], axis=1) / np.maximum(denom[valid], EPS)
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)
    late_cos = late_values(cos)
    late_theta = late_values(theta)
    late_step = late_values(step_norm)

    return {
        "n_points": int(points.shape[0]),
        "n_displacements": int(disp.shape[0]),
        "n_angles": int(theta.size),
        "gcos_mean": float(np.mean(cos)),
        "gcos_std": float(np.std(cos)),
        "gcos_p10": float(np.percentile(cos, 10)),
        "gcos_min": float(np.min(cos)),
        "gcos_late_mean": float(np.mean(late_cos)) if late_cos.size else np.nan,
        "gcos_late_min": float(np.min(late_cos)) if late_cos.size else np.nan,
        "gtheta_mean": float(np.mean(theta)),
        "gtheta_std": float(np.std(theta)),
        "gtheta_late_mean": float(np.mean(late_theta)) if late_theta.size else np.nan,
        "gspike_rate_90": float(np.mean(theta > (math.pi / 2))),
        "gstep_mean": float(np.mean(step_norm)) if step_norm.size else np.nan,
        "gstep_late_mean": float(np.mean(late_step)) if late_step.size else np.nan,
    }


def main() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    layers = parse_ints(args.layers)
    window_sizes = parse_ints(args.window_sizes)
    pools = parse_strings(args.pools)
    if not layers or not window_sizes or not pools:
        raise ValueError("--layers, --window-sizes, and --pools must not be empty.")
    if any(w <= 0 for w in window_sizes):
        raise ValueError("--window-sizes must be positive.")
    for pool in pools:
        if pool not in {"mean", "last"}:
            raise ValueError("--pools may only contain mean,last")

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = select_rows(load_jsonl(input_path), args)
    if not rows:
        raise ValueError("No rows selected.")

    print("Experiment G: Cross-Chunk Angular Dynamics")
    print(f"input: {input_path}")
    print(f"selected rollouts: {len(rows)}")
    print(f"output: {output_path}")
    print(f"model: {args.model}")
    print(f"layers: {layers}")
    print(f"window_sizes: {window_sizes}")
    print(f"pools: {pools}")

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    model.eval()
    device = next(model.parameters()).device

    metric_rows: list[dict[str, Any]] = []
    skipped = 0

    for row in tqdm(rows, desc="cross-chunk angular"):
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
            for layer in layers:
                if layer < 0 or layer >= n_layers:
                    raise ValueError(f"Layer {layer} out of range; n_layers={n_layers}")
                response_hidden = (
                    hidden_states[layer][0, response_start:response_end]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                for window_size in window_sizes:
                    for pool in pools:
                        points = pool_points(response_hidden, window_size, pool)
                        metrics = cross_chunk_angular_metrics(points)
                        metric_rows.append(
                            {
                                "question_id": str(row.get("question_id", "")),
                                "rollout_id": int(row.get("rollout_id", -1)),
                                "source_line": int(row.get("_source_line", -1)),
                                "layer": int(layer),
                                "window_size": int(window_size),
                                "pool": pool,
                                "is_correct": bool(row.get("is_correct", False)),
                                "response_length": int(response_length),
                                "answer": str(row.get("answer", "")),
                                "pred_answer": str(row.get("pred_answer", "")),
                                "model": args.model,
                                **metrics,
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
        print(df[["gcos_mean", "gcos_late_mean", "gcos_p10", "gtheta_late_mean"]].describe())


if __name__ == "__main__":
    main()
