#!/usr/bin/env python3
"""Experiment D: local chunk angular dynamics from token-level Qwen3-VL hidden states.

This script forwards each rollout, pools token-level hidden states into
micro-windows, computes within-chunk angular metrics, and saves scalar metrics
only. It does not save full token hidden states.
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


def angular_metrics_from_windows(
    windows: np.ndarray,
    angle_bins: int = 6,
    min_angles_for_entropy: int = 16,
) -> dict[str, float | int]:
    """Compute local angular metrics from micro-window vectors.

    windows: [n_windows, hidden_dim]. Angles are computed between adjacent
    displacement vectors, so n_angles = n_windows - 2.
    """
    windows = np.asarray(windows, dtype=np.float32)
    empty = {
        "n_windows": int(windows.shape[0]) if windows.ndim >= 1 else 0,
        "n_angles": 0,
        "lad_mean": np.nan,
        "lad_std": np.nan,
        "lad_p90": np.nan,
        "lad_max": np.nan,
        "theta_minus_pi2_mean": np.nan,
        "cos_mean": np.nan,
        "cos_std": np.nan,
        "cos_p10": np.nan,
        "cos_min": np.nan,
        "spike_rate_90": np.nan,
        "spike_rate_105": np.nan,
        "ae_norm_b6": np.nan,
    }
    if windows.ndim != 2 or windows.shape[0] < 3:
        return empty

    disp = windows[1:] - windows[:-1]
    v0 = disp[:-1]
    v1 = disp[1:]
    denom = np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1)
    valid = denom > EPS
    if not np.any(valid):
        return empty

    cos = np.sum(v0[valid] * v1[valid], axis=1) / np.maximum(denom[valid], EPS)
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)

    metrics = dict(empty)
    metrics["n_angles"] = int(theta.size)
    metrics["lad_mean"] = float(np.mean(theta))
    metrics["lad_std"] = float(np.std(theta))
    metrics["lad_p90"] = float(np.percentile(theta, 90))
    metrics["lad_max"] = float(np.max(theta))
    metrics["theta_minus_pi2_mean"] = float(np.mean(theta - (math.pi / 2)))
    metrics["cos_mean"] = float(np.mean(cos))
    metrics["cos_std"] = float(np.std(cos))
    metrics["cos_p10"] = float(np.percentile(cos, 10))
    metrics["cos_min"] = float(np.min(cos))
    metrics["spike_rate_90"] = float(np.mean(theta > (math.pi / 2)))
    metrics["spike_rate_105"] = float(np.mean(theta > ((math.pi / 2) + (math.pi / 12))))

    if theta.size >= min_angles_for_entropy and angle_bins > 1:
        counts, _ = np.histogram(theta, bins=angle_bins, range=(0.0, math.pi))
        total = counts.sum()
        if total > 0:
            probs = counts[counts > 0] / total
            entropy = -float(np.sum(probs * np.log(np.clip(probs, EPS, None))))
            metrics["ae_norm_b6"] = float(entropy / math.log(angle_bins))

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local angular dynamics for Qwen3-VL.")
    parser.add_argument(
        "--input",
        default="data/rollouts_mathverse_full_cot_labeled_mcq_mixed.jsonl",
    )
    parser.add_argument("--output", default="angular_results/local_angular.parquet")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--micro-window", type=int, default=8)
    parser.add_argument("--angle-bins", type=int, default=6)
    parser.add_argument("--min-angles-for-entropy", type=int, default=16)
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


def micro_window_means(hidden: Any, micro_window: int) -> np.ndarray:
    """Pool [tokens, dim] torch tensor into non-overlapping micro-window means."""
    if hidden.shape[0] < micro_window:
        return np.empty((0, int(hidden.shape[-1])), dtype=np.float32)
    n_windows = int(hidden.shape[0]) // micro_window
    trimmed = hidden[: n_windows * micro_window]
    windows = trimmed.float().reshape(n_windows, micro_window, hidden.shape[-1]).mean(dim=1)
    return windows.cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    if args.chunk_size <= 0 or args.micro_window <= 0:
        raise ValueError("--chunk-size and --micro-window must be positive.")
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

    print("Experiment D: Local Chunk Angular Dynamics")
    print(f"input: {input_path}")
    print(f"selected rollouts: {len(rows)}")
    print(f"output: {output_path}")
    print(f"model: {args.model}")
    print(f"layers: {layers}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"micro_window: {args.micro_window}")

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

    for row in tqdm(rows, desc="local angular"):
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
                    windows = micro_window_means(chunk_hidden, args.micro_window)
                    metrics = angular_metrics_from_windows(
                        windows,
                        angle_bins=args.angle_bins,
                        min_angles_for_entropy=args.min_angles_for_entropy,
                    )
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
                            "is_correct": bool(row.get("is_correct", False)),
                            "response_length": response_length,
                            "num_chunks": num_chunks,
                            "answer": str(row.get("answer", "")),
                            "pred_answer": str(row.get("pred_answer", "")),
                            "model": args.model,
                            "chunk_size": args.chunk_size,
                            "micro_window": args.micro_window,
                            "angle_bins": args.angle_bins,
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
        cols = [
            "cos_mean",
            "cos_p10",
            "spike_rate_90",
            "lad_mean",
            "ae_norm_b6",
        ]
        print(df[cols].describe())


if __name__ == "__main__":
    main()
