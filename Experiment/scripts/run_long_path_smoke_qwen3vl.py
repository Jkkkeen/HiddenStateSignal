#!/usr/bin/env python3
"""Long-CoT Step 3 pilot: think-segment chunk path dynamics.

This script forwards labeled Qwen3-VL Thinking rollouts, extracts only the
implicit think segment before `</think>`, computes 256-token chunk means for
layers 24/36, and immediately reduces them to path-dynamics scalars.  It does
not save full token hidden states or chunk hidden tensors.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_path_dynamics import (
    aucs_by_question,
    bootstrap_ci,
    evaluate_summary_features,
    path_dynamics_from_points,
    summarize_path_dynamics,
)


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
PRIMARY_FEATURES = [
    "path_length",
    "d_mean",
    "d_late_mean",
    "d_ratio_late_early",
    "pv_late_max",
    "d_hist_late_min",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-CoT path smoke on think segment.")
    parser.add_argument("--input", required=True, help="Labeled long rollout JSONL.")
    parser.add_argument("--output-dir", default="long_path_smoke")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--clean-only", action="store_true", default=True)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def select_rows(rows: list[dict[str, Any]], clean_only: bool, limit: int) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if clean_only and row.get("truncated"):
            continue
        if clean_only and not row.get("has_think_close"):
            continue
        if row.get("pred_answer") is None:
            continue
        selected.append(row)
    if limit is not None and limit >= 0:
        selected = selected[:limit]
    return selected


def image_part(image_path: str | None) -> dict[str, Any] | None:
    from PIL import Image

    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return {"type": "image", "image": Image.open(path).convert("RGB")}


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


def find_token_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(values):
        return -1
    last = len(values) - len(needle)
    for idx in range(last + 1):
        if values[idx : idx + len(needle)] == needle:
            return idx
    return -1


def segment_token_span_from_text(response_ids: list[int], close_tag_ids: list[int]) -> tuple[int, int, str]:
    close_idx = find_token_subsequence(response_ids, close_tag_ids)
    if close_idx >= 0:
        return 0, close_idx, "implicit_think_close_only"
    return 0, len(response_ids), "no_think_close"


def chunk_means_for_segment(
    hidden: np.ndarray,
    start: int,
    end: int,
    chunk_size: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim == 2:
        hidden = hidden[None, :, :]
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [layers,tokens,dim] or [tokens,dim], got {hidden.shape}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if end <= start:
        return np.empty((0, hidden.shape[0], hidden.shape[-1]), dtype=np.float32), []

    chunks = []
    spans = []
    rel_len = end - start
    for rel_start in range(0, rel_len, chunk_size):
        rel_end = min(rel_start + chunk_size, rel_len)
        full_start = start + rel_start
        full_end = start + rel_end
        chunks.append(hidden[:, full_start:full_end, :].mean(axis=1))
        spans.append((rel_start, rel_end))
    return np.stack(chunks, axis=0).astype(np.float32, copy=False), spans


def length_bin(value: int | float) -> str:
    v = float(value)
    if v < 2000:
        return "<2k"
    if v < 4000:
        return "2-4k"
    if v < 8000:
        return "4-8k"
    return "8k+"


def evaluate_by_length_bin(summary: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame(rows)
    for (layer, bin_name), group in summary.groupby(["layer", "think_length_bin"]):
        for feature in PRIMARY_FEATURES:
            if feature not in group.columns:
                continue
            aucs = aucs_by_question(group, feature)
            if aucs.size == 0:
                continue
            neg_aucs = 1.0 - aucs
            low, high = bootstrap_ci(aucs, n_boot, seed)
            nlow, nhigh = bootstrap_ci(neg_aucs, n_boot, seed)
            rows.append(
                {
                    "layer": int(layer),
                    "think_length_bin": bin_name,
                    "feature": feature,
                    "n_questions": int(aucs.size),
                    "mean_auc_pos": float(np.mean(aucs)),
                    "ci_low_pos": low,
                    "ci_high_pos": high,
                    "mean_auc_neg": float(np.mean(neg_aucs)),
                    "ci_low_neg": nlow,
                    "ci_high_neg": nhigh,
                    "best_auc": float(max(np.mean(aucs), np.mean(neg_aucs))),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    eval_df: pd.DataFrame,
    bin_eval: pd.DataFrame,
    skipped: int,
) -> Path:
    report = output_dir / "LONG_PATH_SMOKE_RESULTS.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Long-CoT Path Smoke Results\n\n")
        f.write("This pilot computes path dynamics on the implicit think segment before `</think>`.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Rollout-layer rows: {len(summary)}\n")
        f.write(f"- Rollouts: {summary[['question_id', 'rollout_id']].drop_duplicates().shape[0] if not summary.empty else 0}\n")
        f.write(f"- Questions: {summary['question_id'].nunique() if not summary.empty else 0}\n")
        f.write(f"- Skipped rollouts: {skipped}\n")
        if not summary.empty:
            f.write(f"- Mean think tokens: {summary['think_length'].mean():.1f}\n")
            f.write(f"- Mean chunks: {summary['num_chunks'].mean():.2f}\n")

        f.write("\n## Primary Path Metrics\n\n")
        f.write("| layer | feature | questions | AUROC(+feature) | AUROC(-feature) | best AUROC |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        top = eval_df[eval_df["feature"].isin(PRIMARY_FEATURES)].copy()
        if not top.empty:
            top = top.sort_values("best_auc", ascending=False)
        for _, row in top.iterrows():
            f.write(
                f"| {int(row['layer'])} | {row['feature']} | {int(row['n_questions'])} | "
                f"{row['mean_auc_pos']:.4f} | {row['mean_auc_neg']:.4f} | {row['best_auc']:.4f} |\n"
            )

        f.write("\n## Length-Bin AUROC\n\n")
        f.write("| layer | bin | feature | questions | AUROC(+feature) | AUROC(-feature) | best AUROC |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|\n")
        view = bin_eval[bin_eval["feature"].isin(["path_length", "d_mean", "d_late_mean"])].copy()
        if not view.empty:
            view = view.sort_values(["layer", "feature", "think_length_bin"])
        for _, row in view.iterrows():
            f.write(
                f"| {int(row['layer'])} | {row['think_length_bin']} | {row['feature']} | "
                f"{int(row['n_questions'])} | {row['mean_auc_pos']:.4f} | "
                f"{row['mean_auc_neg']:.4f} | {row['best_auc']:.4f} |\n"
            )
    return report


def main() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    layers = parse_layers(args.layers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)

    rows = select_rows(load_jsonl(Path(args.input)), args.clean_only, args.limit)
    if not rows:
        raise ValueError("No rows selected.")

    print("Long-CoT path smoke")
    print(f"input: {args.input}")
    print(f"selected clean rollouts: {len(rows)}")
    print(f"output_dir: {output_dir}")
    print(f"model: {args.model}")
    print(f"layers: {layers}")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    tokenizer = getattr(processor, "tokenizer", processor)
    close_tag_ids = tokenizer.encode("</think>", add_special_tokens=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    model.eval()
    device = next(model.parameters()).device

    dynamics_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    skipped = 0

    for row in tqdm(rows, desc="long path"):
        try:
            prompt_inputs = encode_messages(
                processor, build_prompt_messages(row), add_generation_prompt=True
            )
            full_inputs = encode_messages(
                processor, build_full_messages(row), add_generation_prompt=False
            )
            response_start = common_prefix_len(prompt_inputs["input_ids"], full_inputs["input_ids"])
            response_end = int(full_inputs["input_ids"].shape[-1])
            response_ids = full_inputs["input_ids"][0, response_start:response_end].detach().cpu().tolist()
            seg_start, seg_end, token_segment_status = segment_token_span_from_text(
                response_ids, close_tag_ids
            )
            think_len = seg_end - seg_start
            if think_len < args.chunk_size * 2:
                raise ValueError(f"think segment too short: {think_len}")

            full_inputs = to_device(full_inputs, device)
            with torch.inference_mode():
                outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)

            selected_layers = []
            n_layers = len(outputs.hidden_states)
            for layer in layers:
                if layer < 0 or layer >= n_layers:
                    raise ValueError(f"Layer {layer} out of range; n_layers={n_layers}")
                hidden = (
                    outputs.hidden_states[layer][0, response_start:response_end]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                selected_layers.append(hidden)
            stacked = np.stack(selected_layers, axis=0)
            chunk_array, spans = chunk_means_for_segment(
                stacked,
                start=seg_start,
                end=seg_end,
                chunk_size=args.chunk_size,
            )
            if chunk_array.shape[0] < 2:
                raise ValueError(f"not enough chunks: {chunk_array.shape[0]}")

            base = {
                "question_id": str(row.get("question_id", "")),
                "rollout_id": int(row.get("rollout_id", -1)),
                "is_correct": bool(row.get("is_correct", False)),
                "answer": str(row.get("answer", "")),
                "pred_answer": str(row.get("pred_answer", "")),
                "think_length": int(think_len),
                "think_length_bin": length_bin(think_len),
                "response_length": int(row.get("response_token_count", response_end - response_start)),
                "answer_length": int(row.get("answer_token_count", 0)),
                "num_chunks": int(chunk_array.shape[0]),
                "token_segment_status": token_segment_status,
                "text_segment_status": str(row.get("segment_status", "")),
                "truncated": bool(row.get("truncated", False)),
            }
            chunk_starts = [start for start, _ in spans]
            chunk_ends = [end for _, end in spans]

            for layer_pos, layer in enumerate(layers):
                points = chunk_array[:, layer_pos, :]
                dyn = path_dynamics_from_points(points)
                d = dyn["d"]
                pv = dyn["pv"]
                pa = dyn["pa"]
                if d.size == 0:
                    continue
                layer_base = {**base, "layer": int(layer), "n_steps": int(d.size)}
                summary_rows.append({**layer_base, **summarize_path_dynamics(d)})
                for step_id, d_value in enumerate(d):
                    dynamics_rows.append(
                        {
                            **layer_base,
                            "step_id": int(step_id),
                            "from_chunk_start": int(chunk_starts[step_id]),
                            "from_chunk_end": int(chunk_ends[step_id]),
                            "to_chunk_start": int(chunk_starts[step_id + 1]),
                            "to_chunk_end": int(chunk_ends[step_id + 1]),
                            "relative_pos": float(step_id / max(d.size - 1, 1)),
                            "d": float(d_value),
                            "pv": float(pv[step_id - 1])
                            if step_id >= 1 and step_id - 1 < pv.size
                            else np.nan,
                            "pa": float(pa[step_id - 2])
                            if step_id >= 2 and step_id - 2 < pa.size
                            else np.nan,
                        }
                    )

            del outputs, full_inputs, prompt_inputs, stacked, chunk_array
        except Exception as exc:
            skipped += 1
            print(
                f"[skip] question={row.get('question_id')} rollout={row.get('rollout_id')} reason={exc}",
                flush=True,
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    dynamics = pd.DataFrame(dynamics_rows)
    summary = pd.DataFrame(summary_rows)
    eval_df = evaluate_summary_features(summary, args.bootstrap, args.seed) if not summary.empty else pd.DataFrame()
    bin_eval = evaluate_by_length_bin(summary, args.bootstrap, args.seed) if not summary.empty else pd.DataFrame()

    dynamics.to_parquet(output_dir / "long_path_dynamics.parquet", index=False)
    summary.to_parquet(output_dir / "long_path_features.parquet", index=False)
    eval_df.to_parquet(output_dir / "long_path_eval.parquet", index=False)
    bin_eval.to_parquet(output_dir / "long_path_length_bin_eval.parquet", index=False)
    report = write_report(output_dir, summary, eval_df, bin_eval, skipped)

    print(f"saved dynamics: {output_dir / 'long_path_dynamics.parquet'} ({len(dynamics)} rows)")
    print(f"saved summary: {output_dir / 'long_path_features.parquet'} ({len(summary)} rows)")
    print(f"saved eval: {output_dir / 'long_path_eval.parquet'} ({len(eval_df)} rows)")
    print(f"saved bin eval: {output_dir / 'long_path_length_bin_eval.parquet'} ({len(bin_eval)} rows)")
    print(f"saved report: {report}")
    if not eval_df.empty:
        print(eval_df.head(20))


if __name__ == "__main__":
    main()
