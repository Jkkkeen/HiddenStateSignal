#!/usr/bin/env python3
"""Long-CoT trajectory smoke: online E + G + D metrics on think segment.

One expensive transformers forward per rollout, then:
- E path dynamics from 256-token chunk means.
- G macro angular dynamics from w={32,64,128}, pool={mean,last}.
- D local angular dynamics inside 256-token chunks with micro-window=8.

The script saves scalar parquet files only; it does not save full hidden states.
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

from run_cross_chunk_angular_qwen3vl import cross_chunk_angular_metrics, pool_points
from run_local_angular_qwen3vl import angular_metrics_from_windows
from run_long_path_smoke_qwen3vl import (
    build_full_messages,
    build_prompt_messages,
    chunk_means_for_segment,
    common_prefix_len,
    encode_messages,
    evaluate_by_length_bin,
    find_token_subsequence,
    length_bin,
    load_jsonl,
    parse_layers,
    segment_token_span_from_text,
    select_rows,
    to_device,
)
from run_path_dynamics import (
    aucs_by_question,
    bootstrap_ci,
    path_dynamics_from_points,
    summarize_path_dynamics,
)


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
PRIMARY_PATH_FEATURES = [
    "path_length",
    "d_mean",
    "d_late_mean",
    "d_ratio_late_early",
    "pv_late_max",
    "d_hist_late_min",
]
PRIMARY_G_FEATURES = ["gcos_mean", "gcos_p10", "gcos_min", "gcos_late_mean"]
PRIMARY_D_FEATURES = ["cos_mean", "cos_p10", "spike_rate_90"]
PATH_ID_COLUMNS = {
    "question_id",
    "rollout_id",
    "layer",
    "is_correct",
    "answer",
    "pred_answer",
    "think_length",
    "think_length_bin",
    "response_length",
    "answer_length",
    "num_chunks",
    "n_steps",
    "token_segment_status",
    "text_segment_status",
    "truncated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-CoT trajectory smoke metrics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="long_trajectory_smoke")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--micro-window", type=int, default=8)
    parser.add_argument("--window-sizes", default="32,64,128")
    parser.add_argument("--pools", default="mean,last")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--clean-only", action="store_true", default=True)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def micro_window_points(chunk_hidden: np.ndarray, micro_window: int) -> np.ndarray:
    chunk_hidden = np.asarray(chunk_hidden, dtype=np.float32)
    if chunk_hidden.ndim != 2:
        raise ValueError(f"chunk_hidden must be [tokens,dim], got {chunk_hidden.shape}")
    if chunk_hidden.shape[0] < micro_window:
        return np.empty((0, chunk_hidden.shape[-1]), dtype=np.float32)
    n_windows = chunk_hidden.shape[0] // micro_window
    trimmed = chunk_hidden[: n_windows * micro_window].reshape(
        n_windows, micro_window, chunk_hidden.shape[-1]
    )
    return trimmed.mean(axis=1).astype(np.float32, copy=False)


def local_angular_rows_for_chunks(
    chunks: Any,
    spans: list[tuple[int, int]],
    base: dict[str, Any],
    micro_window: int,
) -> list[dict[str, Any]]:
    rows = []
    for chunk_id, chunk_hidden in enumerate(chunks):
        chunk_hidden = np.asarray(chunk_hidden, dtype=np.float32)
        windows = micro_window_points(chunk_hidden, micro_window)
        metrics = angular_metrics_from_windows(windows)
        start, end = spans[chunk_id]
        rows.append(
            {
                **base,
                "chunk_id": int(chunk_id),
                "chunk_start": int(start),
                "chunk_end": int(end),
                "relative_pos": float(start / max(int(base.get("think_length", 1)), 1)),
                "micro_window": int(micro_window),
                **metrics,
            }
        )
    return rows


def macro_angular_rows_for_layer(
    hidden: np.ndarray,
    base: dict[str, Any],
    window_sizes: list[int],
    pools: list[str],
) -> list[dict[str, Any]]:
    rows = []
    hidden = np.asarray(hidden, dtype=np.float32)
    for window_size in window_sizes:
        for pool in pools:
            points = pool_points(hidden, window_size, pool)
            metrics = cross_chunk_angular_metrics(points)
            rows.append(
                {
                    **base,
                    "window_size": int(window_size),
                    "pool": pool,
                    **metrics,
                }
            )
    return rows


def best_auc_table(
    df: pd.DataFrame,
    group_cols: list[str],
    features: list[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    from run_path_dynamics import aucs_by_question, bootstrap_ci

    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    for key, group in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        for feature in features:
            if feature not in group.columns:
                continue
            aucs = aucs_by_question(group, feature)
            if aucs.size == 0:
                continue
            neg = 1.0 - aucs
            low, high = bootstrap_ci(aucs, n_boot, seed)
            nlow, nhigh = bootstrap_ci(neg, n_boot, seed)
            rows.append(
                {
                    **base,
                    "feature": feature,
                    "n_questions": int(aucs.size),
                    "mean_auc_pos": float(np.mean(aucs)),
                    "ci_low_pos": low,
                    "ci_high_pos": high,
                    "mean_auc_neg": float(np.mean(neg)),
                    "ci_low_neg": nlow,
                    "ci_high_neg": nhigh,
                    "best_auc": float(max(np.mean(aucs), np.mean(neg))),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("best_auc", ascending=False)
    return result


def evaluate_path_features_numeric(summary: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame(rows)
    feature_cols = []
    for col in summary.columns:
        if col in PATH_ID_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(summary[col]):
            feature_cols.append(col)
    for layer in sorted(summary["layer"].unique()):
        sub_layer = summary[summary["layer"] == layer]
        for feature in feature_cols:
            aucs = aucs_by_question(sub_layer, feature)
            if aucs.size == 0:
                continue
            neg_aucs = 1.0 - aucs
            low, high = bootstrap_ci(aucs, n_boot, seed)
            nlow, nhigh = bootstrap_ci(neg_aucs, n_boot, seed)
            rows.append(
                {
                    "layer": int(layer),
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
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("best_auc", ascending=False)
    return result


def should_skip_for_all_metrics(think_len: int, micro_window: int, min_window_size: int) -> bool:
    return think_len < max(micro_window * 3, min_window_size * 3)


def write_report(
    output_dir: Path,
    path_summary: pd.DataFrame,
    path_eval: pd.DataFrame,
    path_bin_eval: pd.DataFrame,
    macro_eval: pd.DataFrame,
    local_eval: pd.DataFrame,
    skipped: int,
) -> Path:
    report = output_dir / "LONG_TRAJECTORY_SMOKE_RESULTS.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Long-CoT Trajectory Smoke Results\n\n")
        f.write("Online E + G + D metrics on the implicit think segment before `</think>`.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Rollout-layer rows: {len(path_summary)}\n")
        f.write(f"- Rollouts: {path_summary[['question_id', 'rollout_id']].drop_duplicates().shape[0] if not path_summary.empty else 0}\n")
        f.write(f"- Questions: {path_summary['question_id'].nunique() if not path_summary.empty else 0}\n")
        f.write(f"- Skipped rollouts: {skipped}\n")
        if not path_summary.empty:
            f.write(f"- Mean think tokens: {path_summary['think_length'].mean():.1f}\n")
            f.write(f"- Mean chunks: {path_summary['num_chunks'].mean():.2f}\n")

        f.write("\n## E Path Primary\n\n")
        f.write("| layer | feature | questions | AUROC(+feature) | AUROC(-feature) | best AUROC |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        top = (
            path_eval[path_eval["feature"].isin(PRIMARY_PATH_FEATURES)].copy()
            if not path_eval.empty and "feature" in path_eval.columns
            else pd.DataFrame()
        )
        if not top.empty:
            top = top.sort_values("best_auc", ascending=False)
        for _, row in top.iterrows():
            f.write(
                f"| {int(row['layer'])} | {row['feature']} | {int(row['n_questions'])} | "
                f"{row['mean_auc_pos']:.4f} | {row['mean_auc_neg']:.4f} | {row['best_auc']:.4f} |\n"
            )

        f.write("\n## G Macro Angular Primary\n\n")
        f.write("| layer | window | pool | feature | questions | AUROC(+feature) | AUROC(-feature) | best AUROC |\n")
        f.write("|---:|---:|---|---|---:|---:|---:|---:|\n")
        for _, row in macro_eval.head(40).iterrows():
            f.write(
                f"| {int(row['layer'])} | {int(row['window_size'])} | {row['pool']} | {row['feature']} | "
                f"{int(row['n_questions'])} | {row['mean_auc_pos']:.4f} | "
                f"{row['mean_auc_neg']:.4f} | {row['best_auc']:.4f} |\n"
            )

        f.write("\n## D Local Angular Primary\n\n")
        f.write("| layer | feature | questions | AUROC(+feature) | AUROC(-feature) | best AUROC |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        for _, row in local_eval.head(30).iterrows():
            f.write(
                f"| {int(row['layer'])} | {row['feature']} | {int(row['n_questions'])} | "
                f"{row['mean_auc_pos']:.4f} | {row['mean_auc_neg']:.4f} | {row['best_auc']:.4f} |\n"
            )

        f.write("\n## Length-Bin Path AUROC\n\n")
        f.write("| layer | bin | feature | questions | AUROC(+feature) | AUROC(-feature) | best AUROC |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|\n")
        view = (
            path_bin_eval[
                path_bin_eval["feature"].isin(["path_length", "d_mean", "d_late_mean"])
            ].copy()
            if not path_bin_eval.empty and "feature" in path_bin_eval.columns
            else pd.DataFrame()
        )
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
    window_sizes = parse_ints(args.window_sizes)
    pools = parse_strings(args.pools)
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

    print("Long-CoT trajectory smoke")
    print(f"input: {args.input}")
    print(f"selected clean rollouts: {len(rows)}")
    print(f"output_dir: {output_dir}")
    print(f"model: {args.model}")
    print(f"layers: {layers}")
    print(f"window_sizes: {window_sizes}")
    print(f"pools: {pools}")

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

    path_dynamics_rows: list[dict[str, Any]] = []
    path_summary_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    skipped = 0

    for row in tqdm(rows, desc="long trajectory"):
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
            if should_skip_for_all_metrics(think_len, args.micro_window, min(window_sizes)):
                raise ValueError(f"think segment too short: {think_len}")

            full_inputs = to_device(full_inputs, device)
            with torch.inference_mode():
                outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)

            n_layers = len(outputs.hidden_states)
            layer_hiddens: dict[int, np.ndarray] = {}
            for layer in layers:
                if layer < 0 or layer >= n_layers:
                    raise ValueError(f"Layer {layer} out of range; n_layers={n_layers}")
                layer_hiddens[layer] = (
                    outputs.hidden_states[layer][0, response_start:response_end]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

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
                "token_segment_status": token_segment_status,
                "text_segment_status": str(row.get("segment_status", "")),
                "truncated": bool(row.get("truncated", False)),
            }

            selected_stack = np.stack([layer_hiddens[layer] for layer in layers], axis=0)
            chunk_array, spans = chunk_means_for_segment(
                selected_stack,
                start=seg_start,
                end=seg_end,
                chunk_size=args.chunk_size,
            )
            chunk_starts = [start for start, _ in spans]
            chunk_ends = [end for _, end in spans]

            for layer_pos, layer in enumerate(layers):
                layer_base = {
                    **base,
                    "layer": int(layer),
                    "num_chunks": int(chunk_array.shape[0]),
                }
                if chunk_array.shape[0] >= 2:
                    points = chunk_array[:, layer_pos, :]
                    dyn = path_dynamics_from_points(points)
                    d = dyn["d"]
                    pv = dyn["pv"]
                    pa = dyn["pa"]
                    path_base = {**layer_base, "n_steps": int(d.size)}
                    path_summary_rows.append({**path_base, **summarize_path_dynamics(d)})
                    for step_id, d_value in enumerate(d):
                        path_dynamics_rows.append(
                            {
                                **path_base,
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

                think_hidden = layer_hiddens[layer][seg_start:seg_end]
                macro_rows.extend(
                    macro_angular_rows_for_layer(
                        think_hidden,
                        layer_base,
                        window_sizes=window_sizes,
                        pools=pools,
                    )
                )
                local_token_chunks = []
                for start, end in spans:
                    local_token_chunks.append(think_hidden[start:end])
                local_rows.extend(
                    local_angular_rows_for_chunks(
                        local_token_chunks,
                        spans,
                        layer_base,
                        micro_window=args.micro_window,
                    )
                )

            del outputs, full_inputs, prompt_inputs, selected_stack, chunk_array, layer_hiddens
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

    path_dynamics = pd.DataFrame(path_dynamics_rows)
    path_summary = pd.DataFrame(path_summary_rows)
    macro = pd.DataFrame(macro_rows)
    local = pd.DataFrame(local_rows)
    path_eval = (
        evaluate_path_features_numeric(path_summary, args.bootstrap, args.seed)
        if not path_summary.empty
        else pd.DataFrame()
    )
    path_bin_eval = (
        evaluate_by_length_bin(path_summary, args.bootstrap, args.seed)
        if not path_summary.empty
        else pd.DataFrame()
    )
    macro_eval = best_auc_table(
        macro,
        ["layer", "window_size", "pool"],
        PRIMARY_G_FEATURES + ["gstep_mean", "gstep_late_mean"],
        args.bootstrap,
        args.seed,
    )
    local_eval = best_auc_table(
        local,
        ["layer"],
        PRIMARY_D_FEATURES,
        args.bootstrap,
        args.seed,
    )

    path_dynamics.to_parquet(output_dir / "long_path_dynamics.parquet", index=False)
    path_summary.to_parquet(output_dir / "long_path_features.parquet", index=False)
    macro.to_parquet(output_dir / "long_macro_angular.parquet", index=False)
    local.to_parquet(output_dir / "long_local_angular.parquet", index=False)
    path_eval.to_parquet(output_dir / "long_path_eval.parquet", index=False)
    path_bin_eval.to_parquet(output_dir / "long_path_length_bin_eval.parquet", index=False)
    macro_eval.to_parquet(output_dir / "long_macro_angular_eval.parquet", index=False)
    local_eval.to_parquet(output_dir / "long_local_angular_eval.parquet", index=False)
    report = write_report(output_dir, path_summary, path_eval, path_bin_eval, macro_eval, local_eval, skipped)

    print(f"saved path summary: {output_dir / 'long_path_features.parquet'} ({len(path_summary)} rows)")
    print(f"saved macro angular: {output_dir / 'long_macro_angular.parquet'} ({len(macro)} rows)")
    print(f"saved local angular: {output_dir / 'long_local_angular.parquet'} ({len(local)} rows)")
    print(f"saved report: {report}")
    if not path_eval.empty:
        print("top path:")
        print(path_eval.head(10))
    if not macro_eval.empty:
        print("top macro:")
        print(macro_eval.head(10))
    if not local_eval.empty:
        print("top local:")
        print(local_eval.head(10))


if __name__ == "__main__":
    main()
