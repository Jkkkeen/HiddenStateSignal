#!/usr/bin/env python3
"""Check chunk-position sample-selection bias for Experiment D.

This script compares angular AUROC at chunks 2 and 3 on:
  1. each chunk's native available question set;
  2. the common question set that has both chunks.

It also reports whether questions missing chunk 3 differ in accuracy/length.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "cos_mean",
    "cos_p10",
    "spike_rate_90",
    "av_cos_mean",
    "cos_min",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze angular chunk selection bias.")
    parser.add_argument("--input", default="angular_results/local_angular_with_dynamics.parquet")
    parser.add_argument("--fallback-glob", default="angular_results/local_angular_gpu*.parquet")
    parser.add_argument("--output-dir", default="angular_results/selection_bias")
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--chunks", default="2,3")
    return parser.parse_args()


def load_data(input_path: str, fallback_glob: str) -> pd.DataFrame:
    path = Path(input_path)
    if path.exists():
        return pd.read_parquet(path)
    paths = sorted(glob.glob(fallback_glob))
    if not paths:
        raise FileNotFoundError(f"Missing {input_path} and no files match {fallback_glob}")
    return pd.concat([pd.read_parquet(item) for item in paths], ignore_index=True)


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(score)
    y_true = y_true[valid]
    score = score[valid]
    pos = score[y_true]
    neg = score[~y_true]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    cmp = pos[:, None] - neg[None, :]
    return float((np.sum(cmp > 0) + 0.5 * np.sum(cmp == 0)) / cmp.size)


def per_question_auroc(df: pd.DataFrame, feature: str) -> tuple[float, int]:
    aucs = []
    for _, group in df.groupby("question_id"):
        y = group["is_correct"].to_numpy(dtype=bool)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        auc = binary_auc(y, group[feature].to_numpy(dtype=np.float64))
        if np.isfinite(auc):
            aucs.append(auc)
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


def question_level_table(df: pd.DataFrame, chunks: list[int]) -> pd.DataFrame:
    base = (
        df.groupby("question_id", as_index=False)
        .agg(
            question_accuracy=("is_correct", "mean"),
            mean_response_length=("response_length", "mean"),
            max_chunk=("chunk_id", "max"),
            total_rows=("chunk_id", "size"),
        )
    )
    for chunk in chunks:
        has = df[df["chunk_id"] == chunk].groupby("question_id").size()
        base[f"has_chunk_{chunk}"] = base["question_id"].isin(set(has.index))
    return base


def summarize_selection(qtab: pd.DataFrame, chunks: list[int]) -> pd.DataFrame:
    rows = []
    for chunk in chunks:
        col = f"has_chunk_{chunk}"
        for value, group in qtab.groupby(col):
            rows.append(
                {
                    "chunk": chunk,
                    "has_chunk": bool(value),
                    "questions": int(len(group)),
                    "mean_question_accuracy": float(group["question_accuracy"].mean()),
                    "median_question_accuracy": float(group["question_accuracy"].median()),
                    "mean_response_length": float(group["mean_response_length"].mean()),
                    "median_response_length": float(group["mean_response_length"].median()),
                }
            )
    return pd.DataFrame(rows)


def compare_native_vs_common(
    df: pd.DataFrame,
    layers: list[int],
    chunks: list[int],
    features: list[str],
) -> pd.DataFrame:
    rows = []
    if len(chunks) != 2:
        raise ValueError("This script currently expects exactly two chunks, e.g. --chunks 2,3")
    c0, c1 = chunks
    for layer in layers:
        layer_df = df[df["layer"] == layer]
        q0 = set(layer_df[layer_df["chunk_id"] == c0]["question_id"].unique())
        q1 = set(layer_df[layer_df["chunk_id"] == c1]["question_id"].unique())
        common = q0 & q1
        for feature in features:
            if feature not in layer_df.columns:
                continue
            for chunk in chunks:
                native = layer_df[layer_df["chunk_id"] == chunk]
                common_df = native[native["question_id"].isin(common)]
                native_auc, native_n = per_question_auroc(native, feature)
                common_auc, common_n = per_question_auroc(common_df, feature)
                rows.append(
                    {
                        "layer": layer,
                        "chunk": chunk,
                        "feature": feature,
                        "native_questions": native_n,
                        "native_auc_pos": native_auc,
                        "native_auc_neg": 1.0 - native_auc if np.isfinite(native_auc) else np.nan,
                        "common_questions": common_n,
                        "common_auc_pos": common_auc,
                        "common_auc_neg": 1.0 - common_auc if np.isfinite(common_auc) else np.nan,
                        "common_pool_size": len(common),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    chunks: list[int],
) -> Path:
    report = out_dir / "ANGULAR_SELECTION_BIAS.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Angular Chunk Selection Bias Check\n\n")
        f.write(
            "This report checks whether later-chunk angular AUROC is affected by "
            "only longer responses reaching later chunks.\n\n"
        )
        f.write("## Question Set Differences\n\n")
        f.write("| chunk | has chunk | questions | mean acc | median acc | mean len | median len |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in selection.iterrows():
            f.write(
                f"| {int(row['chunk'])} | {row['has_chunk']} | {int(row['questions'])} | "
                f"{row['mean_question_accuracy']:.4f} | {row['median_question_accuracy']:.4f} | "
                f"{row['mean_response_length']:.1f} | {row['median_response_length']:.1f} |\n"
            )

        f.write("\n## Native vs Common-Question AUROC\n\n")
        f.write(
            f"Common-question AUROC restricts both chunk {chunks[0]} and chunk {chunks[1]} "
            "to the same question set.\n\n"
        )
        f.write("| layer | chunk | feature | native q | native AUROC+ | common q | common AUROC+ | common AUROC- |\n")
        f.write("|---:|---:|---|---:|---:|---:|---:|---:|\n")
        display = comparison.sort_values("common_auc_pos", ascending=False)
        for _, row in display.iterrows():
            f.write(
                f"| {int(row['layer'])} | {int(row['chunk'])} | {row['feature']} | "
                f"{int(row['native_questions'])} | {row['native_auc_pos']:.4f} | "
                f"{int(row['common_questions'])} | {row['common_auc_pos']:.4f} | "
                f"{row['common_auc_neg']:.4f} |\n"
            )
    return report


def main() -> None:
    args = parse_args()
    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    chunks = [int(item.strip()) for item in args.chunks.split(",") if item.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input, args.fallback_glob)
    df = df[df["layer"].isin(layers)].copy()
    qtab = question_level_table(df, chunks)
    selection = summarize_selection(qtab, chunks)
    comparison = compare_native_vs_common(df, layers, chunks, FEATURES)

    qtab.to_csv(out_dir / "question_chunk_availability.csv", index=False)
    selection.to_csv(out_dir / "selection_summary.csv", index=False)
    comparison.to_csv(out_dir / "native_vs_common_auc.csv", index=False)
    report = write_report(out_dir, selection, comparison, chunks)

    print(f"saved report: {report}")
    print(selection)
    print(comparison.sort_values('common_auc_pos', ascending=False).head(20))


if __name__ == "__main__":
    main()
