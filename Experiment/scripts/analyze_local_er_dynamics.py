#!/usr/bin/env python3
"""Post-process Experiment A local ER into ERV / ERA dynamics.

This script does not run model forward.  It reads local ER parquet files,
computes adjacent and historical-baseline ER velocity/acceleration per rollout,
then evaluates rollout-level dynamic summaries with within-question AUROC.
"""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ERV/ERA from local ER parquet files.")
    parser.add_argument("--input-glob", default="er_local_results/local_er_gpu*.parquet")
    parser.add_argument("--output-dir", default="er_local_dynamics_results")
    parser.add_argument("--metric", default="er_centered")
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_results(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched {pattern}")
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


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


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def compute_dynamics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    keys = ["question_id", "rollout_id", "layer"]
    for (qid, rollout_id, layer), group in df.groupby(keys):
        group = group.sort_values("chunk_id")
        values = group[metric].to_numpy(dtype=np.float64)
        chunk_ids = group["chunk_id"].to_numpy(dtype=np.int64)
        if values.size == 0:
            continue

        erv_adj = np.full(values.shape, np.nan, dtype=np.float64)
        era_adj = np.full(values.shape, np.nan, dtype=np.float64)
        erv_hist = np.full(values.shape, np.nan, dtype=np.float64)
        era_hist = np.full(values.shape, np.nan, dtype=np.float64)

        if values.size >= 2:
            erv_adj[1:] = np.diff(values)
        if values.size >= 3:
            era_adj[2:] = np.diff(erv_adj[1:])

        hist_erv_seen = []
        for idx in range(values.size):
            if idx >= 1:
                erv_hist[idx] = values[idx] - float(np.mean(values[:idx]))
                if hist_erv_seen:
                    era_hist[idx] = erv_hist[idx] - float(np.mean(hist_erv_seen))
                hist_erv_seen.append(float(erv_hist[idx]))

        base = group.iloc[0].to_dict()
        for idx, chunk_id in enumerate(chunk_ids):
            rows.append(
                {
                    "question_id": str(qid),
                    "rollout_id": int(rollout_id),
                    "layer": int(layer),
                    "chunk_id": int(chunk_id),
                    "is_correct": bool(base["is_correct"]),
                    "response_length": int(base["response_length"]),
                    "num_chunks": int(base["num_chunks"]),
                    "er": float(values[idx]),
                    "erv_adj": float(erv_adj[idx]) if np.isfinite(erv_adj[idx]) else np.nan,
                    "era_adj": float(era_adj[idx]) if np.isfinite(era_adj[idx]) else np.nan,
                    "erv_hist": float(erv_hist[idx]) if np.isfinite(erv_hist[idx]) else np.nan,
                    "era_hist": float(era_hist[idx]) if np.isfinite(era_hist[idx]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_rollouts(dyn: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["erv_adj", "era_adj", "erv_hist", "era_hist"]
    for (qid, rollout_id, layer), group in dyn.groupby(["question_id", "rollout_id", "layer"]):
        group = group.sort_values("chunk_id")
        row = {
            "question_id": str(qid),
            "rollout_id": int(rollout_id),
            "layer": int(layer),
            "is_correct": bool(group["is_correct"].iloc[0]),
            "num_chunks": int(group["num_chunks"].iloc[0]),
            "response_length": int(group["response_length"].iloc[0]),
        }
        half = max(0, len(group) // 2)
        late = group.iloc[half:] if len(group) else group
        for metric in metrics:
            arr = group[metric].to_numpy(dtype=np.float64)
            late_arr = late[metric].to_numpy(dtype=np.float64)
            for prefix, values in [("all", arr), ("late", late_arr)]:
                valid = values[np.isfinite(values)]
                if valid.size == 0:
                    row[f"{metric}_{prefix}_mean"] = np.nan
                    row[f"{metric}_{prefix}_abs_mean"] = np.nan
                    row[f"{metric}_{prefix}_std"] = np.nan
                    row[f"{metric}_{prefix}_max"] = np.nan
                    row[f"{metric}_{prefix}_min"] = np.nan
                else:
                    row[f"{metric}_{prefix}_mean"] = float(np.mean(valid))
                    row[f"{metric}_{prefix}_abs_mean"] = float(np.mean(np.abs(valid)))
                    row[f"{metric}_{prefix}_std"] = float(np.std(valid))
                    row[f"{metric}_{prefix}_max"] = float(np.max(valid))
                    row[f"{metric}_{prefix}_min"] = float(np.min(valid))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_features(summary: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    feature_cols = [
        col
        for col in summary.columns
        if col
        not in {
            "question_id",
            "rollout_id",
            "layer",
            "is_correct",
            "num_chunks",
            "response_length",
        }
    ]
    rows = []
    for layer in sorted(summary["layer"].unique()):
        sub_layer = summary[summary["layer"] == layer]
        for feature in feature_cols:
            aucs = []
            for qid, group in sub_layer.groupby("question_id"):
                y = group["is_correct"].to_numpy(dtype=bool)
                if y.sum() == 0 or y.sum() == y.size:
                    continue
                auc = binary_auc(y, group[feature].to_numpy(dtype=np.float64))
                if np.isfinite(auc):
                    aucs.append(auc)
            aucs = np.asarray(aucs, dtype=np.float64)
            if aucs.size == 0:
                continue
            lo, hi = bootstrap_ci(aucs, n_boot, seed)
            rows.append(
                {
                    "layer": int(layer),
                    "feature": feature,
                    "n_questions": int(aucs.size),
                    "mean_auc_pos": float(np.mean(aucs)),
                    "median_auc_pos": float(np.median(aucs)),
                    "ci_low_pos": lo,
                    "ci_high_pos": hi,
                    "mean_auc_neg": float(np.mean(1.0 - aucs)),
                    "median_auc_neg": float(np.median(1.0 - aucs)),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("mean_auc_pos", ascending=False)
    return result


def plot_dynamics_curve(dyn: pd.DataFrame, output_dir: Path) -> list[Path]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    work = dyn.copy()
    work["correctness"] = np.where(work["is_correct"], "correct", "incorrect")
    for metric in ["erv_hist", "era_hist", "erv_adj", "era_adj"]:
        grouped = (
            work.groupby(["layer", "chunk_id", "correctness"], as_index=False)
            .agg(value=(metric, "mean"), n=(metric, "count"))
            .dropna(subset=["value"])
        )
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=grouped,
            x="chunk_id",
            y="value",
            hue="correctness",
            style="layer",
            markers=True,
            dashes=False,
        )
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)
        plt.title(f"{metric} by chunk")
        plt.xlabel("Chunk position")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        path = fig_dir / f"ERD_{metric}_curve.png"
        plt.savefig(path, dpi=220)
        plt.close()
        paths.append(path)
    return paths


def write_report(
    eval_df: pd.DataFrame,
    dyn: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    metric: str,
    figure_paths: list[Path],
) -> Path:
    report = output_dir / "LOCAL_ER_DYNAMICS_RESULTS.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Local ER Dynamics Results\n\n")
        f.write("ERV/ERA are computed from existing local ER parquet files; no model forward is used.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Chunk rows: {len(dyn)}\n")
        f.write(f"- Rollout-layer rows: {len(summary)}\n")
        f.write(f"- Questions: {dyn['question_id'].nunique()}\n")
        f.write(f"- Rollouts: {dyn[['question_id', 'rollout_id']].drop_duplicates().shape[0]}\n")
        f.write(f"- Base metric: `{metric}`\n\n")
        f.write("## Top Rollout-Level Dynamics Features\n\n")
        f.write("| layer | feature | questions | AUROC(+feature) | 95% CI | AUROC(-feature) |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        for _, row in eval_df.head(30).iterrows():
            f.write(
                f"| {int(row['layer'])} | {row['feature']} | {int(row['n_questions'])} | "
                f"{row['mean_auc_pos']:.4f} | [{row['ci_low_pos']:.4f}, {row['ci_high_pos']:.4f}] | "
                f"{row['mean_auc_neg']:.4f} |\n"
            )
        f.write("\n## Figures\n\n")
        for path in figure_paths:
            f.write(f"- `{path}`\n")
    return report


def main() -> None:
    args = parse_args()
    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input_glob)
    df = df[df["layer"].isin(layers)].copy()
    if args.metric not in df.columns:
        raise KeyError(f"Missing metric column {args.metric}")

    dyn = compute_dynamics(df, args.metric)
    summary = summarize_rollouts(dyn)
    eval_df = evaluate_features(summary, args.bootstrap, args.seed)

    dyn_path = output_dir / "local_er_dynamics.parquet"
    summary_path = output_dir / "local_er_dynamics_rollout_summary.parquet"
    eval_path = output_dir / "local_er_dynamics_auc.csv"
    dyn.to_parquet(dyn_path, index=False)
    summary.to_parquet(summary_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    sns.set_theme(style="whitegrid")
    figure_paths = plot_dynamics_curve(dyn, output_dir)
    report_path = write_report(eval_df, dyn, summary, output_dir, args.metric, figure_paths)

    print(f"saved {dyn_path}")
    print(f"saved {summary_path}")
    print(f"saved {eval_path}")
    print(f"saved report: {report_path}")
    print("top 10:")
    print(eval_df.head(10))


if __name__ == "__main__":
    main()
