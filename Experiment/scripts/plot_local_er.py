#!/usr/bin/env python3
"""Plot and evaluate Experiment A: per-rollout local ER."""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Local ER results.")
    parser.add_argument("--input-glob", default="er_local_results/local_er*.parquet")
    parser.add_argument("--output-dir", default="er_local_results")
    parser.add_argument("--metric", default="er_centered")
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--max-chunk-id", type=int, default=8)
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


def per_question_auc(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for (layer, chunk_id, qid), group in df.groupby(["layer", "chunk_id", "question_id"]):
        y = group["is_correct"].to_numpy(dtype=bool)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        auc = binary_auc(y, group[score_col].to_numpy())
        rows.append(
            {
                "layer": int(layer),
                "chunk_id": int(chunk_id),
                "question_id": str(qid),
                "auc_pos": auc,
                "auc_neg": 1.0 - auc if np.isfinite(auc) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def plot_correct_vs_incorrect(df: pd.DataFrame, layer: int, metric: str, fig_dir: Path) -> Path:
    sub = df[(df["layer"] == layer) & (df["chunk_id"] <= df["chunk_id"].max())].copy()
    sub["correctness"] = np.where(sub["is_correct"], "correct", "incorrect")
    grouped = (
        sub.groupby(["correctness", "chunk_id"], as_index=False)
        .agg(mean_er=(metric, "mean"), std_er=(metric, "std"), n=(metric, "size"))
        .sort_values(["correctness", "chunk_id"])
    )
    grouped["sem"] = grouped["std_er"] / np.sqrt(grouped["n"].clip(lower=1))

    palette = {"correct": "#2b7a78", "incorrect": "#b84a4a"}
    plt.figure(figsize=(7.5, 4.8))
    ax = plt.gca()
    for name in ["incorrect", "correct"]:
        item = grouped[grouped["correctness"] == name]
        x = item["chunk_id"].to_numpy()
        y = item["mean_er"].to_numpy()
        sem = item["sem"].fillna(0).to_numpy()
        ax.plot(x, y, marker="o", label=name, color=palette[name], linewidth=2)
        ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, color=palette[name], alpha=0.15)
    ax.set_title(f"Local ER: correct vs incorrect, layer {layer}")
    ax.set_xlabel("Chunk position")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path = fig_dir / f"A1_correct_vs_incorrect_er_curve_layer{layer}.png"
    if layer == 24:
        path = fig_dir / "A2_correct_vs_incorrect_er_curve_layer24.png"
    elif layer == 36:
        path = fig_dir / "A1_correct_vs_incorrect_er_curve_layer36.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_relative_position(df: pd.DataFrame, metric: str, fig_dir: Path) -> Path:
    work = df.copy()
    work["rel_bin"] = pd.cut(
        work["relative_pos"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
        labels=[f"{i * 10}-{(i + 1) * 10}%" for i in range(10)],
    )
    work["correctness"] = np.where(work["is_correct"], "correct", "incorrect")
    grouped = (
        work.groupby(["layer", "correctness", "rel_bin"], observed=True, as_index=False)
        .agg(mean_er=(metric, "mean"), n=(metric, "size"))
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=grouped,
        x="rel_bin",
        y="mean_er",
        hue="correctness",
        style="layer",
        markers=True,
        dashes=False,
    )
    plt.title("Local ER by relative response position")
    plt.xlabel("Relative position bin")
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = fig_dir / "A3_relative_position_er_curve.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_auc_by_chunk(auc_df: pd.DataFrame, fig_dir: Path) -> Path:
    summary = (
        auc_df.groupby(["layer", "chunk_id"], as_index=False)
        .agg(mean_auc_pos=("auc_pos", "mean"), mean_auc_neg=("auc_neg", "mean"), n=("auc_pos", "size"))
    )
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=summary, x="chunk_id", y="mean_auc_pos", hue="layer", marker="o")
    sns.lineplot(
        data=summary,
        x="chunk_id",
        y="mean_auc_neg",
        hue="layer",
        marker="s",
        linestyle="--",
        legend=False,
    )
    plt.axhline(0.5, color="black", linewidth=1, alpha=0.5)
    plt.title("Within-question AUROC by chunk")
    plt.xlabel("Chunk position")
    plt.ylabel("Mean per-question AUROC")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = fig_dir / "A4_within_question_auroc_by_chunk.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_distribution(df: pd.DataFrame, metric: str, fig_dir: Path) -> Path:
    work = df.copy()
    work["correctness"] = np.where(work["is_correct"], "correct", "incorrect")
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=work, x="layer", y=metric, hue="correctness", split=True, inner="quart")
    plt.title("Local ER distribution by correctness")
    plt.xlabel("Layer")
    plt.ylabel(metric)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    path = fig_dir / "A5_er_distribution_by_correctness.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(
    df: pd.DataFrame,
    auc_df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    figures: list[Path],
    n_boot: int,
    seed: int,
) -> Path:
    rows = []
    for (layer, chunk_id), group in auc_df.groupby(["layer", "chunk_id"]):
        pos_values = group["auc_pos"].to_numpy(dtype=np.float64)
        neg_values = group["auc_neg"].to_numpy(dtype=np.float64)
        pos_lo, pos_hi = bootstrap_ci(pos_values, n_boot, seed)
        neg_lo, neg_hi = bootstrap_ci(neg_values, n_boot, seed + 7)
        rows.append(
            {
                "layer": int(layer),
                "chunk_id": int(chunk_id),
                "n_questions": int(len(group)),
                "mean_auc_pos": float(np.nanmean(pos_values)),
                "median_auc_pos": float(np.nanmedian(pos_values)),
                "pos_ci_low": pos_lo,
                "pos_ci_high": pos_hi,
                "mean_auc_neg": float(np.nanmean(neg_values)),
                "median_auc_neg": float(np.nanmedian(neg_values)),
                "neg_ci_low": neg_lo,
                "neg_ci_high": neg_hi,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["mean_auc_pos"], ascending=False)
    summary_path = output_dir / "local_er_auc_by_chunk.csv"
    summary.to_csv(summary_path, index=False)

    report_path = output_dir / "LOCAL_ER_RESULTS.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Local ER Results\n\n")
        f.write("Experiment A computes token-level chunk ER online and saves scalar metrics only.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Rows: {len(df)}\n")
        f.write(f"- Rollouts: {df[['question_id', 'rollout_id']].drop_duplicates().shape[0]}\n")
        f.write(f"- Questions: {df['question_id'].nunique()}\n")
        f.write(f"- Layers: {', '.join(map(str, sorted(df['layer'].unique())))}\n")
        f.write(f"- Metric: `{metric}`\n\n")
        f.write("## Top Chunk-Level AUROC\n\n")
        f.write("| layer | chunk | questions | AUROC(+ER) | 95% CI | AUROC(-ER) | 95% CI |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in summary.head(20).iterrows():
            f.write(
                f"| {int(row['layer'])} | {int(row['chunk_id'])} | {int(row['n_questions'])} | "
                f"{row['mean_auc_pos']:.4f} | [{row['pos_ci_low']:.4f}, {row['pos_ci_high']:.4f}] | "
                f"{row['mean_auc_neg']:.4f} | [{row['neg_ci_low']:.4f}, {row['neg_ci_high']:.4f}] |\n"
            )
        f.write("\n## Figures\n\n")
        for path in figures:
            f.write(f"- `{path}`\n")

    return report_path


def main() -> None:
    args = parse_args()
    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input_glob)
    df = df[df["layer"].isin(layers)].copy()
    df = df[df["chunk_id"] <= args.max_chunk_id].copy()
    if df.empty:
        raise ValueError("No rows after filtering.")
    if args.metric not in df.columns:
        raise KeyError(f"Missing metric column {args.metric}")

    sns.set_theme(style="whitegrid")
    auc_df = per_question_auc(df, args.metric)

    figures: list[Path] = []
    for layer in layers:
        if layer in set(df["layer"]):
            figures.append(plot_correct_vs_incorrect(df[df["layer"] == layer], layer, args.metric, fig_dir))
    figures.append(plot_relative_position(df, args.metric, fig_dir))
    if not auc_df.empty:
        figures.append(plot_auc_by_chunk(auc_df, fig_dir))
    figures.append(plot_distribution(df, args.metric, fig_dir))

    report_path = write_report(
        df,
        auc_df,
        output_dir,
        args.metric,
        figures,
        args.bootstrap,
        args.seed,
    )
    print(f"saved report: {report_path}")
    for path in figures:
        print(f"saved figure: {path}")


if __name__ == "__main__":
    main()
