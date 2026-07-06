#!/usr/bin/env python3
"""Evaluate and plot Experiment G cross-chunk angular dynamics."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PRIMARY_DIRECTIONS = {
    "gcos_mean": "pos",
    "gcos_late_mean": "pos",
    "gcos_p10": "pos",
    "gtheta_late_mean": "neg",
}
SECONDARY_FEATURES = [
    "gcos_std",
    "gcos_min",
    "gcos_late_min",
    "gtheta_mean",
    "gtheta_std",
    "gspike_rate_90",
    "gstep_mean",
    "gstep_late_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Experiment G cross-chunk angular results.")
    parser.add_argument("--input-glob", default="cross_chunk_angular_results/cross_chunk_angular*.parquet")
    parser.add_argument("--output-dir", default="cross_chunk_angular_results")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_parquet(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched {pattern}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


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
        means.append(float(np.mean(rng.choice(values, size=values.size, replace=True))))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def per_question_aucs(df: pd.DataFrame, feature: str, direction: str) -> np.ndarray:
    aucs = []
    for _, group in df.groupby("question_id"):
        y = group["is_correct"].to_numpy(dtype=bool)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        score = group[feature].to_numpy(dtype=np.float64)
        if direction == "neg":
            score = -score
        auc = binary_auc(y, score)
        if np.isfinite(auc):
            aucs.append(auc)
    return np.asarray(aucs, dtype=np.float64)


def evaluate_features(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    feature_specs = {**PRIMARY_DIRECTIONS}
    for feature in SECONDARY_FEATURES:
        if feature not in feature_specs:
            feature_specs[feature] = "both"

    group_cols = ["layer", "window_size", "pool"]
    for (layer, window_size, pool), group in df.groupby(group_cols):
        for feature, direction in feature_specs.items():
            if feature not in group.columns:
                continue
            directions = ["pos", "neg"] if direction == "both" else [direction]
            pos_aucs = per_question_aucs(group, feature, "pos")
            neg_aucs = per_question_aucs(group, feature, "neg")
            pos_low, pos_high = bootstrap_ci(pos_aucs, n_boot, seed)
            neg_low, neg_high = bootstrap_ci(neg_aucs, n_boot, seed)
            committed_auc = (
                float(np.mean(pos_aucs))
                if direction == "pos"
                else float(np.mean(neg_aucs))
                if direction == "neg"
                else float(max(np.mean(pos_aucs), np.mean(neg_aucs)))
            )
            rows.append(
                {
                    "layer": int(layer),
                    "window_size": int(window_size),
                    "pool": str(pool),
                    "feature": feature,
                    "is_primary": feature in PRIMARY_DIRECTIONS,
                    "direction": direction,
                    "n_questions": int(max(pos_aucs.size, neg_aucs.size)),
                    "mean_auc_pos": float(np.mean(pos_aucs)) if pos_aucs.size else np.nan,
                    "ci_low_pos": pos_low,
                    "ci_high_pos": pos_high,
                    "mean_auc_neg": float(np.mean(neg_aucs)) if neg_aucs.size else np.nan,
                    "ci_low_neg": neg_low,
                    "ci_high_neg": neg_high,
                    "committed_auc": committed_auc,
                    "best_auc": float(max(np.mean(pos_aucs), np.mean(neg_aucs)))
                    if pos_aucs.size and neg_aucs.size
                    else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["is_primary", "committed_auc"], ascending=[False, False])
    return result


def plot_primary_sweep(eval_df: pd.DataFrame, fig_dir: Path) -> Path:
    work = eval_df[eval_df["is_primary"]].copy()
    work["combo"] = "L" + work["layer"].astype(str) + " " + work["pool"].astype(str)
    grid = sns.relplot(
        data=work,
        x="window_size",
        y="committed_auc",
        hue="combo",
        col="feature",
        kind="line",
        marker="o",
        col_wrap=2,
        height=4,
        aspect=1.25,
        facet_kws={"sharey": True},
    )
    for ax in grid.axes.flat:
        ax.axhline(0.5, color="black", linewidth=1, alpha=0.45)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Committed AUROC")
        ax.set_xlabel("Window size")
    grid.fig.suptitle("Experiment G primary AUROC sweep", y=1.03)
    path = fig_dir / "G1_primary_auc_sweep.png"
    grid.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(grid.fig)
    return path


def plot_raw_group_means(df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    paths = []
    work = df.copy()
    work["correctness"] = np.where(work["is_correct"], "correct", "incorrect")
    for feature in PRIMARY_DIRECTIONS:
        grouped = (
            work.groupby(["layer", "window_size", "pool", "correctness"], as_index=False)
            .agg(value=(feature, "mean"), n=(feature, "count"))
            .dropna(subset=["value"])
        )
        grid = sns.relplot(
            data=grouped,
            x="window_size",
            y="value",
            hue="correctness",
            style="pool",
            col="layer",
            kind="line",
            marker="o",
            height=4,
            aspect=1.25,
        )
        for ax in grid.axes.flat:
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Window size")
            ax.set_ylabel(feature)
        grid.fig.suptitle(f"Experiment G raw group means: {feature}", y=1.03)
        path = fig_dir / f"G2_raw_{feature}.png"
        grid.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(grid.fig)
        paths.append(path)
    return paths


def write_report(output_dir: Path, df: pd.DataFrame, eval_df: pd.DataFrame, figures: list[Path]) -> Path:
    report = output_dir / "CROSS_CHUNK_ANGULAR_RESULTS.md"
    primary = eval_df[eval_df["is_primary"]].sort_values("committed_auc", ascending=False)
    top = eval_df.sort_values("best_auc", ascending=False).head(30)
    with report.open("w", encoding="utf-8") as f:
        f.write("# Cross-Chunk Angular Dynamics Results\n\n")
        f.write("Experiment G measures macro-scale semantic turning across medium-grain response points.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Rows: {len(df)}\n")
        f.write(f"- Rollouts: {df[['question_id', 'rollout_id']].drop_duplicates().shape[0]}\n")
        f.write(f"- Questions: {df['question_id'].nunique()}\n")
        f.write(f"- Layers: {', '.join(map(str, sorted(df['layer'].unique())))}\n")
        f.write(f"- Window sizes: {', '.join(map(str, sorted(df['window_size'].unique())))}\n")
        f.write(f"- Pools: {', '.join(map(str, sorted(df['pool'].unique())))}\n\n")

        f.write("## Primary Metrics\n\n")
        f.write("| layer | window | pool | feature | direction | questions | committed AUROC | AUROC(+feature) | AUROC(-feature) |\n")
        f.write("|---:|---:|---|---|---|---:|---:|---:|---:|\n")
        for _, row in primary.iterrows():
            f.write(
                f"| {int(row['layer'])} | {int(row['window_size'])} | {row['pool']} | "
                f"{row['feature']} | {row['direction']} | {int(row['n_questions'])} | "
                f"{row['committed_auc']:.4f} | {row['mean_auc_pos']:.4f} | {row['mean_auc_neg']:.4f} |\n"
            )

        f.write("\n## Top Exploratory Features\n\n")
        f.write("| layer | window | pool | feature | primary | questions | best AUROC | AUROC(+feature) | AUROC(-feature) |\n")
        f.write("|---:|---:|---|---|---:|---:|---:|---:|---:|\n")
        for _, row in top.iterrows():
            f.write(
                f"| {int(row['layer'])} | {int(row['window_size'])} | {row['pool']} | "
                f"{row['feature']} | {bool(row['is_primary'])} | {int(row['n_questions'])} | "
                f"{row['best_auc']:.4f} | {row['mean_auc_pos']:.4f} | {row['mean_auc_neg']:.4f} |\n"
            )

        f.write("\n## Figures\n\n")
        for path in figures:
            f.write(f"- `{path}`\n")

        f.write("\n## Interpretation Notes\n\n")
        f.write("- Higher `gcos_*` means more consistent macro-scale direction.\n")
        f.write("- Lower `gtheta_late_mean` means fewer late macro turns.\n")
        f.write("- This is a rollout-level direction signal, complementary to ER and path-length magnitude.\n")
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_parquet(args.input_glob)
    eval_df = evaluate_features(df, args.bootstrap, args.seed)
    eval_path = output_dir / "cross_chunk_angular_eval.parquet"
    eval_df.to_parquet(eval_path, index=False)

    figures = [plot_primary_sweep(eval_df, fig_dir)]
    figures.extend(plot_raw_group_means(df, fig_dir))
    report = write_report(output_dir, df, eval_df, figures)

    print(f"saved {eval_path}: {len(eval_df)} rows")
    print(f"saved report: {report}")
    for path in figures:
        print(f"saved figure: {path}")
    print("top primary:")
    print(eval_df[eval_df["is_primary"]].head(10))


if __name__ == "__main__":
    main()

