#!/usr/bin/env python3
"""Plot and evaluate Experiment D: local chunk angular dynamics."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PRIMARY_FEATURES = ["cos_mean", "cos_p10", "spike_rate_90", "av_cos_mean"]
SECONDARY_FEATURES = [
    "lad_mean",
    "lad_std",
    "lad_p90",
    "lad_max",
    "cos_std",
    "cos_min",
    "spike_rate_105",
    "ae_norm_b6",
    "aa_cos_mean",
    "av_lad_p90",
    "aa_lad_p90",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Local Angular Dynamics.")
    parser.add_argument("--input-glob", default="angular_results/local_angular_gpu*.parquet")
    parser.add_argument("--local-er-glob", default="")
    parser.add_argument("--output-dir", default="angular_results")
    parser.add_argument("--layers", default="24,36")
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
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        means.append(float(np.mean(rng.choice(values, size=values.size, replace=True))))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def add_chunk_velocity(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(["question_id", "rollout_id", "layer", "chunk_id"]).copy()
    for feature in ["cos_mean", "lad_p90"]:
        work[f"av_{feature}"] = np.nan
        work[f"aa_{feature}"] = np.nan

    for _, idx in work.groupby(["question_id", "rollout_id", "layer"]).groups.items():
        idx = list(idx)
        for feature in ["cos_mean", "lad_p90"]:
            values = work.loc[idx, feature].to_numpy(dtype=np.float64)
            av = np.full(values.shape, np.nan)
            aa = np.full(values.shape, np.nan)
            if values.size >= 2:
                av[1:] = np.diff(values)
            if values.size >= 3:
                aa[2:] = np.diff(av[1:])
            work.loc[idx, f"av_{feature}"] = av
            work.loc[idx, f"aa_{feature}"] = aa

    work = work.rename(
        columns={
            "av_cos_mean": "av_cos_mean",
            "aa_cos_mean": "aa_cos_mean",
            "av_lad_p90": "av_lad_p90",
            "aa_lad_p90": "aa_lad_p90",
        }
    )
    return work


def evaluate_chunk_features(
    df: pd.DataFrame,
    features: list[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        for (layer, chunk_id), group in df.groupby(["layer", "chunk_id"]):
            aucs = []
            for _, qgroup in group.groupby("question_id"):
                y = qgroup["is_correct"].to_numpy(dtype=bool)
                if y.sum() == 0 or y.sum() == y.size:
                    continue
                auc = binary_auc(y, qgroup[feature].to_numpy(dtype=np.float64))
                if np.isfinite(auc):
                    aucs.append(auc)
            aucs = np.asarray(aucs, dtype=np.float64)
            if aucs.size == 0:
                continue
            lo, hi = bootstrap_ci(aucs, n_boot, seed)
            rows.append(
                {
                    "layer": int(layer),
                    "chunk_id": int(chunk_id),
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
        result = result.sort_values(["mean_auc_pos"], ascending=False)
    return result


def plot_feature_curves(df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    paths = []
    work = df.copy()
    work["correctness"] = np.where(work["is_correct"], "correct", "incorrect")
    for feature in PRIMARY_FEATURES:
        if feature not in work.columns:
            continue
        grouped = (
            work.groupby(["layer", "chunk_id", "correctness"], as_index=False)
            .agg(value=(feature, "mean"), n=(feature, "count"))
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
        if feature.startswith("cos") or feature.startswith("av_cos"):
            plt.axhline(0, color="black", linewidth=1, alpha=0.5)
        plt.title(f"{feature} by chunk")
        plt.xlabel("Chunk position")
        plt.ylabel(feature)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        path = fig_dir / f"D_{feature}_curve.png"
        plt.savefig(path, dpi=220)
        plt.close()
        paths.append(path)
    return paths


def plot_auc(primary_auc: pd.DataFrame, fig_dir: Path) -> Path:
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=primary_auc,
        x="chunk_id",
        y="mean_auc_pos",
        hue="feature",
        style="layer",
        markers=True,
        dashes=False,
    )
    plt.axhline(0.5, color="black", linewidth=1, alpha=0.5)
    plt.title("Primary angular metrics: within-question AUROC")
    plt.xlabel("Chunk position")
    plt.ylabel("Mean per-question AUROC")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = fig_dir / "D_primary_auc_by_chunk.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_er_angular_joint(df: pd.DataFrame, local_er_glob: str, fig_dir: Path) -> Path | None:
    if not local_er_glob:
        return None
    try:
        er_df = load_parquet(local_er_glob)
    except FileNotFoundError:
        return None
    keys = ["question_id", "rollout_id", "layer", "chunk_id"]
    er_small = er_df[keys + ["er_centered"]]
    merged = df.merge(er_small, on=keys, how="inner")
    if merged.empty:
        return None
    merged = merged[merged["chunk_id"].between(1, 3)].copy()
    merged["correctness"] = np.where(merged["is_correct"], "correct", "incorrect")
    sample = merged.sample(min(len(merged), 6000), random_state=2026)
    plt.figure(figsize=(7, 5.5))
    sns.scatterplot(
        data=sample,
        x="er_centered",
        y="cos_mean",
        hue="correctness",
        style="layer",
        alpha=0.45,
        s=20,
    )
    plt.axhline(0, color="black", linewidth=1, alpha=0.4)
    plt.title("Local ER x angular consistency")
    plt.xlabel("Local ER centered")
    plt.ylabel("cos_mean")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = fig_dir / "D4_er_x_cos_mean_scatter.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(
    output_dir: Path,
    df: pd.DataFrame,
    primary_auc: pd.DataFrame,
    secondary_auc: pd.DataFrame,
    figures: list[Path],
) -> Path:
    report_path = output_dir / "LOCAL_ANGULAR_RESULTS.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Local Chunk Angular Dynamics Results\n\n")
        f.write("Primary metrics are pre-committed: `cos_mean`, `cos_p10`, `spike_rate_90`, `AV_cos_mean`.\n")
        f.write("Secondary metrics are exploratory robustness checks.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Rows: {len(df)}\n")
        f.write(f"- Rollouts: {df[['question_id', 'rollout_id']].drop_duplicates().shape[0]}\n")
        f.write(f"- Questions: {df['question_id'].nunique()}\n")
        f.write(f"- Layers: {', '.join(map(str, sorted(df['layer'].unique())))}\n\n")

        f.write("## Primary AUROC Table\n\n")
        f.write("| layer | chunk | feature | questions | AUROC(+feature) | 95% CI | AUROC(-feature) |\n")
        f.write("|---:|---:|---|---:|---:|---:|---:|\n")
        for _, row in primary_auc.head(40).iterrows():
            f.write(
                f"| {int(row['layer'])} | {int(row['chunk_id'])} | {row['feature']} | "
                f"{int(row['n_questions'])} | {row['mean_auc_pos']:.4f} | "
                f"[{row['ci_low_pos']:.4f}, {row['ci_high_pos']:.4f}] | "
                f"{row['mean_auc_neg']:.4f} |\n"
            )

        f.write("\n## Secondary Exploratory Table\n\n")
        f.write("| layer | chunk | feature | questions | AUROC(+feature) | 95% CI | AUROC(-feature) |\n")
        f.write("|---:|---:|---|---:|---:|---:|---:|\n")
        for _, row in secondary_auc.head(40).iterrows():
            f.write(
                f"| {int(row['layer'])} | {int(row['chunk_id'])} | {row['feature']} | "
                f"{int(row['n_questions'])} | {row['mean_auc_pos']:.4f} | "
                f"[{row['ci_low_pos']:.4f}, {row['ci_high_pos']:.4f}] | "
                f"{row['mean_auc_neg']:.4f} |\n"
            )

        f.write("\n## Figures\n\n")
        for path in figures:
            f.write(f"- `{path}`\n")
    return report_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]

    df = load_parquet(args.input_glob)
    df = df[df["layer"].isin(layers)].copy()
    if df.empty:
        raise ValueError("No rows after layer filtering.")
    df = add_chunk_velocity(df)

    primary_auc = evaluate_chunk_features(df, PRIMARY_FEATURES, args.bootstrap, args.seed)
    secondary_auc = evaluate_chunk_features(df, SECONDARY_FEATURES, args.bootstrap, args.seed)

    dyn_path = output_dir / "local_angular_with_dynamics.parquet"
    primary_path = output_dir / "primary_angular_auc.csv"
    secondary_path = output_dir / "secondary_angular_auc.csv"
    df.to_parquet(dyn_path, index=False)
    primary_auc.to_csv(primary_path, index=False)
    secondary_auc.to_csv(secondary_path, index=False)

    sns.set_theme(style="whitegrid")
    figures = plot_feature_curves(df, fig_dir)
    if not primary_auc.empty:
        figures.append(plot_auc(primary_auc, fig_dir))
    joint = plot_er_angular_joint(df, args.local_er_glob, fig_dir)
    if joint is not None:
        figures.append(joint)

    report = write_report(output_dir, df, primary_auc, secondary_auc, figures)
    print(f"saved {dyn_path}")
    print(f"saved {primary_path}")
    print(f"saved {secondary_path}")
    print(f"saved report: {report}")
    for path in figures:
        print(f"saved figure: {path}")
    print("top primary:")
    print(primary_auc.head(10))


if __name__ == "__main__":
    main()
