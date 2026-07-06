#!/usr/bin/env python3
"""Audit whether Experiment G gstep features duplicate Experiment E path dynamics.

The key question is whether the high exploratory AUROC from `gstep_*` is just a
medium-grain copy of the magnitude signal already captured by Experiment E.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


PATH_FEATURES = ["path_length", "d_late_mean"]
GSTEP_FEATURES = ["gstep_mean", "gstep_late_mean"]
ID_COLS = ["question_id", "rollout_id", "layer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze G gstep overlap with E path dynamics.")
    parser.add_argument(
        "--gstep-glob",
        default="cross_chunk_angular_results/cross_chunk_angular_gpu*.parquet",
    )
    parser.add_argument(
        "--path-features",
        default="path_dynamics_results/path_dynamics_features.parquet",
    )
    parser.add_argument("--output-dir", default="gstep_path_overlap_results")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_gstep(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No gstep parquet files matched {pattern}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return values * 0.0
    return (values - values.mean()) / std


def join_gstep_path(gstep: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    keep_g = [
        "question_id",
        "rollout_id",
        "layer",
        "window_size",
        "pool",
        "is_correct",
        "response_length",
        *GSTEP_FEATURES,
    ]
    keep_p = [
        "question_id",
        "rollout_id",
        "layer",
        "is_correct",
        "num_chunks",
        "n_steps",
        *PATH_FEATURES,
    ]
    g = gstep[[col for col in keep_g if col in gstep.columns]].copy()
    p = path[[col for col in keep_p if col in path.columns]].copy()

    for frame in [g, p]:
        frame["question_id"] = frame["question_id"].astype(str)
        frame["rollout_id"] = frame["rollout_id"].astype(int)
        frame["layer"] = frame["layer"].astype(int)

    joined = g.merge(
        p,
        on=ID_COLS,
        how="inner",
        suffixes=("_gstep", "_path"),
    )
    if "is_correct_gstep" in joined.columns:
        joined["is_correct"] = joined["is_correct_gstep"].astype(bool)
    elif "is_correct" in joined.columns:
        joined["is_correct"] = joined["is_correct"].astype(bool)
    else:
        joined["is_correct"] = joined["is_correct_path"].astype(bool)

    joined["path_length_score"] = -joined["path_length"].astype(float)
    joined["d_late_mean_score"] = -joined["d_late_mean"].astype(float)
    joined["gstep_mean_score"] = joined["gstep_mean"].astype(float)
    joined["gstep_late_mean_score"] = joined["gstep_late_mean"].astype(float)
    return joined


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score)
    y_true = y_true[valid]
    score = score[valid]
    pos = score[y_true]
    neg = score[~y_true]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    cmp = pos[:, None] - neg[None, :]
    return float((np.sum(cmp > 0) + 0.5 * np.sum(cmp == 0)) / cmp.size)


def per_question_auc(df: pd.DataFrame, score_col: str) -> np.ndarray:
    aucs = []
    for _, group in df.groupby("question_id"):
        y = group["is_correct"].to_numpy(dtype=bool)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        auc = binary_auc(y, group[score_col].to_numpy(dtype=float))
        if np.isfinite(auc):
            aucs.append(auc)
    return np.asarray(aucs, dtype=float)


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(values, size=values.size, replace=True)))
        for _ in range(n_boot)
    ]
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def spearman_pair(df: pd.DataFrame, x: str, y: str) -> tuple[float, int]:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return float("nan"), 0
    rho, _ = spearmanr(sub[x], sub[y])
    return float(rho) if np.isfinite(rho) else float("nan"), int(len(sub))


def within_question_spearman(df: pd.DataFrame, x: str, y: str) -> tuple[float, int]:
    rhos = []
    for _, group in df.groupby("question_id"):
        rho, n = spearman_pair(group, x, y)
        if n > 0 and np.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan"), 0
    return float(np.mean(rhos)), len(rhos)


def compute_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["layer", "window_size", "pool"]
    for (layer, window_size, pool), group in joined.groupby(group_cols):
        for path_feature in PATH_FEATURES:
            for gstep_feature in GSTEP_FEATURES:
                pairs = [
                    ("raw", path_feature, gstep_feature),
                    ("oriented", f"{path_feature}_score", f"{gstep_feature}_score"),
                ]
                for space, x_col, y_col in pairs:
                    rho, n = spearman_pair(group, x_col, y_col)
                    rows.append(
                        {
                            "layer": int(layer),
                            "window_size": int(window_size),
                            "pool": str(pool),
                            "path_feature": path_feature,
                            "gstep_feature": gstep_feature,
                            "score_space": space,
                            "scope": "global",
                            "rho": rho,
                            "n": n,
                        }
                    )
                    wrho, wn = within_question_spearman(group, x_col, y_col)
                    rows.append(
                        {
                            "layer": int(layer),
                            "window_size": int(window_size),
                            "pool": str(pool),
                            "path_feature": path_feature,
                            "gstep_feature": gstep_feature,
                            "score_space": space,
                            "scope": "within_question_mean",
                            "rho": wrho,
                            "n": wn,
                        }
                    )
    return pd.DataFrame(rows)


def evaluate_overlap(joined: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    group_cols = ["layer", "window_size", "pool"]
    for (layer, window_size, pool), group in joined.groupby(group_cols):
        for path_feature in PATH_FEATURES:
            path_score = f"{path_feature}_score"
            for gstep_feature in GSTEP_FEATURES:
                gstep_score = f"{gstep_feature}_score"
                needed = ["question_id", "is_correct", path_score, gstep_score]
                sub = group[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
                if sub.empty:
                    continue
                sub["z_path"] = zscore(sub[path_score])
                sub["z_gstep"] = zscore(sub[gstep_score])
                sub["combo_score"] = sub["z_path"] + sub["z_gstep"]

                path_aucs = per_question_auc(sub, path_score)
                gstep_aucs = per_question_auc(sub, gstep_score)
                combo_aucs = per_question_auc(sub, "combo_score")
                if path_aucs.size == 0 or gstep_aucs.size == 0 or combo_aucs.size == 0:
                    continue
                low, high = bootstrap_ci(combo_aucs, n_boot, seed)
                path_auc = float(np.mean(path_aucs))
                gstep_auc = float(np.mean(gstep_aucs))
                combo_auc = float(np.mean(combo_aucs))
                rows.append(
                    {
                        "layer": int(layer),
                        "window_size": int(window_size),
                        "pool": str(pool),
                        "path_feature": path_feature,
                        "gstep_feature": gstep_feature,
                        "questions": int(combo_aucs.size),
                        "path_auc": path_auc,
                        "gstep_auc": gstep_auc,
                        "combo_auc": combo_auc,
                        "combo_ci_low": low,
                        "combo_ci_high": high,
                        "delta_vs_path": combo_auc - path_auc,
                        "delta_vs_best_single": combo_auc - max(path_auc, gstep_auc),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("delta_vs_path", ascending=False)
    return result


def plot_rho_heatmap(corr_df: pd.DataFrame, fig_dir: Path) -> Path:
    work = corr_df[
        (corr_df["score_space"] == "oriented")
        & (corr_df["scope"] == "within_question_mean")
    ].copy()
    work["row"] = (
        "L"
        + work["layer"].astype(str)
        + " "
        + work["path_feature"]
        + " vs "
        + work["gstep_feature"]
    )
    work["col"] = "w" + work["window_size"].astype(str) + " " + work["pool"].astype(str)
    pivot = work.pivot_table(index="row", columns="col", values="rho", aggfunc="mean")
    plt.figure(figsize=(11, max(4, 0.45 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="vlag", center=0.0, vmin=-1, vmax=1)
    plt.title("Within-question oriented Spearman: E path scores vs G gstep scores")
    plt.tight_layout()
    path = fig_dir / "gstep_path_oriented_rho_heatmap.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_combo_delta(eval_df: pd.DataFrame, fig_dir: Path) -> Path:
    work = eval_df.copy()
    work["combo"] = (
        "L"
        + work["layer"].astype(str)
        + " w"
        + work["window_size"].astype(str)
        + " "
        + work["pool"].astype(str)
        + "\n"
        + work["path_feature"]
        + " + "
        + work["gstep_feature"]
    )
    work = work.sort_values("delta_vs_path", ascending=False).head(24)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=work, x="combo", y="delta_vs_path", color="#4c78a8")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Combo AUROC - path AUROC")
    plt.xlabel("")
    plt.title("Does G gstep improve over Experiment E path features?")
    plt.tight_layout()
    path = fig_dir / "gstep_path_combo_delta.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(
    output_dir: Path,
    joined: pd.DataFrame,
    corr_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    figures: list[Path],
) -> Path:
    report = output_dir / "GSTEP_PATH_OVERLAP_RESULTS.md"
    oriented = corr_df[
        (corr_df["score_space"] == "oriented")
        & (corr_df["scope"] == "within_question_mean")
    ].copy()
    oriented["abs_rho"] = oriented["rho"].abs()
    oriented = oriented.sort_values("abs_rho", ascending=False).head(24)
    top_eval = eval_df.sort_values("delta_vs_path", ascending=False).head(24)

    with report.open("w", encoding="utf-8") as f:
        f.write("# Gstep vs Path Dynamics Overlap\n\n")
        f.write(
            "This audit checks whether Experiment G `gstep_*` is a distinct signal or a "
            "medium-grain duplicate of Experiment E path magnitude.\n\n"
        )
        f.write("## Data\n\n")
        f.write(f"- Joined rows: {len(joined)}\n")
        f.write(f"- Rollouts: {joined[ID_COLS[:2]].drop_duplicates().shape[0]}\n")
        f.write(f"- Questions: {joined['question_id'].nunique()}\n")
        f.write(f"- Layers: {', '.join(map(str, sorted(joined['layer'].unique())))}\n")
        f.write(f"- Window sizes: {', '.join(map(str, sorted(joined['window_size'].unique())))}\n")
        f.write(f"- Pools: {', '.join(map(str, sorted(joined['pool'].unique())))}\n\n")

        f.write("## Highest Oriented Within-Question Correlations\n\n")
        f.write("| layer | window | pool | path feature | gstep feature | rho | questions |\n")
        f.write("|---:|---:|---|---|---|---:|---:|\n")
        for _, row in oriented.iterrows():
            f.write(
                f"| {int(row['layer'])} | {int(row['window_size'])} | {row['pool']} | "
                f"{row['path_feature']} | {row['gstep_feature']} | "
                f"{row['rho']:.4f} | {int(row['n'])} |\n"
            )

        f.write("\n## Combo AUROC Gain Over Path\n\n")
        f.write(
            "| layer | window | pool | path feature | gstep feature | questions | "
            "path AUROC | gstep AUROC | combo AUROC | delta vs path | delta vs best single |\n"
        )
        f.write("|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for _, row in top_eval.iterrows():
            f.write(
                f"| {int(row['layer'])} | {int(row['window_size'])} | {row['pool']} | "
                f"{row['path_feature']} | {row['gstep_feature']} | {int(row['questions'])} | "
                f"{row['path_auc']:.4f} | {row['gstep_auc']:.4f} | "
                f"{row['combo_auc']:.4f} | {row['delta_vs_path']:.4f} | "
                f"{row['delta_vs_best_single']:.4f} |\n"
            )

        f.write("\n## Figures\n\n")
        for path in figures:
            f.write(f"- `{path}`\n")

        f.write("\n## Interpretation Guide\n\n")
        f.write("- High oriented rho plus near-zero combo gain means `gstep` is mostly path magnitude again.\n")
        f.write("- Low rho or positive combo gain means `gstep` may carry independent fine-grain magnitude information.\n")
        f.write("- This audit does not judge angular direction features such as `gcos_p10`.\n")
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    gstep = load_gstep(args.gstep_glob)
    path = pd.read_parquet(args.path_features)
    joined = join_gstep_path(gstep, path)
    corr_df = compute_correlations(joined)
    eval_df = evaluate_overlap(joined, args.bootstrap, args.seed)

    joined.to_parquet(output_dir / "gstep_path_joined.parquet", index=False)
    corr_df.to_parquet(output_dir / "gstep_path_correlations.parquet", index=False)
    corr_df.to_csv(output_dir / "gstep_path_correlations.csv", index=False)
    eval_df.to_parquet(output_dir / "gstep_path_combo_auc.parquet", index=False)
    eval_df.to_csv(output_dir / "gstep_path_combo_auc.csv", index=False)

    figures = [plot_rho_heatmap(corr_df, fig_dir), plot_combo_delta(eval_df, fig_dir)]
    report = write_report(output_dir, joined, corr_df, eval_df, figures)

    print(f"saved joined: {output_dir / 'gstep_path_joined.parquet'} ({len(joined)} rows)")
    print(f"saved correlations: {output_dir / 'gstep_path_correlations.csv'}")
    print(f"saved combo auc: {output_dir / 'gstep_path_combo_auc.csv'}")
    print(f"saved report: {report}")
    print("top combo gains:")
    print(eval_df.head(10))


if __name__ == "__main__":
    main()
