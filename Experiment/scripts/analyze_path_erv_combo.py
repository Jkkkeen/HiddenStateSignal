#!/usr/bin/env python3
"""Analyze relation and complementarity between Method 3 path length and ERV.

Inputs:
  scores/M3_L36_path_length.npz
  er_local_dynamics_results/local_er_dynamics_rollout_summary.parquet

Outputs:
  combo_results/PATH_ERV_COMBO_RESULTS.md
  combo_results/path_erv_joined.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path length vs ERV combo analysis.")
    parser.add_argument("--path-score", default="scores/M3_L36_path_length.npz")
    parser.add_argument(
        "--erv-summary",
        default="er_local_dynamics_results/local_er_dynamics_rollout_summary.parquet",
    )
    parser.add_argument("--output-dir", default="combo_results")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_path_score(path: Path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    return pd.DataFrame(
        {
            "question_id": z["id"].astype(str),
            "rollout_id": z["roll_index"].astype(int),
            "is_correct_path": z["is_correct"].astype(bool),
            "path_value_raw": z["value"].astype(float),
            "path_score": z["score"].astype(float),
        }
    )


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return values * 0
    return (values - values.mean()) / std


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
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        means.append(float(np.mean(rng.choice(values, size=values.size, replace=True))))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def evaluate_scores(df: pd.DataFrame, columns: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for col in columns:
        aucs = per_question_auc(df, col)
        if aucs.size == 0:
            continue
        low, high = bootstrap_ci(aucs, n_boot, seed)
        rows.append(
            {
                "score": col,
                "questions": int(aucs.size),
                "mean_auc": float(np.mean(aucs)),
                "median_auc": float(np.median(aucs)),
                "ci_low": low,
                "ci_high": high,
                "opposite_mean_auc": float(np.mean(1.0 - aucs)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False)


def within_question_spearman(df: pd.DataFrame, x: str, y: str) -> tuple[float, int]:
    rhos = []
    for _, group in df.groupby("question_id"):
        sub = group[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
            continue
        rho, _ = spearmanr(sub[x], sub[y])
        if np.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan"), 0
    return float(np.mean(rhos)), len(rhos)


def plot_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sample = df.sample(min(len(df), 8000), random_state=2026)
    sample["correctness"] = np.where(sample["is_correct"], "correct", "incorrect")
    plt.figure(figsize=(7, 5.5))
    sns.scatterplot(
        data=sample,
        x="path_score",
        y="erv36_adj_late_min",
        hue="correctness",
        alpha=0.45,
        s=24,
    )
    plt.title("M3 path score vs ERV late-min")
    plt.xlabel("M3 L36 path score (-path_length)")
    plt.ylabel("L36 ERV_adj late min")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = fig_dir / "path_score_vs_erv36_late_min.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(
    out_dir: Path,
    joined: pd.DataFrame,
    corr_rows: list[dict[str, object]],
    eval_df: pd.DataFrame,
    figure_path: Path,
) -> Path:
    report = out_dir / "PATH_ERV_COMBO_RESULTS.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Path Length vs ERV Combo Results\n\n")
        f.write("This analysis checks whether Method 3 path-length and Local ER dynamics are redundant or complementary.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Joined rollout rows: {len(joined)}\n")
        f.write(f"- Questions: {joined['question_id'].nunique()}\n")
        f.write(f"- Rollouts: {joined[['question_id', 'rollout_id']].drop_duplicates().shape[0]}\n\n")

        f.write("## Correlations\n\n")
        f.write("| x | y | scope | rho | n |\n")
        f.write("|---|---|---|---:|---:|\n")
        for row in corr_rows:
            f.write(
                f"| {row['x']} | {row['y']} | {row['scope']} | "
                f"{float(row['rho']):.4f} | {int(row['n'])} |\n"
            )

        f.write("\n## AUROC\n\n")
        f.write("| score | questions | mean AUROC | median | 95% CI | opposite mean |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for _, row in eval_df.iterrows():
            f.write(
                f"| {row['score']} | {int(row['questions'])} | {row['mean_auc']:.4f} | "
                f"{row['median_auc']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
                f"{row['opposite_mean_auc']:.4f} |\n"
            )

        f.write("\n## Figure\n\n")
        f.write(f"- `{figure_path}`\n")
    return report


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_df = load_path_score(Path(args.path_score))
    erv = pd.read_parquet(args.erv_summary)
    wanted = [
        "question_id",
        "rollout_id",
        "layer",
        "is_correct",
        "erv_adj_late_min",
        "erv_hist_late_min",
        "era_adj_late_min",
    ]
    erv = erv[wanted].copy()
    erv36 = erv[erv["layer"] == 36].drop(columns=["layer"]).rename(
        columns={
            "erv_adj_late_min": "erv36_adj_late_min",
            "erv_hist_late_min": "erv36_hist_late_min",
            "era_adj_late_min": "era36_adj_late_min",
            "is_correct": "is_correct_erv36",
        }
    )
    erv24 = erv[erv["layer"] == 24].drop(columns=["layer"]).rename(
        columns={
            "erv_adj_late_min": "erv24_adj_late_min",
            "erv_hist_late_min": "erv24_hist_late_min",
            "era_adj_late_min": "era24_adj_late_min",
            "is_correct": "is_correct_erv24",
        }
    )

    joined = path_df.merge(erv36, on=["question_id", "rollout_id"], how="inner")
    joined = joined.merge(erv24, on=["question_id", "rollout_id"], how="left")
    joined["is_correct"] = joined["is_correct_path"].astype(bool)

    for col in ["path_score", "erv36_adj_late_min", "erv36_hist_late_min", "erv24_hist_late_min"]:
        joined[f"z_{col}"] = zscore(joined[col])

    joined["combo_path_erv36_adj"] = joined["z_path_score"] + joined["z_erv36_adj_late_min"]
    joined["combo_path_erv36_hist"] = joined["z_path_score"] + joined["z_erv36_hist_late_min"]
    joined["combo_path_erv24_hist"] = joined["z_path_score"] + joined["z_erv24_hist_late_min"]

    joined.to_parquet(out_dir / "path_erv_joined.parquet", index=False)

    corr_rows = []
    for ycol in ["erv36_adj_late_min", "erv36_hist_late_min", "erv24_hist_late_min"]:
        valid = joined[["path_score", ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, _ = spearmanr(valid["path_score"], valid[ycol])
        corr_rows.append({"x": "path_score", "y": ycol, "scope": "global", "rho": rho, "n": len(valid)})
        wrho, wn = within_question_spearman(joined, "path_score", ycol)
        corr_rows.append({"x": "path_score", "y": ycol, "scope": "within_question_mean", "rho": wrho, "n": wn})

    eval_cols = [
        "path_score",
        "erv36_adj_late_min",
        "erv36_hist_late_min",
        "erv24_hist_late_min",
        "combo_path_erv36_adj",
        "combo_path_erv36_hist",
        "combo_path_erv24_hist",
    ]
    eval_df = evaluate_scores(joined, eval_cols, args.bootstrap, args.seed)
    eval_df.to_csv(out_dir / "path_erv_combo_auc.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out_dir / "path_erv_correlations.csv", index=False)
    figure_path = plot_scatter(joined, out_dir)
    report = write_report(out_dir, joined, corr_rows, eval_df, figure_path)

    print(f"saved report: {report}")
    print(eval_df)
    print(pd.DataFrame(corr_rows))


if __name__ == "__main__":
    main()
