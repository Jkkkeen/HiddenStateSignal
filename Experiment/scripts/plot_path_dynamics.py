#!/usr/bin/env python3
"""Plot Experiment E path velocity and acceleration results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from run_path_dynamics import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Experiment E path dynamics.")
    parser.add_argument("--input-dir", default="path_dynamics_results")
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--erv-summary", default="er_local_dynamics_results/local_er_dynamics_rollout_summary.parquet")
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dynamics = pd.read_parquet(input_dir / "path_dynamics.parquet")
    summary = pd.read_parquet(input_dir / "path_dynamics_features.parquet")
    eval_df = pd.read_parquet(input_dir / "path_dynamics_eval.parquet")
    chunk_auc = pd.read_parquet(input_dir / "path_dynamics_chunk_auc.parquet")
    return dynamics, summary, eval_df, chunk_auc


def plot_curve(
    df: pd.DataFrame,
    feature: str,
    layers: list[int],
    fig_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
) -> Path:
    work = df[df["layer"].isin(layers)].copy()
    work["correctness"] = np.where(work["is_correct"], "correct", "incorrect")
    grouped = (
        work.groupby(["layer", "step_id", "correctness"], as_index=False)
        .agg(value=(feature, "mean"), n=(feature, "count"))
        .dropna(subset=["value"])
    )
    plt.figure(figsize=(8.5, 5.2))
    sns.lineplot(
        data=grouped,
        x="step_id",
        y="value",
        hue="correctness",
        style="layer",
        markers=True,
        dashes=False,
    )
    if feature in {"pv", "pa"}:
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)
    plt.title(title)
    plt.xlabel("Adjacent chunk step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = fig_dir / filename
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_chunk_auc(chunk_auc: pd.DataFrame, layers: list[int], fig_dir: Path) -> Path:
    work = chunk_auc[(chunk_auc["layer"].isin(layers)) & (chunk_auc["feature"] == "d")].copy()
    plt.figure(figsize=(8.5, 5.2))
    sns.lineplot(
        data=work,
        x="step_id",
        y="mean_auc_neg",
        hue="layer",
        marker="o",
        palette="Set2",
    )
    plt.axhline(0.5, color="black", linewidth=1, alpha=0.5)
    plt.title("Experiment E: within-question AUROC by chunk step")
    plt.xlabel("Adjacent chunk step")
    plt.ylabel("AUROC using -d")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = fig_dir / "E3_auroc_by_chunk.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_path_dynamics_vs_erv(summary: pd.DataFrame, erv_path: Path, fig_dir: Path) -> Path | None:
    if not erv_path.exists():
        return None
    erv = pd.read_parquet(erv_path)
    wanted = ["question_id", "rollout_id", "layer", "erv_adj_late_min", "erv_hist_late_min"]
    missing = [col for col in wanted if col not in erv.columns]
    if missing:
        return None

    path36 = summary[summary["layer"] == 36].copy()
    erv36 = erv[erv["layer"] == 36][wanted].copy()
    merged = path36.merge(erv36, on=["question_id", "rollout_id", "layer"], how="inner")
    if merged.empty:
        return None
    sample = merged.sample(min(len(merged), 8000), random_state=2026)
    sample["correctness"] = np.where(sample["is_correct"], "correct", "incorrect")

    plt.figure(figsize=(7.2, 5.6))
    sns.scatterplot(
        data=sample,
        x="pv_late_max",
        y="erv_adj_late_min",
        hue="correctness",
        alpha=0.45,
        s=24,
    )
    plt.axhline(0, color="black", linewidth=1, alpha=0.35)
    plt.axvline(0, color="black", linewidth=1, alpha=0.35)
    plt.title("Experiment E: path late acceleration vs ERV")
    plt.xlabel("L36 pv_late_max")
    plt.ylabel("L36 ERV_adj late min")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = fig_dir / "E4_path_dynamics_vs_erv.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    fig_dir = input_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)

    dynamics, summary, eval_df, chunk_auc = load_inputs(input_dir)
    figure_paths: list[Path] = []
    figure_paths.append(
        plot_curve(
            dynamics,
            feature="d",
            layers=layers,
            fig_dir=fig_dir,
            filename="E1_d_curve.png",
            title="Experiment E: adjacent path displacement",
            ylabel="d = ||h[k+1] - h[k]||",
        )
    )
    figure_paths.append(
        plot_curve(
            dynamics,
            feature="pv",
            layers=layers,
            fig_dir=fig_dir,
            filename="E2_pv_curve.png",
            title="Experiment E: path velocity change",
            ylabel="pv = d[k] - d[k-1]",
        )
    )
    figure_paths.append(plot_chunk_auc(chunk_auc, layers, fig_dir))
    scatter = plot_path_dynamics_vs_erv(summary, Path(args.erv_summary), fig_dir)
    if scatter is not None:
        figure_paths.append(scatter)

    report = write_report(input_dir, dynamics, summary, eval_df, chunk_auc, figure_paths)
    print(f"saved report: {report}")
    for path in figure_paths:
        print(f"saved figure: {path}")


if __name__ == "__main__":
    main()

