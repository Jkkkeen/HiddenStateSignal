#!/usr/bin/env python3
"""Plot and summarize Experiment B: Cross-rollout ER."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Cross-rollout ER results.")
    parser.add_argument("--input-dir", default="er_results")
    parser.add_argument("--metric", default="er_group_centered")
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--max-heatmap-chunks", type=int, default=12)
    return parser.parse_args()


def phase_name(relative_pos: float) -> str:
    if relative_pos < 1 / 3:
        return "early"
    if relative_pos < 2 / 3:
        return "mid"
    return "late"


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan, np.nan, int(len(data))
    rho, pval = spearmanr(data["x"], data["y"])
    return float(rho), float(pval), int(len(data))


def plot_group_curve(df: pd.DataFrame, layer: int, metric: str, fig_dir: Path) -> Path:
    sub = df[df["layer"] == layer].copy()
    grouped = (
        sub.groupby(["accuracy_group", "chunk_pos"], as_index=False)
        .agg(
            mean_er=(metric, "mean"),
            std_er=(metric, "std"),
            n=("question_id", "nunique"),
        )
        .sort_values(["accuracy_group", "chunk_pos"])
    )
    grouped["sem"] = grouped["std_er"] / np.sqrt(grouped["n"].clip(lower=1))

    order = ["low", "mid", "high"]
    palette = {"low": "#b84a4a", "mid": "#6f6f6f", "high": "#2b7a78"}

    plt.figure(figsize=(7.5, 4.8))
    ax = plt.gca()
    for group in order:
        item = grouped[grouped["accuracy_group"] == group]
        if item.empty:
            continue
        x = item["chunk_pos"].to_numpy()
        y = item["mean_er"].to_numpy()
        sem = item["sem"].fillna(0).to_numpy()
        ax.plot(x, y, marker="o", label=group, color=palette[group], linewidth=2)
        ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, color=palette[group], alpha=0.15)

    ax.set_title(f"Cross-rollout ER by question accuracy, layer {layer}")
    ax.set_xlabel("Chunk position")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(title="accuracy group")
    plt.tight_layout()

    path = fig_dir / f"B1_er_group_by_accuracy_layer{layer}.png"
    if layer == 24:
        path = fig_dir / "B2_er_group_by_accuracy_layer24.png"
    elif layer == 36:
        path = fig_dir / "B1_er_group_by_accuracy_layer36.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def question_phase_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    work = df.copy()
    work["phase"] = work["relative_pos_mean"].map(phase_name)
    phase = (
        work.groupby(["question_id", "layer", "phase"], as_index=False)
        .agg(
            er_mean=(metric, "mean"),
            question_accuracy=("question_accuracy", "first"),
            answer_entropy_norm=("answer_entropy_norm", "first"),
            majority_confidence=("majority_confidence", "first"),
            unique_pred_answers=("unique_pred_answers", "first"),
            mean_response_len=("mean_response_len", "first"),
            mean_num_chunks=("mean_num_chunks", "first"),
        )
    )
    return phase


def plot_er_vs_accuracy(phase_df: pd.DataFrame, metric: str, fig_dir: Path) -> Path:
    phases = ["early", "mid", "late"]
    layers = sorted(phase_df["layer"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, phase in zip(axes, phases):
        sub = phase_df[phase_df["phase"] == phase]
        sns.scatterplot(
            data=sub,
            x="question_accuracy",
            y="er_mean",
            hue="layer",
            hue_order=layers,
            palette="viridis",
            alpha=0.55,
            s=24,
            ax=ax,
        )
        ax.set_title(phase)
        ax.set_xlabel("Question accuracy")
        ax.set_ylabel(metric if ax is axes[0] else "")
        ax.grid(True, alpha=0.25)
        if ax is not axes[-1]:
            ax.get_legend().remove()
    plt.tight_layout()
    path = fig_dir / "B3_er_vs_question_accuracy.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_er_vs_entropy(phase_df: pd.DataFrame, metric: str, fig_dir: Path) -> Path:
    sub = phase_df[phase_df["phase"] == "late"].copy()
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=sub,
        x="answer_entropy_norm",
        y="er_mean",
        hue="layer",
        palette="viridis",
        alpha=0.6,
        s=28,
    )
    plt.title("Late Cross-rollout ER vs answer entropy")
    plt.xlabel("Normalized answer entropy")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = fig_dir / "B4_er_vs_answer_entropy.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_heatmap(df: pd.DataFrame, layer: int, metric: str, fig_dir: Path, max_chunks: int) -> Path:
    sub = df[(df["layer"] == layer) & (df["chunk_pos"] < max_chunks)].copy()
    q_order = (
        sub.groupby("question_id")
        .agg(question_accuracy=("question_accuracy", "first"))
        .sort_values("question_accuracy")
        .index
        .tolist()
    )
    pivot = sub.pivot_table(index="question_id", columns="chunk_pos", values=metric, aggfunc="mean")
    pivot = pivot.reindex(q_order)

    plt.figure(figsize=(8, 7.5))
    sns.heatmap(pivot, cmap="mako", cbar_kws={"label": metric})
    plt.title(f"Cross-rollout ER heatmap, layer {layer}")
    plt.xlabel("Chunk position")
    plt.ylabel("Questions sorted by accuracy")
    plt.tight_layout()
    path = fig_dir / f"B5_er_group_heatmap_layer{layer}.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(
    output_dir: Path,
    df: pd.DataFrame,
    phase_df: pd.DataFrame,
    metric: str,
    figure_paths: list[Path],
) -> Path:
    rows = []
    for layer in sorted(phase_df["layer"].unique()):
        for phase in ["early", "mid", "late"]:
            sub = phase_df[(phase_df["layer"] == layer) & (phase_df["phase"] == phase)]
            for target in ["question_accuracy", "answer_entropy_norm", "majority_confidence"]:
                rho, pval, n = safe_spearman(sub["er_mean"], sub[target])
                rows.append(
                    {
                        "layer": layer,
                        "phase": phase,
                        "target": target,
                        "spearman_rho": rho,
                        "p_value": pval,
                        "n": n,
                    }
                )
    corr = pd.DataFrame(rows)
    corr_path = output_dir / "cross_rollout_er_correlations.csv"
    corr.to_csv(corr_path, index=False)

    report_path = output_dir / "CROSS_ROLLOUT_ER_RESULTS.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Cross-rollout ER Results\n\n")
        f.write("Current version uses chunk-mean hidden states from `h_chunk`.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Rows: {len(df)}\n")
        f.write(f"- Questions: {df['question_id'].nunique()}\n")
        f.write(f"- Layers: {', '.join(map(str, sorted(df['layer'].unique())))}\n")
        f.write(f"- Metric plotted: `{metric}`\n\n")

        f.write("## Spearman Correlations\n\n")
        f.write("| layer | phase | target | rho | p-value | n |\n")
        f.write("|---:|---|---|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['layer']} | {row['phase']} | {row['target']} | "
                f"{row['spearman_rho']:.4f} | {row['p_value']:.3g} | {row['n']} |\n"
            )

        f.write("\n## Figures\n\n")
        for path in figure_paths:
            f.write(f"- `{path}`\n")

        f.write("\n## Interpretation Notes\n\n")
        f.write(
            "- Higher centered ER means same-question rollouts are more dispersed at that chunk position.\n"
        )
        f.write(
            "- This is a question-level uncertainty/consensus signal, not a single-rollout reward.\n"
        )
        f.write(
            "- A last-token version is pending new forward output with `h_last`.\n"
        )

    return report_path


def main() -> None:
    args = parse_args()
    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    output_dir = Path(args.input_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cross_path = output_dir / "cross_rollout_er.parquet"
    if not cross_path.exists():
        raise FileNotFoundError(f"Missing {cross_path}; run run_cross_rollout_er.py first.")

    df = pd.read_parquet(cross_path)
    df = df[df["layer"].isin(layers)].copy()
    if args.metric not in df.columns:
        raise KeyError(f"Metric column not found: {args.metric}")

    sns.set_theme(style="whitegrid")
    phase_df = question_phase_table(df, args.metric)

    figure_paths: list[Path] = []
    for layer in layers:
        if layer in set(df["layer"]):
            figure_paths.append(plot_group_curve(df, layer, args.metric, fig_dir))
    figure_paths.append(plot_er_vs_accuracy(phase_df, args.metric, fig_dir))
    figure_paths.append(plot_er_vs_entropy(phase_df, args.metric, fig_dir))
    if 36 in set(df["layer"]):
        figure_paths.append(
            plot_heatmap(df, 36, args.metric, fig_dir, args.max_heatmap_chunks)
        )
    else:
        figure_paths.append(
            plot_heatmap(df, layers[-1], args.metric, fig_dir, args.max_heatmap_chunks)
        )

    report_path = write_report(output_dir, df, phase_df, args.metric, figure_paths)
    print(f"saved report: {report_path}")
    for path in figure_paths:
        print(f"saved figure: {path}")


if __name__ == "__main__":
    main()
