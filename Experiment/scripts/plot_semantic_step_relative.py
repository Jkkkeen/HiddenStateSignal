#!/usr/bin/env python3
"""Plot relative-step correct-answer basin dynamics.

This reads the semantic-step parquet output from run_semantic_step_basin_qwen3vl.py
and compares correct vs incorrect rollouts within the same question over relative
thinking progress bins.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_FEATURES = [
    "align_correct_state",
    "align_correct_step",
    "state_margin",
    "turn_to_correct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot relative semantic-step basin dynamics.")
    parser.add_argument("--input-dir", default="server_results/angle_results_smoke500")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = finite(values)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or n_boot <= 0:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[idx] = np.mean(sample)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


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


def add_relative_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    out = df.copy()
    pos = out["relative_step_pos"].astype(float).clip(0.0, 1.0)
    bins = np.floor(pos.to_numpy() * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    out["rel_bin"] = bins
    out["rel_bin_mid"] = (bins + 0.5) / n_bins
    return out


def add_angle_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["state_correct_angle_deg"] = np.degrees(
        np.arccos(np.clip(out["align_correct_state"].astype(float), -1.0, 1.0))
    )
    out["turn_correct_angle_deg"] = np.degrees(
        np.arccos(np.clip(out["align_correct_step"].astype(float), -1.0, 1.0))
    )
    return out


def mixed_question_ids(df: pd.DataFrame) -> set[str]:
    flags = (
        df[["question_id", "rollout_id", "is_correct"]]
        .drop_duplicates()
        .groupby("question_id")["is_correct"]
        .agg(["sum", "count"])
    )
    mixed = flags[(flags["sum"] > 0) & (flags["sum"] < flags["count"])]
    return set(mixed.index.astype(str))


def rollout_bin_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    cols = ["question_id", "rollout_id", "layer", "rel_bin", "rel_bin_mid", "is_correct"]
    return (
        df[cols + [score_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[score_col])
        .groupby(cols, as_index=False)[score_col]
        .mean()
    )


def correct_wrong_gap_by_bin(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rb = rollout_bin_table(df, score_col)
    per_question = []
    for (layer, rel_bin, rel_mid, qid), group in rb.groupby(
        ["layer", "rel_bin", "rel_bin_mid", "question_id"]
    ):
        y = group["is_correct"].astype(bool).to_numpy()
        if y.sum() == 0 or y.sum() == y.size:
            continue
        correct_mean = float(group.loc[y, score_col].mean())
        wrong_mean = float(group.loc[~y, score_col].mean())
        per_question.append(
            {
                "layer": int(layer),
                "rel_bin": int(rel_bin),
                "rel_bin_mid": float(rel_mid),
                "question_id": str(qid),
                "correct_mean": correct_mean,
                "wrong_mean": wrong_mean,
                "gap": correct_mean - wrong_mean,
            }
        )
    qdf = pd.DataFrame(per_question)
    if qdf.empty:
        return pd.DataFrame(
            columns=["layer", "rel_bin", "rel_bin_mid", "questions", "gap_mean", "gap_se"]
        )
    rows = []
    for (layer, rel_bin, rel_mid), group in qdf.groupby(["layer", "rel_bin", "rel_bin_mid"]):
        gaps = group["gap"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "layer": int(layer),
                "rel_bin": int(rel_bin),
                "rel_bin_mid": float(rel_mid),
                "questions": int(gaps.size),
                "gap_mean": float(np.mean(gaps)),
                "gap_se": float(np.std(gaps) / math.sqrt(gaps.size)) if gaps.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["layer", "rel_bin"]).reset_index(drop=True)


def within_question_auc_by_bin(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rb = rollout_bin_table(df, score_col)
    per_question = []
    for (layer, rel_bin, rel_mid, qid), group in rb.groupby(
        ["layer", "rel_bin", "rel_bin_mid", "question_id"]
    ):
        y = group["is_correct"].astype(bool).to_numpy()
        if y.sum() == 0 or y.sum() == y.size:
            continue
        auc = binary_auc(y, group[score_col].to_numpy(dtype=np.float64))
        if not np.isfinite(auc):
            continue
        per_question.append(
            {
                "layer": int(layer),
                "rel_bin": int(rel_bin),
                "rel_bin_mid": float(rel_mid),
                "question_id": str(qid),
                "auc": auc,
            }
        )
    qdf = pd.DataFrame(per_question)
    if qdf.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "rel_bin",
                "rel_bin_mid",
                "questions",
                "auc_mean",
                "auc_neg_mean",
                "best_auc",
            ]
        )
    rows = []
    for (layer, rel_bin, rel_mid), group in qdf.groupby(["layer", "rel_bin", "rel_bin_mid"]):
        aucs = group["auc"].to_numpy(dtype=np.float64)
        auc_mean = float(np.mean(aucs))
        auc_neg = float(np.mean(1.0 - aucs))
        rows.append(
            {
                "layer": int(layer),
                "rel_bin": int(rel_bin),
                "rel_bin_mid": float(rel_mid),
                "questions": int(aucs.size),
                "auc_mean": auc_mean,
                "auc_neg_mean": auc_neg,
                "best_auc": max(auc_mean, auc_neg),
            }
        )
    return pd.DataFrame(rows).sort_values(["layer", "rel_bin"]).reset_index(drop=True)


def summarize_curves(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rb = rollout_bin_table(df, score_col)
    rows = []
    for (layer, rel_bin, rel_mid, is_correct), group in rb.groupby(
        ["layer", "rel_bin", "rel_bin_mid", "is_correct"]
    ):
        vals = group[score_col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "layer": int(layer),
                "rel_bin": int(rel_bin),
                "rel_bin_mid": float(rel_mid),
                "is_correct": bool(is_correct),
                "rollouts": int(vals.size),
                "mean": float(np.mean(vals)),
                "se": float(np.std(vals) / math.sqrt(vals.size)) if vals.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["layer", "is_correct", "rel_bin"]).reset_index(drop=True)


def segment_label(steps: pd.DataFrame) -> str:
    if "segment" not in steps.columns:
        return "thinking"
    values = [str(v) for v in steps["segment"].dropna().unique()]
    if len(values) == 1 and values[0]:
        return values[0]
    return "semantic-step"


def progress_label(segment: str) -> str:
    if segment == "answer":
        return "relative answer progress"
    if segment in {"think", "thinking"}:
        return "relative thinking progress"
    return "relative semantic-step progress"


def plot_correct_wrong_curves(
    curves: pd.DataFrame,
    score_col: str,
    ylabel: str,
    title: str,
    path: Path,
    xlabel: str,
) -> None:
    import matplotlib.pyplot as plt

    layers = sorted(curves["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(6.0 * len(layers), 4.2), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    colors = {True: "#2F6BFF", False: "#D94B4B"}
    labels = {True: "correct rollout", False: "wrong rollout"}
    for ax, layer in zip(axes, layers):
        sub = curves[curves["layer"] == layer]
        for is_correct in [True, False]:
            part = sub[sub["is_correct"] == is_correct]
            if part.empty:
                continue
            x = part["rel_bin_mid"].to_numpy(dtype=np.float64)
            y = part["mean"].to_numpy(dtype=np.float64)
            se = part["se"].to_numpy(dtype=np.float64)
            ax.plot(x, y, marker="o", color=colors[is_correct], label=labels[is_correct])
            ax.fill_between(x, y - se, y + se, color=colors[is_correct], alpha=0.16, linewidth=0)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.0, 1.0)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gap(gap: pd.DataFrame, score_col: str, ylabel: str, title: str, path: Path, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for layer, part in gap.groupby("layer"):
        x = part["rel_bin_mid"].to_numpy(dtype=np.float64)
        y = part["gap_mean"].to_numpy(dtype=np.float64)
        se = part["gap_se"].to_numpy(dtype=np.float64)
        ax.plot(x, y, marker="o", label=f"Layer {layer}")
        ax.fill_between(x, y - se, y + se, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color="#222222", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_auc(auc_df: pd.DataFrame, title: str, path: Path, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for layer, part in auc_df.groupby("layer"):
        ax.plot(
            part["rel_bin_mid"].to_numpy(dtype=np.float64),
            part["best_auc"].to_numpy(dtype=np.float64),
            marker="o",
            label=f"Layer {layer}",
        )
    ax.axhline(0.5, color="#222222", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("within-question AUROC (best direction)")
    ax.set_ylim(0.45, max(0.75, float(auc_df["best_auc"].max()) + 0.03))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, value_col: str, title: str, path: Path, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    pivot = table.pivot(index="layer", columns="rel_bin", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="viridis", vmin=0.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c) * 10}-{int(c + 1) * 10}%" for c in pivot.columns], rotation=45, ha="right")
    ax.set_xlabel(f"{xlabel} bin")
    ax.set_ylabel("layer")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summaries: dict[str, pd.DataFrame],
    figures: list[Path],
    segment: str,
    xlabel: str,
) -> None:
    report = output_dir / "SEMANTIC_STEP_RELATIVE_ANALYSIS.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Semantic-Step Relative Analysis\n\n")
        f.write(
            "This analysis compares correct vs wrong rollouts within the same question "
            f"across {xlabel} bins.\n\n"
        )
        f.write(f"- Segment: `{segment}`\n")
        f.write(f"- Progress axis: `{xlabel}`\n\n")
        f.write("## Figures\n\n")
        for fig in figures:
            f.write(f"- `figures/{fig.name}`\n")
        f.write("\n## Top Relative-Bin AUROC\n\n")
        auc = summaries["state_margin_auc"].sort_values("best_auc", ascending=False).head(10)
        f.write("| layer | rel bin | questions | AUROC(+margin) | AUROC(-margin) | best |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for row in auc.itertuples(index=False):
            f.write(
                f"| {row.layer} | {row.rel_bin} | {row.questions} | "
                f"{row.auc_mean:.4f} | {row.auc_neg_mean:.4f} | {row.best_auc:.4f} |\n"
            )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    steps = pd.read_parquet(input_dir / "semantic_step_basin_steps.parquet")
    steps = steps[steps["layer"].isin(layers)].copy()
    steps["question_id"] = steps["question_id"].astype(str)
    segment = segment_label(steps)
    xlabel = progress_label(segment)
    steps = add_angle_columns(add_relative_bins(steps, args.bins))
    mixed = mixed_question_ids(steps)
    steps = steps[steps["question_id"].isin(mixed)].copy()

    summaries: dict[str, pd.DataFrame] = {}
    figures: list[Path] = []

    state_angle_curves = summarize_curves(steps, "state_correct_angle_deg")
    turn_angle_curves = summarize_curves(steps, "turn_correct_angle_deg")
    state_margin_curves = summarize_curves(steps, "state_margin")
    step_gain_curves = summarize_curves(steps, "step_correct_gain")
    turn_margin_curves = summarize_curves(steps, "turn_to_correct")
    state_gap = correct_wrong_gap_by_bin(steps, "state_margin")
    turn_gap = correct_wrong_gap_by_bin(steps, "turn_to_correct")
    state_auc = within_question_auc_by_bin(steps, "state_margin")
    turn_auc = within_question_auc_by_bin(steps, "turn_to_correct")

    summaries["state_margin_auc"] = state_auc
    summaries["turn_to_correct_auc"] = turn_auc

    state_angle_curves.to_csv(output_dir / "relative_state_angle_curves.csv", index=False)
    turn_angle_curves.to_csv(output_dir / "relative_turn_angle_curves.csv", index=False)
    state_gap.to_csv(output_dir / "relative_state_margin_gap.csv", index=False)
    turn_gap.to_csv(output_dir / "relative_turn_to_correct_gap.csv", index=False)
    state_auc.to_csv(output_dir / "relative_state_margin_auc.csv", index=False)
    turn_auc.to_csv(output_dir / "relative_turn_to_correct_auc.csv", index=False)

    fig = figures_dir / "G1_relative_state_angle_curve.png"
    plot_correct_wrong_curves(
        state_angle_curves,
        "state_correct_angle_deg",
        "angle to correct-answer direction (deg)",
        "Cumulative state angle to correct-answer direction",
        fig,
        xlabel,
    )
    figures.append(fig)

    fig = figures_dir / "G2_relative_turn_angle_curve.png"
    plot_correct_wrong_curves(
        turn_angle_curves,
        "turn_correct_angle_deg",
        "step angle to correct-answer direction (deg)",
        "Per-step movement angle to correct-answer direction",
        fig,
        xlabel,
    )
    figures.append(fig)

    fig = figures_dir / "G3_correct_wrong_gap_by_relative_step.png"
    plot_gap(
        state_gap,
        "state_margin",
        "correct - wrong rollout state margin",
        f"Within-question state-margin gap over {segment} progress",
        fig,
        xlabel,
    )
    figures.append(fig)

    fig = figures_dir / "G4_within_question_auc_by_relative_step.png"
    plot_auc(state_auc, f"Within-question AUROC by relative {segment} progress", fig, xlabel)
    figures.append(fig)

    fig = figures_dir / "G5_margin_vs_angle_relative_heatmap.png"
    plot_heatmap(state_auc, "best_auc", "Best state-margin AUROC by layer and relative bin", fig, xlabel)
    figures.append(fig)

    fig = figures_dir / "G6_turn_to_correct_gap_by_relative_step.png"
    plot_gap(
        turn_gap,
        "turn_to_correct",
        "correct - wrong rollout turn margin",
        f"Within-question turn-to-correct gap over {segment} progress",
        fig,
        xlabel,
    )
    figures.append(fig)

    fig = figures_dir / "G7_turn_to_correct_auc_by_relative_step.png"
    plot_auc(turn_auc, f"Turn-to-correct AUROC by relative {segment} progress", fig, xlabel)
    figures.append(fig)

    fig = figures_dir / "G8_state_margin_curve.png"
    plot_correct_wrong_curves(
        state_margin_curves,
        "state_margin",
        "correct-answer state margin",
        "State margin over relative progress",
        fig,
        xlabel,
    )
    figures.append(fig)

    fig = figures_dir / "G9_step_correct_gain_curve.png"
    plot_correct_wrong_curves(
        step_gain_curves,
        "step_correct_gain",
        "step correct-margin gain",
        "Step correct-margin gain over relative progress",
        fig,
        xlabel,
    )
    figures.append(fig)

    write_report(output_dir, summaries, figures, segment, xlabel)
    print(f"input: {input_dir}")
    print(f"mixed questions: {len(mixed)}")
    print(f"saved report: {output_dir / 'SEMANTIC_STEP_RELATIVE_ANALYSIS.md'}")
    for fig in figures:
        print(f"saved figure: {fig}")


if __name__ == "__main__":
    main()
