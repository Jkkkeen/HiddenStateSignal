#!/usr/bin/env python3
"""Plot and embed supplemental diagnostics for Experiment O."""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = ("A", "B", "C", "D")
LOGIT_COLUMNS = tuple(f"logit_{label}" for label in LABELS)
ID_COLUMNS = ("question_id", "rollout_id")
COLORS = {
    "A": "#2563eb",
    "B": "#d97706",
    "C": "#059669",
    "D": "#dc2626",
    "all": "#111827",
}
FIGURE_NAMES = {
    "O4_CHOSEN_MARGIN": "O4_selected_answer_margin_diagnostics.png",
    "O5_ABCD_LOGITS": "O5_abcd_logit_trajectories.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def prepare_probe_frame(probes: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        probes,
        ID_COLUMNS
        + (
            "segment",
            "frac",
            "is_correct",
            "pred_answer",
            "chosen_margin_max",
        )
        + LOGIT_COLUMNS,
    )
    frame = probes.copy()
    frame["pred_answer"] = frame["pred_answer"].astype(str)
    invalid = sorted(set(frame["pred_answer"]) - set(LABELS))
    if invalid:
        raise ValueError(f"invalid pred_answer labels: {invalid}")
    if frame[list(LOGIT_COLUMNS) + ["chosen_margin_max"]].isna().any().any():
        raise ValueError("probe logits and chosen_margin_max must be complete")
    return frame


def add_prompt_deltas(frame: pd.DataFrame, value_columns: tuple[str, ...]) -> pd.DataFrame:
    prompt = frame[frame["segment"].eq("prompt")].copy()
    duplicates = prompt.duplicated(list(ID_COLUMNS), keep=False)
    if duplicates.any():
        raise ValueError("expected exactly one prompt probe per rollout")
    prompt_columns = list(ID_COLUMNS) + list(value_columns)
    prompt = prompt[prompt_columns].rename(
        columns={column: f"{column}_prompt" for column in value_columns}
    )
    thinking = frame[frame["segment"].eq("think")].copy()
    thinking = thinking.merge(prompt, on=list(ID_COLUMNS), how="left", validate="many_to_one")
    for column in value_columns:
        baseline = f"{column}_prompt"
        if thinking[baseline].isna().any():
            raise ValueError(f"missing prompt baseline for {column}")
        thinking[f"delta_{column}"] = thinking[column] - thinking[baseline]
    return thinking


def mean_and_ci(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    summary = (
        frame.groupby(group_columns, sort=True)[value_column]
        .agg(count="count", mean="mean", std="std")
        .reset_index()
    )
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    return summary


def draw_summary_line(
    ax: plt.Axes,
    summary: pd.DataFrame,
    label: str,
    color: str,
    *,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha_band: float = 0.10,
) -> None:
    x = summary["frac"].to_numpy(dtype=float)
    mean = summary["mean"].to_numpy(dtype=float)
    ci95 = summary["ci95"].to_numpy(dtype=float)
    ax.plot(
        x,
        mean,
        marker="o",
        markersize=4,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        label=label,
    )
    if alpha_band > 0:
        ax.fill_between(x, mean - ci95, mean + ci95, color=color, alpha=alpha_band, linewidth=0)


def style_axis(ax: plt.Axes, ylabel: str, *, zero_reference: bool = True) -> None:
    if zero_reference:
        ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_xlim(0.03, 1.02)
    ax.set_xticks((0.05, 0.20, 0.40, 0.60, 0.80, 1.00))
    ax.set_xlabel("Thinking progress")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def plot_selected_answer_margin(thinking: pd.DataFrame, output_path: Path) -> None:
    wrong = thinking[~thinking["is_correct"].astype(bool)].copy()
    if wrong.empty:
        raise ValueError("no incorrect rollouts available for selected-answer diagnostics")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    panels = (
        ("chosen_margin_max", "Absolute selected-answer margin"),
        ("delta_chosen_margin_max", "Change from prompt baseline"),
    )
    for ax, (value_column, title) in zip(axes, panels, strict=True):
        for label in LABELS:
            subset = wrong[wrong["pred_answer"].eq(label)]
            summary = mean_and_ci(subset, ["frac"], value_column)
            rollout_count = subset[list(ID_COLUMNS)].drop_duplicates().shape[0]
            draw_summary_line(
                ax,
                summary,
                f"Selected {label} (n={rollout_count})",
                COLORS[label],
            )
        overall = mean_and_ci(wrong, ["frac"], value_column)
        draw_summary_line(
            ax,
            overall,
            f"All incorrect (n={wrong[list(ID_COLUMNS)].drop_duplicates().shape[0]})",
            COLORS["all"],
            linestyle="--",
            linewidth=2.4,
            alpha_band=0.06,
        )
        style_axis(ax, "Chosen logit - max(other logits)")
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.legend(frameon=False, fontsize=8, ncol=1)

    fig.suptitle(
        "O4. Final selected-answer margin on incorrect rollouts",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Bands show pointwise 95% normal-approximation confidence intervals; groups use the rollout's final selected label.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_abcd_logits(thinking: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    panels = (
        ("raw", "Raw option-token logits"),
        ("delta", "Change from prompt baseline"),
    )
    for ax, (mode, title) in zip(axes, panels, strict=True):
        for label, logit_column in zip(LABELS, LOGIT_COLUMNS, strict=True):
            value_column = logit_column if mode == "raw" else f"delta_{logit_column}"
            summary = mean_and_ci(thinking, ["frac"], value_column)
            draw_summary_line(ax, summary, f"Option {label}", COLORS[label])
        ylabel = "Mean raw logit" if mode == "raw" else "Mean logit change from prompt"
        style_axis(ax, ylabel, zero_reference=mode != "raw")
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.legend(frameon=False, fontsize=9, ncol=2)

    fig.suptitle(
        "O5. A/B/C/D option-logit trajectories",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "All 3,858 rollouts; bands show pointwise 95% normal-approximation confidence intervals.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def embed_figures(report_path: Path, figures: dict[str, Path]) -> None:
    with report_path.open("r", encoding="utf-8", newline="") as handle:
        html = handle.read()
    for figure_id, figure_path in figures.items():
        payload = base64.b64encode(figure_path.read_bytes()).decode("ascii")
        data_uri = f"data:image/png;base64,{payload}"
        pattern = re.compile(
            rf'(<img\b[^>]*\bdata-figure-id="{re.escape(figure_id)}"[^>]*\bsrc=")[^"]*(")'
        )
        html, count = pattern.subn(rf"\g<1>{data_uri}\g<2>", html, count=1)
        if count != 1:
            raise ValueError(f"expected one HTML image placeholder for {figure_id}, found {count}")
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(html)


def main() -> None:
    args = parse_args()
    probes = prepare_probe_frame(pd.read_parquet(args.input))
    thinking = add_prompt_deltas(probes, ("chosen_margin_max",) + LOGIT_COLUMNS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figures = {key: args.output_dir / name for key, name in FIGURE_NAMES.items()}
    plot_selected_answer_margin(thinking, figures["O4_CHOSEN_MARGIN"])
    plot_abcd_logits(thinking, figures["O5_ABCD_LOGITS"])
    if args.report is not None:
        embed_figures(args.report, figures)

    incorrect_rollouts = (
        thinking.loc[~thinking["is_correct"].astype(bool), list(ID_COLUMNS)]
        .drop_duplicates()
        .shape[0]
    )
    print(f"incorrect rollouts: {incorrect_rollouts}")
    for figure_id, path in figures.items():
        print(f"saved {figure_id}: {path} ({path.stat().st_size} bytes)")
    if args.report is not None:
        print(f"embedded figures in: {args.report}")


if __name__ == "__main__":
    main()
