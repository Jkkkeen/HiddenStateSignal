#!/usr/bin/env python3
"""Plot rollout-level Experiment 0 signals over relative reasoning progress."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CORRECT_COLOR = "#167d5a"
WRONG_COLOR = "#c04a3f"
ADDENDUM_MARKER = "## Rollout-Progress Spaghetti Addendum"


@dataclass(frozen=True)
class SignalSpec:
    feature: str
    representation: str
    layer: int
    panel_title: str
    y_label: str


FROZEN_SIGNALS = (
    SignalSpec(
        "span_movement_norm_median",
        "mean_w128_s64",
        24,
        "Horizontal movement amplitude",
        "Median span movement norm",
    ),
    SignalSpec(
        "span_turn_cos_median",
        "mean_w128_s64",
        24,
        "Horizontal movement turning",
        "Median span turning cosine",
    ),
    SignalSpec(
        "vertical_norm_p90",
        "token",
        15,
        "Vertical activity burst",
        "P90 vertical update norm",
    ),
    SignalSpec(
        "coordinate_entropy_mean",
        "token",
        17,
        "Vertical coordinate entropy",
        "Mean normalized coordinate entropy",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def select_signal(frame: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    required = {
        "question_id",
        "rollout_id",
        "is_correct",
        "representation",
        "layer",
        "progress_bin",
        spec.feature,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    columns = [
        "question_id",
        "rollout_id",
        "is_correct",
        "representation",
        "layer",
        "progress_bin",
        spec.feature,
    ]
    selected = frame.loc[
        (frame["representation"] == spec.representation)
        & (frame["layer"] == spec.layer),
        columns,
    ].copy()
    selected = selected.rename(columns={spec.feature: "value"})
    selected["progress_bin"] = pd.to_numeric(
        selected["progress_bin"], errors="coerce"
    )
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    selected = selected[
        np.isfinite(selected["progress_bin"])
        & np.isfinite(selected["value"])
    ].copy()
    selected["progress_bin"] = selected["progress_bin"].astype(int)
    if not selected["progress_bin"].between(0, 9).all():
        invalid = sorted(selected.loc[~selected["progress_bin"].between(0, 9), "progress_bin"].unique())
        raise ValueError(f"progress bins outside 0-9 for {spec.feature}: {invalid}")
    if selected.empty:
        raise ValueError(
            f"no finite rows for {spec.feature} | {spec.representation} | L{spec.layer}"
        )
    keys = ["question_id", "rollout_id", "progress_bin"]
    if selected.duplicated(keys).any():
        raise ValueError(f"duplicate rollout/bin rows for {spec.feature}")

    labels_per_rollout = selected.groupby(
        ["question_id", "rollout_id"], observed=True
    )["is_correct"].nunique()
    if (labels_per_rollout > 1).any():
        raise ValueError(f"inconsistent labels within rollout for {spec.feature}")
    selected["is_correct"] = selected["is_correct"].astype(bool)
    return selected.sort_values(keys, kind="stable").reset_index(drop=True)


def question_weighted_summary(
    selected: pd.DataFrame,
    spec: SignalSpec,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Compute pointwise means after giving each question one equal-weight value."""
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    question_means = (
        selected.groupby(
            ["question_id", "is_correct", "progress_bin"],
            as_index=False,
            observed=True,
        )["value"]
        .mean()
    )
    rollout_coverage = (
        selected[["question_id", "rollout_id", "is_correct", "progress_bin"]]
        .drop_duplicates()
        .groupby(["is_correct", "progress_bin"], observed=True)
        .size()
    )

    rows: list[dict[str, object]] = []
    grouped = question_means.groupby(
        ["is_correct", "progress_bin"], sort=True, observed=True
    )
    for offset, ((label, progress_bin), group) in enumerate(grouped):
        values = group["value"].to_numpy(dtype=np.float64)
        rng = np.random.default_rng(seed + offset)
        sampled_means = rng.choice(
            values, size=(bootstrap, len(values)), replace=True
        ).mean(axis=1)
        rows.append(
            {
                "feature": spec.feature,
                "representation": spec.representation,
                "layer": spec.layer,
                "is_correct": bool(label),
                "progress_bin": int(progress_bin),
                "mean": float(values.mean()),
                "ci_low": float(np.quantile(sampled_means, 0.025)),
                "ci_high": float(np.quantile(sampled_means, 0.975)),
                "n_questions": int(group["question_id"].nunique()),
                "n_rollouts": int(rollout_coverage.loc[(label, progress_bin)]),
            }
        )
    return pd.DataFrame(rows)


def rollout_curve(rollout: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return bins 0-9 and raw values, retaining absent bins as visible gaps."""
    values = rollout.set_index("progress_bin")["value"].reindex(range(10))
    return np.arange(10), values.to_numpy(dtype=np.float64)


def _summary_curve(
    summary: pd.DataFrame, spec: SignalSpec, label: bool
) -> pd.DataFrame:
    curve = summary.loc[
        (summary["feature"] == spec.feature)
        & (summary["is_correct"] == label)
    ].set_index("progress_bin")
    return curve.reindex(range(10))


def plot_spaghetti_panels(
    selected_by_signal: list[tuple[SignalSpec, pd.DataFrame]],
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    colors = {True: CORRECT_COLOR, False: WRONG_COLOR}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

    for axis, (spec, selected) in zip(axes.ravel(), selected_by_signal):
        grouped = selected.groupby(["question_id", "rollout_id"], sort=False)
        for _, rollout in grouped:
            if rollout["progress_bin"].nunique() < 2:
                continue
            label = bool(rollout.iloc[0]["is_correct"])
            x, y = rollout_curve(rollout)
            axis.plot(
                x,
                y,
                color=colors[label],
                alpha=0.10,
                linewidth=0.75,
                zorder=1,
            )

        for label in (True, False):
            curve = _summary_curve(summary, spec, label)
            x = np.arange(10, dtype=float)
            mean = curve["mean"].to_numpy(dtype=float)
            low = curve["ci_low"].to_numpy(dtype=float)
            high = curve["ci_high"].to_numpy(dtype=float)
            axis.fill_between(
                x,
                low,
                high,
                color=colors[label],
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
            axis.plot(
                x,
                mean,
                color=colors[label],
                linewidth=2.6,
                marker="o",
                markersize=4,
                zorder=3,
            )

        axis.set_title(
            f"{spec.panel_title}\n{spec.feature} | {spec.representation} | L{spec.layer}",
            fontsize=11,
        )
        axis.set_ylabel(spec.y_label)
        axis.set_xticks(range(10))
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes[-1]:
        axis.set_xlabel("Relative-progress bin")

    handles = [
        Line2D(
            [0], [0], color=colors[True], alpha=0.25, linewidth=1,
            label="Correct rollout",
        ),
        Line2D(
            [0], [0], color=colors[False], alpha=0.25, linewidth=1,
            label="Wrong rollout",
        ),
        Line2D(
            [0], [0], color=colors[True], linewidth=2.6,
            label="Correct question-weighted mean",
        ),
        Line2D(
            [0], [0], color=colors[False], linewidth=2.6,
            label="Wrong question-weighted mean",
        ),
        Patch(
            facecolor="#777777", alpha=0.18,
            label="95% question-bootstrap CI",
        ),
    ]
    fig.suptitle(
        "Qwen3-VL-8B-Thinking: rollout-level hidden dynamics",
        fontsize=15,
        y=0.99,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.955),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.89), h_pad=2.3, w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _format_value(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.6g}"


def _label_bin_mean(
    panel: pd.DataFrame, label: bool, progress_bin: int
) -> float | None:
    values = panel.loc[
        (panel["is_correct"] == label)
        & (panel["progress_bin"] == progress_bin),
        "mean",
    ]
    return None if values.empty else float(values.iloc[0])


def _panel_description(spec: SignalSpec, summary: pd.DataFrame) -> str:
    panel = summary[summary["feature"] == spec.feature]
    correct = panel[panel["is_correct"]][["progress_bin", "mean"]].rename(
        columns={"mean": "correct_mean"}
    )
    wrong = panel[~panel["is_correct"]][["progress_bin", "mean"]].rename(
        columns={"mean": "wrong_mean"}
    )
    paired = correct.merge(wrong, on="progress_bin", how="inner")
    if paired.empty:
        largest = "No bin has both labels."
    else:
        paired["gap"] = paired["correct_mean"] - paired["wrong_mean"]
        row = paired.loc[paired["gap"].abs().idxmax()]
        largest = (
            f"Largest absolute displayed mean gap: bin {int(row['progress_bin'])}, "
            f"correct - wrong = {_format_value(float(row['gap']))}."
        )
    return (
        f"- `{spec.feature}` (`{spec.representation}`, L{spec.layer}): "
        f"bin 0 correct={_format_value(_label_bin_mean(panel, True, 0))}, "
        f"wrong={_format_value(_label_bin_mean(panel, False, 0))}; "
        f"bin 9 correct={_format_value(_label_bin_mean(panel, True, 9))}, "
        f"wrong={_format_value(_label_bin_mean(panel, False, 9))}. {largest}"
    )


def write_report(
    path: Path,
    frame: pd.DataFrame,
    selected_by_signal: list[tuple[SignalSpec, pd.DataFrame]],
    summary: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> None:
    cohort = frame[["question_id", "rollout_id", "is_correct"]].drop_duplicates()
    if cohort.duplicated(["question_id", "rollout_id"]).any():
        raise ValueError("input has inconsistent labels for a rollout")
    correct_count = int(cohort["is_correct"].astype(bool).sum())
    wrong_count = int(len(cohort) - correct_count)

    coverage_lines = []
    for spec, selected in selected_by_signal:
        bins_per_rollout = selected.groupby(
            ["question_id", "rollout_id"], observed=True
        )["progress_bin"].nunique()
        coverage_lines.append(
            f"- `{spec.feature}`: {int((bins_per_rollout >= 2).sum())} plotted, "
            f"{int((bins_per_rollout < 2).sum())} excluded for fewer than two finite bins."
        )

    descriptions = "\n".join(
        _panel_description(spec, summary) for spec, _ in selected_by_signal
    )
    coverage = "\n".join(coverage_lines)
    text = f"""# E0-F8 Rollout-Progress Spaghetti Curves

## Cohort And Frozen Specification

- Input cohort: {cohort['question_id'].nunique()} questions and {len(cohort)} rollouts ({correct_count} correct, {wrong_count} wrong).
- Model/setting: Qwen3-VL-8B-Thinking long-response Experiment 0.
- Progress: the saved relative-progress bins 0-9; no smoothing, interpolation, normalization, response generation, or hidden-state forward pass.
- Bootstrap: {bootstrap:,} question resamples, seed `{seed}`; bands are pointwise 95% percentile intervals.

The four panels were frozen before viewing E0-F8: horizontal movement amplitude (`span_movement_norm_median`, `mean_w128_s64`, L24), horizontal turning (`span_turn_cos_median`, `mean_w128_s64`, L24), vertical activity burst (`vertical_norm_p90`, `token`, L15), and vertical coordinate entropy (`coordinate_entropy_mean`, `token`, L17).

## Computation

Each faint line is one rollout's saved raw feature value across progress bins. If a bin is absent, its value is `NaN`, so the line has a gap. A rollout with fewer than two finite bins is not drawn.

For question `q`, label `y`, and bin `b`, the first aggregation is

`question_mean(q,y,b) = mean over eligible rollouts r of x(q,r,b)`.

The thick displayed curve is the unweighted mean of these question means, so every question contributes equally even if it has a different number of correct or wrong rollouts. The confidence band resamples questions, not individual rollout rows.

## Line Coverage

{coverage}

Per-bin `n_questions` and `n_rollouts` are recorded in `long_experiment_0_spaghetti_summary.csv`.

## Descriptive Readout

{descriptions}

These are descriptive trajectory plots. Visual separation or confidence-band overlap is not a hypothesis test and is not used here as a significance claim.

## Artifacts

- `figures/E0_F8_rollout_progress_spaghetti.png`
- `long_experiment_0_spaghetti_summary.csv`
"""
    path.write_text(text, encoding="utf-8")


def append_parent_report(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if ADDENDUM_MARKER in text:
        return
    addition = (
        "\n## Rollout-Progress Spaghetti Addendum\n\n"
        "- `figures/E0_F8_rollout_progress_spaghetti.png`\n"
        "- `long_experiment_0_spaghetti_summary.csv`\n"
        "- `E0_F8_ROLLOUT_PROGRESS_REPORT.md`\n"
    )
    path.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    frame = pd.read_parquet(args.input)
    selected_by_signal: list[tuple[SignalSpec, pd.DataFrame]] = []
    summaries = []
    for index, spec in enumerate(FROZEN_SIGNALS):
        selected = select_signal(frame, spec)
        selected_by_signal.append((spec, selected))
        summaries.append(
            question_weighted_summary(
                selected,
                spec,
                bootstrap=args.bootstrap,
                seed=args.seed + index * 100,
            )
        )
    summary = pd.concat(summaries, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "figures/E0_F8_rollout_progress_spaghetti.png"
    summary_path = args.output_dir / "long_experiment_0_spaghetti_summary.csv"
    report_path = args.output_dir / "E0_F8_ROLLOUT_PROGRESS_REPORT.md"
    plot_spaghetti_panels(selected_by_signal, summary, figure_path)
    summary.to_csv(summary_path, index=False)
    write_report(
        report_path,
        frame,
        selected_by_signal,
        summary,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    for artifact in (figure_path, summary_path, report_path):
        if not artifact.exists() or artifact.stat().st_size == 0:
            raise RuntimeError(f"missing or empty artifact: {artifact}")
    append_parent_report(args.output_dir / "LONG_EXPERIMENT_0_RESULTS.md")

    cohort = frame[["question_id", "rollout_id"]].drop_duplicates()
    print(
        f"Completed E0-F8: {frame['question_id'].nunique()} questions, "
        f"{len(cohort)} rollouts, {len(FROZEN_SIGNALS)} signals, "
        "H200 accessed: False"
    )


if __name__ == "__main__":
    main()
