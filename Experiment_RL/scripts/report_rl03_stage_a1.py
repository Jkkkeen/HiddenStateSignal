#!/usr/bin/env python3
"""Build the local final report for the completed RL03 Stage A1 audit."""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_CELL = {
    "interface_id": "I2",
    "representation": "content_sequence",
    "prompt_mode": "neutralized_letter_instruction",
}
CONTENT_INTERFACES = ("I1", "I2")
CORRECT_COLORS = {False: "#d95f02", True: "#1b6ca8"}
CELL_COLORS = {
    "I0": "#6b7280",
    "I1": "#0f766e",
    "I2": "#7c3aed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--html-report", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def cell_mask(frame: pd.DataFrame, cell: dict[str, str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in cell.items():
        mask &= frame[column].eq(value)
    return mask


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_level_trajectory(t1: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    panels = ("raw", "within_question_centered")
    for row_index, interface_id in enumerate(CONTENT_INTERFACES):
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            view = t1[
                t1["interface_id"].eq(interface_id)
                & t1["representation"].eq("content_sequence")
                & t1["prompt_mode"].eq("neutralized_letter_instruction")
                & t1["panel"].eq(panel)
            ]
            for is_correct, group in view.groupby("is_correct"):
                group = group.sort_values("frac")
                label = "Correct" if bool(is_correct) else "Incorrect"
                color = CORRECT_COLORS[bool(is_correct)]
                axis.plot(
                    group["frac"],
                    group["estimate"],
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=color,
                    label=label,
                )
                axis.fill_between(
                    group["frac"],
                    group["ci_low"],
                    group["ci_high"],
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
            axis.axvspan(0.90, 1.00, color="#f3f4f6", alpha=0.8, zorder=-1)
            axis.set_title(
                f"{interface_id}: "
                + ("Raw gold level" if panel == "raw" else "Within-question centered"),
                fontsize=11,
                fontweight="semibold",
            )
            axis.set_ylabel("Gold answer log-likelihood")
            axis.set_xlim(-0.01, 1.01)
            axis.set_xticks((0.0, 0.25, 0.50, 0.75, 1.0))
            style_axis(axis)
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Thinking progress")
    figure.suptitle(
        "F1. Gold answer-likelihood level separates late, across neutral interfaces",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def within_question_center(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    centered = frame.copy()
    for column in columns:
        centered[f"{column}_centered"] = centered[column] - centered.groupby("question_id")[column].transform("mean")
    return centered


def plot_gain_vs_level(features: pd.DataFrame, output_path: Path) -> None:
    view = features[cell_mask(features, PRIMARY_CELL)].copy()
    if view.empty:
        raise ValueError("primary I2 content cell is missing")
    view = within_question_center(view, ("gold_level_90", "gold_gain_25_90"))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    panels = (
        ("gold_level_90", "gold_gain_25_90", "Raw rollout values"),
        ("gold_level_90_centered", "gold_gain_25_90_centered", "Within-question centered"),
    )
    for axis, (x_column, y_column, title) in zip(axes, panels, strict=True):
        for is_correct, group in view.groupby("is_correct"):
            axis.scatter(
                group[x_column],
                group[y_column],
                s=13,
                alpha=0.24,
                linewidths=0,
                color=CORRECT_COLORS[bool(is_correct)],
                label="Correct" if bool(is_correct) else "Incorrect",
            )
        correlation = view[x_column].corr(view[y_column], method="spearman")
        axis.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1)
        axis.axvline(0.0, color="#9ca3af", linestyle="--", linewidth=1)
        axis.set_title(f"{title}\nSpearman rho = {correlation:.3f}", fontsize=11, fontweight="semibold")
        axis.set_xlabel("Gold level at 90%")
        axis.set_ylabel("Gold gain from 25% to 90%")
        style_axis(axis)
    axes[0].legend(frameon=False)
    figure.suptitle(
        "F2. The proposed gain is strongly coupled to late confidence",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def cell_label(row: dict[str, Any]) -> str:
    prompt = "legacy" if row["prompt_mode"] == "legacy_letter_instruction" else "neutral"
    representation = {
        "content_sequence": "content",
        "letter_first": "letter-first",
        "letter_sequence": "letter-seq",
    }.get(str(row["representation"]), str(row["representation"]))
    return f"{row['interface_id']} {representation} ({prompt})"


def plot_incremental_auc(summary: dict[str, Any], output_path: Path) -> None:
    paired = [dict(row) for row in summary["paired_gain_minus_level"]]
    crossfit = {
        (row["interface_id"], row["representation"], row["prompt_mode"]): row
        for row in summary["cross_fitted_comparisons"]
    }
    paired.sort(
        key=lambda row: (
            row["representation"] != "content_sequence",
            row["interface_id"],
            row["representation"],
            row["prompt_mode"],
        )
    )

    figure, axis = plt.subplots(figsize=(10.5, 6.1))
    y_positions = np.arange(len(paired))[::-1]
    for y_position, row in zip(y_positions, paired, strict=True):
        delta = float(row["mean_delta_auc"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        color = CELL_COLORS[str(row["interface_id"])]
        axis.errorbar(
            delta,
            y_position,
            xerr=np.asarray([[delta - ci_low], [ci_high - delta]]),
            fmt="o",
            markersize=6,
            capsize=3,
            color=color,
        )
        key = (row["interface_id"], row["representation"], row["prompt_mode"])
        improvement = float(crossfit[key]["improvement"])
        axis.scatter(improvement, y_position, marker="D", s=30, color="#111827", zorder=3)

    axis.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1.2)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([cell_label(row) for row in paired], fontsize=9)
    axis.set_xlabel("Increment over matched late level (AUC points)")
    axis.set_title(
        "F3. Gain does not add reliable discrimination beyond late level\n"
        "Colored circle: paired AUC delta with 95% CI; black diamond: cross-fit improvement",
        fontsize=13,
        fontweight="bold",
    )
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_interval_auc(t4: pd.DataFrame, output_path: Path) -> None:
    cells = (
        ("I0", "letter_first", "legacy_letter_instruction", "I0 letter anchor"),
        ("I1", "content_sequence", "neutralized_letter_instruction", "I1 gold content"),
        ("I2", "content_sequence", "neutralized_letter_instruction", "I2 gold content"),
    )
    figure, axis = plt.subplots(figsize=(9.5, 5.3))
    for interface_id, representation, prompt_mode, label in cells:
        view = t4[
            t4["interface_id"].eq(interface_id)
            & t4["representation"].eq(representation)
            & t4["prompt_mode"].eq(prompt_mode)
        ].sort_values("frac_mid")
        color = CELL_COLORS[interface_id]
        axis.plot(
            view["frac_mid"],
            view["estimate"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            label=label,
        )
        axis.fill_between(
            view["frac_mid"],
            view["ci_low"],
            view["ci_high"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    axis.axhline(0.5, color="#6b7280", linestyle="--", linewidth=1.2)
    axis.axvspan(0.90, 1.00, color="#f3f4f6", alpha=0.9, zorder=-1)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.43, 0.86)
    axis.set_xlabel("Adjacent interval midpoint")
    axis.set_ylabel("Within-question AUC of interval gain")
    axis.set_title(
        "F4. Interval discrimination is concentrated near the end of thinking",
        fontsize=14,
        fontweight="bold",
    )
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_final_report(run_root: Path, summary: dict[str, Any]) -> Path:
    gate_inputs = summary["gate_inputs"]
    pipeline = summary["pipeline"]
    path = run_root / "RL03_STAGE_A1_FINAL_REPORT.md"
    lines = [
        "# RL03 Stage A1 Final Decision Report",
        "",
        "## Decision",
        "",
        "**LEVEL-EQUIVALENT**",
        "",
        "The full Stage A1 audit completed successfully, but the answer-likelihood gain does not provide incremental discrimination beyond matched late confidence. The current process-reward branch stops before Stage B and Stage C.",
        "",
        "## Run Integrity",
        "",
        "| item | result |",
        "|---|---:|",
        f"| expected atomic scores | {int(pipeline['expected_requests'])} |",
        f"| completed atomic scores | {int(pipeline['completed_requests'])} |",
        f"| completion rate | {float(pipeline['completion_rate']):.4f} |",
        f"| finite score rate | {float(pipeline['finite_score_rate']):.4f} |",
        f"| scoring failures | {int(pipeline['score_failures'])} |",
        f"| questions | {int(summary['questions'])} |",
        f"| rollouts | {int(summary['rollouts'])} |",
        "",
        "## Primary Evidence",
        "",
        "| metric | result |",
        "|---|---:|",
        f"| I2 gold-content gain AUC | {float(gate_inputs['gain_auc']):.4f} |",
        f"| gain AUC 95% CI | [{float(gate_inputs['gain_auc_ci_low']):.4f}, {float(gate_inputs['gain_auc_ci_high']):.4f}] |",
        f"| positive-correlation fraction | {float(gate_inputs['positive_correlation_fraction']):.4f} |",
        f"| matched late-level AUC | {float(gate_inputs['level_auc']):.4f} |",
        f"| trimmed-level AUC | {float(gate_inputs['trimmed_auc']):.4f} |",
        f"| paired gain-minus-level AUC | {float(gate_inputs['paired_delta_auc']):.4f} |",
        f"| paired delta 95% CI | [{float(gate_inputs['paired_delta_ci_low']):.4f}, {float(gate_inputs['paired_delta_ci_high']):.4f}] |",
        f"| cross-fit level+gain improvement | {float(gate_inputs['crossfit_improvement']):.4f} |",
        f"| I1-I2 content gain AUC difference | {float(gate_inputs['i1_i2_auc_difference']):.4f} |",
        "",
        "The gain clears its standalone AUC threshold, but the paired incremental effect is negative with a confidence interval below zero, and the cross-fitted model is worse after adding gain. This is stronger evidence for level equivalence than a confidence interval that merely crosses zero.",
        "",
        "## Figures",
        "",
        "![F1 level trajectory](figures/F1_primary_level_trajectory.png)",
        "",
        "![F2 gain versus level](figures/F2_gain_vs_level.png)",
        "",
        "![F3 incremental AUC](figures/F3_incremental_auc.png)",
        "",
        "![F4 interval AUC](figures/F4_interval_auc.png)",
        "",
        "## Scientific Interpretation",
        "",
        "Allowed claim:",
        "",
        "> Answer likelihood is a useful intermediate and final verifier signal.",
        "",
        "Not supported:",
        "",
        "> The current answer-likelihood gain provides trajectory-specific process supervision.",
        "",
        "The dense interval audit further shows that most interval-level discrimination appears after 80% of thinking and is strongest from 95% to 100%. Selecting a new favorable endpoint interval from this result would be post-hoc and would not repair the process claim.",
        "",
        "## Decision Path",
        "",
        "1. Do not start the current Stage B free-form experiment.",
        "2. Do not start Stage C GRPO with the current gold gain.",
        "3. Preserve the verifier result as Outcome B in the paper narrative.",
        "4. If process supervision remains the target, test one pre-registered non-telescoping instability feature on new questions.",
        "",
        "See `RL03_NON_TELESCOPING_REVISION_PLAN.md` for the proposed revision protocol.",
    ]
    discovery_summary_path = run_root / "non_telescope_discovery" / "metric_summary.json"
    if discovery_summary_path.is_file():
        discovery = load_json(discovery_summary_path)
        metrics = {row["feature"]: row for row in discovery["metrics"]}
        primary = metrics["primary_score"]
        drawdown = metrics["negative_max_drawdown"]
        primary_increment = discovery["incremental"]["primary_score"]
        lines.extend(
            [
                "",
                "## Non-Telescoping Discovery Follow-Up",
                "",
                f"Decision: **{discovery['decision']['label']}**",
                "",
                f"The frozen `-excess_total_variation` feature achieved AUC {primary['mean_auc']:.4f}, 95% CI [{primary['ci_low']:.4f}, {primary['ci_high']:.4f}]. Its paired delta versus `gold_level_90` was {primary_increment['paired']['mean_delta_auc']:.4f}, with 95% CI [{primary_increment['paired']['ci_low']:.4f}, {primary_increment['paired']['ci_high']:.4f}].",
                "",
                f"The strongest secondary diagnostic, `-maximum_drawdown`, achieved AUC {drawdown['mean_auc']:.4f}, still below the frozen gate and below late level. It is not promoted after outcome inspection.",
                "",
                "![Non-telescoping discovery](non_telescope_discovery/figures/N1_non_telescope_auc.png)",
                "",
                "No new H200 confirmation set is authorized for this revision.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_revision_plan(run_root: Path) -> Path:
    path = run_root / "RL03_NON_TELESCOPING_REVISION_PLAN.md"
    text = """# RL03 Non-Telescoping Revision Plan

## Status

The completed Stage A1 result is `LEVEL-EQUIVALENT`. The frozen primary feature below was evaluated on the existing discovery data and failed:

```text
-excess_total_variation AUC = 0.5080
95% CI = [0.4656, 0.5489]
paired delta versus gold_level_90 = -0.2253
paired 95% CI = [-0.2875, -0.1622]
cross-fit improvement = 0.0001
```

Decision: `STOP-NO-CONFIRMATION`. This document records the tested protocol; it does not authorize Stage B, Stage C, a new MathVerse holdout, or a new H200 run.

## Mechanistic Hypothesis

Correct reasoning should move toward the gold answer with less reversal and excess motion before the final conclusion. Endpoint gain cannot test this because:

```text
gain(25, 90) = level(90) - level(25)
```

The revised primary feature measures path inefficiency rather than endpoint displacement.

## Frozen Primary Feature

Use the neutralized I2 interface and gold answer-content sequence score. Use only numeric thinking probes:

```text
fractions = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
```

Exclude 95%, 100%, and trimmed-final from the primary feature.

For score sequence `s_0 ... s_k`:

```text
delta_i = s_(i+1) - s_i
total_variation = sum_i abs(delta_i)
net_displacement = abs(s_k - s_0)
excess_total_variation = total_variation - net_displacement
primary_score = -excess_total_variation
```

Higher `primary_score` means less backtracking. The frozen direction is positive for correctness.

## Secondary Diagnostics

These do not replace the primary feature:

```text
oscillation_fraction = excess_total_variation / (total_variation + 1e-8)
maximum_drawdown
sign_reversal_rate
mean_absolute_second_difference
gold_positive_gain_rate
gold_monotonicity_violation_rate
```

## Required Controls

```text
matched level control = gold_level_90
matched net-gain control = gold_gain_25_90
response length and thinking-token count
answer-label stratification
I1 versus I2 neutral interface comparison
surface-form robustness
question-clustered bootstrap
label permutation
```

Incremental comparisons must be question-held-out and cross-fitted. No threshold, interval, direction, or normalization may be changed after confirmatory outcomes are opened.

## Data Roles

The completed 434-question Stage A1 set is discovery-only for this revision because it motivated the feature. It cannot confirm the new claim.

Had discovery passed, the confirmatory set would have required new MathVerse questions excluded from all existing smoke500 and RL03 feature development. Discovery did not pass, so this data collection is cancelled.

## Confirmation Gate

The branch passes only if all conditions hold for `primary_score`:

```text
within-question AUC >= 0.70
question-bootstrap CI lower bound > 0.60
positive-correlation fraction >= 0.65
I1 versus I2 AUC difference <= 0.03
paired AUC delta over gold_level_90 > 0
paired-delta 95% CI lower bound > 0
cross-fitted level + primary_score improves over level-only
length-controlled direction remains positive
label-permutation AUC change <= 0.02
```

## Stop Rule

The feature failed before confirmation. Stop trajectory-likelihood reward development. Do not search additional intervals or promote `-maximum_drawdown` after observing its AUC. Stage B and Stage C remain blocked.
"""
    path.write_text(text, encoding="utf-8")
    return path


def embed_html_figures(report_path: Path, figures: dict[str, Path]) -> None:
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
    run_root = args.run_root
    audit = run_root / "audit"
    figures = run_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    summary = load_json(audit / "metric_summary.json")
    t1 = pd.read_csv(audit / "T1_level_trajectory.csv")
    t4 = pd.read_csv(audit / "T4_interval_discrimination.csv")
    features = pd.read_csv(audit / "rollout_features.csv")

    figure_paths = {
        "RL03_F1_LEVEL": figures / "F1_primary_level_trajectory.png",
        "RL03_F2_GAIN_LEVEL": figures / "F2_gain_vs_level.png",
        "RL03_F3_INCREMENT": figures / "F3_incremental_auc.png",
        "RL03_F4_INTERVAL": figures / "F4_interval_auc.png",
    }
    plot_level_trajectory(t1, figure_paths["RL03_F1_LEVEL"])
    plot_gain_vs_level(features, figure_paths["RL03_F2_GAIN_LEVEL"])
    plot_incremental_auc(summary, figure_paths["RL03_F3_INCREMENT"])
    plot_interval_auc(t4, figure_paths["RL03_F4_INTERVAL"])
    report_path = write_final_report(run_root, summary)
    plan_path = write_revision_plan(run_root)
    if args.html_report is not None:
        html_figures = dict(figure_paths)
        discovery_figure = run_root / "non_telescope_discovery" / "figures" / "N1_non_telescope_auc.png"
        if discovery_figure.is_file():
            html_figures["RL03_N1_NON_TELESCOPE"] = discovery_figure
        embed_html_figures(args.html_report, html_figures)

    print(f"gate: {summary['gate']['label']}")
    print(f"report: {report_path}")
    print(f"revision plan: {plan_path}")
    if args.html_report is not None:
        print(f"embedded figures in: {args.html_report}")
    for figure in sorted(figures.glob("*.png")):
        print(f"figure: {figure} ({figure.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
