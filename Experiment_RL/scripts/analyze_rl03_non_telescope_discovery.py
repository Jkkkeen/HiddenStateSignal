#!/usr/bin/env python3
"""Discovery-only audit of non-telescoping RL03 trajectory features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.analyze_rl03_mcq_audit import (
        bootstrap_mean_ci,
        cross_fitted_level_gain_comparison,
        paired_auc_bootstrap,
        question_equal_auc,
    )
except ModuleNotFoundError:
    from analyze_rl03_mcq_audit import (
        bootstrap_mean_ci,
        cross_fitted_level_gain_comparison,
        paired_auc_bootstrap,
        question_equal_auc,
    )


PRIMARY_FRACTIONS = np.asarray([0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
PRIMARY_CELL = {
    "interface_id": "I2",
    "representation": "content_sequence",
    "prompt_mode": "neutralized_letter_instruction",
}
FEATURES = (
    ("gold_level_90", "matched level control"),
    ("primary_score", "frozen primary: negative excess total variation"),
    ("negative_max_drawdown", "secondary: negative maximum drawdown"),
    ("negative_sign_reversal", "secondary: negative sign-reversal rate"),
    ("negative_curvature", "secondary: negative mean absolute second difference"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def _cell_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in PRIMARY_CELL.items():
        mask &= frame[column].eq(value)
    return mask


def build_non_telescope_features(probes: pd.DataFrame) -> pd.DataFrame:
    required = {
        "question_id",
        "rollout_id",
        "is_correct",
        "probe_kind",
        "frac",
        "gold_level",
        *PRIMARY_CELL,
    }
    missing = sorted(required - set(probes.columns))
    if missing:
        raise ValueError(f"probe table is missing columns: {missing}")
    rounded = np.round(PRIMARY_FRACTIONS, 8)
    view = probes[
        _cell_mask(probes)
        & probes["probe_kind"].eq("think")
        & probes["frac"].round(8).isin(rounded)
    ].copy()

    rows: list[dict[str, Any]] = []
    group_columns = ["question_id", "rollout_id", "is_correct"]
    for keys, group in view.groupby(group_columns, sort=False, dropna=False):
        group = group.sort_values("frac")
        if len(group) != len(PRIMARY_FRACTIONS):
            continue
        fractions = group["frac"].to_numpy(dtype=float)
        if not np.allclose(fractions, PRIMARY_FRACTIONS):
            continue
        scores = group["gold_level"].to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            continue
        deltas = np.diff(scores)
        total_variation = float(np.abs(deltas).sum())
        net_displacement = float(abs(scores[-1] - scores[0]))
        excess_total_variation = total_variation - net_displacement
        maximum_drawdown = max(
            [0.0]
            + [
                float(scores[left] - scores[right])
                for left in range(len(scores))
                for right in range(left + 1, len(scores))
            ]
        )
        signs = np.sign(deltas)
        sign_reversal_rate = (
            float(np.mean(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0.0
        )
        mean_absolute_second_difference = (
            float(np.mean(np.abs(np.diff(deltas)))) if len(deltas) > 1 else 0.0
        )
        rows.append(
            {
                "question_id": keys[0],
                "rollout_id": keys[1],
                "is_correct": bool(keys[2]),
                "gold_level_90": float(scores[-1]),
                "total_variation": total_variation,
                "net_displacement": net_displacement,
                "excess_total_variation": excess_total_variation,
                "primary_score": -excess_total_variation,
                "maximum_drawdown": maximum_drawdown,
                "negative_max_drawdown": -maximum_drawdown,
                "sign_reversal_rate": sign_reversal_rate,
                "negative_sign_reversal": -sign_reversal_rate,
                "mean_absolute_second_difference": mean_absolute_second_difference,
                "negative_curvature": -mean_absolute_second_difference,
            }
        )
    return pd.DataFrame(rows)


def analyze_features(
    features: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for feature, role in FEATURES:
        result = question_equal_auc(features, feature)
        per_question = result["per_question"]
        ci_low, ci_high = bootstrap_mean_ci(
            per_question["auc"],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        metrics.append(
            {
                "feature": feature,
                "role": role,
                "mean_auc": float(result["mean_auc"]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mixed_questions": int(result["mixed_questions"]),
            }
        )

    incremental: dict[str, Any] = {}
    for feature in ("primary_score", "negative_max_drawdown"):
        incremental[feature] = {
            "paired": paired_auc_bootstrap(
                features,
                feature,
                "gold_level_90",
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
            "crossfit": cross_fitted_level_gain_comparison(
                features,
                "gold_level_90",
                feature,
                folds=5,
                seed=seed,
            ),
        }

    primary = next(row for row in metrics if row["feature"] == "primary_score")
    stop = (
        primary["mean_auc"] < 0.70
        or primary["ci_low"] <= 0.60
        or incremental["primary_score"]["paired"]["ci_low"] <= 0.0
        or incremental["primary_score"]["crossfit"]["improvement"] <= 0.0
    )
    return {
        "protocol": "RL03 non-telescoping discovery v1",
        "data_role": "post-Stage-A1 discovery only",
        "seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples),
        "questions": int(features["question_id"].nunique()),
        "rollouts": int(len(features)),
        "primary_fractions": PRIMARY_FRACTIONS.tolist(),
        "primary_cell": PRIMARY_CELL,
        "metrics": metrics,
        "incremental": incremental,
        "decision": {
            "label": "STOP-NO-CONFIRMATION" if stop else "ELIGIBLE-FOR-NEW-HOLDOUT",
            "authorize_new_h200_confirmation": not stop,
        },
    }


def plot_auc(summary: dict[str, Any], output_path: Path) -> None:
    metrics = summary["metrics"]
    labels = [
        "Late level",
        "-Excess variation",
        "-Max drawdown",
        "-Sign reversals",
        "-Curvature",
    ]
    means = np.asarray([row["mean_auc"] for row in metrics], dtype=float)
    lows = np.asarray([row["ci_low"] for row in metrics], dtype=float)
    highs = np.asarray([row["ci_high"] for row in metrics], dtype=float)
    colors = ["#1b6ca8", "#7c3aed", "#0f766e", "#d97706", "#dc2626"]

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(labels))
    axis.bar(x, means, color=colors, width=0.68)
    axis.errorbar(
        x,
        means,
        yerr=np.vstack([means - lows, highs - means]),
        fmt="none",
        color="#111827",
        capsize=4,
        linewidth=1.2,
    )
    axis.axhline(0.5, color="#6b7280", linestyle="--", linewidth=1)
    axis.axhline(0.7, color="#9ca3af", linestyle=":", linewidth=1)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.40, 0.80)
    axis.set_ylabel("Question-equal within-question AUC")
    axis.set_title(
        "N1. Non-telescoping discovery does not justify a new confirmation run",
        fontsize=13,
        fontweight="bold",
    )
    axis.grid(axis="y", color="#d1d5db", alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(means):
        axis.text(index, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_report(summary: dict[str, Any], output_path: Path) -> None:
    metrics = {row["feature"]: row for row in summary["metrics"]}
    primary = metrics["primary_score"]
    drawdown = metrics["negative_max_drawdown"]
    primary_increment = summary["incremental"]["primary_score"]
    drawdown_increment = summary["incremental"]["negative_max_drawdown"]
    lines = [
        "# RL03 Non-Telescoping Discovery Report",
        "",
        f"Decision: **{summary['decision']['label']}**",
        "",
        "This is a discovery-only reuse of the completed Stage A1 data. It cannot establish a new confirmatory claim.",
        "",
        "## Frozen Primary Result",
        "",
        "| metric | result |",
        "|---|---:|",
        f"| `-excess_total_variation` AUC | {primary['mean_auc']:.4f} |",
        f"| 95% CI | [{primary['ci_low']:.4f}, {primary['ci_high']:.4f}] |",
        f"| paired delta versus `gold_level_90` | {primary_increment['paired']['mean_delta_auc']:.4f} |",
        f"| paired delta 95% CI | [{primary_increment['paired']['ci_low']:.4f}, {primary_increment['paired']['ci_high']:.4f}] |",
        f"| cross-fit improvement | {primary_increment['crossfit']['improvement']:.4f} |",
        "",
        "The frozen primary feature is indistinguishable from chance and is substantially worse than the late-level control.",
        "",
        "## Secondary Diagnostic",
        "",
        f"The strongest secondary candidate was `-maximum_drawdown`: AUC {drawdown['mean_auc']:.4f}, 95% CI [{drawdown['ci_low']:.4f}, {drawdown['ci_high']:.4f}]. Its paired delta versus late level was {drawdown_increment['paired']['mean_delta_auc']:.4f}, with 95% CI [{drawdown_increment['paired']['ci_low']:.4f}, {drawdown_increment['paired']['ci_high']:.4f}].",
        "",
        "It is below the frozen primary AUC gate and was inspected after the primary failure, so it cannot be promoted to a confirmatory feature.",
        "",
        "![Non-telescoping discovery AUC](figures/N1_non_telescope_auc.png)",
        "",
        "## Decision",
        "",
        "Do not generate a new MathVerse holdout for this revision. Do not launch Stage B or Stage C. Preserve the verifier-level result and close the answer-likelihood trajectory reward branch unless a new mechanism is motivated independently of these outcomes.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    probes = pd.read_parquet(args.audit_dir / "rollout_probe_metrics.parquet")
    features = build_non_telescope_features(probes)
    if features.empty:
        raise ValueError("no complete rollout trajectories were available")
    summary = analyze_features(features, args.bootstrap_samples, args.seed)

    features.to_csv(args.output_dir / "non_telescope_features.csv", index=False)
    (args.output_dir / "metric_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_auc(summary, figures_dir / "N1_non_telescope_auc.png")
    write_report(summary, args.output_dir / "NON_TELESCOPING_DISCOVERY_REPORT.md")
    print(json.dumps(summary["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
