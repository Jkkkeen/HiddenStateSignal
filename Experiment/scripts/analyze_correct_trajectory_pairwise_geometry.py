#!/usr/bin/env python3
"""Analyze whether correct rollouts share more similar hidden-state movements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from correct_trajectory_pairwise_geometry import (
    EPS,
    PRIMARY_METRICS,
    adjusted_max_stat_pvalues,
    aggregate_directed_pairs,
    compute_pair_metrics,
    directed_four_cell_interactions,
    pair_type_means,
    permutation_inference,
    question_contrasts,
    summarize_contrasts,
    symmetrize_pairs,
)


REQUIRED_COLUMNS = {
    "question_id",
    "rollout_id",
    "is_correct",
    "representation",
    "span_id",
    "relative_progress",
    "progress_bin",
    "layer",
    "displacement_norm",
    "reference_rollout_id",
    "reference_is_correct",
    "reference_span_id",
    "reference_progress",
    "reference_norm",
    "cosine_similarity",
}
PAIR_TYPES = ("++", "--", "+-")
PAIR_COLORS = {"++": "#157a6e", "--": "#b33f40", "+-": "#4c6085"}
PAIR_LABELS = {
    "++": "correct-correct",
    "--": "wrong-wrong",
    "+-": "correct-wrong",
}
METRIC_LABELS = {
    "angle_deg": "Direction angle (degrees)",
    "delta_rel": "Relative vector distance",
    "delta_amp": "Log-amplitude distance",
}
CONTRAST_LABELS = {
    "pp_minus_mm": "++ minus --",
    "pp_minus_pm": "++ minus +-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def validate_frozen_input(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if set(frame["representation"].dropna().unique()) != {"mean_w128_s64"}:
        raise ValueError("expected only mean_w128_s64")
    if set(frame["layer"].dropna().astype(int).unique()) != {24, 36}:
        raise ValueError("expected hidden-state indexes 24 and 36")
    if (frame["rollout_id"] == frame["reference_rollout_id"]).any():
        raise ValueError("self-pairs are not allowed")
    if frame.empty:
        raise ValueError("input is empty")


def _exclusion_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    left = frame["displacement_norm"].to_numpy(dtype=np.float64)
    right = frame["reference_norm"].to_numpy(dtype=np.float64)
    cosine = frame["cosine_similarity"].to_numpy(dtype=np.float64)
    invalid = ~(
        np.isfinite(left)
        & np.isfinite(right)
        & np.isfinite(cosine)
        & (left > EPS)
        & (right > EPS)
    )
    view = frame[["layer", "progress_bin"]].copy()
    view["excluded"] = invalid.astype(int)
    coverage = (
        view.groupby(["layer", "progress_bin"], as_index=False, observed=True)
        .agg(input_rows=("excluded", "size"), excluded_rows=("excluded", "sum"))
        .sort_values(["layer", "progress_bin"], kind="stable")
    )
    coverage["valid_rows"] = coverage["input_rows"] - coverage["excluded_rows"]
    coverage["excluded_fraction"] = coverage["excluded_rows"] / coverage["input_rows"]
    return coverage


def _bootstrap_pair_type_curves(
    progress_means: pd.DataFrame,
    metric: str,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    keys = ["layer", "progress_bin", "pair_type"]
    for key, group in progress_means.groupby(keys, sort=True, observed=True):
        values = group[metric].dropna().to_numpy(dtype=np.float64)
        if not len(values):
            continue
        samples = rng.choice(values, size=(bootstrap, len(values)), replace=True).mean(axis=1)
        rows.append(
            {
                **dict(zip(keys, key)),
                "mean": float(values.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "n_questions": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _plot_progress_metric(
    progress_means: pd.DataFrame,
    metric: str,
    output_path: Path,
    bootstrap: int,
    seed: int,
) -> None:
    curves = _bootstrap_pair_type_curves(progress_means, metric, bootstrap, seed)
    layers = sorted(curves["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(11.5, 4.2), sharey=True, squeeze=False)
    for axis, layer in zip(axes[0], layers):
        layer_view = curves[curves["layer"] == layer]
        for pair_type in PAIR_TYPES:
            view = layer_view[layer_view["pair_type"] == pair_type].sort_values(
                "progress_bin"
            )
            if view.empty:
                continue
            x = view["progress_bin"].to_numpy(dtype=float)
            mean = view["mean"].to_numpy(dtype=float)
            low = view["ci_low"].to_numpy(dtype=float)
            high = view["ci_high"].to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=PAIR_COLORS[pair_type],
                label=PAIR_LABELS[pair_type],
            )
            axis.fill_between(x, low, high, color=PAIR_COLORS[pair_type], alpha=0.16)
        axis.set_title(f"Hidden-state index {int(layer)}")
        axis.set_xlabel("Relative-progress bin")
        axis.set_xticks(sorted(layer_view["progress_bin"].unique()))
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].set_ylabel(METRIC_LABELS[metric])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Same-question rollout-pair geometry over reasoning progress", y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_whole_contrasts(summary: pd.DataFrame, output_path: Path) -> None:
    whole = summary[summary["scope"] == "whole"].copy()
    display_metrics = ("angle_rad", "delta_rel", "delta_amp")
    titles = {
        "angle_rad": "Direction angle (radians)",
        "delta_rel": "Relative vector distance",
        "delta_amp": "Log-amplitude distance",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for axis, metric in zip(axes, display_metrics):
        view = whole[whole["metric"] == metric]
        positions = np.arange(4, dtype=float)
        labels = []
        for position, (layer, contrast) in enumerate(
            [(24, "pp_minus_mm"), (24, "pp_minus_pm"), (36, "pp_minus_mm"), (36, "pp_minus_pm")]
        ):
            row = view[(view["layer"] == layer) & (view["contrast"] == contrast)]
            labels.append(f"L{layer}\n{CONTRAST_LABELS[contrast]}")
            if row.empty:
                continue
            record = row.iloc[0]
            mean = float(record["mean_contrast"])
            errors = np.asarray(
                [[mean - float(record["ci_low"])], [float(record["ci_high"]) - mean]]
            )
            axis.errorbar(
                position,
                mean,
                yerr=errors,
                fmt="o",
                capsize=4,
                color="#222222",
                ecolor="#555555",
            )
        axis.axhline(0.0, color="#777777", linewidth=1, linestyle="--")
        axis.set_xticks(positions, labels, fontsize=8)
        axis.set_title(titles[metric])
        axis.set_ylabel("Mean paired contrast")
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Whole-trajectory correct-correct cohesion (negative means closer)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _attach_permutation_pvalues(
    summary: pd.DataFrame,
    nulls: pd.DataFrame,
) -> pd.DataFrame:
    result = summary.copy()
    result["permutation_p"] = np.nan
    for index, row in result.iterrows():
        selected = nulls[
            (nulls["scope"] == row["scope"])
            & (nulls["metric"] == row["metric"])
            & (nulls["contrast"] == row["contrast"])
            & (nulls["layer"] == row["layer"])
        ]
        if row["scope"] == "progress":
            selected = selected[selected["progress_bin"] == row["progress_bin"]]
        values = selected["mean_contrast"].dropna().to_numpy(dtype=np.float64)
        if len(values):
            result.at[index, "permutation_p"] = (
                1 + int((np.abs(values) >= abs(float(row["mean_contrast"]))).sum())
            ) / (1 + len(values))
    return result


def _format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No estimable rows."
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    return display.to_markdown(index=False)


def _write_report(
    output_path: Path,
    meta: dict[str, object],
    summary: pd.DataFrame,
    interactions: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    whole = summary[summary["scope"] == "whole"].sort_values(
        ["metric", "layer", "contrast"]
    )
    robust = summary[
        (summary["scope"] == "progress") & (summary["max_stat_p"] < 0.05)
    ].sort_values(["metric", "contrast", "layer", "progress_bin"])
    whole_supported = whole[
        (whole["ci_high"] < 0)
        & (whole["permutation_p"] < 0.05)
    ]
    strongest_progress = summary[summary["scope"] == "progress"].sort_values(
        "permutation_p"
    ).head(3)
    interaction_summary = (
        interactions.groupby(["metric", "layer", "progress_bin"], as_index=False)[
            "interaction"
        ]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", key=lambda values: values.abs(), ascending=False)
        .head(12)
    )
    whole_columns = [
        "metric",
        "layer",
        "contrast",
        "mean_contrast",
        "ci_low",
        "ci_high",
        "negative_sign_fraction",
        "n_questions",
        "permutation_p",
    ]
    robust_columns = whole_columns[:3] + ["progress_bin"] + whole_columns[3:] + [
        "max_stat_p"
    ]
    if robust.empty:
        robust_text = "No progress cell survived the within-family max-stat correction."
    else:
        robust_text = _format_table(robust, robust_columns)
    if whole_supported.empty:
        whole_conclusion = (
            "None of the 12 whole-trajectory metric/layer/control comparisons had both a "
            "bootstrap interval below zero and a permutation p-value below 0.05."
        )
    else:
        whole_conclusion = (
            f"{len(whole_supported)} whole-trajectory comparisons supported closer correct-correct "
            "geometry under both the bootstrap and permutation criteria."
        )
    coverage_display = coverage.copy()
    coverage_display["excluded_fraction"] *= 100.0
    lines = [
        "# Correct-Trajectory Pairwise Hidden Geometry",
        "",
        "## Scope and interpretation",
        "",
        f"This discovery analysis uses {meta['questions']} questions, hidden-state indexes "
        f"{meta['layers']}, and progress bins {meta['progress_bins']} from the frozen "
        "`mean_w128_s64` pairwise-geometry parquet. It did not access H200 or run a model forward pass.",
        "",
        "For every distance, the primary contrast is `correct-correct minus control`. A negative "
        "value means correct trajectories are closer than the wrong-wrong or correct-wrong control. "
        "Questions, rather than pair rows, are the independent units.",
        "",
        "## Whole-trajectory results",
        "",
        whole_conclusion,
        "",
        _format_table(whole, whole_columns),
        "",
        "## Progress cells surviving max-stat correction",
        "",
        robust_text,
        "",
        "The max-stat family is the 2 layers by all progress bins within each metric and contrast. "
        "Pointwise permutation p-values are reported for description; `max_stat_p` is the inferential "
        "quantity for progress localization.",
        "",
        "The strongest unadjusted progress signals are shown below only as hypotheses for a new "
        "confirmatory cohort; they are not discoveries after grid correction.",
        "",
        _format_table(
            strongest_progress,
            [
                "metric",
                "layer",
                "progress_bin",
                "contrast",
                "mean_contrast",
                "ci_low",
                "ci_high",
                "negative_sign_fraction",
                "permutation_p",
                "max_stat_p",
            ],
        ),
        "",
        "## Directed four-cell diagnostic",
        "",
        "The interaction is `(A++ - A+-) - (A-+ - A--)`. It checks whether the apparent correct "
        "reference advantage is specific to correct queries instead of being induced by an asymmetric "
        "query/reference construction.",
        "",
        _format_table(
            interaction_summary,
            ["metric", "layer", "progress_bin", "mean", "count"],
        ),
        "",
        "## Exclusion coverage",
        "",
        "Rows with either movement norm at most `1e-12`, or any non-finite norm/cosine, were "
        "excluded because their direction angle is undefined.",
        "",
        _format_table(
            coverage_display,
            [
                "layer",
                "progress_bin",
                "input_rows",
                "excluded_rows",
                "valid_rows",
                "excluded_fraction",
            ],
        ),
        "",
        "## Metric reading guide",
        "",
        "- `angle_rad`: pure directional agreement after removing movement magnitude.",
        "- `delta_rel`: full vector separation normalized by the sum of movement norms; it combines direction and relative scale.",
        "- `delta_amp`: absolute log-ratio of movement norms; it isolates magnitude agreement.",
        "",
        "A direction-only effect supports shared semantic movement; an amplitude-only effect supports "
        "shared activation scale; a `delta_rel` effect with both components helps identify combined "
        "geometry. These are discovery results and require confirmatory data before use as an RL reward.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.bootstrap <= 0 or args.permutations <= 0:
        raise ValueError("bootstrap and permutations must be positive")
    raw = pd.read_parquet(args.input)
    validate_frozen_input(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    coverage = _exclusion_coverage(raw)
    metrics, exclusions = compute_pair_metrics(raw)
    directed = aggregate_directed_pairs(metrics)
    pairs = symmetrize_pairs(directed)

    progress_means = pair_type_means(pairs, whole_trajectory=False)
    progress_means.insert(0, "scope", "progress")
    whole_means = pair_type_means(pairs, whole_trajectory=True)
    whole_means.insert(0, "scope", "whole")
    means = pd.concat([progress_means, whole_means], ignore_index=True, sort=False)

    progress_contrasts = question_contrasts(progress_means)
    progress_contrasts.insert(0, "scope", "progress")
    whole_contrasts = question_contrasts(whole_means)
    whole_contrasts.insert(0, "scope", "whole")
    contrasts = pd.concat([progress_contrasts, whole_contrasts], ignore_index=True, sort=False)
    interactions = directed_four_cell_interactions(directed)

    progress_summary = summarize_contrasts(
        progress_contrasts, bootstrap=args.bootstrap, seed=args.seed
    )
    progress_summary.insert(0, "scope", "progress")
    whole_summary = summarize_contrasts(
        whole_contrasts, bootstrap=args.bootstrap, seed=args.seed + 1
    )
    whole_summary.insert(0, "scope", "whole")

    labels = raw[["question_id", "rollout_id", "is_correct"]].drop_duplicates()
    permutation = permutation_inference(
        pairs,
        labels,
        permutations=args.permutations,
        seed=args.seed + 10_000,
    )
    progress_summary = adjusted_max_stat_pvalues(progress_summary, permutation.nulls)
    whole_summary["max_stat_p"] = np.nan
    summary = pd.concat([progress_summary, whole_summary], ignore_index=True, sort=False)
    summary = _attach_permutation_pvalues(summary, permutation.nulls)

    directed.to_parquet(args.output_dir / "directed_pair_metrics.parquet", index=False)
    pairs.to_parquet(args.output_dir / "undirected_rollout_pairs.parquet", index=False)
    means.to_csv(args.output_dir / "question_pair_type_means.csv", index=False)
    contrasts.to_csv(args.output_dir / "question_contrasts.csv", index=False)
    interactions.to_csv(
        args.output_dir / "directed_four_cell_interactions.csv", index=False
    )
    summary.to_csv(args.output_dir / "contrast_summary.csv", index=False)
    permutation.nulls.to_parquet(args.output_dir / "permutation_null.parquet", index=False)
    coverage.to_csv(args.output_dir / "exclusion_coverage.csv", index=False)

    _plot_progress_metric(
        progress_means,
        "angle_deg",
        figure_dir / "G1_direction_angle_progress.png",
        args.bootstrap,
        args.seed + 101,
    )
    _plot_progress_metric(
        progress_means,
        "delta_rel",
        figure_dir / "G2_relative_vector_progress.png",
        args.bootstrap,
        args.seed + 102,
    )
    _plot_progress_metric(
        progress_means,
        "delta_amp",
        figure_dir / "G3_amplitude_progress.png",
        args.bootstrap,
        args.seed + 103,
    )
    _plot_whole_contrasts(summary, figure_dir / "G4_whole_trajectory_contrasts.png")

    label_counts = labels.groupby("question_id")["is_correct"].agg(["sum", "count"])
    meta: dict[str, object] = {
        "input": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "representation": "mean_w128_s64",
        "questions": int(raw["question_id"].nunique()),
        "rollouts": int(labels.shape[0]),
        "layers": sorted(raw["layer"].astype(int).unique().tolist()),
        "progress_bins": sorted(raw["progress_bin"].astype(int).unique().tolist()),
        "raw_rows": int(len(raw)),
        "valid_rows": int(len(metrics)),
        "directed_pair_rows": int(len(directed)),
        "undirected_pair_rows": int(len(pairs)),
        "bootstrap": int(args.bootstrap),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "exclusions": exclusions,
        "all_permutations_preserved_label_counts": bool(
            permutation.label_count_checks.all()
        ),
        "questions_with_both_labels": int(
            ((label_counts["sum"] > 0) & (label_counts["sum"] < label_counts["count"])).sum()
        ),
        "h200_accessed": False,
    }
    (args.output_dir / "analysis_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        args.output_dir / "CORRECT_TRAJECTORY_PAIRWISE_GEOMETRY_REPORT.md",
        meta,
        summary,
        interactions,
        coverage,
    )
    return meta


def main() -> None:
    args = parse_args()
    meta = _run(args)
    print(
        "Completed correct-trajectory pairwise geometry: "
        f"{meta['questions']} questions, layers {meta['layers']}, "
        f"progress bins {meta['progress_bins']}, H200 accessed: {meta['h200_accessed']}"
    )


if __name__ == "__main__":
    main()
