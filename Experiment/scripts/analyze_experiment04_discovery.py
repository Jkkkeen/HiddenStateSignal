#!/usr/bin/env python3
"""Question-equal discovery analysis for Experiment 04."""

from __future__ import annotations

import argparse
import html
import json
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


META_COLUMNS = {
    "question_id",
    "rollout_id",
    "is_correct",
    "think_length",
    "response_length",
    "representation",
    "layer",
    "chunk_id",
    "chunk_start",
    "chunk_end",
    "relative_progress",
    "end_aligned_chunk",
    "is_partial",
}
EXCLUDED_CELL_FEATURES = {"turn_angle"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 04 discovery metrics.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--permutations", type=int, default=4000)
    parser.add_argument("--progress-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--top-figures", type=int, default=12)
    parser.add_argument(
        "--cluster-top-families",
        type=int,
        default=0,
        help="0 runs cluster permutation for every family.",
    )
    return parser.parse_args()


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    comparisons = pos[:, None] - neg[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def question_equal_curve(
    frame: pd.DataFrame,
    feature: str,
    x: str,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    per_question = (
        frame.dropna(subset=[feature])
        .groupby(["question_id", "is_correct", x], as_index=False, observed=True)[feature]
        .mean()
    )
    if per_question.empty:
        return pd.DataFrame(columns=[x, "is_correct", "mean", "ci_low", "ci_high", "questions"])
    groups = []
    rng = np.random.default_rng(seed)
    question_ids = np.asarray(sorted(per_question["question_id"].astype(str).unique()))
    for (position, label), group in per_question.groupby([x, "is_correct"], sort=True):
        values = group.set_index(group["question_id"].astype(str))[feature]
        observed = float(values.mean())
        samples = []
        available = values.index.to_numpy()
        if bootstrap > 0 and available.size:
            for _ in range(bootstrap):
                selected = rng.choice(question_ids, size=question_ids.size, replace=True)
                sample = values.reindex(selected).dropna()
                if not sample.empty:
                    samples.append(float(sample.mean()))
        groups.append(
            {
                x: position,
                "is_correct": bool(label),
                "mean": observed,
                "ci_low": float(np.quantile(samples, 0.025)) if samples else np.nan,
                "ci_high": float(np.quantile(samples, 0.975)) if samples else np.nan,
                "questions": int(available.size),
            }
        )
    return pd.DataFrame(groups)


def cluster_components(statistic: np.ndarray, threshold: float = 2.0) -> list[dict[str, Any]]:
    values = np.asarray(statistic, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("statistic must be a 2D layer-by-position grid")
    visited = np.zeros(values.shape, dtype=bool)
    clusters = []
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if visited[row, col] or not np.isfinite(values[row, col]) or abs(values[row, col]) < threshold:
                continue
            sign = 1 if values[row, col] > 0 else -1
            queue = deque([(row, col)])
            visited[row, col] = True
            cells = []
            while queue:
                current = queue.popleft()
                cells.append(current)
                for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (current[0] + drow, current[1] + dcol)
                    if not (0 <= neighbor[0] < values.shape[0] and 0 <= neighbor[1] < values.shape[1]):
                        continue
                    if visited[neighbor]:
                        continue
                    value = values[neighbor]
                    if np.isfinite(value) and abs(value) >= threshold and np.sign(value) == sign:
                        visited[neighbor] = True
                        queue.append(neighbor)
            clusters.append(
                {
                    "sign": sign,
                    "cells": cells,
                    "mass": float(sum(abs(values[cell]) for cell in cells)),
                }
            )
    return clusters


def _t_stat(question_effects: np.ndarray) -> np.ndarray:
    count = np.sum(np.isfinite(question_effects), axis=0)
    mean = np.nanmean(question_effects, axis=0)
    std = np.nanstd(question_effects, axis=0, ddof=1)
    return np.divide(
        mean,
        std / np.sqrt(np.maximum(count, 1)),
        out=np.full_like(mean, np.nan),
        where=(count >= 2) & (std > 0),
    )


def _cell_statistics(frame: pd.DataFrame, feature: str, position: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["question_id", "rollout_id", "is_correct", "representation", "layer", position]
    rollout = frame.dropna(subset=[feature]).groupby(keys, as_index=False, observed=True)[feature].mean()
    label_mean = rollout.groupby(
        ["question_id", "representation", "layer", position, "is_correct"],
        as_index=False,
        observed=True,
    )[feature].mean()
    pivot = label_mean.pivot_table(
        index=["question_id", "representation", "layer", position],
        columns="is_correct",
        values=feature,
    ).reset_index()
    if True not in pivot.columns or False not in pivot.columns:
        return pd.DataFrame(), rollout
    pivot["question_effect"] = pivot[True] - pivot[False]

    auc_rows = []
    for key, question_cell in rollout.groupby(
        ["question_id", "representation", "layer", position],
        sort=False,
        observed=True,
    ):
        auc_rows.append(
            {
                "question_id": key[0],
                "representation": key[1],
                "layer": key[2],
                position: key[3],
                "question_auc": pairwise_auc(
                    question_cell.loc[question_cell["is_correct"], feature].to_numpy(),
                    question_cell.loc[~question_cell["is_correct"], feature].to_numpy(),
                ),
            }
        )
    auc_frame = pd.DataFrame(auc_rows)
    auc_lookup = {
        key: group["question_auc"].dropna().to_numpy(dtype=np.float64)
        for key, group in auc_frame.groupby(
            ["representation", "layer", position], sort=False, observed=True
        )
    }

    rows = []
    for (representation, layer, cell), group in pivot.groupby(
        ["representation", "layer", position], sort=True, observed=True
    ):
        effects = group["question_effect"].to_numpy(dtype=np.float64)
        effects = effects[np.isfinite(effects)]
        aucs = auc_lookup.get((representation, layer, cell), np.empty(0, dtype=np.float64))
        rows.append(
            {
                "feature": feature,
                "representation": representation,
                "layer": int(layer),
                position: int(cell),
                "questions": int(effects.size),
                "mean_correct_minus_wrong": float(np.mean(effects)) if effects.size else np.nan,
                "t_stat": float(_t_stat(effects[:, None])[0]) if effects.size else np.nan,
                "sign_fraction": float(np.mean(effects > 0)) if effects.size else np.nan,
                "within_q_auc": float(np.mean(aucs)) if aucs.size else np.nan,
                "oriented_auc": float(max(np.mean(aucs), 1.0 - np.mean(aucs)))
                if aucs.size
                else np.nan,
            }
        )
    return pd.DataFrame(rows), rollout


def _question_effect_cube(
    frame: pd.DataFrame,
    feature: str,
    representation: str,
    position: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    selected = frame[frame["representation"] == representation]
    keys = ["question_id", "rollout_id", "is_correct", "layer", position]
    rollout = selected.dropna(subset=[feature]).groupby(keys, as_index=False)[feature].mean()
    label = rollout.groupby(["question_id", "layer", position, "is_correct"], as_index=False)[
        feature
    ].mean()
    pivot = label.pivot_table(
        index=["question_id", "layer", position], columns="is_correct", values=feature
    ).reset_index()
    if True not in pivot.columns or False not in pivot.columns:
        return np.empty((0, 0, 0)), [], []
    pivot["effect"] = pivot[True] - pivot[False]
    questions = sorted(pivot["question_id"].astype(str).unique())
    layers = sorted(int(value) for value in pivot["layer"].unique())
    positions = sorted(int(value) for value in pivot[position].unique())
    q_index = {value: idx for idx, value in enumerate(questions)}
    l_index = {value: idx for idx, value in enumerate(layers)}
    p_index = {value: idx for idx, value in enumerate(positions)}
    cube = np.full((len(questions), len(layers), len(positions)), np.nan, dtype=np.float64)
    for row in pivot.itertuples(index=False):
        cube[q_index[str(row.question_id)], l_index[int(row.layer)], p_index[int(getattr(row, position))]] = float(
            row.effect
        )
    return cube, layers, positions


def signflip_cluster_test(
    question_effects: np.ndarray,
    *,
    permutations: int,
    seed: int,
    threshold: float = 2.0,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    observed_stat = _t_stat(question_effects)
    observed = cluster_components(observed_stat, threshold=threshold)
    rng = np.random.default_rng(seed)
    null = np.zeros(permutations, dtype=np.float64)
    for permutation in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=(question_effects.shape[0], 1, 1))
        statistic = _t_stat(question_effects * signs)
        clusters = cluster_components(statistic, threshold=threshold)
        null[permutation] = max((cluster["mass"] for cluster in clusters), default=0.0)
    for cluster in observed:
        cluster["p_value"] = float((1 + np.sum(null >= cluster["mass"])) / (permutations + 1))
    return observed, null


def label_permutation_cluster_test(
    frame: pd.DataFrame,
    feature: str,
    representation: str,
    position: str,
    *,
    permutations: int,
    seed: int,
    threshold: float = 2.0,
) -> tuple[list[dict[str, Any]], np.ndarray, list[int], list[int]]:
    """Exact within-question label permutation with fixed correct/wrong counts."""
    selected = frame[frame["representation"] == representation]
    keys = ["question_id", "rollout_id", "is_correct", "layer", position]
    rollout = selected.dropna(subset=[feature]).groupby(keys, as_index=False)[feature].mean()
    observed_cube, layers, positions = _question_effect_cube(
        frame, feature, representation, position
    )
    if observed_cube.size == 0:
        return [], np.empty(0), layers, positions
    observed_clusters = cluster_components(_t_stat(observed_cube), threshold=threshold)
    columns = pd.MultiIndex.from_product([layers, positions], names=["layer", position])
    cell_count = len(columns)
    sum_effect = np.zeros((permutations, cell_count), dtype=np.float64)
    square_effect = np.zeros_like(sum_effect)
    valid_count = np.zeros_like(sum_effect, dtype=np.int16)
    rng = np.random.default_rng(seed)

    def weighted_sum(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
        result = np.zeros((weights.shape[0], values.shape[1]), dtype=np.float64)
        for rollout_index in range(weights.shape[1]):
            result += weights[:, rollout_index, None] * values[rollout_index][None, :]
        return result

    for _, question in rollout.groupby("question_id", sort=True):
        labels = (
            question[["rollout_id", "is_correct"]]
            .drop_duplicates("rollout_id")
            .sort_values("rollout_id")
        )
        rollout_ids = labels["rollout_id"].to_numpy()
        observed_labels = labels["is_correct"].to_numpy(dtype=bool)
        values = question.pivot_table(
            index="rollout_id", columns=["layer", position], values=feature, aggfunc="mean"
        ).reindex(index=rollout_ids, columns=columns)
        matrix = values.to_numpy(dtype=np.float64)
        mask = np.isfinite(matrix).astype(np.float64)
        finite_values = np.nan_to_num(matrix, nan=0.0)
        permuted_labels = np.stack(
            [rng.permutation(observed_labels) for _ in range(permutations)], axis=0
        ).astype(np.float64)
        negative_labels = 1.0 - permuted_labels
        positive_count = weighted_sum(permuted_labels, mask)
        negative_count = weighted_sum(negative_labels, mask)
        positive_mean = np.divide(
            weighted_sum(permuted_labels, finite_values),
            positive_count,
            out=np.full((permutations, cell_count), np.nan),
            where=positive_count > 0,
        )
        negative_mean = np.divide(
            weighted_sum(negative_labels, finite_values),
            negative_count,
            out=np.full((permutations, cell_count), np.nan),
            where=negative_count > 0,
        )
        effects = positive_mean - negative_mean
        valid = np.isfinite(effects)
        sum_effect += np.nan_to_num(effects, nan=0.0)
        square_effect += np.nan_to_num(np.square(effects), nan=0.0)
        valid_count += valid

    mean = np.divide(
        sum_effect,
        valid_count,
        out=np.full_like(sum_effect, np.nan),
        where=valid_count > 0,
    )
    variance = np.divide(
        square_effect - np.divide(
            np.square(sum_effect),
            valid_count,
            out=np.zeros_like(sum_effect),
            where=valid_count > 0,
        ),
        valid_count - 1,
        out=np.full_like(sum_effect, np.nan),
        where=valid_count > 1,
    )
    statistic = np.divide(
        mean,
        np.sqrt(variance / np.maximum(valid_count, 1)),
        out=np.full_like(mean, np.nan),
        where=(valid_count > 1) & (variance > 0),
    )
    null = np.zeros(permutations, dtype=np.float64)
    for permutation in range(permutations):
        grid = statistic[permutation].reshape(len(layers), len(positions))
        clusters = cluster_components(grid, threshold=threshold)
        null[permutation] = max((cluster["mass"] for cluster in clusters), default=0.0)
    for cluster in observed_clusters:
        cluster["p_value"] = float((1 + np.sum(null >= cluster["mass"])) / (permutations + 1))
    return observed_clusters, null, layers, positions


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in META_COLUMNS
        and column not in EXCLUDED_CELL_FEATURES
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _plot_heatmap(table: pd.DataFrame, value: str, output: Path, title: str) -> None:
    pivot = table.pivot(index="layer", columns="progress_bin", values=value).sort_index()
    fig, axis = plt.subplots(figsize=(10, 6))
    bound = max(float(np.nanmax(np.abs(pivot.to_numpy()))), 0.1) if value != "questions" else None
    image = axis.imshow(
        pivot,
        origin="lower",
        aspect="auto",
        cmap="coolwarm" if value != "questions" else "viridis",
        vmin=-bound if bound is not None else None,
        vmax=bound if bound is not None else None,
    )
    axis.set_title(title)
    axis.set_xlabel("relative-progress bin")
    axis.set_ylabel("hidden-state index")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_curve(curve: pd.DataFrame, x: str, output: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for label, color, name in ((True, "#1b8a5a", "correct"), (False, "#c74c4c", "wrong")):
        group = curve[curve["is_correct"] == label].sort_values(x)
        axis.plot(group[x], group["mean"], color=color, label=name)
        axis.fill_between(group[x], group["ci_low"], group["ci_high"], color=color, alpha=0.18)
    axis.set_title(title)
    axis.set_xlabel(x)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _summary_effects(summary: pd.DataFrame) -> pd.DataFrame:
    features = [
        "step_norm_mean",
        "step_norm_median",
        "path_length",
        "net_displacement",
        "straightness",
        "log_detour",
        "direction_consistency",
        "turn_angle_std",
        "turn_angle_late_std",
    ]
    rows = []
    keys = ["direction", "representation", "layer", "chunk_id", "is_partial"]
    for feature in features:
        for key, group in summary.groupby(keys, sort=True, observed=True):
            question_rows = []
            for question_id, question in group.groupby("question_id", sort=False):
                positive = question.loc[question["is_correct"], feature].dropna().to_numpy()
                negative = question.loc[~question["is_correct"], feature].dropna().to_numpy()
                if positive.size and negative.size:
                    question_rows.append((question_id, positive.mean() - negative.mean(), pairwise_auc(positive, negative)))
            if not question_rows:
                continue
            effects = np.asarray([row[1] for row in question_rows], dtype=np.float64)
            aucs = np.asarray([row[2] for row in question_rows], dtype=np.float64)
            rows.append(
                {
                    "feature": feature,
                    "direction": key[0],
                    "representation": key[1],
                    "layer": key[2],
                    "chunk_id": key[3],
                    "is_partial": key[4],
                    "questions": effects.size,
                    "mean_correct_minus_wrong": effects.mean(),
                    "t_stat": _t_stat(effects[:, None])[0],
                    "sign_fraction": np.mean(effects > 0),
                    "within_q_auc": np.mean(aucs),
                    "oriented_auc": max(np.mean(aucs), 1 - np.mean(aucs)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    horizontal = pd.read_parquet(input_dir / "horizontal_metrics.parquet")
    vertical = pd.read_parquet(input_dir / "vertical_metrics.parquet")
    summary = pd.read_parquet(input_dir / "summary_metrics.parquet")
    horizontal = horizontal[~horizontal["is_partial"]].copy()
    vertical = vertical[~vertical["is_partial"]].copy()
    horizontal["progress_bin"] = np.minimum(
        (horizontal["relative_progress"] * args.progress_bins).astype(int), args.progress_bins - 1
    )
    vertical["progress_bin"] = np.minimum(
        (vertical["relative_progress"] * args.progress_bins).astype(int), args.progress_bins - 1
    )

    all_effects = []
    frame_by_direction = {"horizontal": horizontal, "vertical": vertical}
    for direction, frame in frame_by_direction.items():
        for feature in _feature_columns(frame):
            effects, _ = _cell_statistics(frame, feature, "progress_bin")
            if effects.empty:
                continue
            effects["direction"] = direction
            all_effects.append(effects)
    effects = pd.concat(all_effects, ignore_index=True)
    n_questions = int(horizontal["question_id"].nunique())
    minimum_support = max(20, int(np.ceil(0.5 * n_questions)))
    effects["primary_support"] = effects["questions"] >= minimum_support
    effects.to_csv(output_dir / "cell_effects.csv", index=False)

    summary_effects = _summary_effects(summary)
    summary_effects.to_csv(output_dir / "summary_effects.csv", index=False)
    candidates = effects[effects["primary_support"]].copy()
    candidates["abs_t"] = candidates["t_stat"].abs()
    family_rank = (
        candidates.groupby(["direction", "representation", "feature"], as_index=False)["abs_t"]
        .max()
        .sort_values("abs_t", ascending=False)
    )
    if args.cluster_top_families > 0:
        family_rank = family_rank.head(args.cluster_top_families)

    cluster_rows = []
    for family_index, family in enumerate(family_rank.itertuples(index=False), start=1):
        frame = frame_by_direction[family.direction]
        clusters, _, layers, positions = label_permutation_cluster_test(
            frame,
            family.feature,
            family.representation,
            "progress_bin",
            permutations=args.permutations,
            seed=args.seed + family_index,
        )
        for cluster_id, cluster in enumerate(clusters):
            cluster_rows.append(
                {
                    "direction": family.direction,
                    "representation": family.representation,
                    "feature": family.feature,
                    "cluster_id": cluster_id,
                    "sign": cluster["sign"],
                    "mass": cluster["mass"],
                    "p_value": cluster["p_value"],
                    "layers": json.dumps(sorted({layers[cell[0]] for cell in cluster["cells"]})),
                    "progress_bins": json.dumps(
                        sorted({positions[cell[1]] for cell in cluster["cells"]})
                    ),
                    "cell_count": len(cluster["cells"]),
                }
            )
    clusters = pd.DataFrame(cluster_rows)
    clusters.to_csv(output_dir / "cluster_permutation.csv", index=False)

    top = candidates.sort_values("abs_t", ascending=False).head(args.top_figures)
    figure_paths = []
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        family = effects[
            (effects["direction"] == row.direction)
            & (effects["representation"] == row.representation)
            & (effects["feature"] == row.feature)
        ]
        heatmap = figure_dir / f"F{rank:02d}_{row.direction}_{row.representation}_{row.feature}_atlas.png"
        _plot_heatmap(family, "t_stat", heatmap, f"{row.direction} | {row.representation} | {row.feature}")
        figure_paths.append(heatmap)
        frame = frame_by_direction[row.direction]
        selected = frame[
            (frame["representation"] == row.representation) & (frame["layer"] == row.layer)
        ]
        curve = question_equal_curve(
            selected,
            row.feature,
            "progress_bin",
            bootstrap=args.bootstrap,
            seed=args.seed + rank,
        )
        curve_path = figure_dir / f"F{rank:02d}_{row.direction}_{row.representation}_{row.feature}_curve.png"
        _plot_curve(curve, "progress_bin", curve_path, f"L{row.layer} correct/wrong trajectory")
        figure_paths.append(curve_path)

    candidate_columns = [
        "direction",
        "representation",
        "feature",
        "layer",
        "progress_bin",
        "questions",
        "mean_correct_minus_wrong",
        "t_stat",
        "sign_fraction",
        "within_q_auc",
        "oriented_auc",
    ]
    candidates.sort_values("abs_t", ascending=False)[candidate_columns].to_csv(
        output_dir / "candidate_worksheet.csv", index=False
    )
    report = [
        "# Experiment 04 Discovery Report",
        "",
        f"- Questions: {n_questions}",
        f"- Rollouts: {horizontal[['question_id', 'rollout_id']].drop_duplicates().shape[0]}",
        f"- Primary-support threshold: {minimum_support} questions",
        f"- Cell families scanned: {len(family_rank)}",
        f"- Cluster permutations: {args.permutations}",
        "- Cluster null: within-question correct/wrong label permutation with label counts fixed.",
        "- This is discovery-only; confirm remains blind.",
        "",
        "## Strongest Cells",
        "",
    ]
    for row in top.itertuples(index=False):
        report.append(
            f"- {row.direction} / {row.representation} / {row.feature} / L{row.layer} / "
            f"bin{row.progress_bin}: t={row.t_stat:.3f}, AUC={row.within_q_auc:.3f}, "
            f"Nq={row.questions}"
        )
    (output_dir / "DISCOVERY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    links = "\n".join(
        f'<li><a href="figures/{html.escape(path.name)}">{html.escape(path.name)}</a></li>'
        for path in figure_paths
    )
    (output_dir / "DISCOVERY_REPORT.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>Experiment 04 Discovery</title>"
        f"<h1>Experiment 04 Discovery</h1><p>Confirm remains blind.</p><ul>{links}</ul>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
