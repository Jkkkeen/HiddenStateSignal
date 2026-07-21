#!/usr/bin/env python3
"""Balanced-LOO four-cell analysis for long success-trajectory span shards."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from long_success_trajectory_common import balanced_reference_subsets, length_bin, pairwise_auc


@dataclass(frozen=True)
class PreparedQuery:
    rollout_id: int
    progress_bin: int
    relative_progress: float
    span_start: int
    displacement_norm: float
    think_length: int
    similarities: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze long success-trajectory hidden shards.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-bins", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def interaction_from_cells(cells: dict[str, float]) -> float:
    return float((cells["A_pp"] - cells["A_pn"]) - (cells["A_np"] - cells["A_nn"]))


def amplitude_score(norm: float, s_pos: float, s_neg: float) -> float:
    return float(norm * (s_pos - s_neg))


def _normalized(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return np.zeros_like(vector, dtype=np.float32)
    return vector / norm


def build_prepared_queries(
    shard: Any,
    representation: str,
    layer: int,
    progress_bins: int,
) -> tuple[list[PreparedQuery], dict[int, bool]]:
    if progress_bins <= 0:
        raise ValueError("progress_bins must be positive")
    layers = shard["layers"].astype(int).tolist()
    if layer not in layers:
        raise ValueError(f"layer {layer} not found in shard layers={layers}")
    layer_pos = layers.index(layer)
    key = f"span_{representation}"
    hidden = shard[key][:, layer_pos, :].astype(np.float32)
    rollout_ids = shard["rollout_id"].astype(int)
    labels_by_span = shard["is_correct"].astype(bool)
    starts = shard["span_start"].astype(int)
    progress = shard["relative_progress"].astype(float)
    lengths = shard["think_length"].astype(int)

    labels: dict[int, bool] = {}
    records: list[dict[str, Any]] = []
    for rollout_id in sorted(np.unique(rollout_ids).tolist()):
        indices = np.flatnonzero(rollout_ids == rollout_id)
        indices = indices[np.argsort(starts[indices])]
        labels[int(rollout_id)] = bool(labels_by_span[indices[0]])
        points = hidden[indices]
        displacements = points[1:] - points[:-1]
        for local_index, vector in enumerate(displacements, start=1):
            rho = float(progress[indices[local_index]])
            bin_id = min(int(rho * progress_bins), progress_bins - 1)
            norm = float(np.linalg.norm(vector))
            records.append(
                {
                    "rollout_id": int(rollout_id),
                    "progress_bin": int(bin_id),
                    "relative_progress": rho,
                    "span_start": int(starts[indices[local_index]]),
                    "displacement_norm": norm,
                    "think_length": int(lengths[indices[local_index]]),
                    "unit_vector": _normalized(vector),
                }
            )

    by_rollout_bin: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_rollout_bin[(record["rollout_id"], record["progress_bin"])].append(record)

    prepared = []
    for record in records:
        similarities: dict[int, float] = {}
        for reference_id in labels:
            if reference_id == record["rollout_id"]:
                continue
            candidates = by_rollout_bin.get((reference_id, record["progress_bin"]), [])
            if not candidates:
                continue
            candidate = min(
                candidates,
                key=lambda item: abs(item["relative_progress"] - record["relative_progress"]),
            )
            similarities[reference_id] = float(
                np.dot(record["unit_vector"], candidate["unit_vector"])
            )
        prepared.append(
            PreparedQuery(
                rollout_id=record["rollout_id"],
                progress_bin=record["progress_bin"],
                relative_progress=record["relative_progress"],
                span_start=record["span_start"],
                displacement_norm=record["displacement_norm"],
                think_length=record["think_length"],
                similarities=similarities,
            )
        )
    return prepared, labels


def _empty_interaction(question_id: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "A_pp": np.nan,
        "A_pn": np.nan,
        "A_np": np.nan,
        "A_nn": np.nan,
        "A_amp_pp": np.nan,
        "A_amp_pn": np.nan,
        "A_amp_np": np.nan,
        "A_amp_nn": np.nan,
        "I_angle": np.nan,
        "I_amp": np.nan,
        "pairwise_auc_angle": np.nan,
        "pairwise_auc_theta": np.nan,
        "pairwise_auc_amp": np.nan,
        "n_rollouts": 0,
    }


def score_prepared_queries(
    question_id: str,
    prepared: list[PreparedQuery],
    labels: dict[int, bool],
    excluded_rollout: int | None = None,
    return_details: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    active_labels = {
        rollout_id: label
        for rollout_id, label in labels.items()
        if rollout_id != excluded_rollout
    }
    span_rows = []
    bin_accumulator: dict[tuple[int, int], dict[str, float]] = {}
    coverage_counts: Counter[tuple[int, bool]] = Counter()

    for query in prepared:
        if query.rollout_id not in active_labels:
            continue
        subsets = balanced_reference_subsets(active_labels, query.rollout_id)
        subset_scores = []
        selected_positive: Counter[int] = Counter()
        selected_negative: Counter[int] = Counter()
        for positive_ids, negative_ids in subsets:
            if any(rid not in query.similarities for rid in positive_ids + negative_ids):
                continue
            positive_values = [(rid, query.similarities[rid]) for rid in positive_ids]
            negative_values = [(rid, query.similarities[rid]) for rid in negative_ids]
            positive_ref, s_pos = max(positive_values, key=lambda item: (item[1], -item[0]))
            negative_ref, s_neg = max(negative_values, key=lambda item: (item[1], -item[0]))
            selected_positive[positive_ref] += 1
            selected_negative[negative_ref] += 1
            subset_scores.append((float(s_pos), float(s_neg)))
        if not subset_scores:
            continue
        s_pos = float(np.mean([item[0] for item in subset_scores]))
        s_neg = float(np.mean([item[1] for item in subset_scores]))
        d_angle = float(s_pos - s_neg)
        theta_pos = float(np.degrees(np.arccos(np.clip(s_pos, -1.0, 1.0))))
        theta_neg = float(np.degrees(np.arccos(np.clip(s_neg, -1.0, 1.0))))
        d_theta = float(theta_neg - theta_pos)
        p_pos = float(query.displacement_norm * s_pos)
        p_neg = float(query.displacement_norm * s_neg)
        d_amp = float(p_pos - p_neg)
        selected_pos_id = selected_positive.most_common(1)[0][0]
        selected_neg_id = selected_negative.most_common(1)[0][0]
        coverage_counts.update((rid, True) for rid, count in selected_positive.items() for _ in range(count))
        coverage_counts.update((rid, False) for rid, count in selected_negative.items() for _ in range(count))
        row = {
            "question_id": question_id,
            "rollout_id": query.rollout_id,
            "is_correct": active_labels[query.rollout_id],
            "progress_bin": query.progress_bin,
            "relative_progress": query.relative_progress,
            "span_start": query.span_start,
            "think_length": query.think_length,
            "think_length_bin": length_bin(query.think_length),
            "displacement_norm": query.displacement_norm,
            "s_pos": s_pos,
            "s_neg": s_neg,
            "d_angle": d_angle,
            "theta_pos_deg": theta_pos,
            "theta_neg_deg": theta_neg,
            "d_theta_deg": d_theta,
            "p_pos": p_pos,
            "p_neg": p_neg,
            "d_amp": d_amp,
            "selected_ref_pos": selected_pos_id,
            "selected_ref_neg": selected_neg_id,
            "balanced_subset_count": len(subset_scores),
        }
        if return_details:
            span_rows.append(row)
        key = (query.rollout_id, query.progress_bin)
        accumulator = bin_accumulator.setdefault(
            key,
            {
                "count": 0.0,
                "think_length": float(query.think_length),
                "displacement_norm": 0.0,
                "s_pos": 0.0,
                "s_neg": 0.0,
                "d_angle": 0.0,
                "theta_pos_deg": 0.0,
                "theta_neg_deg": 0.0,
                "d_theta_deg": 0.0,
                "p_pos": 0.0,
                "p_neg": 0.0,
                "d_amp": 0.0,
            },
        )
        accumulator["count"] += 1.0
        for name in (
            "displacement_norm",
            "s_pos",
            "s_neg",
            "d_angle",
            "theta_pos_deg",
            "theta_neg_deg",
            "d_theta_deg",
            "p_pos",
            "p_neg",
            "d_amp",
        ):
            accumulator[name] += float(row[name])

    rollout_bin_rows = []
    by_rollout: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (rollout_id, progress_bin), values in sorted(bin_accumulator.items()):
        count = values["count"]
        row = {
            "question_id": question_id,
            "rollout_id": rollout_id,
            "is_correct": active_labels[rollout_id],
            "progress_bin": progress_bin,
            "think_length": int(values["think_length"]),
            "think_length_bin": length_bin(values["think_length"]),
            "n_span_steps": int(count),
        }
        for name in (
            "displacement_norm",
            "s_pos",
            "s_neg",
            "d_angle",
            "theta_pos_deg",
            "theta_neg_deg",
            "d_theta_deg",
            "p_pos",
            "p_neg",
            "d_amp",
        ):
            row[name] = float(values[name] / count)
        rollout_bin_rows.append(row)
        by_rollout[rollout_id].append(row)

    rollout_rows = []
    for rollout_id, bins in sorted(by_rollout.items()):
        row = {
            "question_id": question_id,
            "rollout_id": rollout_id,
            "is_correct": active_labels[rollout_id],
            "think_length": int(bins[0]["think_length"]),
            "think_length_bin": bins[0]["think_length_bin"],
            "n_progress_bins": len(bins),
        }
        for name in (
            "displacement_norm",
            "s_pos",
            "s_neg",
            "d_angle",
            "theta_pos_deg",
            "theta_neg_deg",
            "d_theta_deg",
            "p_pos",
            "p_neg",
            "d_amp",
        ):
            row[name] = float(np.mean([item[name] for item in bins]))
        rollout_rows.append(row)

    rollout_df = pd.DataFrame(rollout_rows)
    if rollout_df.empty or rollout_df["is_correct"].nunique() < 2:
        interaction = _empty_interaction(question_id)
    else:
        positive = rollout_df[rollout_df["is_correct"]]
        negative = rollout_df[~rollout_df["is_correct"]]
        cells = {
            "A_pp": float(positive["s_pos"].mean()),
            "A_pn": float(positive["s_neg"].mean()),
            "A_np": float(negative["s_pos"].mean()),
            "A_nn": float(negative["s_neg"].mean()),
        }
        amp_cells = {
            "A_amp_pp": float(positive["p_pos"].mean()),
            "A_amp_pn": float(positive["p_neg"].mean()),
            "A_amp_np": float(negative["p_pos"].mean()),
            "A_amp_nn": float(negative["p_neg"].mean()),
        }
        interaction = {
            "question_id": question_id,
            **cells,
            **amp_cells,
            "I_angle": interaction_from_cells(cells),
            "I_amp": float(
                (amp_cells["A_amp_pp"] - amp_cells["A_amp_pn"])
                - (amp_cells["A_amp_np"] - amp_cells["A_amp_nn"])
            ),
            "pairwise_auc_angle": pairwise_auc(
                positive["d_angle"].to_numpy(), negative["d_angle"].to_numpy()
            ),
            "pairwise_auc_theta": pairwise_auc(
                positive["d_theta_deg"].to_numpy(), negative["d_theta_deg"].to_numpy()
            ),
            "pairwise_auc_amp": pairwise_auc(
                positive["d_amp"].to_numpy(), negative["d_amp"].to_numpy()
            ),
            "n_rollouts": int(len(rollout_df)),
        }

    coverage_rows = []
    totals_by_label = {
        label: sum(count for (rid, key_label), count in coverage_counts.items() if key_label == label)
        for label in (True, False)
    }
    for (reference_id, reference_correct), count in sorted(coverage_counts.items()):
        total = totals_by_label[reference_correct]
        coverage_rows.append(
            {
                "question_id": question_id,
                "reference_id": reference_id,
                "reference_correct": reference_correct,
                "selection_count": int(count),
                "selection_fraction": float(count / total) if total else np.nan,
            }
        )
    return (
        pd.DataFrame(span_rows),
        pd.DataFrame(rollout_bin_rows) if return_details else pd.DataFrame(),
        rollout_df if return_details else pd.DataFrame(),
        interaction,
        pd.DataFrame(coverage_rows) if return_details else pd.DataFrame(),
    )


def bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if n_boot <= 0:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return mean, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_interactions(interactions: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for (representation, layer), group in interactions.groupby(["representation", "layer"]):
        row: dict[str, Any] = {
            "representation": representation,
            "layer": int(layer),
            "n_questions": int(group["question_id"].nunique()),
        }
        for offset, feature in enumerate(
            ["I_angle", "I_amp", "pairwise_auc_angle", "pairwise_auc_theta", "pairwise_auc_amp"]
        ):
            mean, low, high = bootstrap_mean(group[feature].to_numpy(), n_boot, seed + offset)
            row[f"mean_{feature}"] = mean
            row[f"ci_low_{feature}"] = low
            row[f"ci_high_{feature}"] = high
        for cell in ["A_pp", "A_pn", "A_np", "A_nn"]:
            row[f"mean_{cell}"] = float(group[cell].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["representation", "layer"]).reset_index(drop=True)


def run_permutations(
    cache: dict[tuple[str, str, int], tuple[list[PreparedQuery], dict[int, bool]]],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if n_permutations <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    records = []
    configurations = sorted({(representation, layer) for _, representation, layer in cache})
    for permutation_id in range(n_permutations):
        permuted_by_question: dict[str, dict[int, bool]] = {}
        for (question_id, _, _), (_, labels) in sorted(cache.items()):
            if question_id in permuted_by_question:
                continue
            rollout_ids = sorted(labels)
            shuffled = np.asarray([labels[rid] for rid in rollout_ids], dtype=bool)
            rng.shuffle(shuffled)
            permuted_by_question[question_id] = {
                rid: bool(label) for rid, label in zip(rollout_ids, shuffled)
            }
        by_config: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for (question_id, representation, layer), (prepared, labels) in sorted(cache.items()):
            _, _, _, interaction, _ = score_prepared_queries(
                question_id,
                prepared,
                permuted_by_question[question_id],
                return_details=False,
            )
            by_config[(representation, layer)].append(interaction)
        for representation, layer in configurations:
            frame = pd.DataFrame(by_config[(representation, layer)])
            records.append(
                {
                    "permutation_id": permutation_id,
                    "representation": representation,
                    "layer": layer,
                    "mean_I_angle": float(frame["I_angle"].mean()),
                    "mean_I_amp": float(frame["I_amp"].mean()),
                }
            )
    return pd.DataFrame(records)


def add_permutation_pvalues(summary: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["permutation_p_I_angle"] = np.nan
    summary["permutation_p_I_amp"] = np.nan
    if nulls.empty:
        return summary
    for index, row in summary.iterrows():
        subset = nulls[
            (nulls["representation"] == row["representation"])
            & (nulls["layer"] == row["layer"])
        ]
        for feature in ("I_angle", "I_amp"):
            observed = row[f"mean_{feature}"]
            values = subset[f"mean_{feature}"].to_numpy()
            p_value = float((1 + np.sum(values >= observed)) / (len(values) + 1))
            summary.loc[index, f"permutation_p_{feature}"] = p_value
    return summary


def write_figures(summary: pd.DataFrame, rollout_bins: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if not summary.empty:
        labels = [f"{row.representation}-L{int(row.layer)}" for row in summary.itertuples()]
        values = summary["mean_I_amp"].to_numpy()
        lower = values - summary["ci_low_I_amp"].to_numpy()
        upper = summary["ci_high_I_amp"].to_numpy() - values
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#267365")
        ax.errorbar(labels, values, yerr=np.vstack([lower, upper]), fmt="none", color="black", capsize=4)
        ax.axhline(0.0, color="#444444", linewidth=1)
        ax.set_ylabel("I_amp")
        ax.set_title("Long success-trajectory amplitude interaction")
        fig.tight_layout()
        path = figure_dir / "E1_amplitude_interaction.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    if not rollout_bins.empty:
        primary = rollout_bins[
            (rollout_bins["representation"] == "last") & (rollout_bins["layer"] == 24)
        ]
        curve = primary.groupby(["progress_bin", "is_correct"], as_index=False)["d_amp"].mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for correct, group in curve.groupby("is_correct"):
            ax.plot(
                (group["progress_bin"] + 0.5)
                / max(int(primary["progress_bin"].max()) + 1, 1),
                group["d_amp"],
                marker="o",
                label="correct" if correct else "wrong",
            )
        ax.axhline(0.0, color="#444444", linewidth=1)
        ax.set_xlabel("Relative think progress")
        ax.set_ylabel("D_amp")
        ax.set_title("Primary L24 span-last progress curve")
        ax.legend()
        fig.tight_layout()
        path = figure_dir / "E2_primary_progress_curve.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(
    summary: pd.DataFrame,
    interactions: pd.DataFrame,
    coverage: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    figures: list[Path],
) -> Path:
    path = output_dir / "LONG_EXPERIMENT_1_RESULTS.md"
    lines = [
        "# Long Experiment 1 Smoke Results",
        "",
        "Balanced leave-one-rollout-out success/failure trajectory interactions on fixed long responses.",
        "",
        "## Data",
        "",
        f"- Questions: {interactions['question_id'].nunique() if not interactions.empty else 0}",
        f"- Progress bins: {args.progress_bins}",
        f"- Bootstrap resamples: {args.bootstrap}",
        f"- Label permutations: {args.permutations}",
        "- New generation: no",
        "",
        "## Primary And Sensitivity Results",
        "",
        "| representation | layer | questions | I_angle [95% CI] | I_amp [95% CI] | pair-AUC angle | pair-AUC amp | perm p(I_amp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.representation} | {int(row.layer)} | {int(row.n_questions)} | "
            f"{row.mean_I_angle:.4f} [{row.ci_low_I_angle:.4f}, {row.ci_high_I_angle:.4f}] | "
            f"{row.mean_I_amp:.4f} [{row.ci_low_I_amp:.4f}, {row.ci_high_I_amp:.4f}] | "
            f"{row.mean_pairwise_auc_angle:.4f} | {row.mean_pairwise_auc_amp:.4f} | "
            f"{row.permutation_p_I_amp:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Four-Cell Interpretation",
            "",
            "Positive interaction means correct queries prefer observed correct references more strongly than wrong queries do, after symmetric correct/wrong reference control.",
            "",
            "## Reference Coverage",
            "",
            f"- Reference coverage rows: {len(coverage)}",
            "- Selected reference IDs and dominance fractions are saved in `reference_coverage.parquet`.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{figure.relative_to(output_dir).as_posix()}`" for figure in figures)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise FileNotFoundError(f"No shards matched: {args.input_glob}")

    all_span = []
    all_rollout_bins = []
    all_rollout = []
    all_interactions = []
    all_coverage = []
    cache: dict[tuple[str, str, int], tuple[list[PreparedQuery], dict[int, bool]]] = {}

    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            question_id = str(shard["question_id"].item())
            layers = shard["layers"].astype(int).tolist()
            for representation in ("last", "mean"):
                for layer in layers:
                    prepared, labels = build_prepared_queries(
                        shard,
                        representation=representation,
                        layer=layer,
                        progress_bins=args.progress_bins,
                    )
                    cache[(question_id, representation, layer)] = (prepared, labels)
                    span, rollout_bins, rollout, interaction, coverage = score_prepared_queries(
                        question_id, prepared, labels
                    )
                    for frame in (span, rollout_bins, rollout, coverage):
                        if not frame.empty:
                            frame["representation"] = representation
                            frame["layer"] = layer
                    interaction["representation"] = representation
                    interaction["layer"] = layer

                    positive_coverage = coverage[coverage["reference_correct"]] if not coverage.empty else coverage
                    if not positive_coverage.empty:
                        top_reference = int(
                            positive_coverage.sort_values(
                                ["selection_count", "reference_id"], ascending=[False, True]
                            ).iloc[0]["reference_id"]
                        )
                        _, _, _, robust, _ = score_prepared_queries(
                            question_id,
                            prepared,
                            labels,
                            excluded_rollout=top_reference,
                            return_details=False,
                        )
                        interaction["top_correct_reference"] = top_reference
                        interaction["I_angle_drop_top_correct"] = robust["I_angle"]
                        interaction["I_amp_drop_top_correct"] = robust["I_amp"]
                    else:
                        interaction["top_correct_reference"] = np.nan
                        interaction["I_angle_drop_top_correct"] = np.nan
                        interaction["I_amp_drop_top_correct"] = np.nan
                    all_span.append(span)
                    all_rollout_bins.append(rollout_bins)
                    all_rollout.append(rollout)
                    all_interactions.append(interaction)
                    all_coverage.append(coverage)

    span_scores = pd.concat(all_span, ignore_index=True) if all_span else pd.DataFrame()
    rollout_bins = pd.concat(all_rollout_bins, ignore_index=True) if all_rollout_bins else pd.DataFrame()
    rollout_features = pd.concat(all_rollout, ignore_index=True) if all_rollout else pd.DataFrame()
    interactions = pd.DataFrame(all_interactions)
    coverage = pd.concat(all_coverage, ignore_index=True) if all_coverage else pd.DataFrame()
    summary = summarize_interactions(interactions, args.bootstrap, args.seed)
    nulls = run_permutations(cache, args.permutations, args.seed + 1000)
    summary = add_permutation_pvalues(summary, nulls)

    span_scores.to_parquet(output_dir / "span_scores.parquet", index=False)
    rollout_bins.to_parquet(output_dir / "rollout_progress_bin_scores.parquet", index=False)
    rollout_features.to_parquet(output_dir / "rollout_features.parquet", index=False)
    interactions.to_csv(output_dir / "question_interactions.csv", index=False)
    coverage.to_parquet(output_dir / "reference_coverage.parquet", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    nulls.to_csv(output_dir / "permutation_null.csv", index=False)
    figures = write_figures(summary, rollout_bins, output_dir)
    report = write_report(summary, interactions, coverage, output_dir, args, figures)
    metadata = {
        "input_glob": args.input_glob,
        "shards": len(paths),
        "questions": int(interactions["question_id"].nunique()),
        "progress_bins": args.progress_bins,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
        "new_generation": False,
        "report": report.name,
    }
    (output_dir / "analysis_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
