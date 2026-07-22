#!/usr/bin/env python3
"""Pure metrics for long-response Experiment 0 hidden dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from long_success_trajectory_common import balanced_reference_subsets, full_span_bounds


EPS = 1e-12


def coordinate_energy_entropy(
    update: np.ndarray, eps: float = EPS
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized coordinate-energy entropy and effective dimensions."""
    values = np.asarray(update, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("update must have at least two hidden coordinates")
    centered = values - values.mean(axis=-1, keepdims=True)
    energy = centered * centered
    total = energy.sum(axis=-1, keepdims=True)
    probabilities = np.divide(
        energy,
        total,
        out=np.zeros_like(energy),
        where=total > eps,
    )
    raw_entropy = -(probabilities * np.log(probabilities + eps)).sum(axis=-1)
    normalized_entropy = raw_entropy / np.log(values.shape[-1])
    effective_dimensions = np.exp(raw_entropy)
    return normalized_entropy, effective_dimensions


def _cosine(left: np.ndarray, right: np.ndarray, eps: float = EPS) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= eps:
        return float("nan")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _unit_rows(vectors: np.ndarray, eps: float = EPS) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > eps)


def prototype_shrinkage(unit_vectors: np.ndarray, eps: float = EPS) -> float:
    """Measure how much a mean direction shrinks before normalization."""
    vectors = np.asarray(unit_vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return float("nan")
    centroid_norm = float(np.linalg.norm(vectors.mean(axis=0)))
    mean_norm = float(np.linalg.norm(vectors, axis=1).mean())
    if mean_norm <= eps:
        return 0.0
    return float(np.clip(centroid_norm / mean_norm, 0.0, 1.0))


def pool_span_vectors(
    hidden: np.ndarray,
    window: int,
    stride: int,
    mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool `[layers,tokens,dim]` hidden states into full-window spans."""
    values = np.asarray(hidden, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"hidden must be [layers,tokens,dim], got {values.shape}")
    if mode not in {"mean", "last"}:
        raise ValueError("mode must be 'mean' or 'last'")
    spans = full_span_bounds(values.shape[1], window=window, stride=stride)
    starts = np.asarray([start for start, _ in spans], dtype=np.int32)
    ends = np.asarray([end for _, end in spans], dtype=np.int32)
    if mode == "mean":
        pooled = np.stack(
            [values[:, start:end].mean(axis=1) for start, end in spans], axis=0
        )
    else:
        pooled = np.stack([values[:, end - 1] for _, end in spans], axis=0)
    progress = ((starts.astype(np.float64) + ends) / 2.0) / values.shape[1]
    return pooled.astype(np.float32), starts, ends, progress.astype(np.float32)


def build_span_direction_records(
    pooled: np.ndarray,
    layers: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    relative_progress: np.ndarray,
    rollout_id: int,
    is_correct: bool,
    representation: str,
    progress_bins: int,
    question_id: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build horizontal displacement and vertical layer-turn records."""
    values = np.asarray(pooled, dtype=np.float32)
    layer_values = np.asarray(layers, dtype=np.int32)
    if values.ndim != 3 or values.shape[1] != layer_values.size:
        raise ValueError("pooled must be [spans,layers,dim] and match layers")
    if values.shape[0] != len(starts) or values.shape[0] != len(relative_progress):
        raise ValueError("span metadata length mismatch")
    if progress_bins <= 0:
        raise ValueError("progress_bins must be positive")

    time_displacements = values[1:] - values[:-1]
    horizontal_rows: list[dict[str, Any]] = []
    for span_id in range(1, values.shape[0]):
        progress_bin = min(
            int(float(relative_progress[span_id]) * progress_bins), progress_bins - 1
        )
        for layer_index, layer in enumerate(layer_values):
            displacement = time_displacements[span_id - 1, layer_index].astype(np.float32)
            turn = float("nan")
            if span_id >= 2:
                turn = _cosine(
                    time_displacements[span_id - 1, layer_index],
                    time_displacements[span_id - 2, layer_index],
                )
            horizontal_rows.append(
                {
                    "question_id": str(question_id),
                    "rollout_id": int(rollout_id),
                    "is_correct": bool(is_correct),
                    "representation": str(representation),
                    "span_id": int(span_id),
                    "span_start": int(starts[span_id]),
                    "span_end": int(ends[span_id]),
                    "relative_progress": float(relative_progress[span_id]),
                    "progress_bin": int(progress_bin),
                    "layer": int(layer),
                    "displacement": displacement,
                    "displacement_norm": float(np.linalg.norm(displacement)),
                    "span_turn_cos": turn,
                }
            )

    vertical_rows: list[dict[str, Any]] = []
    layer_updates = values[:, 1:] - values[:, :-1]
    for span_id in range(values.shape[0]):
        progress_bin = min(
            int(float(relative_progress[span_id]) * progress_bins), progress_bins - 1
        )
        for layer_index in range(1, layer_values.size):
            update = layer_updates[span_id, layer_index - 1]
            turn = float("nan")
            if layer_index >= 2:
                turn = _cosine(
                    layer_updates[span_id, layer_index - 1],
                    layer_updates[span_id, layer_index - 2],
                )
            vertical_rows.append(
                {
                    "question_id": str(question_id),
                    "rollout_id": int(rollout_id),
                    "is_correct": bool(is_correct),
                    "representation": str(representation),
                    "span_id": int(span_id),
                    "span_start": int(starts[span_id]),
                    "span_end": int(ends[span_id]),
                    "relative_progress": float(relative_progress[span_id]),
                    "progress_bin": int(progress_bin),
                    "layer": int(layer_values[layer_index]),
                    "span_layer_update_norm": float(np.linalg.norm(update)),
                    "span_layer_turn_cos": turn,
                }
            )
    return pd.DataFrame(horizontal_rows), pd.DataFrame(vertical_rows)


def _summaries(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_count": 0,
        }
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p90": float(np.quantile(finite, 0.9)),
        f"{prefix}_count": int(finite.size),
    }


def aggregate_token_dynamics(hidden: np.ndarray, progress_bins: int = 10) -> pd.DataFrame:
    """Reference NumPy reducer for all token-amplitude Experiment 0 metrics."""
    values = np.asarray(hidden, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"hidden must be [layers,tokens,dim], got {values.shape}")
    if progress_bins <= 0:
        raise ValueError("progress_bins must be positive")
    n_layers, n_tokens, _ = values.shape
    token_bins = np.minimum(
        ((np.arange(n_tokens) + 1) * progress_bins) // n_tokens,
        progress_bins - 1,
    ).astype(int)
    rows: list[dict[str, Any]] = []
    previous_vertical_norm: np.ndarray | None = None
    for layer in range(n_layers):
        hidden_norm = np.linalg.norm(values[layer], axis=1)
        horizontal_update = values[layer, 1:] - values[layer, :-1]
        horizontal_norm = np.linalg.norm(horizontal_update, axis=1)
        horizontal_delta = np.diff(horizontal_norm)
        token_turn = np.asarray(
            [
                _cosine(horizontal_update[index], horizontal_update[index - 1])
                for index in range(1, horizontal_update.shape[0])
            ],
            dtype=np.float64,
        )
        vertical_norm = np.asarray([], dtype=np.float64)
        vertical_delta = np.asarray([], dtype=np.float64)
        entropy = np.asarray([], dtype=np.float64)
        effective = np.asarray([], dtype=np.float64)
        if layer >= 1:
            vertical_update = values[layer] - values[layer - 1]
            vertical_norm = np.linalg.norm(vertical_update, axis=1)
            entropy, effective = coordinate_energy_entropy(vertical_update)
            if previous_vertical_norm is not None:
                vertical_delta = vertical_norm - previous_vertical_norm
            previous_vertical_norm = vertical_norm

        for progress_bin in range(progress_bins):
            token_mask = token_bins == progress_bin
            horizontal_mask = token_bins[1:] == progress_bin
            horizontal_delta_mask = token_bins[2:] == progress_bin
            row: dict[str, Any] = {
                "layer": int(layer),
                "progress_bin": int(progress_bin),
                "token_count": int(token_mask.sum()),
                "horizontal_token_count": int(horizontal_mask.sum()),
                "hidden_norm_mean": float(np.mean(hidden_norm[token_mask]))
                if token_mask.any()
                else float("nan"),
            }
            row.update(_summaries(horizontal_norm[horizontal_mask], "horizontal_norm"))
            row.update(
                _summaries(horizontal_delta[horizontal_delta_mask], "horizontal_norm_delta")
            )
            row.update(_summaries(token_turn[horizontal_delta_mask], "token_turn_cos"))
            if layer >= 1:
                row.update(_summaries(vertical_norm[token_mask], "vertical_norm"))
                row.update(_summaries(entropy[token_mask], "coordinate_entropy"))
                row.update(_summaries(effective[token_mask], "effective_dimensions"))
            else:
                row.update(_summaries(np.asarray([]), "vertical_norm"))
                row.update(_summaries(np.asarray([]), "coordinate_entropy"))
                row.update(_summaries(np.asarray([]), "effective_dimensions"))
            if layer >= 2:
                row.update(_summaries(vertical_delta[token_mask], "vertical_norm_delta"))
            else:
                row.update(_summaries(np.asarray([]), "vertical_norm_delta"))
            rows.append(row)
    return pd.DataFrame(rows)


def score_cross_rollout_queries(
    records: pd.DataFrame,
    kappa_min: float = 0.20,
    eps: float = EPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score same-bin cross-rollout direction and movement-length support."""
    required = {
        "question_id",
        "rollout_id",
        "is_correct",
        "representation",
        "span_id",
        "relative_progress",
        "progress_bin",
        "layer",
        "displacement",
        "displacement_norm",
    }
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"missing cross-rollout columns: {sorted(missing)}")
    score_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    group_columns = ["question_id", "representation", "layer", "progress_bin"]
    for _, raw_group in records.groupby(group_columns, sort=True, dropna=False):
        group = raw_group.sort_values(
            ["rollout_id", "relative_progress", "span_id"], kind="stable"
        ).reset_index(drop=True)
        rollout_ids = group["rollout_id"].to_numpy(dtype=np.int64)
        progress = group["relative_progress"].to_numpy(dtype=np.float64)
        vectors = np.stack(group["displacement"].to_numpy()).astype(np.float64, copy=False)
        norms = group["displacement_norm"].to_numpy(dtype=np.float64)
        scalar_records = group.drop(columns=["displacement"]).to_dict("records")
        indices_by_rollout = {
            int(rollout_id): np.flatnonzero(rollout_ids == rollout_id)
            for rollout_id in np.unique(rollout_ids)
        }
        label_by_rollout = {
            rollout_id: bool(group.iloc[indices[0]]["is_correct"])
            for rollout_id, indices in indices_by_rollout.items()
        }
        subsets_by_rollout = {
            rollout_id: balanced_reference_subsets(label_by_rollout, rollout_id)
            for rollout_id in indices_by_rollout
        }
        positive_available_by_rollout = {
            rollout_id: sum(
                label for reference_id, label in label_by_rollout.items() if reference_id != rollout_id
            )
            for rollout_id in indices_by_rollout
        }
        negative_available_by_rollout = {
            rollout_id: sum(
                not label
                for reference_id, label in label_by_rollout.items()
                if reference_id != rollout_id
            )
            for rollout_id in indices_by_rollout
        }
        for query_index, query in enumerate(scalar_records):
            query_id = int(query["rollout_id"])
            subsets = subsets_by_rollout[query_id]
            nearest_index: dict[int, int] = {}
            query_progress = progress[query_index]
            for reference_id, indices in indices_by_rollout.items():
                if reference_id == query_id:
                    continue
                distances = np.abs(progress[indices] - query_progress)
                nearest_index[reference_id] = int(indices[int(np.argmin(distances))])
            subset_scores: list[dict[str, float]] = []
            query_vector = vectors[query_index]
            query_norm = float(norms[query_index])
            query_unit = query_vector / max(query_norm, eps)
            for subset_id, (positive_ids, negative_ids) in enumerate(subsets):
                positive_indices = np.asarray(
                    [nearest_index[reference_id] for reference_id in positive_ids], dtype=np.int64
                )
                negative_indices = np.asarray(
                    [nearest_index[reference_id] for reference_id in negative_ids], dtype=np.int64
                )
                positive_vectors = vectors[positive_indices]
                negative_vectors = vectors[negative_indices]
                positive_units = _unit_rows(positive_vectors, eps)
                negative_units = _unit_rows(negative_vectors, eps)
                positive_similarity = float(np.max(positive_units @ query_unit))
                negative_similarity = float(np.max(negative_units @ query_unit))
                kappa_pos = prototype_shrinkage(positive_units, eps)
                kappa_neg = prototype_shrinkage(negative_units, eps)
                prototype_valid = bool(kappa_pos >= kappa_min and kappa_neg >= kappa_min)
                proto_score = float("nan")
                if prototype_valid:
                    positive_centroid = positive_units.mean(axis=0)
                    negative_centroid = negative_units.mean(axis=0)
                    positive_centroid /= max(float(np.linalg.norm(positive_centroid)), eps)
                    negative_centroid /= max(float(np.linalg.norm(negative_centroid)), eps)
                    proto_score = float(
                        np.dot(query_unit, positive_centroid)
                        - np.dot(query_unit, negative_centroid)
                    )
                positive_lengths = norms[positive_indices]
                negative_lengths = norms[negative_indices]
                positive_center = float(np.exp(np.mean(np.log(positive_lengths + eps))))
                negative_center = float(np.exp(np.mean(np.log(negative_lengths + eps))))
                length_support = float(
                    abs(np.log(query_norm + eps) - np.log(negative_center + eps))
                    - abs(np.log(query_norm + eps) - np.log(positive_center + eps))
                )
                subset_scores.append(
                    {
                        "set_direction": positive_similarity - negative_similarity,
                        "prototype_direction": proto_score,
                        "length_support": length_support,
                        "kappa_pos": kappa_pos,
                        "kappa_neg": kappa_neg,
                        "prototype_valid": float(prototype_valid),
                    }
                )
                diagnostic_rows.append(
                    {
                        "question_id": str(query["question_id"]),
                        "rollout_id": query_id,
                        "is_correct": bool(query["is_correct"]),
                        "representation": str(query["representation"]),
                        "span_id": int(query["span_id"]),
                        "relative_progress": float(query["relative_progress"]),
                        "progress_bin": int(query["progress_bin"]),
                        "layer": int(query["layer"]),
                        "subset_id": int(subset_id),
                        "positive_reference_ids": ",".join(map(str, positive_ids)),
                        "negative_reference_ids": ",".join(map(str, negative_ids)),
                        "n_positive_references": len(positive_ids),
                        "n_negative_references": len(negative_ids),
                        "kappa_pos": kappa_pos,
                        "kappa_neg": kappa_neg,
                        "prototype_valid_010": bool(kappa_pos >= 0.10 and kappa_neg >= 0.10),
                        "prototype_valid_020": bool(kappa_pos >= 0.20 and kappa_neg >= 0.20),
                        "prototype_valid_030": bool(kappa_pos >= 0.30 and kappa_neg >= 0.30),
                    }
                )
            if not subset_scores:
                continue
            subset_frame = pd.DataFrame(subset_scores)
            valid_proto = subset_frame["prototype_direction"].dropna()
            base = dict(query)
            base.update(
                {
                    "cross_set_direction": float(subset_frame["set_direction"].mean()),
                    "cross_prototype_direction": float(valid_proto.mean())
                    if not valid_proto.empty
                    else float("nan"),
                    "cross_length_support": float(subset_frame["length_support"].mean()),
                    "prototype_kappa_pos_mean": float(subset_frame["kappa_pos"].mean()),
                    "prototype_kappa_pos_min": float(subset_frame["kappa_pos"].min()),
                    "prototype_kappa_neg_mean": float(subset_frame["kappa_neg"].mean()),
                    "prototype_kappa_neg_min": float(subset_frame["kappa_neg"].min()),
                    "prototype_valid_fraction": float(subset_frame["prototype_valid"].mean()),
                    "balanced_subset_count": int(len(subset_frame)),
                    "reference_rollout_count_pos": int(len(subsets[0][0])),
                    "reference_rollout_count_neg": int(len(subsets[0][1])),
                    "available_reference_rollout_count_pos": int(
                        positive_available_by_rollout[query_id]
                    ),
                    "available_reference_rollout_count_neg": int(
                        negative_available_by_rollout[query_id]
                    ),
                }
            )
            score_rows.append(base)
    return pd.DataFrame(score_rows), pd.DataFrame(diagnostic_rows)


def build_pairwise_geometry(
    records: pd.DataFrame,
    layers: tuple[int, ...] | None = (24, 36),
    eps: float = EPS,
) -> pd.DataFrame:
    """Persist scalar nearest-reference geometry for leakage-safe label permutations."""
    required = {
        "question_id",
        "rollout_id",
        "is_correct",
        "representation",
        "span_id",
        "relative_progress",
        "progress_bin",
        "layer",
        "displacement",
        "displacement_norm",
    }
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"missing pairwise geometry columns: {sorted(missing)}")
    view = records if layers is None else records[records["layer"].isin(layers)]
    rows: list[dict[str, Any]] = []
    group_columns = ["question_id", "representation", "layer", "progress_bin"]
    for _, raw_group in view.groupby(group_columns, sort=True, dropna=False):
        group = raw_group.sort_values(
            ["rollout_id", "relative_progress", "span_id"], kind="stable"
        ).reset_index(drop=True)
        rollout_ids = group["rollout_id"].to_numpy(dtype=np.int64)
        progress = group["relative_progress"].to_numpy(dtype=np.float64)
        vectors = np.stack(group["displacement"].to_numpy()).astype(np.float64, copy=False)
        units = _unit_rows(vectors, eps)
        norms = group["displacement_norm"].to_numpy(dtype=np.float64)
        indices_by_rollout = {
            int(rollout_id): np.flatnonzero(rollout_ids == rollout_id)
            for rollout_id in np.unique(rollout_ids)
        }
        scalar_records = group.drop(columns=["displacement"]).to_dict("records")
        for query_index, query in enumerate(scalar_records):
            query_id = int(query["rollout_id"])
            for reference_id, indices in indices_by_rollout.items():
                if reference_id == query_id:
                    continue
                distances = np.abs(progress[indices] - progress[query_index])
                reference_index = int(indices[int(np.argmin(distances))])
                rows.append(
                    {
                        **query,
                        "reference_rollout_id": int(reference_id),
                        "reference_is_correct": bool(
                            scalar_records[reference_index]["is_correct"]
                        ),
                        "reference_span_id": int(
                            scalar_records[reference_index]["span_id"]
                        ),
                        "reference_progress": float(progress[reference_index]),
                        "reference_norm": float(norms[reference_index]),
                        "cosine_similarity": float(
                            np.dot(units[query_index], units[reference_index])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def score_set_direction_from_geometry(
    geometry: pd.DataFrame,
    label_overrides: dict[tuple[str, int], bool] | None = None,
    eps: float = EPS,
) -> pd.DataFrame:
    """Rebuild balanced reference scores from scalar geometry under new labels."""
    required = {
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
        "reference_norm",
        "cosine_similarity",
    }
    missing = required.difference(geometry.columns)
    if missing:
        raise ValueError(f"missing scalar geometry columns: {sorted(missing)}")
    label_overrides = label_overrides or {}
    rows: list[dict[str, Any]] = []
    query_columns = [
        "question_id",
        "rollout_id",
        "representation",
        "span_id",
        "relative_progress",
        "progress_bin",
        "layer",
    ]
    for keys, group in geometry.groupby(query_columns, sort=True, observed=True):
        question_id = str(keys[0])
        query_id = int(keys[1])
        query_label = label_overrides.get(
            (question_id, query_id), bool(group.iloc[0]["is_correct"])
        )
        label_by_rollout = {query_id: query_label}
        for row in group.itertuples():
            reference_id = int(row.reference_rollout_id)
            label_by_rollout[reference_id] = label_overrides.get(
                (question_id, reference_id), bool(row.reference_is_correct)
            )
        subsets = balanced_reference_subsets(label_by_rollout, query_id)
        if not subsets:
            continue
        reference = group.set_index("reference_rollout_id", verify_integrity=True)
        set_scores = []
        length_scores = []
        query_norm = float(group.iloc[0]["displacement_norm"])
        for positive_ids, negative_ids in subsets:
            positive = reference.loc[list(positive_ids)]
            negative = reference.loc[list(negative_ids)]
            set_scores.append(
                float(
                    positive["cosine_similarity"].max()
                    - negative["cosine_similarity"].max()
                )
            )
            positive_center = float(
                np.exp(np.mean(np.log(positive["reference_norm"].to_numpy() + eps)))
            )
            negative_center = float(
                np.exp(np.mean(np.log(negative["reference_norm"].to_numpy() + eps)))
            )
            length_scores.append(
                float(
                    abs(np.log(query_norm + eps) - np.log(negative_center + eps))
                    - abs(np.log(query_norm + eps) - np.log(positive_center + eps))
                )
            )
        base = group.iloc[0].drop(
            labels=[
                "reference_rollout_id",
                "reference_is_correct",
                "reference_span_id",
                "reference_progress",
                "reference_norm",
                "cosine_similarity",
            ]
        ).to_dict()
        base["is_correct"] = bool(query_label)
        base["cross_set_direction"] = float(np.mean(set_scores))
        base["cross_length_support"] = float(np.mean(length_scores))
        base["balanced_subset_count"] = int(len(subsets))
        rows.append(base)
    return pd.DataFrame(rows)
