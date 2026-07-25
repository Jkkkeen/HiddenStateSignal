#!/usr/bin/env python3
"""Pure numerical metrics for Experiment 04 fixed-chunk hidden dynamics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


EPS = 1e-12
REPRESENTATIONS = ("last", "last25", "mean")


def coordinate_entropy(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Normalized coordinate-energy entropy after per-vector centering."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 2:
        raise ValueError("values must end in at least two hidden coordinates")
    centered = array - array.mean(axis=-1, keepdims=True)
    energy = np.square(centered)
    total = energy.sum(axis=-1, keepdims=True)
    probabilities = np.divide(
        energy,
        total,
        out=np.zeros_like(energy),
        where=total > eps,
    )
    raw = -(probabilities * np.log(probabilities + eps)).sum(axis=-1)
    normalized = np.clip(raw / np.log(array.shape[-1]), 0.0, 1.0)
    return np.where(total[..., 0] > eps, normalized, 0.0)


def _bounds(token_count: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    full_count = token_count // chunk_size
    full = np.asarray(
        [(idx * chunk_size, (idx + 1) * chunk_size) for idx in range(full_count)],
        dtype=np.int64,
    ).reshape(-1, 2)
    partial_start = full_count * chunk_size
    partial = (
        np.asarray([(partial_start, token_count)], dtype=np.int64)
        if partial_start < token_count
        else np.empty((0, 2), dtype=np.int64)
    )
    return full, partial


def _pool(hidden: np.ndarray, bounds: np.ndarray, last_n: int) -> dict[str, np.ndarray]:
    if bounds.size == 0:
        shape = (0, hidden.shape[-1])
        return {name: np.empty(shape, dtype=np.float32) for name in REPRESENTATIONS}
    last = np.stack([hidden[end - 1] for start, end in bounds]).astype(np.float32)
    mean = np.stack([hidden[start:end].mean(axis=0) for start, end in bounds]).astype(
        np.float32
    )
    last25 = np.stack(
        [hidden[max(start, end - last_n) : end].mean(axis=0) for start, end in bounds]
    ).astype(np.float32)
    return {"last": last, "last25": last25, "mean": mean}


def chunk_representations(
    hidden: np.ndarray,
    chunk_size: int = 256,
    last_n: int = 25,
) -> dict[str, Any]:
    """Pool one [tokens, hidden_dim] trajectory into full and partial chunks."""
    array = np.asarray(hidden, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("hidden must have shape [tokens, hidden_dim]")
    if last_n <= 0:
        raise ValueError("last_n must be positive")
    full, partial = _bounds(array.shape[0], chunk_size)
    return {
        "full_bounds": full,
        "partial_bounds": partial,
        "full": _pool(array, full, last_n),
        "partial": _pool(array, partial, last_n),
    }


def late_window(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 8:
        return np.empty(0, dtype=np.float64)
    size = max(4, int(np.ceil(0.25 * array.size)))
    return array[-size:]


def _cosines(left: np.ndarray, right: np.ndarray, threshold: float) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    denominator = left_norm * right_norm
    valid = (
        np.isfinite(denominator)
        & (left_norm > threshold)
        & (right_norm > threshold)
    )
    result = np.full(left_norm.shape, np.nan, dtype=np.float64)
    result[valid] = np.sum(left[valid] * right[valid], axis=-1) / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def state_angles(
    points: np.ndarray,
    common: np.ndarray | None = None,
    threshold: float = EPS,
) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("points must have shape [positions, hidden_dim]")
    if array.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    if common is not None:
        center = np.asarray(common, dtype=np.float64)
        if center.shape != (array.shape[1],):
            raise ValueError("common must match hidden_dim")
        array = array - center
    return np.arccos(_cosines(array[:-1], array[1:], threshold))


def _integral(displacements: np.ndarray, eps: float) -> tuple[float, float, float, float]:
    path_length = float(np.linalg.norm(displacements, axis=-1).sum())
    net_displacement = float(np.linalg.norm(displacements.sum(axis=0)))
    straightness = float(net_displacement / (path_length + eps))
    log_detour = float(np.log(path_length + eps) - np.log(net_displacement + eps))
    return path_length, net_displacement, straightness, log_detour


def path_metrics(
    points: np.ndarray,
    rolling: int = 4,
    threshold: float = EPS,
    eps: float = EPS,
) -> dict[str, np.ndarray | float]:
    """Local and integrated geometry for one ordered hidden-state path."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError("points must have shape [positions, hidden_dim]")
    if rolling < 2:
        raise ValueError("rolling must contain at least two displacements")
    if array.shape[0] < 2:
        empty = np.empty(0, dtype=np.float64)
        return {
            "displacement": np.empty((0, array.shape[1]), dtype=np.float64),
            "step_norm": empty,
            "turn_cos": empty,
            "turn_angle": empty,
            "direction_consistency": float("nan"),
            "turn_angle_std": float("nan"),
            "turn_angle_late_std": float("nan"),
            "path_length": float("nan"),
            "net_displacement": float("nan"),
            "straightness": float("nan"),
            "log_detour": float("nan"),
            "rolling_path_length": empty,
            "rolling_net_displacement": empty,
            "rolling_straightness": empty,
            "rolling_log_detour": empty,
        }
    displacement = np.diff(array, axis=0)
    step_norm = np.linalg.norm(displacement, axis=-1)
    turn_cos = (
        _cosines(displacement[:-1], displacement[1:], threshold)
        if displacement.shape[0] >= 2
        else np.empty(0, dtype=np.float64)
    )
    turn_angle = np.arccos(turn_cos)
    finite_turn = turn_angle[np.isfinite(turn_angle)]
    late_turn = late_window(finite_turn)
    path_length, net_displacement, straightness, log_detour = _integral(
        displacement, eps
    )

    rolling_values = []
    if displacement.shape[0] >= rolling:
        for start in range(displacement.shape[0] - rolling + 1):
            rolling_values.append(_integral(displacement[start : start + rolling], eps))
    rolling_array = (
        np.asarray(rolling_values, dtype=np.float64)
        if rolling_values
        else np.empty((0, 4), dtype=np.float64)
    )
    return {
        "displacement": displacement,
        "step_norm": step_norm,
        "turn_cos": turn_cos,
        "turn_angle": turn_angle,
        "direction_consistency": float(np.nanmean(turn_cos))
        if np.any(np.isfinite(turn_cos))
        else float("nan"),
        "turn_angle_std": float(np.std(finite_turn, ddof=1))
        if finite_turn.size >= 2
        else float("nan"),
        "turn_angle_late_std": float(np.std(late_turn, ddof=1))
        if late_turn.size >= 2
        else float("nan"),
        "path_length": path_length,
        "net_displacement": net_displacement,
        "straightness": straightness,
        "log_detour": log_detour,
        "rolling_path_length": rolling_array[:, 0],
        "rolling_net_displacement": rolling_array[:, 1],
        "rolling_straightness": rolling_array[:, 2],
        "rolling_log_detour": rolling_array[:, 3],
    }


def vertical_scale(
    rollout_update_norms: Iterable[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Median of per-rollout chunk medians, preserving equal rollout weight."""
    summaries = []
    for values in rollout_update_norms:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("each rollout update norm array must be [chunks, updates]")
        summaries.append(np.nanmedian(array, axis=0))
    if not summaries:
        raise ValueError("at least one rollout is required")
    scale = np.nanmedian(np.stack(summaries, axis=0), axis=0)
    threshold = np.maximum(1e-8, 1e-4 * scale)
    return scale, threshold


def rollout_equal_common(rollout_vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Mean of per-rollout chunk means for arrays shaped [chunks, layers, dim]."""
    summaries = []
    for vectors in rollout_vectors:
        array = np.asarray(vectors, dtype=np.float64)
        if array.ndim != 3 or array.shape[0] == 0:
            raise ValueError("rollout vectors must have shape [chunks, layers, dim]")
        summaries.append(array.mean(axis=0))
    if not summaries:
        raise ValueError("at least one rollout is required")
    return np.mean(np.stack(summaries, axis=0), axis=0)


def aggregate_distribution(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {name: float("nan") for name in ("mean", "median", "p90", "top25mean")}
    top_count = min(25, finite.size)
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "top25mean": float(np.mean(np.partition(finite, -top_count)[-top_count:])),
    }
