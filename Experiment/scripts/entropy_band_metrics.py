#!/usr/bin/env python3
"""Pure metrics for the frozen raw-activation entropy band."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


EPS = 1e-12


def progress_bin_ids(length: int, n_bins: int = 10) -> np.ndarray:
    """Match the relative-progress convention used in Experiment 02."""
    if length <= 0 or n_bins <= 0:
        raise ValueError("length and n_bins must be positive")
    return np.minimum(
        ((np.arange(length, dtype=np.int64) + 1) * n_bins) // length,
        n_bins - 1,
    ).astype(np.int16)


def centered_coordinate_entropy(
    values: np.ndarray,
    *,
    eps: float = EPS,
) -> np.ndarray:
    """Compute normalized coordinate-energy entropy after per-token centering."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] < 2:
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
    entropy = -(probabilities * np.log(probabilities + eps)).sum(axis=-1)
    return np.clip(entropy / np.log(array.shape[-1]), 0.0, 1.0)


def frozen_band_score(
    layer_values: Mapping[int, float],
    *,
    layers: Sequence[int] = tuple(range(14, 20)),
) -> float:
    """Average the six preregistered layer means without reweighting."""
    requested = tuple(int(layer) for layer in layers)
    if not requested or any(layer not in layer_values for layer in requested):
        raise ValueError("all requested layer values must be present")
    values = np.asarray([layer_values[layer] for layer in requested], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("layer values must be finite")
    return float(values.mean())
