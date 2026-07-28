#!/usr/bin/env python3
"""Pure metrics for the Experiment 02 replication."""

from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1e-12


def coordinate_energy_entropy(
    values: np.ndarray,
    *,
    center_coordinates: bool,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized coordinate-energy entropy and effective dimensions."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] < 2:
        raise ValueError("values must end in at least two hidden coordinates")
    if center_coordinates:
        array = array - array.mean(axis=-1, keepdims=True)
    energy = np.square(array)
    total = energy.sum(axis=-1, keepdims=True)
    probabilities = np.divide(
        energy,
        total,
        out=np.zeros_like(energy),
        where=total > eps,
    )
    raw_entropy = -(probabilities * np.log(probabilities + eps)).sum(axis=-1)
    normalized = np.clip(raw_entropy / np.log(array.shape[-1]), 0.0, 1.0)
    return normalized, np.exp(raw_entropy)


def trajectory_z_entropy(
    hidden: np.ndarray,
    *,
    sigma_scale: float = 1e-4,
    clip: float = 8.0,
    eps: float = EPS,
) -> dict[str, Any]:
    """Standardize each coordinate over one trajectory, then compute energy entropy."""
    values = np.asarray(hidden, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("hidden must have shape [tokens, dimensions] with both axes >= 2")
    if sigma_scale < 0 or clip <= 0:
        raise ValueError("sigma_scale must be non-negative and clip must be positive")
    mean = values.mean(axis=0)
    sigma = values.std(axis=0, ddof=1)
    median_sigma = float(np.median(sigma))
    sigma_floor = max(float(sigma_scale * median_sigma), eps)
    effective_sigma = np.maximum(sigma, sigma_floor)
    z = np.clip((values - mean) / effective_sigma, -clip, clip)
    entropy, effective = coordinate_energy_entropy(
        z,
        center_coordinates=False,
        eps=eps,
    )
    return {
        "entropy": entropy,
        "effective_dimensions": effective,
        "z": z,
        "mean": mean,
        "sigma": sigma,
        "median_sigma": median_sigma,
        "sigma_floor": sigma_floor,
        "low_variance_fraction": float(np.mean(sigma <= sigma_floor)),
    }


def _path_row(
    displacements: np.ndarray,
    *,
    scope: str,
    progress_bin: int,
    eps: float,
) -> dict[str, float | int | str]:
    path_length = float(np.linalg.norm(displacements, axis=1).sum())
    net_displacement = float(np.linalg.norm(displacements.sum(axis=0)))
    return {
        "scope": scope,
        "progress_bin": int(progress_bin),
        "step_count": int(displacements.shape[0]),
        "path_length": path_length,
        "net_displacement": net_displacement,
        "straightness": float(net_displacement / (path_length + eps)),
        "log_detour": float(
            np.log(path_length + eps) - np.log(net_displacement + eps)
        ),
    }


def path_integrals(
    displacements: np.ndarray,
    progress_bins: np.ndarray,
    *,
    n_bins: int = 10,
    min_block_steps: int = 3,
    eps: float = EPS,
) -> list[dict[str, float | int | str]]:
    """Compute whole-trajectory and progress-block path integrals."""
    vectors = np.asarray(displacements, dtype=np.float64)
    bins = np.asarray(progress_bins, dtype=np.int16)
    if vectors.ndim != 2 or vectors.shape[1] < 1:
        raise ValueError("displacements must have shape [steps, dimensions]")
    if bins.shape != (vectors.shape[0],):
        raise ValueError("progress_bins must contain one value per displacement")
    if vectors.shape[0] == 0:
        return []
    if n_bins <= 0 or min_block_steps <= 0:
        raise ValueError("n_bins and min_block_steps must be positive")
    if np.any((bins < 0) | (bins >= n_bins)):
        raise ValueError("progress bin outside configured range")

    rows = [_path_row(vectors, scope="whole", progress_bin=-1, eps=eps)]
    for progress_bin in range(n_bins):
        selected = vectors[bins == progress_bin]
        if selected.shape[0] < min_block_steps:
            continue
        rows.append(
            _path_row(
                selected,
                scope="progress_block",
                progress_bin=progress_bin,
                eps=eps,
            )
        )
    return rows
