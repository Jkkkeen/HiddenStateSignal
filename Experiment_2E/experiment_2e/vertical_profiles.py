from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import EPS, NORM_FLOOR, STAGES, energy_entropy_rows, stage_mask


PROFILE_COLUMNS = (
    "v1_raw_update_norm",
    "v1_relative_update_norm",
    "v3_demean_state_angle",
    "v4_layer_update_turning_angle",
    "v6_raw_activation_entropy",
    "v7_layer_difference_entropy",
)


def _adjacent_angles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    output = np.full(len(values), np.nan, dtype=float)
    if len(values) < 2:
        return output
    valid = (norms[1:] >= NORM_FLOOR) & (norms[:-1] >= NORM_FLOOR)
    cosine = np.sum(values[1:] * values[:-1], axis=1) / (
        norms[1:] * norms[:-1] + EPS
    )
    adjacent = np.full(len(values) - 1, np.nan, dtype=float)
    adjacent[valid] = np.arccos(np.clip(cosine[valid], -1.0, 1.0))
    output[1:] = adjacent
    return output


def local_profile_arrays(
    layers: np.ndarray,
    base_common: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(layers, dtype=np.float32)
    common = np.asarray(base_common, dtype=np.float32)
    if values.ndim != 2 or common.shape != values.shape:
        raise ValueError("layers and base_common must share shape (L+1,D)")
    if values.shape[0] < 2:
        raise ValueError("at least two hidden-state positions are required")

    updates = np.diff(values, axis=0)
    raw = np.linalg.norm(updates, axis=1)
    relative = raw / (np.linalg.norm(values[:-1], axis=1) + EPS)
    v1_raw = np.r_[np.nan, raw]
    v1_relative = np.r_[np.nan, relative]
    v3 = _adjacent_angles(values - common)
    update_angles = _adjacent_angles(updates)
    v4 = np.full(len(values), np.nan, dtype=float)
    v4[2:] = update_angles[1:]
    v6 = energy_entropy_rows(values, center=False, zero_value=np.nan)
    v7 = np.r_[
        np.nan,
        energy_entropy_rows(updates, center=True, zero_value=np.nan),
    ]
    return dict(
        zip(
            PROFILE_COLUMNS,
            (v1_raw, v1_relative, v3, v4, v6, v7),
            strict=True,
        )
    )


def _stage_medians(
    trajectory: np.ndarray,
    indices: np.ndarray,
    base_common: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    layer_count = trajectory.shape[1]
    medians = {
        metric: np.full(layer_count, np.nan, dtype=float) for metric in PROFILE_COLUMNS
    }
    counts = {metric: np.zeros(layer_count, dtype=int) for metric in PROFILE_COLUMNS}
    if not len(indices):
        return medians, counts
    chunk_profiles = [local_profile_arrays(trajectory[index], base_common) for index in indices]
    for metric in PROFILE_COLUMNS:
        values = np.stack([profile[metric] for profile in chunk_profiles])
        finite = np.isfinite(values)
        counts[metric] = finite.sum(axis=0)
        valid_layers = counts[metric] > 0
        if valid_layers.any():
            medians[metric][valid_layers] = np.nanmedian(
                values[:, valid_layers],
                axis=0,
            )
    return medians, counts


def reduce_vertical_profiles(
    trajectory: np.ndarray,
    progress: np.ndarray,
    *,
    representation: str,
    metadata: dict[str, Any],
    base_common: np.ndarray,
) -> pd.DataFrame:
    values = np.asarray(trajectory, dtype=np.float32)
    coordinates = np.asarray(progress, dtype=float)
    common = np.asarray(base_common, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(coordinates):
        raise ValueError("trajectory must be (K,L+1,D) and align with progress")
    if common.shape != values.shape[1:]:
        raise ValueError("base_common must have shape (L+1,D)")
    decoder_layers = values.shape[1] - 1
    if decoder_layers < 1:
        raise ValueError("trajectory must include embedding and decoder states")

    rows: list[dict[str, Any]] = []
    for stage in range(len(STAGES)):
        indices = np.flatnonzero(stage_mask(coordinates, stage))
        medians, counts = _stage_medians(values, indices, common)
        for layer_index in range(values.shape[1]):
            row = {
                **metadata,
                "representation": representation,
                "stage": stage,
                "layer_index": layer_index,
                "relative_depth": float(layer_index / decoder_layers),
            }
            row.update(
                {metric: float(medians[metric][layer_index]) for metric in PROFILE_COLUMNS}
            )
            row["profile_chunk_count"] = int(len(indices))
            row.update(
                {
                    f"profile_coverage_count_{metric}": int(counts[metric][layer_index])
                    for metric in PROFILE_COLUMNS
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
