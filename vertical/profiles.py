from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .pooling import STAGES, stage_mask


EPS = 1e-12
NORM_FLOOR = 1e-6
PROFILE_COLUMNS = (
    "v1_raw_update_norm",
    "v1_relative_update_norm",
    "v3_demean_state_angle",
    "v4_layer_update_turning_angle",
    "v6_raw_activation_entropy",
    "v7_layer_difference_entropy",
)


def select_decoder_backbone(
    values: np.ndarray,
    *,
    num_decoder_layers: int,
    layer_kind: np.ndarray | None = None,
) -> np.ndarray:
    arrays = np.asarray(values)
    if arrays.ndim < 2:
        raise ValueError("hidden values must include layer and hidden dimensions")
    hidden_state_count = arrays.shape[-2]
    expected = num_decoder_layers + 1
    if hidden_state_count == expected and layer_kind is None:
        return arrays
    if layer_kind is None:
        raise ValueError("extra hidden-state positions require explicit layer_kind metadata")
    kinds = np.asarray(layer_kind, dtype=str)
    if kinds.ndim != 1 or len(kinds) != hidden_state_count:
        raise ValueError("layer_kind must align with hidden-state positions")
    normalized = np.char.lower(kinds)
    embedding = np.flatnonzero(normalized == "embedding")
    decoder = np.flatnonzero(normalized == "decoder")
    if len(embedding) != 1 or len(decoder) != num_decoder_layers:
        raise ValueError(
            "layer_kind must identify exactly one embedding and num_decoder_layers decoder states"
        )
    indices = np.r_[embedding, decoder]
    if not np.all(np.diff(indices) > 0):
        raise ValueError("embedding and decoder layer_kind positions must preserve model order")
    return np.take(arrays, indices, axis=-2)


def energy_entropy_rows(
    states: np.ndarray,
    *,
    center: bool,
    zero_value: float,
) -> np.ndarray:
    values = np.array(states, dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("states must have shape (N,D) with D >= 2")
    if center:
        values -= values.mean(axis=1, keepdims=True)
    np.square(values, out=values)
    totals = values.sum(axis=1, dtype=np.float64)
    energy_log_energy = np.zeros_like(values)
    np.log(values, out=energy_log_energy, where=values > 0)
    energy_log_energy *= values
    weighted_logs = energy_log_energy.sum(axis=1, dtype=np.float64)
    output = np.full(values.shape[0], float(zero_value), dtype=float)
    valid = np.isfinite(totals) & (totals > EPS)
    output[valid] = (
        np.log(totals[valid]) - weighted_logs[valid] / totals[valid]
    ) / np.log(values.shape[1])
    return output


def _adjacent_angles(values: np.ndarray) -> np.ndarray:
    arrays = np.asarray(values, dtype=np.float32)
    if arrays.ndim != 2:
        raise ValueError("angle values must have shape (layer, dim)")
    norms = np.linalg.norm(arrays, axis=1)
    output = np.full(len(arrays), np.nan, dtype=float)
    if len(arrays) < 2:
        return output
    valid = (norms[1:] >= NORM_FLOOR) & (norms[:-1] >= NORM_FLOOR)
    cosine = np.sum(arrays[1:] * arrays[:-1], axis=1) / (
        norms[1:] * norms[:-1] + EPS
    )
    adjacent = np.full(len(arrays) - 1, np.nan, dtype=float)
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
    chunk_profiles = [
        local_profile_arrays(trajectory[index], base_common) for index in indices
    ]
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


def reduce_stage_profiles(
    trajectory: np.ndarray,
    progress: np.ndarray,
    *,
    representation: str,
    metadata: Mapping[str, Any],
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
            row: dict[str, Any] = {
                **metadata,
                "representation": representation,
                "stage": stage,
                "layer_index": layer_index,
                "relative_depth": float(layer_index / decoder_layers),
                "profile_chunk_count": int(len(indices)),
            }
            row.update(
                {metric: float(medians[metric][layer_index]) for metric in PROFILE_COLUMNS}
            )
            row.update(
                {
                    f"profile_coverage_count_{metric}": int(counts[metric][layer_index])
                    for metric in PROFILE_COLUMNS
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)

