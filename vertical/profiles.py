from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .pooling import STAGES, stage_mask


EPS = 1e-12
NORM_FLOOR = 1e-6
SVD_RTOL = 1e-6
STABLE_PROFILE_COLUMNS = (
    "v1_raw_update_norm",
    "v1_relative_update_norm",
    "v3_demean_state_angle",
    "v4_layer_update_turning_angle",
    "v6_raw_activation_entropy",
    "v7_layer_difference_entropy",
)
GEOMETRY_VALUE_COLUMNS = (
    "v5_path_length",
    "v5_net_displacement",
    "v5_straightness",
    "v5_log_detour",
    "v8_layer_state_er",
    "v8_layer_update_er",
    "v9_layer_state_ed",
    "v9_state_high_order_fraction",
    "v9_state_linear_fraction",
    "v9_state_relative_fit_rmse",
    "v9_state_condition_number",
    "v9_layer_update_ed",
    "v9_update_high_order_fraction",
    "v9_update_linear_fraction",
    "v9_update_relative_fit_rmse",
    "v9_update_condition_number",
)


def _window_metric(window: str, metric: str) -> str:
    family, name = metric.split("_", 1)
    return f"{family}_{window}_{name}"


EXTENDED_PROFILE_COLUMNS = (
    "v2_raw_state_angle",
    *(_window_metric("rolling", metric) for metric in GEOMETRY_VALUE_COLUMNS),
    *(_window_metric("cumulative", metric) for metric in GEOMETRY_VALUE_COLUMNS),
)
PROFILE_COLUMNS = STABLE_PROFILE_COLUMNS + EXTENDED_PROFILE_COLUMNS
METRIC_REGISTRY = {
    **{
        metric: {"status": "stable", "release": "v0.1"}
        for metric in STABLE_PROFILE_COLUMNS
    },
    **{
        metric: {"status": "experimental", "release": "v0.2"}
        for metric in EXTENDED_PROFILE_COLUMNS
    },
}
DEFAULT_REPORT_METRICS = STABLE_PROFILE_COLUMNS + (
    "v2_raw_state_angle",
    "v5_cumulative_straightness",
    "v8_cumulative_layer_update_er",
    "v9_cumulative_layer_state_ed",
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


def v2_raw_state_angle(layers: np.ndarray) -> np.ndarray:
    values = np.asarray(layers, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("layers must have shape (L+1,D) with at least two states")
    return _adjacent_angles(values)


def vertical_effective_rank(
    layers: np.ndarray,
    centered: bool,
    *,
    rtol: float = SVD_RTOL,
) -> float:
    values = np.asarray(layers, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        return np.nan
    if rtol <= 0:
        raise ValueError("rtol must be positive")
    if centered:
        values = values - values.mean(axis=0, keepdims=True)
    if not np.isfinite(values).all():
        return np.nan
    singular_values = np.linalg.svd(values, compute_uv=False)
    singular_values = singular_values[np.isfinite(singular_values)]
    if not singular_values.size:
        return np.nan
    singular_values = singular_values[
        singular_values > singular_values.max() * rtol
    ]
    if not singular_values.size:
        return np.nan
    probabilities = singular_values / (singular_values.sum() + EPS)
    return float(
        np.exp(-np.sum(probabilities * np.log(probabilities + EPS)))
    )


def _invalid_degree(n_valid: int, degree: int) -> dict[str, Any]:
    return {
        "coverage_ok": False,
        "n_valid": int(n_valid),
        "polynomial_degree": int(degree),
        "polynomial_rank": 0,
        "condition_number": np.nan,
        "raw_trajectory_ed": np.nan,
        "normalized_trajectory_ed": np.nan,
        "high_order_fraction": np.nan,
        "linear_fraction": np.nan,
        "relative_fit_rmse": np.nan,
    }


def _effective_degree(
    points: np.ndarray,
    *,
    degree: int,
    centered: bool,
) -> dict[str, Any]:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2:
        raise ValueError("layers must have shape (N,D)")
    if degree < 1:
        raise ValueError("degree must be at least 1")
    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]
    required = degree + 1
    if len(values) < required:
        return _invalid_degree(len(values), degree)
    positions = np.linspace(0.0, 1.0, len(values))
    scaled = 2.0 * positions - 1.0
    target = values - values.mean(axis=0, keepdims=True) if centered else values
    design = np.polynomial.chebyshev.chebvander(scaled, degree)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    if rank < degree + 1 or not np.all(np.isfinite(coefficients)):
        invalid = _invalid_degree(len(values), degree)
        invalid["polynomial_rank"] = int(rank)
        return invalid
    strengths = np.linalg.norm(coefficients, axis=1)
    nonconstant_strength = float(strengths[1:].sum())
    high_order_strength = float(strengths[2:].sum()) if degree >= 2 else 0.0
    strength_floor = max(EPS, float(strengths.max()) * SVD_RTOL)
    if nonconstant_strength <= strength_floor:
        raw_ed = 0.0
        normalized_ed = 0.0
        high_order_fraction = 0.0
        linear_fraction = 0.0
    else:
        orders = np.arange(degree + 1, dtype=float)
        raw_ed = float(orders @ strengths)
        normalized_ed = raw_ed / nonconstant_strength
        high_order_fraction = high_order_strength / nonconstant_strength
        linear_fraction = float(strengths[1]) / nonconstant_strength
    fitted = design @ coefficients
    target_norm = float(np.linalg.norm(target))
    residual_norm = float(np.linalg.norm(target - fitted))
    relative_rmse = (
        0.0
        if target_norm <= EPS and residual_norm <= EPS
        else residual_norm / (target_norm + EPS)
    )
    condition_number = (
        float(singular_values.max() / singular_values.min())
        if singular_values.size and singular_values.min() > EPS
        else np.inf
    )
    return {
        "coverage_ok": True,
        "n_valid": int(len(values)),
        "polynomial_degree": int(degree),
        "polynomial_rank": int(rank),
        "condition_number": condition_number,
        "raw_trajectory_ed": raw_ed,
        "normalized_trajectory_ed": normalized_ed,
        "high_order_fraction": high_order_fraction,
        "linear_fraction": linear_fraction,
        "relative_fit_rmse": relative_rmse,
    }


def vertical_effective_degree(
    layers: np.ndarray,
    degree: int = 3,
) -> Mapping[str, float | int | bool]:
    return _effective_degree(layers, degree=degree, centered=True)


def _empty_geometry_row(
    *,
    window_type: str,
    start_layer: int,
    end_layer: int,
    window_updates: int,
    actual_updates: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "window_type": window_type,
        "start_layer": int(start_layer),
        "end_layer": int(end_layer),
        "relative_depth": np.nan,
        "window_updates": int(window_updates),
        "actual_window_updates": int(actual_updates),
        "coverage_ok": False,
        "v5_coverage_ok": False,
        "v8_coverage_ok": False,
        "v9_state_coverage_ok": False,
        "v9_update_coverage_ok": False,
        "invalid_reason": reason,
        **{metric: np.nan for metric in GEOMETRY_VALUE_COLUMNS},
    }


def _geometry_row(
    states: np.ndarray,
    *,
    window_type: str,
    start_layer: int,
    end_layer: int,
    window_updates: int,
    decoder_layers: int,
) -> dict[str, Any]:
    values = np.asarray(states, dtype=float)
    updates = np.diff(values, axis=0)
    row = _empty_geometry_row(
        window_type=window_type,
        start_layer=start_layer,
        end_layer=end_layer,
        window_updates=window_updates,
        actual_updates=len(updates),
        reason="",
    )
    row["relative_depth"] = float(end_layer / decoder_layers)
    if not len(updates) or not np.isfinite(values).all():
        row["invalid_reason"] = "nonfinite_or_empty_window"
        return row

    update_norms = np.linalg.norm(updates, axis=1)
    path_length = float(update_norms.sum())
    net_displacement = float(np.linalg.norm(updates.sum(axis=0)))
    row.update(
        {
            "v5_coverage_ok": True,
            "v5_path_length": path_length,
            "v5_net_displacement": net_displacement,
            "v5_straightness": net_displacement / (path_length + EPS),
            "v5_log_detour": float(
                np.log(path_length + EPS) - np.log(net_displacement + EPS)
            ),
        }
    )

    state_er = vertical_effective_rank(values, centered=True)
    update_er = vertical_effective_rank(updates, centered=False)
    v8_ok = (
        len(updates) >= 2
        and int((update_norms >= NORM_FLOOR).sum()) >= 2
        and np.isfinite(state_er)
        and np.isfinite(update_er)
    )
    if v8_ok:
        row.update(
            {
                "v8_coverage_ok": True,
                "v8_layer_state_er": state_er,
                "v8_layer_update_er": update_er,
            }
        )

    state_degree = _effective_degree(values, degree=3, centered=True)
    update_degree = _effective_degree(updates, degree=3, centered=False)
    if state_degree["coverage_ok"]:
        row.update(
            {
                "v9_state_coverage_ok": True,
                "v9_layer_state_ed": state_degree["normalized_trajectory_ed"],
                "v9_state_high_order_fraction": state_degree["high_order_fraction"],
                "v9_state_linear_fraction": state_degree["linear_fraction"],
                "v9_state_relative_fit_rmse": state_degree["relative_fit_rmse"],
                "v9_state_condition_number": state_degree["condition_number"],
            }
        )
    if update_degree["coverage_ok"]:
        row.update(
            {
                "v9_update_coverage_ok": True,
                "v9_layer_update_ed": update_degree["normalized_trajectory_ed"],
                "v9_update_high_order_fraction": update_degree["high_order_fraction"],
                "v9_update_linear_fraction": update_degree["linear_fraction"],
                "v9_update_relative_fit_rmse": update_degree["relative_fit_rmse"],
                "v9_update_condition_number": update_degree["condition_number"],
            }
        )
    row["coverage_ok"] = bool(
        row["v5_coverage_ok"]
        and row["v8_coverage_ok"]
        and row["v9_state_coverage_ok"]
        and row["v9_update_coverage_ok"]
    )
    if not row["coverage_ok"]:
        row["invalid_reason"] = "insufficient_component_coverage"
    return row


def rolling_vertical_geometry(
    layers: np.ndarray,
    window_updates: int = 4,
) -> pd.DataFrame:
    values = np.asarray(layers, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("layers must have shape (L+1,D) with at least two states")
    if window_updates < 1:
        raise ValueError("window_updates must be positive")
    decoder_layers = values.shape[0] - 1
    rows: list[dict[str, Any]] = []
    for end_layer in range(1, decoder_layers + 1):
        start_layer = max(0, end_layer - window_updates)
        if end_layer < window_updates:
            row = _empty_geometry_row(
                window_type="rolling",
                start_layer=start_layer,
                end_layer=end_layer,
                window_updates=window_updates,
                actual_updates=end_layer - start_layer,
                reason="insufficient_updates",
            )
            row["relative_depth"] = float(end_layer / decoder_layers)
        else:
            row = _geometry_row(
                values[start_layer : end_layer + 1],
                window_type="rolling",
                start_layer=start_layer,
                end_layer=end_layer,
                window_updates=window_updates,
                decoder_layers=decoder_layers,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def cumulative_vertical_geometry(layers: np.ndarray) -> pd.DataFrame:
    values = np.asarray(layers, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("layers must have shape (L+1,D) with at least two states")
    decoder_layers = values.shape[0] - 1
    return pd.DataFrame(
        [
            _geometry_row(
                values[: end_layer + 1],
                window_type="cumulative",
                start_layer=0,
                end_layer=end_layer,
                window_updates=end_layer,
                decoder_layers=decoder_layers,
            )
            for end_layer in range(1, decoder_layers + 1)
        ]
    )


def _extended_profile_arrays(layers: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(layers, dtype=float)
    layer_count = values.shape[0]
    output = {
        metric: np.full(layer_count, np.nan, dtype=float)
        for metric in EXTENDED_PROFILE_COLUMNS
    }
    output["v2_raw_state_angle"] = v2_raw_state_angle(values)
    for window, frame in (
        ("rolling", rolling_vertical_geometry(values)),
        ("cumulative", cumulative_vertical_geometry(values)),
    ):
        for row in frame.itertuples(index=False):
            end_layer = int(row.end_layer)
            for source in GEOMETRY_VALUE_COLUMNS:
                output[_window_metric(window, source)][end_layer] = float(
                    getattr(row, source)
                )
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
    stable = dict(
        zip(
            STABLE_PROFILE_COLUMNS,
            (v1_raw, v1_relative, v3, v4, v6, v7),
            strict=True,
        )
    )
    return {**stable, **_extended_profile_arrays(values)}


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
