from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from . import metrics


HORIZONTAL_FAMILIES: dict[str, tuple[str, Callable[..., dict[str, Any]], str]] = {
    "H1": ("movement", metrics.h1_movement, metrics.PRIMARY_KEYS["H1"]),
    "H2": ("path", metrics.h2_path_local, metrics.PRIMARY_KEYS["H2"]),
    "H3": ("turning", metrics.h3_turning, metrics.PRIMARY_KEYS["H3"]),
    "H4": ("angular_velocity", metrics.h4_angular_velocity, metrics.PRIMARY_KEYS["H4"]),
    "H5": ("state_er_dynamics", metrics.h5_er_dynamics, metrics.PRIMARY_KEYS["H5"]),
    "H6": ("directional_spectrum", metrics.h6_directional_er, metrics.PRIMARY_KEYS["H6"]),
    "H7": ("length_angle_coupling", metrics.h7_weighted_turning, metrics.PRIMARY_KEYS["H7"]),
}


def trajectory_endpoints(token_count: int, *, window: int = 128, stride: int = 32) -> np.ndarray:
    if token_count < 1:
        return np.asarray([], dtype=int)
    regular = list(range(window, token_count + 1, stride)) if token_count >= window else []
    if not regular or regular[-1] != token_count:
        regular.append(token_count)
    return np.asarray(regular, dtype=int)


def pool_token_hidden(
    token_hidden: np.ndarray,
    endpoints: np.ndarray,
    *,
    window: int = 128,
) -> dict[str, np.ndarray]:
    """Pool response-only hidden states into mean and last trajectories.

    token_hidden has shape (T, L+1, D); endpoints are 1-based exclusive.
    """
    values = np.asarray(token_hidden)
    endpoints = np.asarray(endpoints, dtype=int)
    if values.ndim != 3:
        raise ValueError("token_hidden must have shape (T,L+1,D)")
    if np.any(endpoints < 1) or np.any(endpoints > values.shape[0]):
        raise ValueError("trajectory endpoint falls outside response tokens")
    means = []
    lasts = []
    for endpoint in endpoints:
        start = max(0, int(endpoint) - window)
        means.append(values[start:int(endpoint)].mean(axis=0))
        lasts.append(values[int(endpoint) - 1])
    return {"mean_w128_s32": np.asarray(means), "last_s32": np.asarray(lasts)}


def _metric_rows(
    *,
    base: dict[str, Any],
    family_id: str,
    family: str,
    axis: str,
    result: dict[str, Any],
    primary_key: str,
    aggregation_mode: str,
) -> list[dict[str, Any]]:
    ignored = {
        "coverage_ok",
        "n_valid",
        "anchor_layer",
        "representation",
        "stage",
        "excluded_zero_norm",
        "excluded_omega_gap",
    }
    scalar_keys = [
        key
        for key, value in result.items()
        if key not in ignored and isinstance(value, (int, float, np.integer, np.floating))
    ]
    if primary_key not in scalar_keys:
        scalar_keys.append(primary_key)
    rows = []
    for key in sorted(set(scalar_keys)):
        value = result.get(key, np.nan)
        rows.append(
            {
                **base,
                "axis": axis,
                "family_id": family_id,
                "family": family,
                "aggregation_mode": aggregation_mode,
                "metric": key,
                "value": float(value) if np.isfinite(value) else np.nan,
                "is_primary_metric": key == primary_key,
                "coverage": bool(result.get("coverage_ok", False) and np.isfinite(value)),
                "n_valid": int(result.get("n_valid", 0)),
                "excluded_zero_norm": int(result.get("excluded_zero_norm", 0)),
                "excluded_omega_gap": int(result.get("excluded_omega_gap", 0)),
            }
        )
    return rows

def reduce_horizontal(
    trajectory: np.ndarray,
    progress: np.ndarray,
    *,
    representation: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    trajectory = np.asarray(trajectory, dtype=float)
    progress = np.asarray(progress, dtype=float)
    if trajectory.ndim != 3 or trajectory.shape[0] != len(progress):
        raise ValueError("trajectory must be (K,L+1,D) and align with progress")
    rows = []
    for anchor in metrics.resolve_anchor_layers(trajectory.shape[1]):
        points = trajectory[:, anchor, :]
        for stage in range(len(metrics.STAGES)):
            base = {
                **metadata,
                "representation": representation,
                "anchor_layer": anchor,
                "stage": stage,
            }
            for family_id, (family, function, primary_key) in HORIZONTAL_FAMILIES.items():
                result = function(points, progress, stage)
                mode = "cumulative" if family_id in {"H5", "H6"} else "local"
                rows.extend(
                    _metric_rows(
                        base=base,
                        family_id=family_id,
                        family=family,
                        axis="horizontal",
                        result=result,
                        primary_key=primary_key,
                        aggregation_mode=mode,
                    )
                )
                if family_id == "H2":
                    cumulative = metrics.h2_path_cumulative(points, progress, stage)
                    rows.extend(
                        _metric_rows(
                            base=base,
                            family_id=family_id,
                            family=family,
                            axis="horizontal",
                            result=cumulative,
                            primary_key=primary_key,
                            aggregation_mode="cumulative_sensitivity",
                        )
                    )
    return rows


def _aggregate_chunk_results(
    trajectory: np.ndarray,
    progress: np.ndarray,
    stage: int,
    function: Callable[[np.ndarray], dict[str, Any]],
    primary_key: str,
) -> dict[str, Any]:
    results = [function(trajectory[index]) for index in np.flatnonzero(metrics.stage_mask(progress, stage))]
    valid_results = [result for result in results if result.get("coverage_ok")]
    if not valid_results:
        return {"coverage_ok": False, "n_valid": 0, primary_key: np.nan}
    keys = set().union(*(result.keys() for result in valid_results))
    output: dict[str, Any] = {"coverage_ok": True, "n_valid": len(valid_results)}
    for key in keys - {"coverage_ok", "n_valid"}:
        values = [result.get(key, np.nan) for result in valid_results]
        values = np.asarray([value for value in values if isinstance(value, (int, float)) and np.isfinite(value)])
        if values.size:
            output[key] = float(np.median(values))
    output.setdefault(primary_key, np.nan)
    return output


def reduce_vertical(
    trajectory: np.ndarray,
    progress: np.ndarray,
    *,
    representation: str,
    metadata: dict[str, Any],
    base_common: np.ndarray | None,
    coordinate_common: np.ndarray | None,
    coordinate_sigma: np.ndarray | None,
) -> list[dict[str, Any]]:
    trajectory = np.asarray(trajectory, dtype=float)
    progress = np.asarray(progress, dtype=float)
    families: list[tuple[str, str, Callable[[np.ndarray], dict[str, Any]], str]] = [
        ("V1", "layer_update_norm", metrics.v1_layer_update_norm, metrics.PRIMARY_KEYS["V1"]),
        ("V2", "raw_state_angle", metrics.v2_raw_state_angle, metrics.PRIMARY_KEYS["V2"]),
        (
            "V3",
            "demean_state_angle",
            (lambda layers: metrics.v3_state_angle_demean(layers, base_common))
            if base_common is not None
            else (lambda layers: {"coverage_ok": False, "n_valid": 0}),
            metrics.PRIMARY_KEYS["V3"],
        ),
        ("V4", "layer_update_turning", metrics.v4_layer_update_turning, metrics.PRIMARY_KEYS["V4"]),
        ("V5", "vertical_path", metrics.v5_vertical_path, metrics.PRIMARY_KEYS["V5"]),
        ("V6", "raw_activation_entropy", metrics.v6_raw_activation_entropy, metrics.PRIMARY_KEYS["V6"]),
        (
            "V7",
            "centered_difference_robust_entropy",
            lambda layers: metrics.v7_entropies(layers, coordinate_common, coordinate_sigma),
            metrics.PRIMARY_KEYS["V7"],
        ),
        ("V8", "vertical_spectrum", metrics.v8_vertical_er, metrics.PRIMARY_KEYS["V8"]),
    ]
    rows = []
    for stage in range(len(metrics.STAGES)):
        base = {
            **metadata,
            "representation": representation,
            "anchor_layer": "all",
            "stage": stage,
        }
        for family_id, family, function, primary_key in families:
            result = _aggregate_chunk_results(trajectory, progress, stage, function, primary_key)
            rows.extend(
                _metric_rows(
                    base=base,
                    family_id=family_id,
                    family=family,
                    axis="vertical",
                    result=result,
                    primary_key=primary_key,
                    aggregation_mode="local",
                )
            )
    return rows


def reduce_h8(
    token_hidden: np.ndarray,
    endpoints: np.ndarray,
    progress: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for result in metrics.h8_token_entropy_all_stages(token_hidden, endpoints, progress):
        base = {
            **metadata,
            "representation": "token",
            "anchor_layer": int(result["anchor_layer"]),
            "stage": int(result["stage"]),
        }
        rows.extend(
            _metric_rows(
                base=base,
                family_id="H8",
                family="token_entropy_dynamics",
                axis="horizontal",
                result=result,
                primary_key=metrics.PRIMARY_KEYS["H8"],
                aggregation_mode="local",
            )
        )
    return rows
