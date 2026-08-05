"""Pure NumPy implementation of the frozen Experiment 2E metrics.

The module deliberately has no model, vLLM, or filesystem dependency.  Model
forward code supplies pooled trajectories and response-only token states; this
module reduces them to scalar dictionaries and coverage diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


EPS = 1e-12
NORM_FLOOR = 1e-6
Z_CLIP = 8.0
SIGMA_FLOOR = 1e-4
WINDOW = 128
STAGES = ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0))
HORIZONTAL_ANCHORS: tuple[int | str, ...] = (3, 12, 24, "final")


def stage_mask(progress: np.ndarray, stage: int) -> np.ndarray:
    progress = np.asarray(progress, dtype=float)
    lo, hi = STAGES[stage]
    if stage == len(STAGES) - 1:
        return (progress >= lo) & (progress <= hi)
    return (progress >= lo) & (progress < hi)


def cumulative_index(progress: np.ndarray, stage: int) -> int | None:
    progress = np.asarray(progress, dtype=float)
    indices = np.flatnonzero(progress <= STAGES[stage][1])
    return int(indices[-1]) if indices.size else None


def resolve_anchor_layers(n_hidden_states: int) -> tuple[int, ...]:
    final = n_hidden_states - 1
    anchors = (3, 12, 24, final)
    if final < 24 or len(set(anchors)) != len(anchors):
        raise ValueError("model must expose distinct L3/L12/L24/Lfinal states")
    return anchors


def displacements(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must have shape (K, D)")
    d = np.diff(points, axis=0)
    r = np.linalg.norm(d, axis=1)
    u = d / (r[:, None] + EPS)
    return d, r, u


def turn_angles(u: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, int]:
    """Return position-aligned turns; invalid positions remain NaN.

    If ``u`` has K-1 displacements, ``theta[i-1]`` is the turn from
    displacement i-1 to i and aligns with the original endpoint at index i+1.
    Keeping NaN placeholders prevents stage reassignment and prevents angular
    velocity from connecting non-adjacent turns.
    """
    u = np.asarray(u, dtype=float)
    r = np.asarray(r, dtype=float)
    if u.ndim != 2 or r.ndim != 1 or len(u) != len(r):
        raise ValueError("u must be (N,D) and r must be (N,)")
    theta = np.full(max(len(u) - 1, 0), np.nan, dtype=float)
    valid = r >= NORM_FLOOR
    for index in range(1, len(u)):
        if valid[index] and valid[index - 1]:
            theta[index - 1] = np.arccos(
                np.clip(float(u[index] @ u[index - 1]), -1.0, 1.0)
            )
    return theta, int(np.isnan(theta).sum())


def spectral_effective_rank(matrix: np.ndarray, *, center_rows: bool = False) -> float:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        return np.nan
    if center_rows:
        values = values - values.mean(axis=0, keepdims=True)
    sigma = np.linalg.svd(values, compute_uv=False)
    sigma = sigma[np.isfinite(sigma) & (sigma > EPS)]
    if sigma.size == 0:
        return np.nan
    probabilities = sigma / (sigma.sum() + EPS)
    return float(np.exp(-np.sum(probabilities * np.log(probabilities + EPS))))


def historical_deviation(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(1, len(values)):
        previous = values[:index]
        if np.all(np.isfinite(previous)) and np.isfinite(values[index]):
            result[index] = values[index] - previous.mean()
    return result


def energy_entropy(vector: np.ndarray, *, center: bool, zero_value: float = np.nan) -> float:
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or vector.size < 1:
        return np.nan
    values = vector - vector.mean() if center else vector
    energy = values * values
    total = float(energy.sum())
    if not np.isfinite(total) or total <= EPS:
        return zero_value
    probabilities = energy / (total + EPS)
    return float(-np.sum(probabilities * np.log(probabilities + EPS)) / np.log(vector.size))


def _invalid(n_valid: int = 0, **extra: Any) -> dict[str, Any]:
    return {"coverage_ok": False, "n_valid": int(n_valid), **extra}


def h1_movement(points: np.ndarray, progress: np.ndarray, stage: int) -> dict[str, Any]:
    _, norms, _ = displacements(points)
    base_norm = np.linalg.norm(np.asarray(points[:-1], dtype=float), axis=1)
    selection = stage_mask(progress[1:], stage)
    if not selection.any():
        return _invalid()
    relative = norms / (base_norm + EPS)
    rms = norms / np.sqrt(points.shape[1])
    selected = lambda values: np.asarray(values)[selection]
    return {
        "coverage_ok": True,
        "n_valid": int(selection.sum()),
        "median_relative_movement": float(np.median(selected(relative))),
        "mean_relative_movement": float(np.mean(selected(relative))),
        "p90_relative_movement": float(np.percentile(selected(relative), 90)),
        "median_rms_movement": float(np.median(selected(rms))),
        "mean_rms_movement": float(np.mean(selected(rms))),
        "p90_rms_movement": float(np.percentile(selected(rms), 90)),
        "median_raw_movement": float(np.median(selected(norms))),
        "mean_raw_movement": float(np.mean(selected(norms))),
        "p90_raw_movement": float(np.percentile(selected(norms), 90)),
    }


def h2_path(displacement_values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(displacement_values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        return _invalid()
    lengths = np.linalg.norm(values, axis=1)
    path = float(lengths.sum())
    net = float(np.linalg.norm(values.sum(axis=0)))
    return {
        "coverage_ok": True,
        "n_valid": int(values.shape[0]),
        "straightness": net / (path + EPS),
        "path_length": path,
        "net_displacement": net,
        "log_detour": float(np.log(path + EPS) - np.log(net + EPS)),
    }


def h2_path_local(points: np.ndarray, progress: np.ndarray, stage: int) -> dict[str, Any]:
    displacements_, _, _ = displacements(points)
    return h2_path(displacements_[stage_mask(progress[1:], stage)])


def h2_path_cumulative(points: np.ndarray, progress: np.ndarray, stage: int) -> dict[str, Any]:
    endpoint = cumulative_index(progress, stage)
    if endpoint is None or endpoint < 1:
        return _invalid()
    displacements_, _, _ = displacements(points[: endpoint + 1])
    return h2_path(displacements_)


def h3_turning(points: np.ndarray, progress: np.ndarray, stage: int) -> dict[str, Any]:
    _, norms, directions = displacements(points)
    theta, excluded = turn_angles(directions, norms)
    selection = stage_mask(progress[2:], stage) & np.isfinite(theta)
    if not selection.any():
        return _invalid(excluded_zero_norm=excluded)
    values = theta[selection]
    return {
        "coverage_ok": True,
        "n_valid": int(values.size),
        "excluded_zero_norm": excluded,
        "median_turn_angle": float(np.median(values)),
        "mean_turn_angle": float(np.mean(values)),
        "std_turn_angle": float(np.std(values)),
        "p90_turn_angle": float(np.percentile(values, 90)),
    }


def h4_angular_velocity(points: np.ndarray, progress: np.ndarray, stage: int) -> dict[str, Any]:
    _, norms, directions = displacements(points)
    theta, excluded = turn_angles(directions, norms)
    if theta.size < 2:
        return _invalid(excluded_zero_norm=excluded, excluded_omega_gap=0)
    omega = np.diff(theta)
    valid_gap = np.isfinite(theta[:-1]) & np.isfinite(theta[1:])
    selection = stage_mask(progress[3:], stage) & valid_gap & np.isfinite(omega)
    if not selection.any():
        return _invalid(excluded_zero_norm=excluded, excluded_omega_gap=int((~valid_gap).sum()))
    signed = omega[selection]
    absolute = np.abs(signed)
    return {
        "coverage_ok": True,
        "n_valid": int(selection.sum()),
        "excluded_zero_norm": excluded,
        "excluded_omega_gap": int((~valid_gap).sum()),
        "p90_abs_angular_velocity": float(np.percentile(absolute, 90)),
        "median_abs_angular_velocity": float(np.median(absolute)),
        "mean_signed_angular_velocity": float(np.mean(signed)),
        "median_signed_angular_velocity": float(np.median(signed)),
    }


def _er_series(points: np.ndarray, *, centered: bool) -> np.ndarray:
    return np.asarray(
        [spectral_effective_rank(points[: index + 1], center_rows=centered)
         for index in range(1, len(points))],
        dtype=float,
    )


def h5_er_dynamics(points: np.ndarray, progress: np.ndarray, stage: int) -> dict[str, Any]:
    endpoint = cumulative_index(progress, stage)
    if endpoint is None or endpoint < 1:
        return _invalid()
    centered = _er_series(points[: endpoint + 1], centered=True)
    uncentered = _er_series(points[: endpoint + 1], centered=False)
    centered = centered[np.isfinite(centered)]
    uncentered = uncentered[np.isfinite(uncentered)]
    if centered.size < 2:
        return _invalid(n_valid=int(centered.size))
    centered_delta = historical_deviation(centered)
    uncentered_delta = historical_deviation(uncentered)
    centered_delta = centered_delta[np.isfinite(centered_delta)]
    uncentered_delta = uncentered_delta[np.isfinite(uncentered_delta)]
    out = {
        "coverage_ok": True,
        "n_valid": int(centered.size),
        "centered_ER_final": float(centered[-1]),
        "centered_ERV": float(centered_delta.mean()) if centered_delta.size else np.nan,
        "centered_ERA": float(np.diff(centered_delta).mean()) if centered_delta.size >= 2 else np.nan,
        "uncentered_ER_final": float(uncentered[-1]) if uncentered.size else np.nan,
        "uncentered_ERV": float(uncentered_delta.mean()) if uncentered_delta.size else np.nan,
        "uncentered_ERA": float(np.diff(uncentered_delta).mean()) if uncentered_delta.size >= 2 else np.nan,
    }
    return out


def h6_directional_er(points: np.ndarray, progress: np.ndarray, stage: int) -> dict[str, Any]:
    endpoint = cumulative_index(progress, stage)
    if endpoint is None or endpoint < 1:
        return _invalid()
    displacements_, norms, directions = displacements(points[: endpoint + 1])
    valid = norms >= NORM_FLOOR
    if not valid.any():
        return _invalid()
    weights = norms[valid] / (norms[valid].sum() + EPS)
    rows = np.sqrt(weights)[:, None] * directions[valid]
    spectrum = np.linalg.svd(rows, compute_uv=False) ** 2
    spectrum = spectrum[spectrum > EPS]
    if not spectrum.size:
        return _invalid(n_valid=int(valid.sum()))
    probabilities = spectrum / (spectrum.sum() + EPS)
    return {
        "coverage_ok": True,
        "n_valid": int(valid.sum()),
        "directional_ER": float(np.exp(-np.sum(probabilities * np.log(probabilities + EPS)))),
        "companion_path_length": float(np.linalg.norm(displacements_, axis=1).sum()),
    }


def h7_weighted_turning(points: np.ndarray, progress: np.ndarray, stage: int) -> dict[str, Any]:
    _, norms, directions = displacements(points)
    theta, excluded = turn_angles(directions, norms)
    if theta.size == 0:
        return _invalid(excluded_zero_norm=excluded)
    r_bar = 0.5 * (norms[1:] + norms[:-1])
    selection = stage_mask(progress[2:], stage) & np.isfinite(theta) & np.isfinite(r_bar)
    if not selection.any():
        return _invalid(excluded_zero_norm=excluded)
    denominator = r_bar[selection].sum() + EPS
    return {
        "coverage_ok": True,
        "n_valid": int(selection.sum()),
        "excluded_zero_norm": excluded,
        "weighted_turning": float((r_bar[selection] * theta[selection]).sum() / denominator),
    }


def token_entropy_series(token_states: np.ndarray) -> dict[str, np.ndarray]:
    token_states = np.asarray(token_states)
    if token_states.ndim != 2 or token_states.shape[0] < 1:
        raise ValueError("token_states must have shape (T, D), T >= 1")
    state_centered = energy_entropy_rows(token_states, center=True, zero_value=0.0)
    state_raw = energy_entropy_rows(token_states, center=False, zero_value=0.0)
    time_difference = np.full(token_states.shape[0], np.nan, dtype=float)
    if token_states.shape[0] >= 2:
        time_difference[1:] = energy_entropy_rows(
            np.diff(token_states, axis=0), center=True, zero_value=0.0
        )
    return {
        "token_time_diff_coordinate_entropy": time_difference,
        "token_state_coordinate_entropy": state_centered,
        "token_state_raw_energy_entropy": state_raw,
    }


def energy_entropy_rows(states: np.ndarray, *, center: bool, zero_value: float) -> np.ndarray:
    """Vectorized coordinate-energy entropy without materializing normalized probabilities."""
    values = np.array(states, dtype=np.float32, copy=True)
    if values.ndim != 2:
        raise ValueError("states must have shape (N,D)")
    if center:
        values -= values.mean(axis=1, keepdims=True)
    np.square(values, out=values)
    totals = values.sum(axis=1, dtype=np.float64)
    energy_log_energy = np.zeros_like(values)
    np.log(values, out=energy_log_energy, where=values > 0)
    energy_log_energy *= values
    weighted_logs = energy_log_energy.sum(axis=1, dtype=np.float64)
    output = np.full(values.shape[0], float(zero_value), dtype=float)
    valid = totals > EPS
    output[valid] = (
        np.log(totals[valid]) - weighted_logs[valid] / totals[valid]
    ) / np.log(values.shape[1])
    return output


def h8_token_entropy_stage(
    token_hidden: np.ndarray,
    endpoints: np.ndarray,
    progress: np.ndarray,
    stage: int,
    layer: int,
) -> dict[str, Any]:
    token_hidden = np.asarray(token_hidden)
    endpoints = np.asarray(endpoints, dtype=int)
    progress = np.asarray(progress, dtype=float)
    if token_hidden.ndim != 3 or not 0 <= layer < token_hidden.shape[1]:
        raise ValueError("token_hidden must have shape (T,L+1,D) and a valid layer")
    if endpoints.shape != progress.shape:
        raise ValueError("endpoints and progress must have equal shape")
    series = token_entropy_series(token_hidden[:, layer, :])
    values: dict[str, list[float]] = {key: [] for key in series}
    for index in np.flatnonzero(stage_mask(progress, stage)):
        end = min(int(endpoints[index]), token_hidden.shape[0])
        start = max(0, end - WINDOW)
        for key, sequence in series.items():
            finite = sequence[start:end]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                values[key].append(float(np.median(finite)))
    primary = values["token_time_diff_coordinate_entropy"]
    if not primary:
        return _invalid(anchor_layer=layer, representation="token")
    out: dict[str, Any] = {
        "coverage_ok": True,
        "n_valid": len(primary),
        "anchor_layer": layer,
        "representation": "token",
    }
    for key, sequence in values.items():
        if sequence:
            arr = np.asarray(sequence)
            out[f"median_{key}"] = float(np.median(arr))
            out[f"mean_{key}"] = float(np.mean(arr))
            out[f"p90_{key}"] = float(np.percentile(arr, 90))
    return out


def h8_token_entropy_all_stages(
    token_hidden: np.ndarray,
    endpoints: np.ndarray,
    progress: np.ndarray,
    layers: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Compute H8 window scalars once per layer, then aggregate all stages."""
    token_hidden = np.asarray(token_hidden)
    endpoints = np.asarray(endpoints, dtype=int)
    progress = np.asarray(progress, dtype=float)
    if token_hidden.ndim != 3:
        raise ValueError("token_hidden must have shape (T,L+1,D)")
    if endpoints.shape != progress.shape:
        raise ValueError("endpoints and progress must have equal shape")
    selected_layers = layers if layers is not None else tuple(range(token_hidden.shape[1]))
    rows: list[dict[str, Any]] = []
    for layer in selected_layers:
        if not 0 <= layer < token_hidden.shape[1]:
            raise ValueError(f"invalid H8 layer: {layer}")
        series = token_entropy_series(token_hidden[:, layer, :])
        windows = {key: np.full(len(endpoints), np.nan, dtype=float) for key in series}
        for endpoint_index, endpoint in enumerate(endpoints):
            end = min(int(endpoint), token_hidden.shape[0])
            start = max(0, end - WINDOW)
            for key, sequence in series.items():
                finite = sequence[start:end]
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    windows[key][endpoint_index] = float(np.median(finite))
        for stage in range(len(STAGES)):
            stage_selection = stage_mask(progress, stage)
            row: dict[str, Any] = {
                "coverage_ok": False,
                "n_valid": 0,
                "anchor_layer": layer,
                "representation": "token",
                "stage": stage,
            }
            for key, values in windows.items():
                finite = values[stage_selection & np.isfinite(values)]
                if finite.size:
                    row[f"median_{key}"] = float(np.median(finite))
                    row[f"mean_{key}"] = float(np.mean(finite))
                    row[f"p90_{key}"] = float(np.percentile(finite, 90))
                if key == "token_time_diff_coordinate_entropy" and finite.size:
                    row["coverage_ok"] = True
                    row["n_valid"] = int(finite.size)
            rows.append(row)
    return rows


def vertical_stage_reduce(
    trajectory: np.ndarray,
    progress: np.ndarray,
    stage: int,
    per_chunk_fn: Callable[[np.ndarray], dict[str, Any]],
    key: str,
    reducer: Callable[[np.ndarray], float] = np.median,
) -> dict[str, Any]:
    values = []
    for index in np.flatnonzero(stage_mask(progress, stage)):
        result = per_chunk_fn(trajectory[index])
        value = result.get(key, np.nan)
        if result.get("coverage_ok") and np.isfinite(value):
            values.append(value)
    if not values:
        return _invalid()
    return {"coverage_ok": True, "n_valid": len(values), key: float(reducer(values))}


def _layer_angles(layers: np.ndarray, common: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    values = np.asarray(layers, dtype=float)
    if common is not None:
        values = values - common
    norms = np.linalg.norm(values, axis=1)
    angles = np.full(max(len(values) - 1, 0), np.nan, dtype=float)
    valid = norms >= NORM_FLOOR
    for index in range(1, len(values)):
        if valid[index] and valid[index - 1]:
            cosine = float(values[index] @ values[index - 1]) / (norms[index] * norms[index - 1] + EPS)
            angles[index - 1] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return angles, int(np.isnan(angles).sum())


def v1_layer_update_norm(layers: np.ndarray) -> dict[str, Any]:
    updates = np.diff(np.asarray(layers, dtype=float), axis=0)
    if updates.shape[0] < 1:
        return _invalid()
    raw = np.linalg.norm(updates, axis=1)
    relative = raw / (np.linalg.norm(layers[:-1], axis=1) + EPS)
    return {
        "coverage_ok": True,
        "n_valid": int(len(raw)),
        "p90_relative_layer_update_norm": float(np.percentile(relative, 90)),
        "mean_relative_layer_update_norm": float(relative.mean()),
        "median_relative_layer_update_norm": float(np.median(relative)),
        "mean_raw_layer_update_norm": float(raw.mean()),
        "median_raw_layer_update_norm": float(np.median(raw)),
        "p90_raw_layer_update_norm": float(np.percentile(raw, 90)),
        "argmax_layer": int(np.argmax(raw) + 1),
        "concentration": float(raw.max() / (raw.sum() + EPS)),
    }


def v2_raw_state_angle(layers: np.ndarray) -> dict[str, Any]:
    angles, excluded = _layer_angles(layers)
    finite = angles[np.isfinite(angles)]
    if not finite.size:
        return _invalid(excluded_zero_norm=excluded)
    return {
        "coverage_ok": True,
        "n_valid": int(finite.size),
        "excluded_zero_norm": excluded,
        "median_raw_state_angle": float(np.median(finite)),
        "mean_raw_state_angle": float(finite.mean()),
        "p90_raw_state_angle": float(np.percentile(finite, 90)),
    }


def base_layer_common(all_base_trajectories: list[np.ndarray]) -> np.ndarray:
    if not all_base_trajectories:
        raise ValueError("at least one base trajectory is required")
    return np.mean(np.stack([np.asarray(traj).mean(axis=0) for traj in all_base_trajectories]), axis=0)


def v3_state_angle_demean(layers: np.ndarray, common: np.ndarray) -> dict[str, Any]:
    angles, excluded = _layer_angles(layers, np.asarray(common, dtype=float))
    finite = angles[np.isfinite(angles)]
    if not finite.size:
        return _invalid(excluded_zero_norm=excluded)
    return {
        "coverage_ok": True,
        "n_valid": int(finite.size),
        "excluded_zero_norm": excluded,
        "median_state_angle_demean": float(np.median(finite)),
        "mean_state_angle_demean": float(finite.mean()),
        "p90_state_angle_demean": float(np.percentile(finite, 90)),
    }


def v4_layer_update_turning(layers: np.ndarray) -> dict[str, Any]:
    updates = np.diff(np.asarray(layers, dtype=float), axis=0)
    if updates.shape[0] < 2:
        return _invalid(excluded_zero_norm=0)
    norms = np.linalg.norm(updates, axis=1)
    directions = updates / (norms[:, None] + EPS)
    angles = np.full(len(updates) - 1, np.nan, dtype=float)
    valid = norms >= NORM_FLOOR
    for index in range(1, len(updates)):
        if valid[index] and valid[index - 1]:
            angles[index - 1] = np.arccos(
                np.clip(float(directions[index] @ directions[index - 1]), -1.0, 1.0)
            )
    finite = angles[np.isfinite(angles)]
    excluded = int(np.isnan(angles).sum())
    if not finite.size:
        return _invalid(excluded_zero_norm=excluded)
    return {
        "coverage_ok": True,
        "n_valid": int(finite.size),
        "excluded_zero_norm": excluded,
        "median_layer_update_turning_angle": float(np.median(finite)),
        "mean_layer_update_turning_angle": float(finite.mean()),
        "std_layer_update_turning_angle": float(finite.std()),
        "p90_layer_update_turning_angle": float(np.percentile(finite, 90)),
    }


def v5_vertical_path(layers: np.ndarray) -> dict[str, Any]:
    updates = np.diff(np.asarray(layers, dtype=float), axis=0)
    if updates.shape[0] < 2:
        return _invalid()
    norms = np.linalg.norm(updates, axis=1)
    valid_count = int((norms >= NORM_FLOOR).sum())
    if valid_count < 2:
        return _invalid(valid_count)
    path = float(norms.sum())
    net = float(np.linalg.norm(updates.sum(axis=0)))
    return {
        "coverage_ok": True,
        "n_valid": valid_count,
        "vertical_straightness": net / (path + EPS),
        "vertical_path_length": path,
        "vertical_net_displacement": net,
        "vertical_log_detour": float(np.log(path + EPS) - np.log(net + EPS)),
    }


def v6_raw_activation_entropy(layers: np.ndarray) -> dict[str, Any]:
    values = np.asarray([energy_entropy(row, center=False) for row in np.asarray(layers)])
    values = values[np.isfinite(values)]
    if not values.size:
        return _invalid()
    return {
        "coverage_ok": True,
        "n_valid": int(values.size),
        "median_raw_activation_entropy": float(np.median(values)),
        "mean_raw_activation_entropy": float(values.mean()),
        "p90_raw_activation_entropy": float(np.percentile(values, 90)),
    }


def base_coordinate_stats(all_base_layers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(all_base_layers, dtype=float)
    if values.ndim != 3:
        raise ValueError("all_base_layers must have shape (N,L+1,D)")
    return values.mean(axis=0), values.std(axis=0)


def v7_entropies(
    layers: np.ndarray,
    common: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(layers, dtype=float)
    centered_state = np.asarray([energy_entropy(row, center=True) for row in values])
    updates = np.diff(values, axis=0)
    difference = np.asarray([energy_entropy(row, center=True) for row in updates])
    difference = difference[np.isfinite(difference)]
    if not difference.size:
        return _invalid()
    out: dict[str, Any] = {
        "coverage_ok": True,
        "n_valid": int(difference.size),
        "median_layer_difference_entropy": float(np.median(difference)),
        "mean_layer_difference_entropy": float(difference.mean()),
    }
    centered_state = centered_state[np.isfinite(centered_state)]
    if centered_state.size:
        out["median_centered_state_entropy"] = float(np.median(centered_state))
    if common is not None and sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        denominator = np.maximum(sigma, SIGMA_FLOOR)
        standardized = np.clip((values - np.asarray(common)) / denominator, -Z_CLIP, Z_CLIP)
        robust = np.asarray([energy_entropy(row, center=False) for row in standardized])
        robust = robust[np.isfinite(robust)]
        if robust.size:
            out["median_robust_z_entropy"] = float(np.median(robust))
        out["low_variance_coordinate_fraction"] = float(np.mean(sigma < SIGMA_FLOOR))
    return out


def v8_vertical_er(layers: np.ndarray) -> dict[str, Any]:
    values = np.asarray(layers, dtype=float)
    if values.shape[0] < 3:
        return _invalid()
    updates = np.diff(values, axis=0)
    if int((np.linalg.norm(updates, axis=1) >= NORM_FLOOR).sum()) < 2:
        return _invalid()
    return {
        "coverage_ok": True,
        "n_valid": int(values.shape[0]),
        "layer_update_ER": spectral_effective_rank(updates, center_rows=False),
        "layer_update_ER_centered": spectral_effective_rank(updates, center_rows=True),
        "layer_state_ER": spectral_effective_rank(values, center_rows=True),
        "layer_state_ER_uncentered_sensitivity": spectral_effective_rank(values, center_rows=False),
    }


PRIMARY_KEYS = {
    "H1": "median_relative_movement",
    "H2": "straightness",
    "H3": "median_turn_angle",
    "H4": "p90_abs_angular_velocity",
    "H5": "centered_ERV",
    "H6": "directional_ER",
    "H7": "weighted_turning",
    "H8": "median_token_time_diff_coordinate_entropy",
    "V1": "p90_relative_layer_update_norm",
    "V2": "median_raw_state_angle",
    "V3": "median_state_angle_demean",
    "V4": "median_layer_update_turning_angle",
    "V5": "vertical_straightness",
    "V6": "median_raw_activation_entropy",
    "V7": "median_layer_difference_entropy",
    "V8": "layer_update_ER",
}
