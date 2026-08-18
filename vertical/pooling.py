from __future__ import annotations

import numpy as np


WINDOW = 128
STRIDE = 32
STAGES = ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0))


def trajectory_endpoints(
    response_length: int,
    *,
    window: int = WINDOW,
    stride: int = STRIDE,
) -> np.ndarray:
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be positive")
    if response_length < 1:
        return np.asarray([], dtype=int)
    regular = (
        list(range(window, response_length + 1, stride))
        if response_length >= window
        else []
    )
    if not regular or regular[-1] != response_length:
        regular.append(response_length)
    return np.asarray(regular, dtype=int)


def pool_response_states(
    response_states: np.ndarray,
    endpoints: np.ndarray,
    *,
    window: int = WINDOW,
) -> dict[str, np.ndarray]:
    values = np.asarray(response_states)
    endpoints = np.asarray(endpoints, dtype=int)
    if values.ndim != 3:
        raise ValueError("response_states must have shape (token, layer, dim)")
    if window < 1:
        raise ValueError("window must be positive")
    if endpoints.ndim != 1:
        raise ValueError("endpoints must be one-dimensional")
    if np.any(endpoints < 1) or np.any(endpoints > values.shape[0]):
        raise ValueError("trajectory endpoint falls outside response tokens")
    if len(endpoints) > 1 and np.any(np.diff(endpoints) <= 0):
        raise ValueError("endpoints must be strictly increasing")

    values32 = values.astype(np.float32, copy=False)
    if not len(endpoints):
        empty = np.empty((0, values.shape[1], values.shape[2]), dtype=np.float32)
        return {"mean_w128_s32": empty.copy(), "last_s32": empty.copy()}
    means = []
    for endpoint in endpoints:
        start = max(0, int(endpoint) - window)
        means.append(values32[start : int(endpoint)].mean(axis=0))
    lasts = values32[endpoints - 1]
    return {
        "mean_w128_s32": np.asarray(means, dtype=np.float32),
        "last_s32": np.asarray(lasts, dtype=np.float32),
    }


def progress_from_endpoints(
    endpoints: np.ndarray,
    response_length: int,
) -> np.ndarray:
    endpoints = np.asarray(endpoints, dtype=int)
    if response_length < 1:
        raise ValueError("response_length must be positive")
    if endpoints.ndim != 1:
        raise ValueError("endpoints must be one-dimensional")
    if np.any(endpoints < 1) or np.any(endpoints > response_length):
        raise ValueError("trajectory endpoint falls outside response tokens")
    return endpoints.astype(np.float64) / float(response_length)


def stage_mask(progress: np.ndarray, stage: int) -> np.ndarray:
    coordinates = np.asarray(progress, dtype=float)
    if stage < 0 or stage >= len(STAGES):
        raise ValueError(f"stage must be in [0, {len(STAGES) - 1}]")
    if coordinates.ndim != 1 or not np.isfinite(coordinates).all():
        raise ValueError("progress must be a finite one-dimensional array")
    if np.any(coordinates < 0) or np.any(coordinates > 1):
        raise ValueError("progress must fall in [0, 1]")
    lower, upper = STAGES[stage]
    if stage == len(STAGES) - 1:
        return (coordinates >= lower) & (coordinates <= upper)
    return (coordinates >= lower) & (coordinates < upper)

