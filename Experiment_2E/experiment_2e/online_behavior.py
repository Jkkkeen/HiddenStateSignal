"""Compact online H2/H10 behavior reduction and query-relative bonus."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from .metrics import EPS
from .reduction import pool_token_hidden


BEHAVIOR_KEYS = (
    "h2",
    "h2_slope",
    "focus_forward",
    "focus_net",
    "orthogonal",
)


def _invalid() -> tuple[np.ndarray, np.ndarray]:
    return np.full(len(BEHAVIOR_KEYS), np.nan, dtype=np.float32), np.zeros(len(BEHAVIOR_KEYS), dtype=bool)


def _trajectory_features(points: np.ndarray, progress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    progress = np.asarray(progress, dtype=np.float64)
    if points.ndim != 2 or len(points) != len(progress) or len(points) < 2:
        return _invalid()
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(progress)):
        return _invalid()

    displacements = np.diff(points, axis=0)
    lengths = np.linalg.norm(displacements, axis=1)
    valid = np.isfinite(lengths) & (lengths > EPS)
    if not np.any(valid):
        return _invalid()
    displacements = displacements[valid]
    lengths = lengths[valid]
    terminal = points[-1] - points[0]
    terminal_norm = float(np.linalg.norm(terminal))
    path_length = float(lengths.sum())
    if terminal_norm <= EPS or path_length <= EPS:
        return _invalid()
    direction = terminal / terminal_norm
    axial = displacements @ direction
    forward = np.maximum(axial, 0.0)
    backward = np.maximum(-axial, 0.0)
    orthogonal = np.sqrt(np.maximum(lengths * lengths - axial * axial, 0.0))

    # H2 is the complete-response straightness. The slope uses all valid
    # cumulative prefixes, so it is a response-level exploration-speed signal.
    h2 = terminal_norm / (path_length + EPS)
    prefix_points = points[1:]
    prefix_displacements = np.diff(points, axis=0)
    prefix_lengths = np.linalg.norm(prefix_displacements, axis=1)
    prefix_path = np.cumsum(prefix_lengths)
    prefix_net = np.linalg.norm(prefix_points - points[0], axis=1)
    prefix_h2 = prefix_net / (prefix_path + EPS)
    prefix_u = progress[1:]
    finite_prefix = np.isfinite(prefix_u) & np.isfinite(prefix_h2)
    if finite_prefix.sum() >= 2 and np.ptp(prefix_u[finite_prefix]) > EPS:
        x = prefix_u[finite_prefix]
        y = prefix_h2[finite_prefix]
        x_centered = x - x.mean()
        h2_slope = float((x_centered * (y - y.mean())).sum() / ((x_centered * x_centered).sum() + EPS))
    else:
        h2_slope = np.nan

    values = np.asarray(
        [
            h2,
            h2_slope,
            float(forward.sum() / (path_length + EPS)),
            float((forward.sum() - backward.sum()) / (path_length + EPS)),
            float(orthogonal.sum() / (path_length + EPS)),
        ],
        dtype=np.float32,
    )
    coverage = np.isfinite(values)
    return values, coverage


def reduce_behavior_response(token_hidden: np.ndarray, endpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one response's ``(tokens, layers, hidden)`` tensor to 5 scalars."""

    values = np.asarray(token_hidden)
    endpoints = np.asarray(endpoints, dtype=np.int64)
    if values.ndim != 3 or values.shape[0] < 1 or len(endpoints) < 2:
        return _invalid()
    pooled = pool_token_hidden(values, endpoints)["mean_w128_s32"]
    points = pooled[:, -1, :]
    progress = endpoints.astype(np.float64) / float(endpoints[-1])
    return _trajectory_features(points, progress)


def _as_numpy(value: Any, *, padding: bool = False) -> np.ndarray:
    if hasattr(value, "to"):
        value = value.to("cpu")
    if hasattr(value, "to_padded_tensor"):
        value = value.to_padded_tensor(padding)
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _dense_array(value: Any) -> np.ndarray:
    """Convert dense or nested torch values to an ordinary numpy array."""
    if hasattr(value, "unbind") and not isinstance(value, np.ndarray):
        try:
            rows = value.unbind()
            if rows:
                return np.stack([_dense_array(row) for row in rows])
        except Exception:
            pass
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _row_arrays(value: Any) -> list[np.ndarray] | None:
    if hasattr(value, "unbind") and not isinstance(value, np.ndarray):
        try:
            rows = value.unbind()
            return [_dense_array(row).reshape(-1) for row in rows]
        except Exception:
            return None
    array = _dense_array(value)
    if array.ndim == 2:
        return [array[index].reshape(-1) for index in range(array.shape[0])]
    return None


def _offsets(value: Any) -> np.ndarray | None:
    if hasattr(value, "offsets"):
        offsets = value.offsets()
        if hasattr(offsets, "detach"):
            offsets = offsets.detach().cpu().numpy()
        return np.asarray(offsets, dtype=np.int64)
    if hasattr(value, "unbind"):
        try:
            row_lengths = [int(row.numel()) for row in value.unbind()]
            return np.concatenate(([0], np.cumsum(row_lengths, dtype=np.int64)))
        except Exception:
            pass
    if hasattr(value, "_nested_tensor_size"):
        size_source = value.to("cpu") if hasattr(value, "to") else value
        sizes = size_source._nested_tensor_size()
        if hasattr(sizes, "detach"):
            sizes = sizes.detach().cpu().numpy()
        sizes = np.asarray(sizes, dtype=np.int64)
        if sizes.ndim == 2 and sizes.shape[1] >= 1:
            return np.concatenate(([0], np.cumsum(sizes[:, 0], dtype=np.int64)))
    return None


def _flat_layer(layer: Any) -> np.ndarray:
    if hasattr(layer, "detach"):
        layer = layer.detach().float().cpu().numpy()
    result = np.asarray(layer)
    if result.ndim == 3 and result.shape[0] == 1:
        return result[0]
    return result


def reduce_behavior_batch(
    hidden_states: Iterable[Any], input_ids: Any, response_mask: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce an actor micro-batch to dense ``(batch, 5)`` behavior values."""

    layers = list(hidden_states)
    if not layers:
        raise ValueError("behavior reduction requires model hidden states")
    final_layer = _flat_layer(layers[-1])
    input_offsets = _offsets(input_ids)
    response_offsets = _offsets(response_mask)
    response_rows = _row_arrays(response_mask)
    mask = None if response_rows is not None else _as_numpy(response_mask, padding=False).astype(bool)
    if final_layer.ndim == 2:
        if input_offsets is None:
            raise ValueError("rmpad input_ids must expose offsets")
        batch_size = len(input_offsets) - 1
        full_lengths = np.diff(input_offsets)
    elif final_layer.ndim == 3:
        batch_size = final_layer.shape[0]
        full_lengths = (
            np.diff(input_offsets)
            if input_offsets is not None
            else np.full(batch_size, final_layer.shape[1], dtype=np.int64)
        )
    else:
        raise ValueError(f"unsupported hidden-state shape: {final_layer.shape}")

    output = np.full((batch_size, len(BEHAVIOR_KEYS)), np.nan, dtype=np.float32)
    coverage = np.zeros_like(output, dtype=bool)
    for index in range(batch_size):
        row_values = response_rows[index] if response_rows is not None else mask[index]
        nonzero = np.flatnonzero(row_values > 0.5)
        response_length = int(nonzero[-1] + 1) if nonzero.size else 0
        if response_length <= 0 or response_length > int(full_lengths[index]):
            continue
        row_mask = np.ones(response_length, dtype=bool) if response_rows is None else row_values[:response_length] > 0.5
        if final_layer.ndim == 2:
            end = int(input_offsets[index + 1])
            start = end - response_length
            row_hidden = final_layer[start:end][row_mask, None, :]
        else:
            start = int(full_lengths[index]) - response_length
            row_hidden = final_layer[index, start : int(full_lengths[index])][row_mask, None, :]
        if row_hidden.shape[0] < 2:
            continue
        endpoints = np.arange(1, row_hidden.shape[0] + 1, dtype=np.int64)
        output[index], coverage[index] = reduce_behavior_response(row_hidden, endpoints)
    return output, coverage


def compute_behavior_advantage(
    values: np.ndarray,
    coverage: np.ndarray,
    group_ids: Sequence[Any],
    *,
    padding: Sequence[bool] | None = None,
    lambda_: float = 0.2,
    expected_group_size: int = 8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute centered H2-conditioned behavior bonuses within rollout groups."""

    values = _dense_array(values).astype(np.float64, copy=False)
    coverage = _dense_array(coverage).astype(bool, copy=False)
    if values.ndim != 2 or values.shape[1] != len(BEHAVIOR_KEYS) or coverage.shape != values.shape:
        raise ValueError("behavior values/coverage must have shape (batch, 5)")
    if len(group_ids) != len(values):
        raise ValueError("group_ids must align with behavior values")
    padding_array = np.zeros(len(values), dtype=bool) if padding is None else np.asarray(padding, dtype=bool)
    if len(padding_array) != len(values):
        raise ValueError("padding must align with behavior values")

    bonus = np.zeros(len(values), dtype=np.float32)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        if not padding_array[index]:
            groups[str(group_id)].append(index)
    valid_groups = 0
    valid_rows = 0
    utility_values: list[float] = []
    for indices in groups.values():
        if len(indices) != expected_group_size:
            continue
        idx = np.asarray(indices, dtype=np.int64)
        valid = coverage[idx].all(axis=1) & np.isfinite(values[idx]).all(axis=1)
        if not valid.all():
            continue
        h = values[idx, 0]
        v = values[idx, 1]
        focus = values[idx, 2]
        ranks = np.asarray(
            [
                (np.sum(h < current) + 0.5 * np.sum(h == current)) / float(len(idx) - 1)
                for current in h
            ],
            dtype=np.float64,
        )
        focus_wins = np.asarray(
            [np.sum(np.sign(current - focus[np.arange(len(idx)) != pos])) / float(len(idx) - 1) for pos, current in enumerate(focus)],
            dtype=np.float64,
        )
        exploration_wins = np.asarray(
            [np.sum(np.sign(v[np.arange(len(idx)) != pos] - current)) / float(len(idx) - 1) for pos, current in enumerate(v)],
            dtype=np.float64,
        )
        utility = (1.0 - ranks) * focus_wins + ranks * exploration_wins
        utility_centered = utility - utility.mean()
        bonus[idx] = (lambda_ * utility_centered).astype(np.float32)
        utility_values.extend(utility_centered.tolist())
        valid_groups += 1
        valid_rows += len(idx)
    metrics = {
        "behavior/groups": float(len(groups)),
        "behavior/valid_groups": float(valid_groups),
        "behavior/coverage_rate": float(valid_rows / len(values)) if len(values) else 0.0,
        "behavior/raw_std": float(np.std(utility_values)) if utility_values else 0.0,
        "behavior/bonus_std": float(np.std(bonus[bonus != 0.0])) if np.any(bonus != 0.0) else 0.0,
        "behavior/bonus_mean": float(np.mean(bonus)) if len(bonus) else 0.0,
    }
    return bonus, metrics
