"""Scalar H2 and terminal-direction H10 reductions for fixed rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import EPS, STAGES, stage_mask


def _nan_stage() -> dict[str, float | bool | int]:
    return {
        "coverage_ok": False,
        "n_valid": 0,
        "path_length": np.nan,
        "net_displacement": np.nan,
        "h2_local": np.nan,
        "h2_cumulative": np.nan,
        "delta_h2": np.nan,
        "forward_length": np.nan,
        "backward_length": np.nan,
        "orthogonal_length": np.nan,
        "forward_ratio": np.nan,
        "backward_ratio": np.nan,
        "orthogonal_ratio": np.nan,
        "h10_p": np.nan,
        "h10_q": np.nan,
        "h10_forward_reward": np.nan,
        "h10_exploration_reward": np.nan,
        "h10_effective_exploration": np.nan,
    }


def _stage_displacements(points: np.ndarray, progress: np.ndarray, stage: int) -> np.ndarray:
    displacements = np.diff(points, axis=0)
    selected = stage_mask(progress[1:], stage)
    return displacements[selected]


def reduce_h2_h10(
    trajectory: np.ndarray,
    progress: np.ndarray,
    *,
    representation: str,
    anchor_layer: int = -1,
) -> list[dict[str, Any]]:
    """Return one scalar row per stage for a pooled response trajectory.

    ``trajectory`` is ``(n_chunks, n_hidden_states, hidden_size)``. The
    terminal direction is defined by the complete response and each stage's
    H10 quantities are local projections onto that fixed direction.
    """

    values = np.asarray(trajectory, dtype=np.float64)
    coordinates = np.asarray(progress, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("trajectory must have shape (K,L,D), K >= 1")
    if coordinates.ndim != 1 or len(coordinates) != values.shape[0]:
        raise ValueError("progress must align with trajectory")
    points = values[:, anchor_layer, :]
    if values.shape[0] < 2:
        rows = []
        for stage in range(len(STAGES)):
            row = _nan_stage()
            row.update({"representation": representation, "anchor_layer": anchor_layer, "stage": stage})
            rows.append(row)
        return rows
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(coordinates)):
        raise ValueError("trajectory and progress must be finite")

    terminal = points[-1] - points[0]
    terminal_norm = float(np.linalg.norm(terminal))
    if terminal_norm <= EPS:
        rows = []
        for stage in range(len(STAGES)):
            row = _nan_stage()
            row.update({"representation": representation, "anchor_layer": anchor_layer, "stage": stage})
            rows.append(row)
        return rows
    direction = terminal / terminal_norm

    rows: list[dict[str, Any]] = []
    previous_h2 = np.nan
    for stage in range(len(STAGES)):
        local = _stage_displacements(points, coordinates, stage)
        row = _nan_stage()
        row.update({"representation": representation, "anchor_layer": anchor_layer, "stage": stage})
        if local.shape[0] == 0:
            rows.append(row)
            continue

        lengths = np.linalg.norm(local, axis=1)
        valid = np.isfinite(lengths) & (lengths > EPS)
        local = local[valid]
        lengths = lengths[valid]
        if local.shape[0] == 0:
            rows.append(row)
            continue
        axial = local @ direction
        forward = np.maximum(axial, 0.0)
        backward = np.maximum(-axial, 0.0)
        orthogonal = np.sqrt(np.maximum(lengths * lengths - axial * axial, 0.0))
        path_length = float(lengths.sum())
        net_displacement = float(np.linalg.norm(local.sum(axis=0)))
        forward_length = float(forward.sum())
        backward_length = float(backward.sum())
        orthogonal_length = float(orthogonal.sum())
        axial_length = forward_length + backward_length
        h2_local = net_displacement / (path_length + EPS)

        endpoint = int(np.flatnonzero(coordinates <= STAGES[stage][1])[-1])
        cumulative = np.diff(points[: endpoint + 1], axis=0)
        cumulative_lengths = np.linalg.norm(cumulative, axis=1)
        cumulative_path = float(cumulative_lengths.sum())
        cumulative_net = float(np.linalg.norm(cumulative.sum(axis=0)))
        h2_cumulative = cumulative_net / (cumulative_path + EPS)
        delta_h2 = float(h2_cumulative - previous_h2) if np.isfinite(previous_h2) else np.nan
        previous_h2 = h2_cumulative

        forward_ratio = forward_length / (path_length + EPS)
        backward_ratio = backward_length / (path_length + EPS)
        orthogonal_ratio = orthogonal_length / (path_length + EPS)
        h10_p = (forward_length - backward_length) / (path_length + EPS)
        h10_q = forward_length / (axial_length + EPS)
        forward_reward = (2.0 * h10_q - 1.0) * axial_length / (path_length + EPS)
        exploration_reward = (
            max(-delta_h2, 0.0) * orthogonal_ratio * max(1.0 - backward_ratio, 0.0)
            if np.isfinite(delta_h2)
            else np.nan
        )
        row.update(
            {
                "coverage_ok": True,
                "n_valid": int(local.shape[0]),
                "path_length": path_length,
                "net_displacement": net_displacement,
                "h2_local": h2_local,
                "h2_cumulative": h2_cumulative,
                "delta_h2": delta_h2,
                "forward_length": forward_length,
                "backward_length": backward_length,
                "orthogonal_length": orthogonal_length,
                "forward_ratio": forward_ratio,
                "backward_ratio": backward_ratio,
                "orthogonal_ratio": orthogonal_ratio,
                "h10_p": h10_p,
                "h10_q": h10_q,
                "h10_forward_reward": forward_reward,
                "h10_exploration_reward": exploration_reward,
            }
        )
        rows.append(row)

    for index in range(len(rows) - 1):
        current = rows[index]
        next_row = rows[index + 1]
        if current["coverage_ok"] and next_row["coverage_ok"]:
            current["h10_effective_exploration"] = float(
                current["h10_exploration_reward"]
                * max(float(next_row["h10_p"]) - float(current["h10_p"]), 0.0)
            )
    return rows


def reduce_forward_output(
    forward: dict[str, Any], *, metadata: dict[str, Any], anchor_layer: int = -1
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation in ("mean_w128_s32", "last_s32"):
        reduced = reduce_h2_h10(
            forward[representation],
            forward["progress"],
            representation=representation,
            anchor_layer=anchor_layer,
        )
        for row in reduced:
            rows.append({**metadata, **row})
    return rows
