from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cross_chunk_angular_qwen3vl import cross_chunk_angular_metrics, pool_points


def test_pool_points_supports_mean_and_last() -> None:
    hidden = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ],
        dtype=np.float32,
    )

    mean_points = pool_points(hidden, window_size=2, pool="mean")
    last_points = pool_points(hidden, window_size=2, pool="last")

    np.testing.assert_allclose(mean_points, [[1.0, 0.0], [5.0, 0.0]])
    np.testing.assert_allclose(last_points, [[2.0, 0.0], [6.0, 0.0]])


def test_straight_cross_chunk_trajectory_has_high_gcos() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = cross_chunk_angular_metrics(points)

    assert metrics["n_points"] == 4
    assert metrics["n_angles"] == 2
    assert math.isclose(metrics["gcos_mean"], 1.0, abs_tol=1e-6)
    assert math.isclose(metrics["gtheta_mean"], 0.0, abs_tol=1e-6)
    assert metrics["gspike_rate_90"] == 0.0


def test_reversal_cross_chunk_trajectory_has_negative_gcos() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = cross_chunk_angular_metrics(points)

    assert metrics["n_angles"] == 2
    assert metrics["gcos_min"] <= -0.999
    assert metrics["gspike_rate_90"] >= 0.5

