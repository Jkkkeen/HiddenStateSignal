from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_path_dynamics import path_dynamics_from_points, summarize_path_dynamics


def test_path_dynamics_uses_adjacent_l2_distances() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
            [6.0, 11.0],
        ],
        dtype=np.float32,
    )

    dyn = path_dynamics_from_points(points)

    np.testing.assert_allclose(dyn["d"], [5.0, 5.0, 3.0])
    np.testing.assert_allclose(dyn["pv"], [0.0, -2.0])
    np.testing.assert_allclose(dyn["pa"], [-2.0])


def test_path_dynamics_summary_tracks_late_and_historical_features() -> None:
    d = np.asarray([10.0, 8.0, 4.0, 2.0], dtype=np.float64)
    summary = summarize_path_dynamics(d)

    assert math.isclose(summary["path_length"], 24.0)
    assert math.isclose(summary["d_mean"], 6.0)
    assert math.isclose(summary["d_late_mean"], 2.0)
    assert math.isclose(summary["d_early_mean"], 10.0)
    assert math.isclose(summary["d_ratio_late_early"], 0.2)
    assert math.isclose(summary["pv_late_max"], -2.0)
    assert math.isclose(summary["d_hist_late_min"], -5.333333333333333)

