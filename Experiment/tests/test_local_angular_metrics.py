from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_local_angular_qwen3vl import angular_metrics_from_windows


def test_straight_windows_have_high_directional_consistency() -> None:
    windows = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = angular_metrics_from_windows(windows, angle_bins=6, min_angles_for_entropy=1)

    assert metrics["n_angles"] == 2
    assert math.isclose(metrics["cos_mean"], 1.0, abs_tol=1e-6)
    assert math.isclose(metrics["lad_mean"], 0.0, abs_tol=1e-6)
    assert metrics["spike_rate_90"] == 0.0


def test_reverse_turn_counts_as_spike() -> None:
    windows = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = angular_metrics_from_windows(windows, angle_bins=6, min_angles_for_entropy=1)

    assert metrics["n_angles"] == 2
    assert metrics["cos_min"] <= -0.999
    assert metrics["spike_rate_90"] >= 0.5
