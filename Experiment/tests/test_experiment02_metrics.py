from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from experiment02_metrics import (  # noqa: E402
    coordinate_energy_entropy,
    path_integrals,
    trajectory_z_entropy,
)


def test_coordinate_entropy_centering_is_per_token() -> None:
    values = np.asarray([[1000.0, 1.0, 1.0], [1001.0, 2.0, 2.0]])
    entropy, effective = coordinate_energy_entropy(values, center_coordinates=True)

    assert entropy.shape == (2,)
    assert np.all(np.isfinite(entropy))
    assert np.all((entropy >= 0.0) & (entropy <= 1.0))
    assert np.all(effective >= 1.0)


def test_trajectory_zscore_removes_persistent_rogue_scale() -> None:
    hidden = np.asarray(
        [
            [1000.0, 0.0, 0.0],
            [1001.0, 10.0, 0.0],
            [999.0, 20.0, 10.0],
        ]
    )
    result = trajectory_z_entropy(hidden, sigma_scale=1e-4, clip=8.0)

    assert np.allclose(result["mean"], [1000.0, 10.0, 10.0 / 3.0])
    assert np.max(np.abs(result["z"])) <= 8.0
    assert abs(result["z"][1, 0]) < 1.1
    assert np.all(np.isfinite(result["entropy"]))
    assert 0.0 <= result["low_variance_fraction"] <= 1.0


def test_trajectory_zscore_handles_constant_coordinates() -> None:
    hidden = np.asarray([[2.0, 7.0], [2.0, 7.0], [2.0, 7.0]])
    result = trajectory_z_entropy(hidden, sigma_scale=1e-4, clip=8.0)

    assert np.all(result["z"] == 0.0)
    assert np.all(result["entropy"] == 0.0)
    assert result["low_variance_fraction"] == 1.0


def test_path_integrals_distinguish_straight_and_returning_paths() -> None:
    bins = np.asarray([0, 0, 0], dtype=np.int16)
    straight = path_integrals(
        np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        bins,
        n_bins=1,
        min_block_steps=3,
    )
    returning = path_integrals(
        np.asarray([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]),
        bins,
        n_bins=1,
        min_block_steps=3,
    )

    straight_whole = next(row for row in straight if row["scope"] == "whole")
    returning_whole = next(row for row in returning if row["scope"] == "whole")
    assert np.isclose(straight_whole["path_length"], 3.0)
    assert np.isclose(straight_whole["net_displacement"], 3.0)
    assert np.isclose(straight_whole["straightness"], 1.0)
    assert returning_whole["straightness"] < straight_whole["straightness"]
    assert returning_whole["log_detour"] > straight_whole["log_detour"]
