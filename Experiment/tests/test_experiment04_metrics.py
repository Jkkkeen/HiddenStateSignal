from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from experiment04_metrics import (  # noqa: E402
    chunk_representations,
    coordinate_entropy,
    late_window,
    path_metrics,
    state_angles,
    vertical_scale,
)


def test_chunk_representations_keep_only_full_chunks_primary() -> None:
    hidden = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
    result = chunk_representations(hidden, chunk_size=4, last_n=3)

    assert result["full_bounds"].tolist() == [[0, 4], [4, 8]]
    assert result["partial_bounds"].tolist() == [[8, 10]]
    assert np.array_equal(result["full"]["last"], hidden[[3, 7]])
    assert np.allclose(result["full"]["mean"][0], hidden[:4].mean(axis=0))
    assert np.allclose(result["full"]["last25"][0], hidden[1:4].mean(axis=0))
    assert np.allclose(result["partial"]["last25"][0], hidden[8:10].mean(axis=0))


def test_coordinate_entropy_is_normalized_and_centered_per_vector() -> None:
    values = np.asarray([[10.0, 10.0, 10.0], [2.0, -1.0, -1.0]])
    entropy = coordinate_entropy(values)

    assert entropy.shape == (2,)
    assert entropy[0] == 0.0
    assert 0.0 < entropy[1] <= 1.0


def test_path_metrics_distinguish_straight_and_returning_paths() -> None:
    straight = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    returning = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])

    straight_result = path_metrics(straight, rolling=2)
    returning_result = path_metrics(returning, rolling=2)

    assert np.allclose(straight_result["step_norm"], 1.0)
    assert np.allclose(straight_result["turn_cos"], 1.0)
    assert np.isclose(straight_result["path_length"], 3.0)
    assert np.isclose(straight_result["net_displacement"], 3.0)
    assert np.isclose(straight_result["straightness"], 1.0)
    assert returning_result["straightness"] < straight_result["straightness"]
    assert returning_result["log_detour"] > straight_result["log_detour"]
    assert returning_result["rolling_straightness"].shape == (2,)


def test_state_angle_demean_removes_shared_component() -> None:
    points = np.asarray([[100.0, 0.0], [100.0, 1.0], [100.0, 2.0]])
    common = np.asarray([100.0, 0.0])

    raw = state_angles(points)
    demean = state_angles(points, common=common)

    assert raw[0] < 0.02
    assert np.isnan(demean[0])
    assert np.isclose(demean[1], 0.0)


def test_late_window_requires_eight_values() -> None:
    assert late_window(np.arange(7, dtype=np.float64)).size == 0
    assert late_window(np.arange(8, dtype=np.float64)).tolist() == [4.0, 5.0, 6.0, 7.0]
    assert late_window(np.arange(20, dtype=np.float64)).tolist() == [15.0, 16.0, 17.0, 18.0, 19.0]


def test_vertical_scale_is_rollout_equal_and_sets_threshold() -> None:
    per_rollout = [np.asarray([[1.0, 100.0], [3.0, 100.0]]), np.asarray([[5.0, 10.0]])]
    scale, threshold = vertical_scale(per_rollout)

    assert np.allclose(scale, [3.5, 55.0])
    assert np.allclose(threshold, np.maximum(1e-8, 1e-4 * scale))
