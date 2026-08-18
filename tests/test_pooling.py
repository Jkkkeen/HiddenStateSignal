import numpy as np
import pytest

from vertical.pooling import (
    STAGES,
    pool_response_states,
    progress_from_endpoints,
    stage_mask,
    trajectory_endpoints,
)


def test_trajectory_endpoints_match_frozen_window_stride():
    assert trajectory_endpoints(0).tolist() == []
    assert trajectory_endpoints(64).tolist() == [64]
    assert trajectory_endpoints(128).tolist() == [128]
    assert trajectory_endpoints(129).tolist() == [128, 129]
    assert trajectory_endpoints(200).tolist() == [128, 160, 192, 200]


def test_pooling_matches_trailing_window_definition():
    tokens = np.arange(10 * 2 * 3, dtype=np.float32).reshape(10, 2, 3)
    endpoints = np.array([2, 5, 10])

    pooled = pool_response_states(tokens, endpoints, window=4)

    assert np.allclose(pooled["last_s32"][1], tokens[4])
    assert np.allclose(pooled["mean_w128_s32"][1], tokens[1:5].mean(axis=0))
    assert pooled["mean_w128_s32"].dtype == np.float32


def test_pooling_rejects_out_of_range_endpoint():
    tokens = np.zeros((4, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="outside response tokens"):
        pool_response_states(tokens, np.array([5]))


def test_progress_uses_one_based_exclusive_endpoints():
    progress = progress_from_endpoints(np.array([2, 5, 10]), response_length=10)

    assert np.allclose(progress, [0.2, 0.5, 1.0])


def test_stage_boundaries_are_left_closed_and_final_right_closed():
    progress = np.array([0.0, 0.249, 0.25, 0.5, 0.75, 1.0])

    assert len(STAGES) == 4
    assert stage_mask(progress, 0).tolist() == [True, True, False, False, False, False]
    assert stage_mask(progress, 1).tolist() == [False, False, True, False, False, False]
    assert stage_mask(progress, 2).tolist() == [False, False, False, True, False, False]
    assert stage_mask(progress, 3).tolist() == [False, False, False, False, True, True]


def test_stage_mask_rejects_unknown_stage():
    with pytest.raises(ValueError, match="stage"):
        stage_mask(np.array([0.5]), 4)

