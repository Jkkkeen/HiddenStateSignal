import numpy as np

from experiment_2e.online_behavior import (
    compute_behavior_advantage,
    reduce_behavior_response,
)


def _token_hidden(points: np.ndarray, tokens_per_chunk: int = 128) -> np.ndarray:
    rows = []
    for point in points:
        rows.extend([point] * tokens_per_chunk)
    return np.asarray(rows, dtype=np.float32)[:, None, :]


def test_straight_response_has_forward_focus():
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    values, coverage = reduce_behavior_response(_token_hidden(points), np.asarray([128, 256, 384]))
    assert coverage.all()
    assert np.isclose(values[0], 1.0, atol=1e-5)
    assert values[2] > 0.99
    assert values[3] > 0.99
    assert values[4] < 1e-5


def test_sideward_response_has_orthogonal_motion():
    points = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    values, coverage = reduce_behavior_response(_token_hidden(points), np.asarray([128, 256, 384]))
    assert coverage.all()
    assert values[0] < 0.8
    assert values[2] < 0.8
    assert values[4] > 0.4


def test_behavior_advantage_is_group_centered():
    values = np.zeros((8, 5), dtype=np.float32)
    values[:, 0] = np.linspace(0.1, 0.8, 8)
    values[:, 1] = -np.linspace(0.01, 0.08, 8)
    values[:, 2] = np.linspace(0.1, 0.8, 8)
    values[:, 3] = values[:, 2]
    values[:, 4] = 1.0 - values[:, 2]
    bonus, metrics = compute_behavior_advantage(values, np.ones_like(values, dtype=bool), ["q"] * 8)
    assert np.isclose(float(bonus.sum()), 0.0, atol=1e-7)
    assert metrics["behavior/valid_groups"] == 1.0
    assert bonus[-1] != bonus[0]
