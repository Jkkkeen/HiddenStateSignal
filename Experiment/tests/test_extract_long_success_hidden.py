from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_long_success_hidden_qwen3vl import (
    pool_span_representations,
    shard_name,
    valid_question_shard,
)


def test_pool_span_last_and_mean() -> None:
    hidden = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)

    last, mean = pool_span_representations(hidden, [(0, 4), (2, 6)])

    assert last.shape == (2, 2, 3)
    assert mean.shape == (2, 2, 3)
    np.testing.assert_allclose(last[0], hidden[:, 3, :])
    np.testing.assert_allclose(last[1], hidden[:, 5, :])
    np.testing.assert_allclose(mean[0], hidden[:, 0:4, :].mean(axis=1))


def test_shard_name_is_stable_and_filesystem_safe() -> None:
    assert shard_name("some/question:id") == shard_name("some/question:id")
    assert "/" not in shard_name("some/question:id")
    assert ":" not in shard_name("some/question:id")


def test_valid_question_shard_checks_required_schema(tmp_path: Path) -> None:
    path = tmp_path / "question.npz"
    np.savez_compressed(
        path,
        question_id=np.asarray("q1"),
        layers=np.asarray([24, 36]),
        span_last=np.zeros((2, 2, 3), dtype=np.float16),
        span_mean=np.zeros((2, 2, 3), dtype=np.float16),
        rollout_id=np.asarray([0, 0]),
        is_correct=np.asarray([True, True]),
        span_start=np.asarray([0, 64]),
        span_end=np.asarray([128, 192]),
        relative_progress=np.asarray([0.2, 0.4]),
        think_length=np.asarray([300, 300]),
    )

    assert valid_question_shard(path) is True

    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(invalid, span_last=np.zeros((1, 1, 1)))
    assert valid_question_shard(invalid) is False
