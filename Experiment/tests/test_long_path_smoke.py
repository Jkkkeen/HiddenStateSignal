from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_long_rollouts import segment_thinking_text
from run_long_trajectory_smoke_qwen3vl import (
    evaluate_path_features_numeric,
    local_angular_rows_for_chunks,
    macro_angular_rows_for_layer,
    should_skip_for_all_metrics,
)
from run_long_path_smoke_qwen3vl import (
    chunk_means_for_segment,
    find_token_subsequence,
    segment_token_span_from_text,
)


def test_implicit_think_close_status() -> None:
    segment = segment_thinking_text("reasoning text</think>\nAnswer: B")

    assert segment["segment_status"] == "implicit_think_close_only"
    assert segment["think_text"] == "reasoning text"
    assert segment["answer_text"].strip() == "Answer: B"


def test_find_token_subsequence_returns_first_match() -> None:
    assert find_token_subsequence([1, 2, 3, 2, 3, 4], [2, 3]) == 1
    assert find_token_subsequence([1, 2, 3], [9]) == -1


def test_segment_token_span_from_text_uses_close_tag_as_boundary() -> None:
    ids = [10, 11, 12, 99, 100, 13, 14]
    close_ids = [99, 100]

    start, end, status = segment_token_span_from_text(ids, close_ids)

    assert start == 0
    assert end == 3
    assert status == "implicit_think_close_only"


def test_chunk_means_for_segment_chunks_only_requested_span() -> None:
    hidden = np.arange(10, dtype=np.float32).reshape(5, 2)

    chunks, spans = chunk_means_for_segment(hidden, start=1, end=5, chunk_size=2)

    assert spans == [(0, 2), (2, 4)]
    np.testing.assert_allclose(chunks, [[[3.0, 4.0]], [[7.0, 8.0]]])


def test_macro_angular_rows_for_layer_computes_multiple_windows_and_pools() -> None:
    hidden = np.stack([np.arange(24, dtype=np.float32), np.zeros(24, dtype=np.float32)], axis=1)
    base = {"question_id": "q1", "rollout_id": 0, "layer": 24, "is_correct": True}

    rows = macro_angular_rows_for_layer(hidden, base, window_sizes=[4, 8], pools=["mean", "last"])

    assert len(rows) == 4
    assert {row["window_size"] for row in rows} == {4, 8}
    assert {row["pool"] for row in rows} == {"mean", "last"}
    assert all("gcos_mean" in row for row in rows)


def test_local_angular_rows_for_chunks_computes_per_chunk_metrics() -> None:
    layer_hidden = np.stack(
        [
            np.arange(32, dtype=np.float32),
            np.zeros(32, dtype=np.float32),
        ],
        axis=1,
    )
    chunks = np.stack([layer_hidden[:16], layer_hidden[16:]], axis=0)
    spans = [(0, 16), (16, 32)]
    base = {"question_id": "q1", "rollout_id": 0, "layer": 24, "is_correct": True}

    rows = local_angular_rows_for_chunks(chunks, spans, base, micro_window=4)

    assert len(rows) == 2
    assert rows[0]["chunk_id"] == 0
    assert rows[1]["chunk_start"] == 16
    assert all("cos_mean" in row for row in rows)


def test_evaluate_path_features_numeric_ignores_string_metadata() -> None:
    import pandas as pd

    rows = []
    for rollout_id, correct in enumerate([False, False, True, True]):
        rows.append(
            {
                "question_id": "q1",
                "rollout_id": rollout_id,
                "layer": 24,
                "is_correct": correct,
                "answer": "A",
                "pred_answer": "A" if correct else "B",
                "path_length": float(4 - rollout_id),
                "d_late_mean": float(4 - rollout_id),
            }
        )

    result = evaluate_path_features_numeric(pd.DataFrame(rows), n_boot=10, seed=1)

    assert set(result["feature"]) == {"path_length", "d_late_mean"}
    assert result["best_auc"].max() == 1.0


def test_short_segment_can_still_keep_gd_metrics() -> None:
    assert should_skip_for_all_metrics(20, micro_window=8, min_window_size=32) is True
    assert should_skip_for_all_metrics(251, micro_window=8, min_window_size=32) is False
