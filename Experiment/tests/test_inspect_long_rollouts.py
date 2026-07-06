from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_long_rollouts import (
    extract_choice,
    label_and_segment_row,
    segment_thinking_text,
    summarize_rows,
)


def test_segment_thinking_text_splits_think_and_answer() -> None:
    response = "<think>first reason\nwait, revise</think>\nThe answer is C."

    segment = segment_thinking_text(response)

    assert segment["has_think_open"] is True
    assert segment["has_think_close"] is True
    assert segment["think_text"] == "first reason\nwait, revise"
    assert "answer is C" in segment["answer_text"]


def test_label_and_segment_row_marks_missing_close_as_truncated() -> None:
    row = {
        "question_id": "q1",
        "rollout_id": 0,
        "answer": "B",
        "response": "<think>unfinished reasoning",
        "finish_reason": "stop",
        "max_tokens": 8192,
        "output_token_count": 200,
    }

    labeled = label_and_segment_row(row)

    assert labeled["has_think_open"] is True
    assert labeled["has_think_close"] is False
    assert labeled["truncated"] is True
    assert labeled["segment_status"] == "missing_think_close"


def test_extract_choice_prefers_answer_segment() -> None:
    text = "<think>I considered A, then B, then C.</think>\nFinal answer: D"

    assert extract_choice(text) == "D"


def test_summarize_rows_counts_mixed_clean_questions() -> None:
    rows = [
        {"question_id": "q1", "is_correct": True, "truncated": False, "think_token_count": 10},
        {"question_id": "q1", "is_correct": False, "truncated": False, "think_token_count": 20},
        {"question_id": "q2", "is_correct": True, "truncated": False, "think_token_count": 30},
        {"question_id": "q2", "is_correct": True, "truncated": True, "think_token_count": 40},
    ]

    summary = summarize_rows(rows)

    assert summary["rollouts"] == 4
    assert summary["questions"] == 2
    assert summary["mixed_questions_full"] == 1
    assert summary["mixed_questions_clean"] == 1
    assert summary["truncation_rate"] == 0.25
