from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_experiment02_replication import build_replication_manifest  # noqa: E402


def _row(question_id: str, rollout_id: int, is_correct: bool) -> dict:
    return {
        "question_id": question_id,
        "rollout_id": rollout_id,
        "is_correct": is_correct,
        "pred_answer": "A",
        "truncated": False,
        "has_think_close": True,
        "think_token_count": 2200 + rollout_id,
    }


def test_manifest_excludes_discovery_and_requires_two_by_two() -> None:
    rows = []
    for question_index in range(8):
        question_id = f"q{question_index}"
        labels = [True, True, False, False]
        if question_id == "q7":
            labels = [True, False, False]
        rows.extend(_row(question_id, rid, label) for rid, label in enumerate(labels))

    result = build_replication_manifest(
        rows,
        excluded_question_ids={"q0"},
        question_limit=5,
        smoke_questions=2,
        seed=20260724,
    )

    assert len(result.selected_questions) == 5
    assert "q0" not in result.selected_questions
    assert "q7" not in result.selected_questions
    assert set(result.smoke_questions).issubset(result.selected_questions)
    selected = set(result.selected_questions)
    assert {row["question_id"] for row in result.selected_rows} == selected
    for question_id in selected:
        labels = [
            bool(row["is_correct"])
            for row in result.selected_rows
            if row["question_id"] == question_id
        ]
        assert sum(labels) >= 2
        assert len(labels) - sum(labels) >= 2
