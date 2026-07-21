from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_long_success_smoke import build_manifest


def _question_rows(question_id: str, n_correct: int, n_wrong: int, base_length: int = 3000):
    rows = []
    for rollout_id in range(n_correct + n_wrong):
        correct = rollout_id < n_correct
        rows.append(
            {
                "question_id": question_id,
                "rollout_id": rollout_id,
                "is_correct": correct,
                "truncated": False,
                "has_think_close": True,
                "pred_answer": "A" if correct else "B",
                "think_token_count": base_length + rollout_id,
                "response": "fixed response</think>Answer: A",
            }
        )
    return rows


def test_build_manifest_keeps_only_primary_discovery_questions() -> None:
    rows = []
    rows.extend(_question_rows("q1", 3, 3, 1000))
    rows.extend(_question_rows("q2", 2, 4, 3000))
    rows.extend(_question_rows("q3", 4, 4, 5000))
    rows.extend(_question_rows("q4", 3, 3, 9000))

    result = build_manifest(rows, smoke_questions=2, discovery_fraction=0.7, seed=7)

    assert set(result.primary_questions) == {"q1", "q3", "q4"}
    assert len(result.smoke_questions) == 2
    assert set(result.smoke_questions).issubset(set(result.discovery_questions))
    assert all(row["question_id"] in result.smoke_questions for row in result.smoke_rows)
    assert all(row["question_id"] != "q2" for row in result.smoke_rows)


def test_build_manifest_is_deterministic() -> None:
    rows = []
    for idx in range(12):
        rows.extend(_question_rows(f"q{idx}", 3, 3, 1000 + idx * 700))

    first = build_manifest(rows, smoke_questions=4, discovery_fraction=0.7, seed=11)
    second = build_manifest(rows, smoke_questions=4, discovery_fraction=0.7, seed=11)

    assert first.smoke_questions == second.smoke_questions
    assert first.discovery_questions == second.discovery_questions
    assert first.confirmatory_questions == second.confirmatory_questions


def test_build_manifest_removes_incomplete_rows_before_counting() -> None:
    rows = _question_rows("q1", 3, 3)
    rows[0]["truncated"] = True

    result = build_manifest(rows, smoke_questions=1, discovery_fraction=0.7, seed=1)

    assert result.primary_questions == []
    assert result.smoke_rows == []
