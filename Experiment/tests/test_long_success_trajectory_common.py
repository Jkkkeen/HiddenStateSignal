from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from long_success_trajectory_common import (
    balanced_reference_subsets,
    eligible_question_summaries,
    full_span_bounds,
    pairwise_auc,
)


def _row(question_id: str, rollout_id: int, correct: bool, **overrides):
    row = {
        "question_id": question_id,
        "rollout_id": rollout_id,
        "is_correct": correct,
        "truncated": False,
        "has_think_close": True,
        "pred_answer": "A" if correct else "B",
        "think_token_count": 3000 + rollout_id,
    }
    row.update(overrides)
    return row


def test_balanced_subsets_for_three_correct_five_wrong_query_wrong() -> None:
    labels = {0: True, 1: True, 2: True, 3: False, 4: False, 5: False, 6: False, 7: False}

    subsets = balanced_reference_subsets(labels, query_id=3)

    assert len(subsets) == 18
    assert all(len(pos) == len(neg) == 2 for pos, neg in subsets)
    assert all(3 not in pos and 3 not in neg for pos, neg in subsets)


def test_balanced_subsets_for_three_three_use_both_remaining_paths() -> None:
    labels = {0: True, 1: True, 2: True, 3: False, 4: False, 5: False}

    subsets = balanced_reference_subsets(labels, query_id=0)

    assert subsets == [((1, 2), (3, 4)), ((1, 2), (3, 5)), ((1, 2), (4, 5))]


def test_full_span_bounds_drop_partial_tail() -> None:
    assert full_span_bounds(300, window=128, stride=64) == [(0, 128), (64, 192), (128, 256)]
    assert full_span_bounds(127, window=128, stride=64) == []


def test_primary_eligibility_requires_three_correct_and_three_wrong() -> None:
    rows = []
    rows.extend(_row("q_primary", idx, idx < 3) for idx in range(6))
    rows.extend(_row("q_secondary", idx, idx < 2) for idx in range(6))
    rows.append(_row("q_primary", 99, True, truncated=True))

    summary = eligible_question_summaries(rows)

    primary = summary.set_index("question_id").loc["q_primary"]
    secondary = summary.set_index("question_id").loc["q_secondary"]
    assert primary["n_correct"] == 3
    assert primary["n_wrong"] == 3
    assert bool(primary["is_primary"]) is True
    assert bool(secondary["is_primary"]) is False
    assert bool(secondary["is_secondary"]) is True


def test_pairwise_auc_handles_ties() -> None:
    assert pairwise_auc(np.array([0.7, 0.5]), np.array([0.5, 0.2])) == 0.875

