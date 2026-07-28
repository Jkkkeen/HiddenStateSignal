from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_entropy_band_confirm120 import (  # noqa: E402
    bootstrap_mean_interval,
    classify_gate_outcome,
    grouped_oof_scores,
    question_aucs,
)


def _frame(n_questions: int = 10) -> pd.DataFrame:
    rows = []
    for question in range(n_questions):
        for rollout, label in enumerate([False, False, True, True]):
            rows.append(
                {
                    "question_id": f"q{question}",
                    "rollout_id": rollout,
                    "is_correct": label,
                    "think_length": 3000 + 10 * rollout + question,
                    "raw_entropy_L14_19_bin6": 0.45 + 0.1 * int(label) + 0.001 * question,
                }
            )
    return pd.DataFrame(rows)


def test_question_auc_is_question_equal_and_pairwise() -> None:
    aucs = question_aucs(_frame())

    assert len(aucs) == 10
    assert np.allclose(aucs.to_numpy(), 1.0)


def test_grouped_oof_has_no_question_overlap() -> None:
    scores, metadata = grouped_oof_scores(_frame(), seed=20260725)

    assert metadata["fold_question_overlap"] == 0
    assert scores[["feature_score", "length_score", "combined_score"]].notna().all().all()
    assert scores.groupby("question_id")["fold_id"].nunique().max() == 1


def test_bootstrap_and_gate_classification_use_lower_bounds() -> None:
    low, high = bootstrap_mean_interval(np.ones(20), bootstrap=200, seed=7)

    assert low == 1.0
    assert high == 1.0
    assert classify_gate_outcome(True, True) == "advance_to_rl_credit_assignment"
    assert classify_gate_outcome(True, False) == "internal_correlate_not_rl_candidate"
    assert classify_gate_outcome(False, True) == "replication_failed"
