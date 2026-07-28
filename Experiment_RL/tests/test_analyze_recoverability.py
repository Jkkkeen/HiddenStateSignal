from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Experiment_RL" / "scripts"))

from analyze_recoverability import analyze_feature, pairwise_concordance


def test_pairwise_concordance_uses_ordered_recovery_pairs() -> None:
    assert pairwise_concordance([0.0, 1.0, 2.0], [0.0, 0.5, 1.0]) == 1.0
    assert pairwise_concordance([2.0, 1.0, 0.0], [0.0, 0.5, 1.0]) == 0.0
    assert pairwise_concordance([0.0, 0.0], [0.0, 1.0]) == 0.5
    assert math.isnan(pairwise_concordance([0.0, 1.0], [0.5, 0.5]))


def test_analyze_feature_reports_within_question_recovery_signal() -> None:
    rows = []
    for question_id in ("q1", "q2"):
        for rollout_id, recovery_rate in enumerate((0.0, 0.25, 0.75, 1.0)):
            rows.append(
                {
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "recovery_rate": recovery_rate,
                    "any_recovery": recovery_rate > 0,
                    "trimmed_margin_mean": float(rollout_id),
                }
            )
    frame = pd.DataFrame(rows)

    result, per_question = analyze_feature(
        frame,
        feature="trimmed_margin_mean",
        direction="pos",
        n_boot=100,
        seed=3,
    )

    assert result["valid_pairwise_questions"] == 2
    assert result["mean_within_pairwise_auc"] == 1.0
    assert result["mean_within_corr"] > 0.98
    assert result["top_recovery_mean"] == 1.0
    assert result["bottom_recovery_mean"] == 0.0
    assert result["top_bottom_recovery_delta"] == 1.0
    assert len(per_question) == 2
