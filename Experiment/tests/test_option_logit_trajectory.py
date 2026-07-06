from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_option_logit_trajectory_qwen3vl import (
    build_rollout_features,
    evaluate_option_logit_features,
)


def test_build_rollout_features_computes_margin_gain_drop_and_prompt_baseline() -> None:
    probes = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "rollout_id": 0,
                "is_correct": True,
                "answer": "C",
                "pred_answer": "C",
                "segment": "prompt",
                "frac": 0.0,
                "correct_margin_max": -2.0,
                "correct_margin_mean": -0.5,
                "abcd_entropy": 1.2,
                "top_option": "A",
            },
            {
                "question_id": "q1",
                "rollout_id": 0,
                "is_correct": True,
                "answer": "C",
                "pred_answer": "C",
                "segment": "think",
                "frac": 0.05,
                "correct_margin_max": -4.0,
                "correct_margin_mean": -1.5,
                "abcd_entropy": 1.3,
                "top_option": "A",
            },
            {
                "question_id": "q1",
                "rollout_id": 0,
                "is_correct": True,
                "answer": "C",
                "pred_answer": "C",
                "segment": "think",
                "frac": 1.0,
                "correct_margin_max": -1.0,
                "correct_margin_mean": 0.5,
                "abcd_entropy": 0.7,
                "top_option": "A",
            },
            {
                "question_id": "q1",
                "rollout_id": 0,
                "is_correct": True,
                "answer": "C",
                "pred_answer": "C",
                "segment": "answer",
                "frac": 0.75,
                "correct_margin_max": -3.0,
                "correct_margin_mean": -1.0,
                "abcd_entropy": 1.0,
                "top_option": "A",
            },
        ]
    )

    features = build_rollout_features(probes)

    assert len(features) == 1
    row = features.iloc[0]
    assert np.isclose(row["prompt_only_margin_max"], -2.0)
    assert np.isclose(row["think_early_margin_max"], -4.0)
    assert np.isclose(row["think_final_margin_max"], -1.0)
    assert np.isclose(row["think_early_to_final_gain_max"], 3.0)
    assert np.isclose(row["think_late_margin_drop_max"], -3.0)
    assert np.isclose(row["think_relative_final_margin_max"], 1.0)
    assert np.isclose(row["think_entropy_delta"], -0.6)
    assert row["think_top_option_switch_count"] == 0


def test_evaluate_option_logit_features_reports_best_auc() -> None:
    rows = []
    for q in range(3):
        for r in range(4):
            is_correct = r >= 2
            rows.append(
                {
                    "question_id": f"q{q}",
                    "rollout_id": r,
                    "is_correct": is_correct,
                    "think_early_to_final_gain_max": float(r),
                    "think_late_margin_drop_max": -float(r),
                    "think_final_margin_max": float(r),
                }
            )
    features = pd.DataFrame(rows)

    eval_df = evaluate_option_logit_features(features, n_boot=0, seed=1)

    assert {"feature", "mean_auc_pos", "mean_auc_neg", "best_auc"}.issubset(eval_df.columns)
    assert eval_df["best_auc"].max() == 1.0
