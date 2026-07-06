from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_option_logit_trimmed_conclusion_qwen3vl import (
    build_trimmed_features,
    evaluate_trimmed_features,
    trim_final_conclusion,
)


def test_trim_final_conclusion_removes_last_answer_sentence() -> None:
    text = (
        "We know the two angles form a linear pair. "
        "So angle 2 is 180 - 50 = 130. "
        "Therefore, the answer is D."
    )

    trimmed, status = trim_final_conclusion(text)

    assert status == "trigger"
    assert "Therefore" not in trimmed
    assert trimmed.endswith("130.")


def test_trim_final_conclusion_falls_back_to_last_sentence() -> None:
    text = "First compute the slope. Then compare the two angles. It is 140 degrees."

    trimmed, status = trim_final_conclusion(text)

    assert status == "last_sentence"
    assert trimmed == "First compute the slope. Then compare the two angles."


def test_build_trimmed_features_joins_original_baselines_and_deltas() -> None:
    trimmed = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "rollout_id": 0,
                "source_line": 1,
                "is_correct": True,
                "answer": "C",
                "pred_answer": "C",
                "correct_margin_max": 1.5,
                "correct_margin_mean": 2.0,
                "abcd_entropy": 0.5,
                "trim_status": "trigger",
            }
        ]
    )
    original = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "rollout_id": 0,
                "prompt_only_margin_max": -1.0,
                "prompt_only_margin_mean": -0.5,
                "think_final_margin_max": 3.0,
                "think_final_margin_mean": 4.0,
            }
        ]
    )

    features = build_trimmed_features(trimmed, original)

    row = features.iloc[0]
    assert np.isclose(row["trimmed_margin_max"], 1.5)
    assert np.isclose(row["trimmed_relative_margin_max"], 2.5)
    assert np.isclose(row["original_minus_trimmed_final_margin_max"], 1.5)
    assert np.isclose(row["trimmed_margin_mean"], 2.0)
    assert np.isclose(row["trimmed_relative_margin_mean"], 2.5)
    assert np.isclose(row["original_minus_trimmed_final_margin_mean"], 2.0)


def test_evaluate_trimmed_features_reports_auc() -> None:
    rows = []
    for q in range(3):
        for r in range(4):
            is_correct = r >= 2
            rows.append(
                {
                    "question_id": f"q{q}",
                    "rollout_id": r,
                    "is_correct": is_correct,
                    "trimmed_relative_margin_max": float(r),
                    "trimmed_margin_max": float(r),
                }
            )
    features = pd.DataFrame(rows)

    eval_df = evaluate_trimmed_features(features, n_boot=0, seed=1)

    assert {"feature", "mean_auc_pos", "mean_auc_neg", "best_auc"}.issubset(eval_df.columns)
    assert eval_df["best_auc"].max() == 1.0
