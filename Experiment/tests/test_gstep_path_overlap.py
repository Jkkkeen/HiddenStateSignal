from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_gstep_path_overlap import (
    evaluate_overlap,
    join_gstep_path,
    within_question_spearman,
)


def test_join_gstep_path_aligns_on_question_rollout_layer() -> None:
    gstep = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2"],
            "rollout_id": [0, 1, 0],
            "layer": [24, 24, 36],
            "window_size": [64, 64, 64],
            "pool": ["mean", "mean", "mean"],
            "is_correct": [True, False, True],
            "gstep_mean": [3.0, 1.0, 2.0],
            "gstep_late_mean": [2.0, 0.5, 1.5],
        }
    )
    path = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2"],
            "rollout_id": [0, 1, 0],
            "layer": [24, 24, 36],
            "is_correct": [True, False, True],
            "path_length": [10.0, 20.0, 30.0],
            "d_late_mean": [4.0, 8.0, 12.0],
        }
    )

    joined = join_gstep_path(gstep, path)

    assert len(joined) == 3
    assert "path_length_score" in joined.columns
    assert "gstep_mean_score" in joined.columns
    assert joined.loc[joined["rollout_id"] == 0, "path_length_score"].iloc[0] == -10.0


def test_within_question_spearman_averages_valid_questions() -> None:
    df = pd.DataFrame(
        {
            "question_id": ["q1"] * 4 + ["q2"] * 4,
            "x": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [2, 4, 6, 8, 4, 3, 2, 1],
        }
    )

    rho, n = within_question_spearman(df, "x", "y")

    assert n == 2
    assert np.isclose(rho, 0.0)


def test_evaluate_overlap_reports_combo_delta() -> None:
    rows = []
    for q in range(3):
        for r in range(4):
            correct = r >= 2
            rows.append(
                {
                    "question_id": f"q{q}",
                    "rollout_id": r,
                    "layer": 24,
                    "window_size": 64,
                    "pool": "mean",
                    "is_correct": correct,
                    "path_length_score": float(r),
                    "d_late_mean_score": float(r),
                    "gstep_mean_score": float(r),
                    "gstep_late_mean_score": float(r),
                }
            )
    joined = pd.DataFrame(rows)

    eval_df = evaluate_overlap(joined, n_boot=20, seed=1)

    assert {"path_feature", "gstep_feature", "combo_auc", "delta_vs_path"}.issubset(eval_df.columns)
    assert eval_df["path_auc"].max() == 1.0
    assert abs(eval_df["delta_vs_path"]).max() < 1e-9
