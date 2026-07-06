from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_think_answer_transition_combo import (
    build_transition_features,
    evaluate_transition_features,
    plot_think_final_margin,
    summarize_step_segment,
)


def test_summarize_step_segment_extracts_early_late_and_stability_features() -> None:
    steps = pd.DataFrame(
        {
            "question_id": ["q1"] * 4,
            "rollout_id": [0] * 4,
            "layer": [36] * 4,
            "is_correct": [True] * 4,
            "relative_step_pos": [0.0, 0.3, 0.8, 1.0],
            "state_margin": [0.1, 0.2, 0.4, 0.6],
            "step_correct_gain": [0.1, 0.1, 0.2, -0.1],
            "turn_to_correct": [0.0, 0.1, 0.2, 0.3],
            "productive_turn": [0.0, 0.0, 0.5, 1.0],
        }
    )

    out = summarize_step_segment(steps, "think")

    row = out.iloc[0]
    assert row["think_step_count"] == 4
    assert np.isclose(row["think_early_margin_mean"], 0.15)
    assert np.isclose(row["think_late_margin_mean"], 0.5)
    assert np.isclose(row["think_mean_gain"], 0.075)
    assert np.isclose(row["think_gain_abs_mean"], 0.125)


def test_build_transition_features_joins_think_and_answer_and_adds_combos() -> None:
    think = pd.DataFrame(
        {
            "question_id": ["q1", "q1"],
            "rollout_id": [0, 1],
            "layer": [36, 36],
            "is_correct": [True, False],
            "relative_step_pos": [0.9, 0.9],
            "state_margin": [0.4, 0.2],
            "step_correct_gain": [0.05, 0.1],
            "turn_to_correct": [0.1, 0.2],
            "productive_turn": [0.0, 0.0],
        }
    )
    answer = pd.DataFrame(
        {
            "question_id": ["q1", "q1"],
            "rollout_id": [0, 1],
            "layer": [36, 36],
            "is_correct": [True, False],
            "relative_step_pos": [0.1, 0.1],
            "state_margin": [0.1, 0.5],
            "step_correct_gain": [-0.02, 0.2],
            "turn_to_correct": [0.0, 0.3],
            "productive_turn": [0.0, 0.1],
        }
    )

    features = build_transition_features(think, answer)

    assert len(features) == 2
    assert "transition_margin_drop" in features.columns
    assert "answer_stability_score" in features.columns
    correct = features[features["is_correct"]].iloc[0]
    assert np.isclose(correct["transition_margin_drop"], -0.3)
    assert np.isclose(correct["answer_stability_score"], -0.02)


def test_evaluate_transition_features_reports_auc_rows() -> None:
    rows = []
    for q in range(3):
        for r in range(4):
            is_correct = r >= 2
            rows.append(
                {
                    "question_id": f"q{q}",
                    "rollout_id": r,
                    "layer": 36,
                    "is_correct": is_correct,
                    "answer_stability_score": float(r),
                    "transition_margin_drop": float(r),
                    "combo_l36_lock_score": float(r),
                }
            )
    features = pd.DataFrame(rows)

    eval_df = evaluate_transition_features(features, n_boot=0, seed=1)

    assert {"feature", "mean_auc_pos", "mean_auc_neg", "best_auc"}.issubset(eval_df.columns)
    assert eval_df["best_auc"].max() == 1.0


def test_plot_think_final_margin_writes_png(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2", "q2"],
            "rollout_id": [0, 1, 0, 1],
            "layer": [24, 24, 36, 36],
            "is_correct": [True, False, True, False],
            "think_final_margin": [0.3, -0.1, 0.2, -0.2],
        }
    )

    out = tmp_path / "margin.png"
    plot_think_final_margin(features, out)

    assert out.is_file()
    assert out.stat().st_size > 0
