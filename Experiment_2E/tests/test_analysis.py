import numpy as np
import pandas as pd

from experiment_2e.analysis import (
    apply_base_standardizers,
    auc_from_scores,
    benjamini_hochberg,
    fit_base_standardizers,
    grouped_oof_length_increment,
    question_bootstrap_mean_ci,
    select_primary_metrics,
    summarize_outcome_auc,
)


def _metric_frame() -> pd.DataFrame:
    rows = []
    for checkpoint, progress in (("base", 0.0), ("final", 1.0)):
        for question in range(6):
            for slot in range(4):
                correct = slot >= 2
                rows.append(
                    {
                        "axis": "horizontal",
                        "family_id": "H1",
                        "family": "movement",
                        "representation": "mean_w128_s32",
                        "anchor_layer": 3,
                        "aggregation_mode": "local",
                        "metric": "median_relative_movement",
                        "stage": 0,
                        "checkpoint": checkpoint,
                        "training_progress": progress,
                        "question_id": f"q{question}",
                        "rollout_id": f"{checkpoint}-{question}-{slot}",
                        "is_correct": correct,
                        "response_length": 100 + 5 * slot,
                        "policy_entropy": 1.0 + 0.01 * slot,
                        "level": "Level 3",
                        "value": float(correct) + progress,
                        "is_primary_metric": True,
                        "coverage": True,
                        "truncated": False,
                    }
                )
    return pd.DataFrame(rows)


def test_primary_selection_and_base_standardization():
    frame = _metric_frame()
    sensitivity = frame.iloc[[0]].copy()
    sensitivity["representation"] = "last_s32"
    selected = select_primary_metrics(pd.concat([frame, sensitivity], ignore_index=True))
    assert len(selected) == len(frame)
    calibrators = fit_base_standardizers(selected)
    standardized = apply_base_standardizers(selected, calibrators)
    base = standardized.loc[standardized["checkpoint"] == "base", "z_value"]
    assert np.isclose(base.mean(), 0.0)
    assert np.isclose(base.std(ddof=0), 1.0)


def test_auc_handles_ties_and_outcome_summary():
    assert auc_from_scores([False, False, True, True], [0.0, 1.0, 1.0, 2.0]) == 0.875
    frame = _metric_frame()
    calibrated = apply_base_standardizers(frame, fit_base_standardizers(frame))
    summary = summarize_outcome_auc(calibrated, n_bootstrap=100, seed=7)
    assert set(summary["question_equal_auc"]) == {1.0}
    assert set(summary["pair_weighted_auc"]) == {1.0}
    assert set(summary["n_pairs"]) == {24}


def test_question_bootstrap_and_bh_are_deterministic():
    first = question_bootstrap_mean_ci(pd.Series([1.0, 2.0, 3.0]), n_bootstrap=100, seed=9)
    second = question_bootstrap_mean_ci(pd.Series([1.0, 2.0, 3.0]), n_bootstrap=100, seed=9)
    assert first == second
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_grouped_oof_increment_uses_question_groups():
    frame = _metric_frame()
    result = grouped_oof_length_increment(frame, metric_column="value", n_splits=3, seed=4)
    assert result["fit_ok"]
    assert result["n_questions"] == 6
    assert result["question_equal_extended_auc"] == 1.0
    assert result["question_equal_delta_auc"] >= 0.0
