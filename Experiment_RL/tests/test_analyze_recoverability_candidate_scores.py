import math

import pandas as pd

from scripts.analyze_recoverability_candidate_scores import (
    FEATURE_SPECS,
    analyze_candidate_features,
    build_analysis_frame,
    paired_feature_comparisons,
)


def _feature_spec(name: str) -> dict[str, str]:
    return next(spec for spec in FEATURE_SPECS if spec["name"] == name)


def test_build_analysis_frame_derives_letter_features_and_merges_content_scores():
    recovery = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "rollout_id": 1,
                "answer": "B",
                "original_prediction": "A",
                "recovery_rate": 0.75,
                "any_recovery": True,
            }
        ]
    )
    probes = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "rollout_id": 1,
                "answer": "B",
                "pred_answer": "A",
                "logit_A": -0.2,
                "logit_B": -1.2,
                "logit_C": -2.2,
                "logit_D": -3.2,
            }
        ]
    )
    content = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "rollout_id": 1,
                "content_gold_margin": -0.4,
                "content_option_labels": "ABCD",
                "content_score_A": -0.2,
                "content_score_B": -1.2,
                "content_score_C": -2.2,
                "content_score_D": -3.2,
                "content_calibrated_score_A": 0.8,
                "content_calibrated_score_B": -0.2,
                "content_calibrated_score_C": -1.2,
                "content_calibrated_score_D": -2.2,
                "content_commitment_gap": 0.4,
                "content_top2_gap": 0.4,
                "content_entropy": 1.0,
                "content_calibrated_gold_margin": -0.6,
                "content_calibrated_commitment_gap": 0.6,
                "content_calibrated_top2_gap": 0.3,
                "content_calibrated_entropy": 1.1,
            }
        ]
    )

    row = build_analysis_frame(recovery, probes, content).iloc[0]

    assert math.isclose(row["letter_gold_margin"], -1.0)
    assert math.isclose(row["letter_gold_margin_mean"], 2.0 / 3.0)
    assert math.isclose(row["letter_commitment_gap"], 1.0)
    assert math.isclose(row["letter_top2_gap"], 1.0)
    assert row["letter_top_option"] == "A"
    assert row["content_gold_margin"] == -0.4
    assert math.isclose(row["content_gold_margin_mean"], 2.0 / 3.0)
    assert math.isclose(row["content_calibrated_gold_margin_mean"], 2.0 / 3.0)
    assert row["recovery_rate"] == 0.75


def test_feature_specs_keep_both_mean_gold_support_and_low_margin_hypotheses():
    support = _feature_spec("letter_gold_mean_support")
    low_margin = _feature_spec("letter_low_gold_mean_margin")

    assert support["column"] == low_margin["column"] == "letter_gold_margin_mean"
    assert support["direction"] == "pos"
    assert low_margin["direction"] == "neg"


def test_analyze_candidate_features_orients_lower_commitment_as_more_recoverable():
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "recovery_rate": 1.0, "any_recovery": True, "letter_commitment_gap": 0.1},
            {"question_id": "q1", "recovery_rate": 0.0, "any_recovery": False, "letter_commitment_gap": 2.0},
            {"question_id": "q2", "recovery_rate": 0.5, "any_recovery": True, "letter_commitment_gap": 0.2},
            {"question_id": "q2", "recovery_rate": 0.0, "any_recovery": False, "letter_commitment_gap": 1.5},
        ]
    )
    spec = _feature_spec("letter_commitment")

    summary, per_question = analyze_candidate_features(
        frame,
        specs=[spec],
        n_boot=100,
        seed=7,
    )

    assert summary.iloc[0]["mean_within_pairwise_auc"] == 1.0
    assert set(per_question["within_pairwise_auc"]) == {1.0}


def test_paired_feature_comparisons_bootstraps_question_level_auc_differences():
    per_question = pd.DataFrame(
        [
            {"question_id": "q1", "feature": "left", "within_pairwise_auc": 0.8},
            {"question_id": "q2", "feature": "left", "within_pairwise_auc": 0.6},
            {"question_id": "q1", "feature": "right", "within_pairwise_auc": 0.5},
            {"question_id": "q2", "feature": "right", "within_pairwise_auc": 0.5},
        ]
    )

    result = paired_feature_comparisons(
        per_question,
        comparisons=[{"name": "left_minus_right", "left": "left", "right": "right"}],
        n_boot=100,
        seed=3,
    )

    assert result.iloc[0]["valid_questions"] == 2
    assert math.isclose(result.iloc[0]["mean_auc_delta"], 0.2)
