import numpy as np
import pandas as pd

from vertical.statistics import (
    summarize_layer_auc,
    summarize_outcome_profiles,
    summarize_policy_profiles,
    within_question_auc,
)


METRIC = "v1_raw_update_norm"


def profile_frame(rows):
    defaults = {
        "model_family": "qwen3",
        "model_name": "Qwen3-Test",
        "condition": "base",
        "checkpoint": None,
        "global_step": None,
        "training_progress": None,
        "stage": 0,
        "representation": "last_s32",
        "layer_index": 1,
        "relative_depth": 0.5,
        f"profile_coverage_count_{METRIC}": 1,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_policy_summary_weights_questions_equally():
    frame = profile_frame(
        [
            {"record_id": "q1:r0", "question_id": "q1", "rollout_id": "r0", METRIC: 0.0},
            {"record_id": "q1:r1", "question_id": "q1", "rollout_id": "r1", METRIC: 2.0},
            {"record_id": "q2:r0", "question_id": "q2", "rollout_id": "r0", METRIC: 3.0},
        ]
    )

    result = summarize_policy_profiles(frame, METRIC)

    assert result["n_questions"].iat[0] == 2
    assert result["n_rollouts"].iat[0] == 3
    assert result["profile_mean"].iat[0] == 2.0


def test_outcome_summary_uses_only_mixed_questions():
    frame = profile_frame(
        [
            {"record_id": "q1:w", "question_id": "q1", "rollout_id": "w", "is_correct": False, METRIC: 1.0},
            {"record_id": "q1:c", "question_id": "q1", "rollout_id": "c", "is_correct": True, METRIC: 3.0},
            {"record_id": "q2:c", "question_id": "q2", "rollout_id": "c", "is_correct": True, METRIC: 100.0},
        ]
    )

    result = summarize_outcome_profiles(frame, METRIC)

    assert result["n_mixed_questions"].iat[0] == 1
    assert result["correct_minus_wrong"].iat[0] == 2.0


def test_within_question_auc_handles_ties_and_excludes_single_outcome():
    frame = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q1", "q1", "q2"],
            "is_correct": [False, False, True, True, True],
            "score": [0.0, 1.0, 1.0, 2.0, 5.0],
        }
    )

    result = within_question_auc(frame, score_column="score")

    assert result["question_id"].tolist() == ["q1"]
    assert result["pair_count"].iat[0] == 4
    assert result["auc"].iat[0] == 0.875


def test_layer_auc_reports_unsupported_without_fabricating_value():
    frame = profile_frame(
        [
            {"record_id": "q1:r0", "question_id": "q1", "rollout_id": "r0", "is_correct": True, METRIC: 1.0},
            {"record_id": "q2:r0", "question_id": "q2", "rollout_id": "r0", "is_correct": True, METRIC: 2.0},
        ]
    )

    result = summarize_layer_auc(frame, METRIC)

    assert result["n_mixed_questions"].iat[0] == 0
    assert np.isnan(result["question_equal_auc"].iat[0])
    assert result["unsupported_reason"].iat[0] == "no mixed-outcome questions"

