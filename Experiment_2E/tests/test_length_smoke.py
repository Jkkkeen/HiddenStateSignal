import pandas as pd

from experiment_2e.length_smoke import analyze_length_records, trajectory_point_count


def test_trajectory_point_count_includes_final_endpoint_once():
    assert trajectory_point_count(100, window=128, stride=64) == 1
    assert trajectory_point_count(832, window=128, stride=64) == 12
    assert trajectory_point_count(900, window=128, stride=64) == 14


def test_stride_64_is_frozen_when_median_has_at_least_12_points():
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "rollout_slot": 0, "token_count": 832, "finish_reason": "stop", "answer_reward": 1},
            {"question_id": "q1", "rollout_slot": 1, "token_count": 900, "finish_reason": "stop", "answer_reward": 0},
            {"question_id": "q2", "rollout_slot": 0, "token_count": 832, "finish_reason": "stop", "answer_reward": 0},
            {"question_id": "q2", "rollout_slot": 1, "token_count": 832, "finish_reason": "stop", "answer_reward": 0},
        ]
    )
    summary = analyze_length_records(frame)
    assert summary["frozen_stride"] == 64
    assert summary["eval_mixed_question_rate_n4"] == 0.5
    assert summary["truncation_rate"] == 0.0


def test_stride_32_is_frozen_and_length_finishes_are_excluded():
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "rollout_slot": 0, "token_count": 400, "finish_reason": "stop", "answer_reward": 1},
            {"question_id": "q1", "rollout_slot": 1, "token_count": 1536, "finish_reason": "length", "answer_reward": 0},
        ]
    )
    summary = analyze_length_records(frame)
    assert summary["frozen_stride"] == 32
    assert summary["n_non_truncated"] == 1
    assert summary["truncation_rate"] == 0.5
