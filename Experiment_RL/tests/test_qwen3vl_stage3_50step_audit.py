import json
from pathlib import Path

from scripts.qwen3vl_stage3_50step_audit import (
    paired_outcome_counts,
    parse_c2_training_log,
    summarize_eval_pair,
)


def write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_paired_outcome_counts_tracks_flips_and_prediction_delta(tmp_path):
    a1_rows = [
        {"question_id": "q1", "answer": "A", "prediction": "A", "correct": True, "output_token_count": 10, "truncated_by_length": False},
        {"question_id": "q2", "answer": "B", "prediction": "C", "correct": False, "output_token_count": 20, "truncated_by_length": False},
        {"question_id": "q3", "answer": "C", "prediction": "C", "correct": True, "output_token_count": 30, "truncated_by_length": True},
    ]
    c2_rows = [
        {"question_id": "q1", "answer": "A", "prediction": "D", "correct": False, "output_token_count": 12, "truncated_by_length": False},
        {"question_id": "q2", "answer": "B", "prediction": "B", "correct": True, "output_token_count": 25, "truncated_by_length": True},
        {"question_id": "q3", "answer": "C", "prediction": "C", "correct": True, "output_token_count": 30, "truncated_by_length": True},
    ]

    paired = paired_outcome_counts(a1_rows, c2_rows)

    assert paired["outcome_counts"] == {
        "both_wrong": 0,
        "a1_wrong_c2_right": 1,
        "a1_right_c2_wrong": 1,
        "both_right": 1,
    }
    assert paired["prediction_delta"]["D"] == 1
    assert paired["prediction_delta"]["A"] == -1
    assert paired["length_delta_mean"] == 7 / 3
    assert paired["truncation_pairs"]["False->True"] == 1


def test_parse_c2_training_log_aggregates_group_diagnostics():
    log_text = "\n".join(
        [
            "step:1 - c2_frac_groups_with_nonzero_b_gain_std:1.0 - c2_frac_groups_where_b_gain_changes_ranking:1.0 - c2_all_same_answer_groups:1.0 - c2_frac_all_same_answer_groups_with_b_gain_ranking:1.0 - c2_within_group_corr_answer_b_gain_mean:0.0 - c2_1/actor_option_probe_failed_mean:0.0 - c2_1/actor_option_gain_raw_mean:0.1 - c2_1/actor_option_gain_raw_std:0.2 - response_length/mean:1000.0",
            "step:2 - c2_frac_groups_with_nonzero_b_gain_std:0.0 - c2_frac_groups_where_b_gain_changes_ranking:0.5 - c2_all_same_answer_groups:0.0 - c2_frac_all_same_answer_groups_with_b_gain_ranking:0.0 - c2_within_group_corr_answer_b_gain_mean:0.5 - c2_1/actor_option_probe_failed_mean:0.0 - c2_1/actor_option_gain_raw_mean:-0.1 - c2_1/actor_option_gain_raw_std:0.3 - response_length/mean:2000.0",
        ]
    )

    summary = parse_c2_training_log(log_text)

    assert summary["num_logged_steps"] == 2
    assert summary["latest_step"] == 2
    assert summary["c2_frac_groups_with_nonzero_b_gain_std"]["mean"] == 0.5
    assert summary["c2_frac_groups_where_b_gain_changes_ranking"]["last"] == 0.5
    assert summary["c2_1/actor_option_probe_failed_mean"]["max"] == 0.0
    assert summary["response_length/mean"]["last"] == 2000.0


def test_summarize_eval_pair_combines_summary_and_generations(tmp_path):
    a1_dir = tmp_path / "a1"
    c2_dir = tmp_path / "c2"
    a1_dir.mkdir()
    c2_dir.mkdir()
    (a1_dir / "summary.json").write_text(json.dumps({"accuracy": 0.5, "balanced_accuracy": 0.6}), encoding="utf-8")
    (c2_dir / "summary.json").write_text(json.dumps({"accuracy": 0.75, "balanced_accuracy": 0.7}), encoding="utf-8")
    write_jsonl(
        a1_dir / "generations.jsonl",
        [
            {"question_id": "q1", "answer": "A", "prediction": "A", "correct": True, "output_token_count": 10, "truncated_by_length": False},
            {"question_id": "q2", "answer": "B", "prediction": "C", "correct": False, "output_token_count": 20, "truncated_by_length": False},
        ],
    )
    write_jsonl(
        c2_dir / "generations.jsonl",
        [
            {"question_id": "q1", "answer": "A", "prediction": "A", "correct": True, "output_token_count": 11, "truncated_by_length": False},
            {"question_id": "q2", "answer": "B", "prediction": "B", "correct": True, "output_token_count": 19, "truncated_by_length": False},
        ],
    )

    result = summarize_eval_pair(a1_dir, c2_dir, c2_log_text="")

    assert result["headline"]["accuracy_delta"] == 0.25
    assert result["headline"]["balanced_accuracy_delta"] == 0.1
    assert result["paired"]["outcome_counts"]["a1_wrong_c2_right"] == 1
