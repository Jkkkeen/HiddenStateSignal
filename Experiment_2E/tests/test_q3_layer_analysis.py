import json

import numpy as np
import pandas as pd

from experiment_2e.q3_layer_analysis import (
    analyze_profiles,
    summarize_depth_change,
    summarize_layer_auc,
    summarize_outcome_profiles,
    summarize_policy_profiles,
)
from experiment_2e.vertical_profiles import PROFILE_COLUMNS


def _profile_frame():
    rows = []
    for global_step in (0, 50):
        for question in range(4):
            for slot in range(4):
                correct = slot >= 2
                for layer in range(4):
                    row = {
                        "model": "Qwen3-test",
                        "checkpoint": f"step{global_step:03d}",
                        "global_step": global_step,
                        "training_progress": global_step / 50,
                        "question_id": f"q{question}",
                        "rollout_id": f"s{global_step}-q{question}-r{slot}",
                        "rollout_slot": slot,
                        "is_correct": correct,
                        "stage": 0,
                        "representation": "mean_w128_s32",
                        "layer_index": layer,
                        "relative_depth": layer / 3,
                    }
                    value = float(correct) + (global_step / 50) * layer
                    for metric in PROFILE_COLUMNS:
                        row[metric] = value
                        row[f"profile_coverage_count_{metric}"] = 1
                    rows.append(row)
    return pd.DataFrame(rows)


def test_question_equal_policy_outcome_and_auc():
    frame = _profile_frame()
    metric = "v1_relative_update_norm"
    policy = summarize_policy_profiles(frame, metric)
    assert policy["n_questions"].min() == 4
    assert policy["n_rollouts"].min() == 16

    outcome = summarize_outcome_profiles(frame, metric)
    assert set(outcome["correct_minus_wrong"]) == {1.0}
    assert set(outcome["n_mixed_questions"]) == {4}

    auc = summarize_layer_auc(frame, metric)
    assert set(auc["question_equal_auc"]) == {1.0}
    assert set(auc["n_mixed_questions"]) == {4}


def test_depth_summary_uses_absolute_change_from_base():
    policy = summarize_policy_profiles(_profile_frame(), "v1_relative_update_norm")
    depth = summarize_depth_change(
        policy,
        metric="v1_relative_update_norm",
        base_step=0,
    )
    base = depth.loc[depth["global_step"] == 0]
    trained = depth.loc[depth["global_step"] == 50]
    assert base["depth_mass"].eq(0.0).all()
    assert trained["peak_depth"].eq(1.0).all()
    assert trained["depth_center"].between(0.0, 1.0).all()
    assert np.allclose(
        trained[["early_mass", "middle_mass", "late_mass"]].sum(axis=1),
        1.0,
    )


def test_analysis_writes_tables_figures_and_passing_audit(tmp_path):
    profiles_dir = tmp_path / "metrics"
    generation_dir = tmp_path / "generations"
    output_dir = tmp_path / "analysis"
    profiles_dir.mkdir()
    generation_dir.mkdir()
    frame = _profile_frame()
    for step in (0, 50):
        frame.loc[frame["global_step"] == step].to_parquet(
            profiles_dir / f"layer_profiles_step{step:03d}.parquet",
            index=False,
        )
        (profiles_dir / f"extraction_audit_step{step:03d}.json").write_text(
            json.dumps({"passed": True, "elapsed_seconds": 2.0}),
            encoding="utf-8",
        )
        generation_rows = [
            {"acc": bool(slot >= 2), "step": step}
            for _question in range(4)
            for slot in range(4)
        ]
        (generation_dir / f"{step}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in generation_rows),
            encoding="utf-8",
        )

    audit = analyze_profiles(
        profiles_dir=profiles_dir,
        generation_dir=generation_dir,
        output_dir=output_dir,
        expected_steps=[0, 50],
        expected_questions=4,
        metrics=("v1_relative_update_norm",),
    )

    assert audit["passed"] is True
    assert audit["figure_count"] == 4
    assert all(path.stat().st_size > 10_000 for path in (output_dir / "figures").glob("*.png"))
    assert (output_dir / "policy_profiles.parquet").is_file()
    assert (output_dir / "analysis_audit.json").is_file()
