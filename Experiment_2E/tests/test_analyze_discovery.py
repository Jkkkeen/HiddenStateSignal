from argparse import Namespace

import pandas as pd
import pytest

from experiment_2e.analyze_discovery import primary_selection_audit, run_analysis


def test_primary_selection_audit_rejects_incomplete_formal_input():
    frame = pd.DataFrame(
        {
            "axis": ["horizontal"],
            "family_id": ["H1"],
            "family": ["movement"],
            "representation": ["mean_w128_s32"],
            "anchor_layer": [3],
            "aggregation_mode": ["local"],
            "metric": ["median_relative_movement"],
            "checkpoint": ["base"],
            "rollout_id": ["r1"],
            "question_id": ["q1"],
        }
    )
    with pytest.raises(ValueError, match="checkpoints"):
        primary_selection_audit(frame, allow_incomplete=False)
    audit = primary_selection_audit(frame, allow_incomplete=True)
    assert audit["n_primary_tests"] == 1


def test_incomplete_analysis_runs_end_to_end(tmp_path):
    rows = []
    controls = []
    for checkpoint, progress in (("base", 0.0), ("final", 1.0)):
        for question in ("q1", "q2"):
            for slot in range(2):
                rollout_id = f"{checkpoint}-{question}-{slot}"
                common = {
                    "run_id": "synthetic",
                    "checkpoint": checkpoint,
                    "training_progress": progress,
                    "question_id": question,
                    "rollout_id": rollout_id,
                    "rollout_slot": slot,
                    "is_correct": bool(slot),
                    "response_length": 128 + slot,
                    "policy_entropy": 1.0 + slot * 0.1,
                    "token_logprob_mean": -1.0,
                    "hidden_norm_mean": 2.0,
                    "truncated": False,
                    "level": "Level 3",
                    "subject": "Algebra",
                }
                rows.append(
                    {
                        **common,
                        "axis": "horizontal",
                        "family_id": "H1",
                        "family": "movement",
                        "representation": "mean_w128_s32",
                        "anchor_layer": 3,
                        "aggregation_mode": "local",
                        "metric": "median_relative_movement",
                        "stage": 0,
                        "value": float(slot + progress),
                        "is_primary_metric": True,
                        "coverage": True,
                    }
                )
                controls.append({**common, "stage": 0})
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pd.DataFrame(rows).to_parquet(metrics_dir / "horizontal_metrics_base.parquet", index=False)
    pd.DataFrame(controls).to_parquet(metrics_dir / "controls_base.parquet", index=False)
    output = tmp_path / "output"
    audit = run_analysis(
        Namespace(
            metrics_dir=metrics_dir,
            output_dir=output,
            bootstrap=50,
            seed=5,
            allow_incomplete=True,
            skip_models=True,
            skip_oof=True,
        )
    )
    assert audit["n_primary_tests"] == 1
    assert audit["n_figures"] == 2
    assert (output / "REPORT.html").exists()
    assert (output / "analysis" / "outcome_effects.csv").exists()
