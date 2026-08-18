import json

import numpy as np
import pandas as pd

from vertical.plotting import render_v01_report


METRIC = "v1_raw_update_norm"


def make_profile_path(tmp_path):
    rows = []
    for condition, model, offset in (
        ("base", "MiMo-Base", 0.0),
        ("sft", "MiMo-SFT", 0.4),
    ):
        for question in ("q1", "q2"):
            for rollout, correct, outcome_shift in (("r0", False, 0.0), ("r1", True, 0.2)):
                for layer in range(3):
                    rows.append(
                        {
                            "record_id": f"{condition}:{question}:{rollout}",
                            "model_family": "mimo",
                            "model_name": model,
                            "condition": condition,
                            "checkpoint": None,
                            "global_step": None,
                            "training_progress": None,
                            "question_id": question,
                            "rollout_id": rollout,
                            "is_correct": correct,
                            "stage": 0,
                            "representation": "last_s32",
                            "layer_index": layer,
                            "relative_depth": layer / 2,
                            METRIC: np.nan if layer == 0 else layer + offset + outcome_shift,
                            f"profile_coverage_count_{METRIC}": 0 if layer == 0 else 1,
                        }
                    )
    path = tmp_path / "profiles.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_v01_report_writes_tables_nonblank_figures_and_audit(tmp_path):
    profile_path = make_profile_path(tmp_path)
    output = tmp_path / "analysis"

    audit = render_v01_report(profile_path, output, [METRIC])

    assert audit.passed
    assert (output / "policy_profiles.parquet").is_file()
    assert (output / "outcome_profiles.parquet").is_file()
    assert (output / "layer_auc.parquet").is_file()
    assert (output / "depth_summaries.parquet").is_file()
    figures = list((output / "figures").glob("*.png"))
    assert len(figures) == 4
    assert all(path.stat().st_size > 10_000 for path in figures)
    saved = json.loads((output / "analysis_audit.json").read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["figure_count"] == 4


def test_v01_report_rejects_missing_metric(tmp_path):
    profile_path = make_profile_path(tmp_path)

    try:
        render_v01_report(profile_path, tmp_path / "analysis", ["missing_metric"])
    except ValueError as exc:
        assert "missing profile metric" in str(exc)
    else:
        raise AssertionError("missing metric should fail")

