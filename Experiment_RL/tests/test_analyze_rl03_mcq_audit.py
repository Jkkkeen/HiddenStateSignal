import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.analyze_rl03_mcq_audit import (
    _logistic_predictions,
    _gate_metrics,
    _spearman,
    assign_stage_a_gate,
    build_dense_tables,
    build_probe_features,
    build_rollout_features,
    build_step_features,
    cross_fitted_level_gain_comparison,
    gain_level_diagnostics,
    paired_auc_bootstrap,
    question_equal_auc,
    run_analysis,
    smoke_gate,
    within_question_auc,
)


def test_analysis_module_imports_when_sklearn_is_unavailable():
    project_dir = Path(__file__).resolve().parents[1]
    code = """
import builtins
real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'sklearn' or name.startswith('sklearn.'):
        raise ModuleNotFoundError('blocked sklearn import')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import scripts.analyze_rl03_mcq_audit
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _scores():
    rows = []
    for rollout_id, correct, early, late in [(0, False, -3.0, -2.0), (1, True, -3.0, -0.5)]:
        for frac, gold_score in [(0.25, early), (0.90, late)]:
            for label, offset in [("A", -1.0), ("B", 0.0), ("C", -2.0), ("D", -3.0)]:
                rows.append(
                    {
                        "request_id": f"q1:{rollout_id}:{frac}:{label}",
                        "question_id": "q1",
                        "rollout_id": rollout_id,
                        "is_correct": correct,
                        "gold_letter": "B",
                        "probe_id": f"think_{frac:.2f}",
                        "probe_kind": "think",
                        "frac": frac,
                        "interface_id": "I1",
                        "representation": "content_sequence",
                        "target_id": f"option_{label}",
                        "is_gold_target": label == "B",
                        "score_mean": gold_score if label == "B" else gold_score + offset,
                    }
                )
    return pd.DataFrame(rows)


def test_probe_and_step_features_keep_gold_level_and_contrastive_margin_separate():
    probes = build_probe_features(_scores())
    steps = build_step_features(_scores())

    correct_late = probes[(probes["rollout_id"] == 1) & (probes["frac"] == 0.90)].iloc[0]
    assert correct_late["gold_level"] == -0.5
    assert correct_late["gold_margin"] == 1.0
    correct_step = steps[steps["rollout_id"] == 1].iloc[0]
    assert correct_step["level_gain"] == 2.5
    assert correct_step["level_gain_rate"] == 2.5 / 0.65


def test_prompt_modes_remain_separate_analysis_cells():
    legacy = _scores().assign(prompt_mode="legacy_letter_instruction")
    neutral = _scores().assign(prompt_mode="neutralized_letter_instruction")

    features = build_rollout_features(pd.concat([legacy, neutral], ignore_index=True))

    assert len(features) == 4
    assert set(features["prompt_mode"]) == {
        "legacy_letter_instruction",
        "neutralized_letter_instruction",
    }


def test_legacy_anchor_uses_letter_margin_gain_from_5_to_100_percent():
    rows = []
    for frac, gold_score in ((0.05, -2.0), (1.0, 0.0)):
        for label, score in (("A", -3.0), ("B", gold_score), ("C", -4.0), ("D", -5.0)):
            rows.append(
                {
                    "question_id": "q1",
                    "rollout_id": 0,
                    "is_correct": True,
                    "gold_letter": "B",
                    "probe_id": f"think_{frac:.2f}",
                    "probe_kind": "think",
                    "frac": frac,
                    "interface_id": "I0",
                    "representation": "letter_first",
                    "prompt_mode": "legacy_letter_instruction",
                    "target_id": f"letter_{label}",
                    "is_gold_target": label == "B",
                    "score_mean": score,
                }
            )

    feature = build_rollout_features(pd.DataFrame(rows)).iloc[0]

    assert feature["gold_margin_5"] == 1.0
    assert feature["gold_margin_100"] == 3.0
    assert feature["letter_margin_gain_5_100"] == 2.0


def test_endpoint_gain_is_median_of_per_surface_gains():
    rows = []
    scores_by_probe = {
        ("think", 0.25): [0.0, 100.0, 101.0],
        ("think", 0.90): [50.0, 51.0, 102.0],
    }
    for (probe_kind, frac), surface_scores in scores_by_probe.items():
        for surface_index in (2, 0, 1):
            score = surface_scores[surface_index]
            rows.append(
                {
                    "question_id": "q1",
                    "rollout_id": 0,
                    "is_correct": True,
                    "gold_letter": "B",
                    "probe_id": f"{probe_kind}_{frac}",
                    "probe_kind": probe_kind,
                    "frac": frac,
                    "interface_id": "I1",
                    "representation": "content_sequence",
                    "target_id": f"gold_surface_{surface_index}",
                    "surface_id": f"gold_surface_{surface_index}",
                    "is_gold_target": True,
                    "target_text": f"surface {surface_index}",
                    "score_mean": score,
                }
            )

    features = build_rollout_features(pd.DataFrame(rows)).iloc[0]

    assert features["gold_level_25"] == 100.0
    assert features["gold_level_90"] == 51.0
    assert features["gold_gain_25_90"] == 1.0
    assert features["gold_gain_25_90_canonical"] == 50.0
    assert features["gold_trajectory_slope"] == pytest.approx(1.0 / 0.65)
    assert features["gold_gain_25_90"] != features["gold_level_90"] - features["gold_level_25"]
    assert "polyfit" not in inspect.getsource(build_rollout_features)


def test_dense_steps_exclude_trimmed_final_and_use_per_surface_gains():
    rows = []
    probes = [
        ("prompt_0.00", "prompt", 0.0, [0.0, 100.0, 101.0]),
        ("think_0.25", "think", 0.25, [25.0, 75.0, 102.0]),
        ("trimmed_final", "trimmed", None, [1000.0, 1000.0, 1000.0]),
    ]
    for probe_id, probe_kind, frac, surface_scores in probes:
        for surface_index, score in enumerate(surface_scores):
            rows.append(
                {
                    "question_id": "q1",
                    "rollout_id": 0,
                    "is_correct": True,
                    "gold_letter": "B",
                    "probe_id": probe_id,
                    "probe_kind": probe_kind,
                    "frac": frac,
                    "interface_id": "I1",
                    "representation": "content_sequence",
                    "target_id": f"gold_surface_{surface_index}",
                    "surface_id": f"gold_surface_{surface_index}",
                    "is_gold_target": True,
                    "target_text": f"surface {surface_index}",
                    "score_mean": score,
                }
            )

    steps = build_step_features(pd.DataFrame(rows))

    assert len(steps) == 1
    assert steps.iloc[0]["frac_start"] == 0.0
    assert steps.iloc[0]["frac_end"] == 0.25
    assert steps.iloc[0]["step_gain"] == 1.0
    assert steps.iloc[0]["step_gain_rate"] == 4.0


def test_within_question_auc_is_question_equal():
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "is_correct": False, "feature": 0.0},
            {"question_id": "q1", "is_correct": True, "feature": 1.0},
            {"question_id": "q2", "is_correct": False, "feature": 2.0},
            {"question_id": "q2", "is_correct": True, "feature": 1.0},
        ]
    )
    assert within_question_auc(frame, "feature") == 0.5


def test_question_equal_auc_ignores_unequal_rollout_and_pair_counts():
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "is_correct": False, "feature": 0.0},
            {"question_id": "q1", "is_correct": True, "feature": 1.0},
            {"question_id": "q2", "is_correct": False, "feature": 3.0},
            {"question_id": "q2", "is_correct": False, "feature": 4.0},
            {"question_id": "q2", "is_correct": False, "feature": 5.0},
            {"question_id": "q2", "is_correct": True, "feature": 1.0},
            {"question_id": "q2", "is_correct": True, "feature": 2.0},
            {"question_id": "all_correct", "is_correct": True, "feature": 99.0},
        ]
    )

    result = question_equal_auc(frame, "feature")

    assert result["mean_auc"] == 0.5
    assert result["mixed_questions"] == 2
    assert result["per_question"].set_index("question_id")["auc"].to_dict() == {
        "q1": 1.0,
        "q2": 0.0,
    }


def test_question_equal_auc_awards_half_credit_to_ties():
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "is_correct": False, "feature": 2.0},
            {"question_id": "q1", "is_correct": True, "feature": 2.0},
        ]
    )

    result = question_equal_auc(frame, "feature")

    assert result["mean_auc"] == 0.5
    assert result["per_question"].iloc[0]["pairs"] == 1


def test_paired_question_bootstrap_is_deterministic_and_preserves_pairs():
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "is_correct": False, "gain": 0.0, "level": 0.0},
            {"question_id": "q1", "is_correct": True, "gain": 1.0, "level": 1.0},
            {"question_id": "q2", "is_correct": False, "gain": 0.0, "level": 1.0},
            {"question_id": "q2", "is_correct": True, "gain": 1.0, "level": 0.0},
            {"question_id": "q3", "is_correct": False, "gain": 1.0, "level": 0.0},
            {"question_id": "q3", "is_correct": True, "gain": 0.0, "level": 1.0},
            {"question_id": "q4", "is_correct": False, "gain": 0.0, "level": 0.0},
            {"question_id": "q4", "is_correct": True, "gain": 0.0, "level": 0.0},
        ]
    )

    first = paired_auc_bootstrap(frame, "gain", "level", bootstrap_samples=500, seed=17)
    second = paired_auc_bootstrap(frame, "gain", "level", bootstrap_samples=500, seed=17)

    assert first == second
    assert first["valid_questions"] == 4
    assert first["mean_delta_auc"] == 0.0
    assert first["ci_low"] <= 0.0 <= first["ci_high"]


def test_paired_auc_bootstrap_handles_no_mixed_questions():
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "is_correct": True, "gain": 1.0, "level": 2.0},
            {"question_id": "q2", "is_correct": False, "gain": 0.0, "level": 1.0},
        ]
    )

    result = paired_auc_bootstrap(frame, "gain", "level", bootstrap_samples=50, seed=1)

    assert result["valid_questions"] == 0
    assert pd.isna(result["mean_delta_auc"])


def test_gate_is_level_equivalent_when_gain_auc_is_high_but_delta_ci_includes_zero():
    gate = assign_stage_a_gate(
        {
            "pipeline_pass": True,
            "gain_auc": 0.75,
            "gain_auc_ci_low": 0.65,
            "paired_delta_auc": 0.01,
            "paired_delta_ci_low": -0.01,
            "paired_delta_ci_high": 0.03,
            "crossfit_improvement": 0.01,
            "level_auc": 0.78,
        }
    )

    assert gate["label"] == "LEVEL-EQUIVALENT"
    assert gate["provisional"] is False
    assert gate["passed"] is False


def test_missing_a0_controls_make_process_candidate_provisional_not_passed():
    gate = assign_stage_a_gate(
        {
            "pipeline_pass": True,
            "gain_auc": 0.75,
            "gain_auc_ci_low": 0.65,
        }
    )

    assert gate["label"] == "PASS-PROCESS"
    assert gate["provisional"] is True
    assert gate["passed"] is False
    assert "label_permutation_auc_change" in gate["missing_controls"]


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({"pipeline_pass": False}, "PIPELINE-FAIL"),
        (
            {
                "pipeline_pass": True,
                "gain_auc": 0.75,
                "gain_auc_ci_low": 0.65,
                "gain_auc_ci_high": 0.85,
                "positive_correlation_fraction": 0.70,
                "i1_i2_auc_difference": 0.02,
                "trimmed_auc": 0.68,
                "label_permutation_auc_change": 0.01,
                "paired_delta_auc": 0.03,
                "paired_delta_ci_low": 0.01,
                "paired_delta_ci_high": 0.05,
                "crossfit_improvement": 0.02,
            },
            "PASS-PROCESS",
        ),
        ({"pipeline_pass": True, "gain_auc": 0.67, "gain_auc_ci_low": 0.61}, "BORDERLINE-PROCESS"),
        (
            {
                "pipeline_pass": True,
                "gain_auc": 0.60,
                "gain_auc_ci_low": 0.45,
                "gain_auc_ci_high": 0.65,
                "level_auc": 0.80,
            },
            "VERIFIER-ONLY",
        ),
        (
            {
                "pipeline_pass": True,
                "gain_auc": 0.60,
                "gain_auc_ci_low": 0.45,
                "gain_auc_ci_high": 0.65,
                "level_auc": 0.60,
            },
            "NO-SIGNAL",
        ),
    ],
)
def test_required_gate_labels(metrics, expected):
    assert assign_stage_a_gate(metrics)["label"] == expected


def test_i0_anchor_below_reproduction_threshold_is_pipeline_fail():
    gate = assign_stage_a_gate(
        {
            "pipeline_pass": True,
            "i0_letter_gain_auc": 0.69,
            "gain_auc": 0.75,
            "gain_auc_ci_low": 0.65,
        }
    )

    assert gate["label"] == "PIPELINE-FAIL"


def test_interface_and_representation_gates_use_matched_prompt_modes():
    interface_gate = assign_stage_a_gate(
        {
            "pipeline_pass": True,
            "i0_letter_gain_auc": 0.75,
            "legacy_interface_letter_gain_auc": 0.60,
            "neutral_letter_gain_auc": 0.80,
            "content_margin_gain_auc": 0.80,
        }
    )
    representation_gate = assign_stage_a_gate(
        {
            "pipeline_pass": True,
            "i0_letter_gain_auc": 0.75,
            "legacy_interface_letter_gain_auc": 0.75,
            "neutral_letter_gain_auc": 0.75,
            "content_margin_gain_auc": 0.55,
        }
    )

    assert interface_gate["label"] == "INTERFACE-FAIL"
    assert representation_gate["label"] == "REPRESENTATION-FAIL"


def test_representation_gate_pairs_letter_first_and_content_within_interface():
    metric_rows = [
        {
            "interface_id": "I1",
            "prompt_mode": "neutralized_letter_instruction",
            "representation": "letter_first",
            "feature": "letter_margin_gain_25_90",
            "mean_auc": 0.80,
        },
        {
            "interface_id": "I1",
            "prompt_mode": "neutralized_letter_instruction",
            "representation": "content_sequence",
            "feature": "content_margin_gain_25_90",
            "mean_auc": 0.55,
        },
        {
            "interface_id": "I2",
            "prompt_mode": "neutralized_letter_instruction",
            "representation": "letter_first",
            "feature": "letter_margin_gain_25_90",
            "mean_auc": 0.50,
        },
        {
            "interface_id": "I2",
            "prompt_mode": "neutralized_letter_instruction",
            "representation": "letter_sequence",
            "feature": "letter_margin_gain_25_90",
            "mean_auc": 0.95,
        },
        {
            "interface_id": "I2",
            "prompt_mode": "neutralized_letter_instruction",
            "representation": "content_sequence",
            "feature": "content_margin_gain_25_90",
            "mean_auc": 0.85,
        },
    ]

    metrics = _gate_metrics(
        {"passed": True, "completion_known": True},
        metric_rows,
        [],
        [],
        [],
    )

    assert metrics["representation_fail"] is True
    assert metrics["representation_comparisons"] == [
        {"interface_id": "I1", "letter_first_auc": 0.80, "content_margin_auc": 0.55},
        {"interface_id": "I2", "letter_first_auc": 0.50, "content_margin_auc": 0.85},
    ]
    assert assign_stage_a_gate(metrics)["label"] == "REPRESENTATION-FAIL"


def test_unknown_pipeline_completion_prevents_final_process_pass():
    gate = assign_stage_a_gate(
        {
            "pipeline_pass": None,
            "gain_auc": 0.75,
            "gain_auc_ci_low": 0.65,
            "gain_auc_ci_high": 0.85,
            "positive_correlation_fraction": 0.70,
            "i1_i2_auc_difference": 0.02,
            "trimmed_auc": 0.68,
            "label_permutation_auc_change": 0.01,
            "paired_delta_auc": 0.03,
            "paired_delta_ci_low": 0.01,
            "paired_delta_ci_high": 0.05,
            "crossfit_improvement": 0.02,
        }
    )

    assert gate["provisional"] is True
    assert gate["passed"] is False
    assert "pipeline_completion" in gate["missing_controls"]


def test_gain_level_diagnostics_report_variance_ratio_and_rank_agreement():
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "early": 1.0, "late": 1.0, "gain": 0.0},
            {"question_id": "q1", "early": 1.0, "late": 2.0, "gain": 1.0},
            {"question_id": "q1", "early": 1.0, "late": 3.0, "gain": 2.0},
            {"question_id": "q2", "early": 0.0, "late": 2.0, "gain": 0.0},
            {"question_id": "q2", "early": 1.0, "late": 1.0, "gain": 1.0},
            {"question_id": "q2", "early": 2.0, "late": 0.0, "gain": 2.0},
        ]
    )

    result = gain_level_diagnostics(frame, "early", "late", "gain")

    assert result["median_early_final_variance_ratio"] == 0.5
    assert result["near_zero_early_variance_fraction"] == 0.5
    assert result["positive_correlation_fraction"] == 0.5
    assert result["positive_correlation_questions"] == 1
    assert result["negative_correlation_questions"] == 1
    assert ".corr(" not in inspect.getsource(_spearman)


def test_question_grouped_cross_fit_is_deterministic_and_uses_held_out_auc():
    rows = []
    for question_index in range(8):
        for is_correct in (False, True):
            rows.append(
                {
                    "question_id": f"q{question_index}",
                    "is_correct": is_correct,
                    "level": float(question_index),
                    "gain": float(is_correct),
                }
            )
    frame = pd.DataFrame(rows)

    first = cross_fitted_level_gain_comparison(frame, "level", "gain", folds=4, seed=23)
    second = cross_fitted_level_gain_comparison(frame, "level", "gain", folds=4, seed=23)

    assert first == second
    assert first["metric"] == "held_out_question_equal_pairwise_auc"
    assert first["level_only_score"] == 0.5
    assert first["level_gain_score"] == 1.0
    assert first["improvement"] == 0.5
    assert "np.linalg" not in inspect.getsource(_logistic_predictions)


def test_dense_tables_retain_every_fraction_interval_and_t1_panel():
    rows = []
    for question_index in range(2):
        for rollout_id, is_correct in enumerate((False, True)):
            for probe_kind, frac in (("prompt", 0.0), ("think", 0.25), ("think", 0.90)):
                rows.append(
                    {
                        "question_id": f"q{question_index}",
                        "rollout_id": rollout_id,
                        "is_correct": is_correct,
                        "gold_letter": "A",
                        "probe_id": f"{probe_kind}_{frac}",
                        "probe_kind": probe_kind,
                        "frac": frac,
                        "interface_id": "I1",
                        "representation": "content_sequence",
                        "target_id": "gold_surface_0",
                        "surface_id": "gold_surface_0",
                        "is_gold_target": True,
                        "target_text": "answer",
                        "score_mean": question_index + float(is_correct) + frac,
                    }
                )
    scores = pd.DataFrame(rows)

    tables = build_dense_tables(
        build_probe_features(scores),
        build_step_features(scores),
        bootstrap_samples=100,
        seed=31,
    )

    assert tables["T1"].shape[0] == 12
    assert set(tables["T1"]["panel"]) == {"raw", "within_question_centered"}
    assert tables["T2"].shape[0] == 4
    assert tables["T3"].shape[0] == 2
    assert tables["T4"].shape[0] == 2
    assert {"estimate", "ci_low", "ci_high", "questions"}.issubset(tables["T4"].columns)


def test_run_analysis_writes_required_stage_a_artifacts(tmp_path):
    rows = []
    for question_index in range(2):
        for rollout_id, is_correct in enumerate((False, True)):
            for probe_id, probe_kind, frac in (
                ("prompt_0.00", "prompt", 0.0),
                ("think_0.25", "think", 0.25),
                ("think_0.90", "think", 0.90),
                ("trimmed_final", "trimmed", None),
            ):
                effective_frac = 0.80 if frac is None else frac
                gold_score = question_index + effective_frac * (1.0 + float(is_correct))
                for target_id, is_gold, offset in (
                    ("gold_surface_0", True, 0.0),
                    ("option_B", False, -0.5),
                ):
                    rows.append(
                        {
                            "question_id": f"q{question_index}",
                            "rollout_id": rollout_id,
                            "is_correct": is_correct,
                            "gold_letter": "A",
                            "probe_id": probe_id,
                            "probe_kind": probe_kind,
                            "frac": frac,
                            "interface_id": "I1",
                            "representation": "content_sequence",
                            "target_id": target_id,
                            "surface_id": "gold_surface_0" if is_gold else None,
                            "is_gold_target": is_gold,
                            "target_text": "answer" if is_gold else "wrong",
                            "score_mean": gold_score + offset,
                        }
                    )
    scores_path = tmp_path / "request_scores.jsonl"
    pd.DataFrame(rows).to_json(scores_path, orient="records", lines=True)
    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "requests": len(rows) * 2,
                "scored_total": len(rows),
                "score_failures": 0,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"

    result = run_analysis(scores_path, output_dir, bootstrap_samples=50, seed=37)

    required = {
        "rollout_probe_metrics.parquet",
        "rollout_probe_metrics.csv",
        "rollout_features.csv",
        "dense_step_metrics.csv",
        "metric_summary.json",
        "STAGE_A_AUDIT_REPORT.md",
        "T1_level_trajectory.csv",
        "T2_adjacent_gain_trajectory.csv",
        "T3_gain_separation.csv",
        "T4_interval_discrimination.csv",
    }
    assert required.issubset({path.name for path in output_dir.iterdir()})
    summary = json.loads((output_dir / "metric_summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "STAGE_A_AUDIT_REPORT.md").read_text(encoding="utf-8")
    assert summary["bootstrap_samples"] == 50
    assert summary["cross_fitted_metric"] == "held_out_question_equal_pairwise_auc"
    assert "## Gain vs Level Diagnostics" in report
    assert "held_out_question_equal_pairwise_auc" in report
    assert result["pipeline"]["completion_known"] is True
    assert result["gate"]["label"] == "PIPELINE-FAIL"
    assert result["gate"]["passed"] is False
    if importlib.util.find_spec("matplotlib") is not None:
        assert {
            "T1_level_trajectory.png",
            "T2_adjacent_gain_trajectory.png",
            "T3_gain_separation.png",
            "T4_interval_discrimination.png",
        }.issubset({path.name for path in output_dir.iterdir()})


def test_smoke_gate_checks_completion_finiteness_and_within_question_variance():
    probes = build_probe_features(_scores())
    steps = build_step_features(probes)
    gate = smoke_gate(_scores(), steps, expected_requests=len(_scores()))

    assert gate["completion_rate"] == 1.0
    assert gate["finite_score_rate"] == 1.0
    assert gate["nonzero_within_question_gain_variance"] is True
    assert gate["passed"] is True


def test_smoke_gate_does_not_confuse_interval_changes_with_rollout_variance():
    scores = pd.DataFrame([{"score_mean": -1.0}, {"score_mean": -1.0}])
    rows = []
    for rollout_id in (0, 1):
        for start, end, gain in ((0.0, 0.25, 1.0), (0.25, 0.90, 2.0)):
            rows.append(
                {
                    "question_id": "q1",
                    "rollout_id": rollout_id,
                    "interface_id": "I1",
                    "representation": "content_sequence",
                    "prompt_mode": "neutralized_letter_instruction",
                    "frac_start": start,
                    "frac_end": end,
                    "level_gain": gain,
                }
            )

    gate = smoke_gate(scores, pd.DataFrame(rows), expected_requests=2)

    assert gate["nonzero_within_question_gain_variance"] is False
    assert gate["passed"] is False
