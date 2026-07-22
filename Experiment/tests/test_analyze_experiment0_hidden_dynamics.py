from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_experiment0_hidden_dynamics import (
    compare_feature_to_length,
    hedges_g,
    main,
    passes_length_gate,
    run_geometry_permutations,
)


def test_hedges_g_has_correct_sign_and_small_sample_correction() -> None:
    value = hedges_g(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0]))

    assert 1.0 < value < 4.0
    assert hedges_g(np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0, 5.0])) == -value


def _predictor_frame(n_questions: int = 10) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(3)
    for question in range(n_questions):
        for rollout in range(6):
            correct = rollout < 3
            rows.append(
                {
                    "question_id": f"q{question}",
                    "rollout_id": rollout,
                    "is_correct": correct,
                    "feature": (2.0 if correct else -2.0) + rng.normal(scale=0.1),
                    "think_length": 3000 + rng.normal(scale=100),
                }
            )
    return pd.DataFrame(rows)


def test_length_baseline_uses_identical_rows_and_grouped_oof_predictions() -> None:
    result = compare_feature_to_length(
        _predictor_frame(), feature="feature", seed=7, bootstrap=50
    )

    assert result["n_rows_feature"] == result["n_rows_length"] == result["n_rows_combined"]
    assert result["fold_question_overlap"] == 0
    assert result["feature_only_auc"] > result["length_only_auc"]
    assert result["combined_auc"] > result["length_only_auc"]


def test_gate_requires_both_paired_ci_lower_bounds_above_zero() -> None:
    assert passes_length_gate(0.01, 0.02) is True
    assert passes_length_gate(-0.01, 0.02) is False
    assert passes_length_gate(0.01, -0.02) is False


def test_geometry_permutation_rebuilds_reference_scores() -> None:
    rows = []
    labels = {0: True, 1: True, 2: True, 3: False, 4: False, 5: False}
    for query_id, query_label in labels.items():
        for reference_id, reference_label in labels.items():
            if reference_id == query_id:
                continue
            rows.append(
                {
                    "question_id": "q1",
                    "rollout_id": query_id,
                    "is_correct": query_label,
                    "representation": "mean_w128_s64",
                    "span_id": 1,
                    "relative_progress": 0.5,
                    "progress_bin": 5,
                    "layer": 24,
                    "displacement_norm": 1.0,
                    "reference_rollout_id": reference_id,
                    "reference_is_correct": reference_label,
                    "reference_span_id": 1,
                    "reference_progress": 0.5,
                    "reference_norm": 1.0,
                    "cosine_similarity": 0.9 if query_label == reference_label else 0.1,
                }
            )
    nulls = run_geometry_permutations(pd.DataFrame(rows), permutations=3, seed=9)

    assert set(nulls["permutation_id"]) == {0, 1, 2}
    assert set(nulls["feature"]) == {"cross_set_direction", "cross_length_support"}
    assert nulls["geometry_recomputed"].all()


def _write_synthetic_question(input_dir: Path, question_index: int) -> None:
    question_id = f"q{question_index}"
    token_rows = []
    span_rows = []
    prototype_rows = []
    geometry_rows = []
    rng = np.random.default_rng(question_index)
    for rollout_id in range(6):
        correct = rollout_id < 3
        sign = 1.0 if correct else -1.0
        think_length = int(2800 + rng.normal(scale=50))
        for layer in (0, 1, 2):
            token_rows.append(
                {
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "is_correct": correct,
                    "representation": "token",
                    "progress_bin": 0,
                    "layer": layer,
                    "think_length": think_length,
                    "token_count": 100,
                    "horizontal_norm_mean": sign + rng.normal(scale=0.05),
                    "horizontal_norm_delta_mean": 0.1 * sign,
                    "vertical_norm_mean": sign + layer if layer else np.nan,
                    "vertical_norm_delta_mean": 0.2 * sign if layer > 1 else np.nan,
                    "coordinate_entropy_mean": 0.5 + 0.1 * sign if layer else np.nan,
                    "effective_dimensions_mean": 4.0 + sign if layer else np.nan,
                    "token_turn_cos_mean": 0.2 * sign,
                    "hidden_norm_mean": 10.0 + layer,
                }
            )
            span_rows.append(
                {
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "is_correct": correct,
                    "representation": "mean_w128_s64",
                    "progress_bin": 0,
                    "layer": layer,
                    "think_length": think_length,
                    "span_count": 10,
                    "span_movement_norm_mean": 2.0 + sign,
                    "span_turn_cos_mean": 0.4 * sign,
                    "span_turn_cos_split_a": 0.4 * sign + 0.01,
                    "span_turn_cos_split_b": 0.4 * sign - 0.01,
                    "span_layer_update_norm_mean": 1.0 + sign if layer else np.nan,
                    "span_layer_turn_cos_mean": 0.3 * sign if layer > 1 else np.nan,
                    "span_layer_turn_cos_split_a": 0.3 * sign + 0.01 if layer > 1 else np.nan,
                    "span_layer_turn_cos_split_b": 0.3 * sign - 0.01 if layer > 1 else np.nan,
                    "cross_set_direction": 1.5 * sign,
                    "cross_prototype_direction": 1.4 * sign,
                    "cross_length_support": 0.5 * sign,
                    "prototype_kappa_pos_mean": 0.8,
                    "prototype_kappa_neg_mean": 0.7,
                    "prototype_valid_fraction": 1.0,
                }
            )
            prototype_rows.append(
                {
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "is_correct": correct,
                    "representation": "mean_w128_s64",
                    "progress_bin": 0,
                    "layer": layer,
                    "subset_id": 0,
                    "kappa_pos": 0.8,
                    "kappa_neg": 0.7,
                    "prototype_valid_010": True,
                    "prototype_valid_020": True,
                    "prototype_valid_030": True,
                }
            )
            for reference_id in range(6):
                if reference_id == rollout_id:
                    continue
                reference_correct = reference_id < 3
                geometry_rows.append(
                    {
                        "question_id": question_id,
                        "rollout_id": rollout_id,
                        "is_correct": correct,
                        "representation": "mean_w128_s64",
                        "span_id": 1,
                        "relative_progress": 0.5,
                        "progress_bin": 0,
                        "layer": layer,
                        "displacement_norm": 1.0,
                        "reference_rollout_id": reference_id,
                        "reference_is_correct": reference_correct,
                        "reference_span_id": 1,
                        "reference_progress": 0.5,
                        "reference_norm": 1.0,
                        "cosine_similarity": 0.9 if correct == reference_correct else 0.1,
                    }
                )
    stem = f"question_{question_id}"
    pd.DataFrame(token_rows).to_parquet(input_dir / "bin_features" / f"{stem}.parquet")
    pd.DataFrame(span_rows).to_parquet(input_dir / "span_features" / f"{stem}.parquet")
    pd.DataFrame(prototype_rows).to_parquet(
        input_dir / "prototype_diagnostics" / f"{stem}.parquet"
    )
    pd.DataFrame(geometry_rows).to_parquet(
        input_dir / "pairwise_geometry" / f"{stem}.parquet"
    )


def test_analyzer_writes_all_declared_outputs(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "extraction"
    output_dir = tmp_path / "results"
    for directory in (
        "bin_features",
        "span_features",
        "prototype_diagnostics",
        "pairwise_geometry",
    ):
        (input_dir / directory).mkdir(parents=True)
    for question_index in range(6):
        _write_synthetic_question(input_dir, question_index)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_experiment0_hidden_dynamics.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--bootstrap",
            "20",
            "--permutations",
            "2",
            "--run-label",
            "Synthetic Smoke",
        ],
    )

    main()

    expected = [
        "LONG_EXPERIMENT_0_RESULTS.md",
        "long_experiment_0_bin_features.parquet",
        "long_experiment_0_question_effects.csv",
        "long_experiment_0_predictor_comparisons.csv",
        "long_experiment_0_prototype_diagnostics.parquet",
        "long_experiment_0_pairwise_geometry.parquet",
        "long_experiment_0_permutation_null.csv",
        "analysis_meta.json",
        "figures/E0_F1_horizontal_length_angle.png",
        "figures/E0_F2_cross_rollout_support.png",
        "figures/E0_F3_vertical_entropy_activity.png",
        "figures/E0_F4_layer_progress_effects.png",
        "figures/E0_F5_angle_snr_reliability.png",
        "figures/E0_F6_length_strata.png",
        "figures/E0_F7_length_baseline.png",
    ]
    assert all((output_dir / name).is_file() for name in expected)
    report = (output_dir / "LONG_EXPERIMENT_0_RESULTS.md").read_text(encoding="utf-8")
    assert report.startswith("# Long Experiment 0 Synthetic Smoke Results")
