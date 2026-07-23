from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from correct_trajectory_pairwise_geometry import (
    adjusted_max_stat_pvalues,
    aggregate_directed_pairs,
    compute_pair_metrics,
    directed_four_cell_interactions,
    pair_type_means,
    permutation_inference,
    question_contrasts,
    summarize_contrasts,
    symmetrize_pairs,
)


def _base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "question_id": "q1",
        "rollout_id": 0,
        "is_correct": True,
        "representation": "mean_w128_s64",
        "span_id": 1,
        "relative_progress": 0.5,
        "progress_bin": 5,
        "layer": 24,
        "displacement_norm": 1.0,
        "reference_rollout_id": 1,
        "reference_is_correct": True,
        "reference_span_id": 1,
        "reference_progress": 0.5,
        "reference_norm": 1.0,
        "cosine_similarity": 1.0,
    }
    row.update(overrides)
    return row


def test_compute_pair_metrics_recovers_known_geometry() -> None:
    frame = pd.DataFrame(
        [
            _base_row(cosine_similarity=1.0),
            _base_row(cosine_similarity=0.0),
            _base_row(cosine_similarity=-1.0),
        ]
    )

    result, exclusions = compute_pair_metrics(frame)

    np.testing.assert_allclose(result["angle_rad"], [0.0, np.pi / 2, np.pi])
    np.testing.assert_allclose(result["delta_vec"], [0.0, np.sqrt(2.0), 2.0])
    np.testing.assert_allclose(result["delta_rel"], [0.0, np.sqrt(2.0) / 2.0, 1.0])
    np.testing.assert_allclose(result["delta_amp"], 0.0)
    assert exclusions == {"input_rows": 3, "excluded_rows": 0, "valid_rows": 3}


def test_compute_pair_metrics_clips_cosine_and_excludes_undefined_directions() -> None:
    frame = pd.DataFrame(
        [
            _base_row(cosine_similarity=1.0 + 1e-8),
            _base_row(displacement_norm=0.0),
            _base_row(reference_norm=np.nan),
        ]
    )

    result, exclusions = compute_pair_metrics(frame)

    assert len(result) == 1
    assert result.iloc[0]["angle_rad"] == pytest.approx(0.0)
    assert exclusions == {"input_rows": 3, "excluded_rows": 2, "valid_rows": 1}


def test_directed_matches_are_aggregated_and_symmetrized_once() -> None:
    raw = pd.DataFrame(
        [
            _base_row(rollout_id=0, reference_rollout_id=1, span_id=1, cosine_similarity=0.8),
            _base_row(rollout_id=0, reference_rollout_id=1, span_id=2, cosine_similarity=0.6),
            _base_row(rollout_id=1, reference_rollout_id=0, span_id=1, cosine_similarity=0.4),
        ]
    )
    metrics, _ = compute_pair_metrics(raw)

    directed = aggregate_directed_pairs(metrics)
    pairs = symmetrize_pairs(directed)

    assert len(directed) == 2
    assert len(pairs) == 1
    assert pairs.iloc[0]["pair_type"] == "++"
    assert pairs.iloc[0]["angle_rad"] == pytest.approx(directed["angle_rad"].mean())


def test_pair_type_means_equalize_unequal_pair_counts() -> None:
    rows = []
    for question_id, pp_values in (("q1", [0.1, 0.3]), ("q2", [0.8])):
        for pair_index, value in enumerate(pp_values):
            rows.append(
                {
                    "question_id": question_id,
                    "layer": 24,
                    "progress_bin": 5,
                    "pair_lo": pair_index,
                    "pair_hi": pair_index + 10,
                    "pair_type": "++",
                    "angle_rad": value,
                    "delta_rel": value,
                    "delta_amp": value,
                    "delta_vec": value,
                    "angle_deg": np.degrees(value),
                }
            )
    means = pair_type_means(pd.DataFrame(rows), whole_trajectory=False)

    assert len(means) == 2
    assert means.loc[means["question_id"] == "q1", "angle_rad"].iloc[0] == pytest.approx(0.2)
    assert means.loc[means["question_id"] == "q2", "angle_rad"].iloc[0] == pytest.approx(0.8)


def test_question_contrasts_have_expected_negative_sign_and_missingness() -> None:
    means = pd.DataFrame(
        {
            "question_id": ["q1"] * 3 + ["q2"] * 2,
            "layer": [24] * 5,
            "progress_bin": [5] * 5,
            "pair_type": ["++", "--", "+-", "++", "+-"],
            "angle_rad": [0.2, 0.8, 1.0, 0.1, 0.7],
        }
    )

    result = question_contrasts(means, metrics=("angle_rad",))
    q1 = result[result["question_id"] == "q1"]
    q2 = result[result["question_id"] == "q2"]

    assert dict(zip(q1["contrast"], q1["value"])) == pytest.approx(
        {"pp_minus_mm": -0.6, "pp_minus_pm": -0.8}
    )
    assert set(q2["contrast"]) == {"pp_minus_pm"}
    assert q2.iloc[0]["value"] == pytest.approx(-0.6)


def test_directed_four_cell_interaction_uses_all_cells() -> None:
    rows = []
    for query_correct, reference_correct, value in (
        (True, True, 0.2),
        (True, False, 0.8),
        (False, True, 0.9),
        (False, False, 0.5),
    ):
        rows.append(
            {
                **_base_row(
                    is_correct=query_correct,
                    reference_is_correct=reference_correct,
                ),
                "angle_rad": value,
                "delta_rel": value,
                "delta_amp": value,
            }
        )

    result = directed_four_cell_interactions(pd.DataFrame(rows))

    angle = result[result["metric"] == "angle_rad"].iloc[0]
    assert angle["interaction"] == pytest.approx((0.2 - 0.8) - (0.9 - 0.5))


def test_question_bootstrap_is_deterministic() -> None:
    contrasts = pd.DataFrame(
        {
            "question_id": [f"q{index}" for index in range(6)],
            "layer": [24] * 6,
            "progress_bin": [5] * 6,
            "metric": ["angle_rad"] * 6,
            "contrast": ["pp_minus_pm"] * 6,
            "value": [-0.1, -0.2, -0.3, -0.4, 0.1, -0.5],
        }
    )

    first = summarize_contrasts(contrasts, bootstrap=100, seed=7)
    second = summarize_contrasts(contrasts, bootstrap=100, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["n_questions"] == 6
    assert first.iloc[0]["negative_sign_fraction"] == pytest.approx(5 / 6)


def _synthetic_permutation_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = []
    pairs = []
    for question_index in range(3):
        question_id = f"q{question_index}"
        label_by_rollout = {0: True, 1: True, 2: False, 3: False}
        labels.extend(
            {
                "question_id": question_id,
                "rollout_id": rollout_id,
                "is_correct": is_correct,
            }
            for rollout_id, is_correct in label_by_rollout.items()
        )
        for layer in (24, 36):
            for progress_bin in (0, 1):
                for left in range(4):
                    for right in range(left + 1, 4):
                        same = label_by_rollout[left] == label_by_rollout[right]
                        value = 0.2 if same else 1.0
                        pair_type = (
                            "++"
                            if label_by_rollout[left] and label_by_rollout[right]
                            else "--"
                            if not label_by_rollout[left] and not label_by_rollout[right]
                            else "+-"
                        )
                        pairs.append(
                            {
                                "question_id": question_id,
                                "layer": layer,
                                "progress_bin": progress_bin,
                                "pair_lo": left,
                                "pair_hi": right,
                                "pair_type": pair_type,
                                "angle_rad": value,
                                "angle_deg": np.degrees(value),
                                "delta_vec": value,
                                "delta_rel": value,
                                "delta_amp": value,
                            }
                        )
    return pd.DataFrame(pairs), pd.DataFrame(labels)


def test_permutation_preserves_label_counts_and_is_deterministic() -> None:
    pairs, labels = _synthetic_permutation_inputs()

    first = permutation_inference(pairs, labels, permutations=5, seed=9)
    second = permutation_inference(pairs, labels, permutations=5, seed=9)

    pd.testing.assert_frame_equal(first.nulls, second.nulls)
    assert first.label_count_checks.all()
    assert set(first.nulls["permutation_id"]) == set(range(5))
    assert set(first.nulls["scope"]) == {"progress", "whole"}


def test_max_stat_adjustment_uses_family_wide_null_maximum() -> None:
    observed = pd.DataFrame(
        {
            "metric": ["angle_rad", "angle_rad"],
            "contrast": ["pp_minus_pm", "pp_minus_pm"],
            "layer": [24, 36],
            "progress_bin": [0, 0],
            "mean_contrast": [-0.8, -0.2],
        }
    )
    nulls = pd.DataFrame(
        {
            "permutation_id": [0, 0, 1, 1],
            "scope": ["progress"] * 4,
            "metric": ["angle_rad"] * 4,
            "contrast": ["pp_minus_pm"] * 4,
            "layer": [24, 36, 24, 36],
            "progress_bin": [0, 0, 0, 0],
            "mean_contrast": [0.3, 0.4, 0.9, 0.1],
        }
    )

    result = adjusted_max_stat_pvalues(observed, nulls)

    assert result.loc[result["layer"] == 24, "max_stat_p"].iloc[0] == pytest.approx(2 / 3)
    assert result.loc[result["layer"] == 36, "max_stat_p"].iloc[0] == pytest.approx(1.0)
