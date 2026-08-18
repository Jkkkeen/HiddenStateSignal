import numpy as np
import pandas as pd
import pytest

from vertical.statistics import (
    cluster_permutation,
    cluster_signflip,
    fit_nuisance_controlled_effects,
    paired_condition_delta,
    question_bootstrap,
    standardize_within_family_base,
)


def test_question_bootstrap_is_reproducible_and_question_clustered():
    frame = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
            "score": [0.0, 2.0, 4.0, 4.0, 9.0, 9.0],
        }
    )

    statistic = lambda sample: float(
        sample.groupby("question_id")["score"].mean().mean()
    )
    first = question_bootstrap(frame, statistic, n_boot=100, seed=7)
    second = question_bootstrap(frame, statistic, n_boot=100, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first["n_questions"].eq(3).all()
    assert first["resampling_unit"].eq("question_id").all()
    assert first["estimate"].nunique() > 1


def test_cluster_permutation_finds_reproducible_contiguous_signal():
    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 0.4, size=(40, 8))
    labels = np.r_[np.zeros(20, dtype=bool), np.ones(20, dtype=bool)]
    values[labels, 2:5] += 2.0

    first = cluster_permutation(values, labels, n_permutations=199, seed=11)
    second = cluster_permutation(values, labels, n_permutations=199, seed=11)

    pd.testing.assert_frame_equal(first, second)
    signal = first.loc[(first["start_index"] <= 2) & (first["end_index"] >= 4)]
    assert len(signal) == 1
    assert signal.iloc[0]["p_value"] <= 0.05
    assert signal.iloc[0]["correction_method"] == "max_cluster_mass"
    assert signal.iloc[0]["n_questions"] == 40


def test_cluster_signflip_uses_question_effects_as_exchangeable_units():
    rng = np.random.default_rng(17)
    effects = rng.normal(0.0, 0.3, size=(30, 7))
    effects[:, 1:4] += 1.5

    result = cluster_signflip(effects, n_permutations=199, seed=3)

    signal = result.loc[(result["start_index"] <= 1) & (result["end_index"] >= 3)]
    assert len(signal) == 1
    assert signal.iloc[0]["p_value"] <= 0.05
    assert signal.iloc[0]["correction_method"].startswith("question_sign_flip")


def test_nuisance_effect_uses_available_finite_controls_and_clustered_se():
    rng = np.random.default_rng(9)
    n_questions = 60
    correct = np.tile([False, True], n_questions)
    question_id = np.repeat([f"q{index}" for index in range(n_questions)], 2)
    response_length = rng.normal(100.0, 12.0, size=len(correct))
    score = 2.0 * correct + 0.03 * response_length + rng.normal(0.0, 0.05, len(correct))
    frame = pd.DataFrame(
        {
            "question_id": question_id,
            "is_correct": correct,
            "score": score,
            "response_token_count": response_length,
        }
    )

    result = fit_nuisance_controlled_effects(
        frame,
        "score",
        ["response_token_count", "missing_control"],
    )

    effect = result.loc[result["term"] == "is_correct"].iloc[0]
    assert effect["coverage_ok"]
    assert np.isclose(effect["estimate"], 2.0, atol=0.05)
    assert effect["n_questions"] == n_questions
    assert effect["controls_used"] == "response_token_count"
    assert effect["controls_omitted"] == "missing_control"


def test_paired_delta_rejects_duplicate_condition_keys():
    frame = pd.DataFrame(
        {
            "condition": ["base", "base", "sft"],
            "question_id": ["q1", "q1", "q1"],
            "rollout_id": ["r0", "r0", "r0"],
            "score": [1.0, 1.5, 3.0],
        }
    )

    with pytest.raises(ValueError, match="one-to-one"):
        paired_condition_delta(frame, "base", "sft", ["question_id", "rollout_id"])


def test_paired_delta_and_within_family_standardization_are_explicit():
    frame = pd.DataFrame(
        {
            "model_family": ["mimo"] * 4,
            "condition": ["base", "base", "sft", "sft"],
            "question_id": ["q1", "q2", "q1", "q2"],
            "rollout_id": ["r0"] * 4,
            "representation": ["last_s32"] * 4,
            "stage": [0] * 4,
            "layer_index": [1] * 4,
            "score": [1.0, 3.0, 4.0, 8.0],
        }
    )

    paired = paired_condition_delta(
        frame,
        "base",
        "sft",
        ["model_family", "question_id", "rollout_id", "representation", "stage", "layer_index"],
    )
    standardized = standardize_within_family_base(frame, "score")

    assert paired["score_delta"].tolist() == [3.0, 5.0]
    assert paired["pairing_status"].eq("complete").all()
    base = standardized.loc[standardized["condition"] == "base"]
    assert np.isclose(base["base_standardized_score"].mean(), 0.0)
    assert standardized["standardization_ok"].all()
