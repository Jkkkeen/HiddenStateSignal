import numpy as np
import pytest

from vertical.profiles import (
    PROFILE_COLUMNS,
    energy_entropy_rows,
    local_profile_arrays,
    reduce_stage_profiles,
    select_decoder_backbone,
)


def test_local_profiles_have_frozen_layer_semantics():
    layers = np.array(
        [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 3.0]],
        dtype=float,
    )

    result = local_profile_arrays(layers, np.zeros_like(layers))

    assert tuple(result) == PROFILE_COLUMNS
    assert np.isnan(result["v1_raw_update_norm"][0])
    assert np.allclose(result["v1_raw_update_norm"][1:], [1.0, 1.0, 2.0])
    assert np.isnan(result["v4_layer_update_turning_angle"][:2]).all()
    assert np.allclose(
        result["v4_layer_update_turning_angle"][2:],
        [np.pi / 2, np.pi / 2],
    )


def test_stage_reduction_preserves_depth_coverage_and_medians():
    base = np.arange(4 * 3, dtype=float).reshape(4, 3) + 1.0
    trajectory = np.stack([base, base * 2, base * 3, base * 4], axis=0)

    frame = reduce_stage_profiles(
        trajectory,
        np.array([0.10, 0.30, 0.60, 1.00]),
        representation="mean_w128_s32",
        metadata={"record_id": "q:r", "question_id": "q", "rollout_id": "r"},
        base_common=np.zeros_like(base),
    )

    assert len(frame) == 4 * 4
    assert frame["stage"].value_counts().sort_index().tolist() == [4, 4, 4, 4]
    assert np.allclose(
        frame["relative_depth"].drop_duplicates(),
        [0.0, 1 / 3, 2 / 3, 1.0],
    )
    assert frame.loc[frame["layer_index"] == 0, "v1_raw_update_norm"].isna().all()
    assert frame["profile_chunk_count"].eq(1).all()


def test_stage_reduction_uses_chunk_median_per_layer():
    first = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 3.0]])
    second = first * 3.0
    trajectory = np.stack([first, second], axis=0)

    frame = reduce_stage_profiles(
        trajectory,
        np.array([0.10, 0.20]),
        representation="last_s32",
        metadata={"record_id": "q:r"},
        base_common=np.zeros_like(first),
    )

    stage_zero = frame.loc[frame["stage"] == 0].sort_values("layer_index")
    expected = np.r_[np.nan, [2.0, 4.0]]
    assert np.allclose(stage_zero["v1_raw_update_norm"], expected, equal_nan=True)
    assert stage_zero["profile_chunk_count"].unique().tolist() == [2]


def test_entropy_is_normalized_and_does_not_mutate_input():
    states = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    original = states.copy()

    entropy = energy_entropy_rows(states, center=False, zero_value=np.nan)

    assert np.allclose(states, original)
    assert np.allclose(entropy, [1.0, 0.0], atol=1e-6)


def test_decoder_backbone_excludes_mtp_positions():
    values = np.arange(2 * 6 * 3, dtype=float).reshape(2, 6, 3)
    kinds = np.array(["embedding", "decoder", "decoder", "decoder", "decoder", "mtp"])

    selected = select_decoder_backbone(values, num_decoder_layers=4, layer_kind=kinds)

    assert selected.shape == (2, 5, 3)
    assert np.array_equal(selected, values[:, :5, :])


def test_extra_positions_without_layer_kind_fail_closed():
    values = np.zeros((2, 6, 3))

    with pytest.raises(ValueError, match="layer_kind"):
        select_decoder_backbone(values, num_decoder_layers=4)

