import numpy as np

from experiment_2e.metrics import (
    v1_layer_update_norm,
    v3_state_angle_demean,
    v4_layer_update_turning,
    v6_raw_activation_entropy,
    v7_entropies,
)
from experiment_2e.vertical_profiles import local_profile_arrays, reduce_vertical_profiles


def test_local_profiles_have_frozen_layer_semantics():
    layers = np.array(
        [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 3.0]],
        dtype=float,
    )
    out = local_profile_arrays(layers, np.zeros_like(layers))

    assert len(out["v6_raw_activation_entropy"]) == 4
    assert np.isnan(out["v1_raw_update_norm"][0])
    assert np.allclose(out["v1_raw_update_norm"][1:], [1.0, 1.0, 2.0])
    assert np.isnan(out["v4_layer_update_turning_angle"][:2]).all()
    assert np.allclose(
        out["v4_layer_update_turning_angle"][2:],
        [np.pi / 2, np.pi / 2],
    )


def test_stage_reduction_is_wide_and_preserves_relative_depth():
    base = np.arange(4 * 3, dtype=float).reshape(4, 3) + 1.0
    trajectory = np.stack([base, base * 2, base * 3, base * 4], axis=0)
    frame = reduce_vertical_profiles(
        trajectory,
        np.array([0.10, 0.30, 0.60, 1.00]),
        representation="mean_w128_s32",
        metadata={"checkpoint": "step000", "question_id": "q", "rollout_id": "r"},
        base_common=np.zeros_like(base),
    )

    assert len(frame) == 4 * 4
    assert frame["stage"].value_counts().sort_index().tolist() == [4, 4, 4, 4]
    assert np.allclose(frame["relative_depth"].drop_duplicates(), [0.0, 1 / 3, 2 / 3, 1.0])
    metric_columns = [column for column in frame if column.startswith("v")]
    assert metric_columns == [
        "v1_raw_update_norm",
        "v1_relative_update_norm",
        "v3_demean_state_angle",
        "v4_layer_update_turning_angle",
        "v6_raw_activation_entropy",
        "v7_layer_difference_entropy",
    ]
    assert frame.loc[frame["layer_index"] == 0, "v1_raw_update_norm"].isna().all()
    assert frame.loc[frame["layer_index"] < 2, "v4_layer_update_turning_angle"].isna().all()


def test_stage_reduction_takes_chunk_median_per_layer():
    first = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 3.0]])
    second = first * 3.0
    trajectory = np.stack([first, second], axis=0)
    frame = reduce_vertical_profiles(
        trajectory,
        np.array([0.10, 0.20]),
        representation="last_s32",
        metadata={"checkpoint": "step000", "question_id": "q", "rollout_id": "r"},
        base_common=np.zeros_like(first),
    )
    stage_zero = frame.loc[frame["stage"] == 0].sort_values("layer_index")
    stacked = np.stack(
        [
            local_profile_arrays(first, np.zeros_like(first))["v1_raw_update_norm"],
            local_profile_arrays(second, np.zeros_like(second))["v1_raw_update_norm"],
        ]
    )
    expected = np.r_[np.nan, np.median(stacked[:, 1:], axis=0)]
    assert np.allclose(stage_zero["v1_raw_update_norm"], expected, equal_nan=True)
    assert stage_zero["profile_chunk_count"].unique().tolist() == [2]


def test_local_profiles_match_existing_per_chunk_aggregates():
    rng = np.random.default_rng(17)
    layers = rng.normal(size=(7, 11)) + 0.5
    common = rng.normal(size=layers.shape) * 0.1
    out = local_profile_arrays(layers, common)

    v1 = v1_layer_update_norm(layers)
    assert np.isclose(np.nanmean(out["v1_raw_update_norm"]), v1["mean_raw_layer_update_norm"])
    assert np.isclose(
        np.nanmedian(out["v1_relative_update_norm"]),
        v1["median_relative_layer_update_norm"],
    )
    v3 = v3_state_angle_demean(layers, common)
    assert np.isclose(np.nanmedian(out["v3_demean_state_angle"]), v3["median_state_angle_demean"])
    v4 = v4_layer_update_turning(layers)
    assert np.isclose(
        np.nanmedian(out["v4_layer_update_turning_angle"]),
        v4["median_layer_update_turning_angle"],
    )
    v6 = v6_raw_activation_entropy(layers)
    assert np.isclose(
        np.nanmedian(out["v6_raw_activation_entropy"]),
        v6["median_raw_activation_entropy"],
    )
    v7 = v7_entropies(layers)
    assert np.isclose(
        np.nanmedian(out["v7_layer_difference_entropy"]),
        v7["median_layer_difference_entropy"],
    )
