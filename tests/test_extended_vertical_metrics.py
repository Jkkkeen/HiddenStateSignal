import numpy as np

from vertical.profiles import (
    EXTENDED_PROFILE_COLUMNS,
    METRIC_REGISTRY,
    cumulative_vertical_geometry,
    reduce_stage_profiles,
    rolling_vertical_geometry,
    v2_raw_state_angle,
    vertical_effective_degree,
    vertical_effective_rank,
)


def test_v2_aligns_adjacent_state_angles_to_destination_layer():
    layers = np.array(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]],
        dtype=float,
    )

    values = v2_raw_state_angle(layers)

    assert np.isnan(values[0])
    assert len(values) == layers.shape[0]
    assert np.allclose(values[1:], [np.pi / 4, np.pi / 4, np.pi / 4])


def test_rolling_window_uses_exactly_four_consecutive_updates():
    layers = np.column_stack(
        [np.arange(7, dtype=float), np.zeros(7), np.zeros(7)]
    )

    result = rolling_vertical_geometry(layers, window_updates=4)

    assert result["window_updates"].eq(4).all()
    assert result["end_layer"].is_monotonic_increasing
    assert result["end_layer"].tolist() == list(range(1, 7))
    assert result.loc[result["end_layer"] < 4, "coverage_ok"].eq(False).all()
    valid = result.loc[result["coverage_ok"]]
    assert (valid["end_layer"] - valid["start_layer"]).eq(4).all()
    assert np.allclose(valid["v5_path_length"], 4.0)
    assert np.allclose(valid["v5_straightness"], 1.0)


def test_cumulative_geometry_starts_at_embedding_and_preserves_early_coverage():
    layers = np.column_stack(
        [np.arange(7, dtype=float), np.zeros(7), np.zeros(7)]
    )

    result = cumulative_vertical_geometry(layers)

    assert result["start_layer"].eq(0).all()
    assert result["end_layer"].tolist() == list(range(1, 7))
    assert np.allclose(result["v5_path_length"], result["end_layer"])
    assert np.allclose(result["v5_straightness"], 1.0)
    assert result.loc[result["end_layer"] < 4, "v9_update_coverage_ok"].eq(False).all()
    assert result.loc[result["end_layer"] >= 4, "v9_update_coverage_ok"].all()


def test_effective_rank_and_degree_have_expected_invariances():
    assert np.isclose(vertical_effective_rank(np.eye(3), centered=False), 3.0)
    positions = np.linspace(0.0, 1.0, 9)
    scaled = 2.0 * positions - 1.0
    cubic = np.polynomial.chebyshev.chebval(scaled, [0.0, 0.0, 0.0, 1.0])
    layers = np.column_stack([cubic, 2.0 * cubic, -cubic])

    result = vertical_effective_degree(layers, degree=3)
    shifted = vertical_effective_degree(layers + np.array([8.0, -3.0, 2.0]), degree=3)

    assert result["coverage_ok"] is True
    assert np.isclose(result["normalized_trajectory_ed"], 3.0, atol=1e-8)
    assert np.isclose(
        result["normalized_trajectory_ed"],
        shifted["normalized_trajectory_ed"],
        atol=1e-8,
    )
    assert result["high_order_fraction"] > 0.99
    assert result["relative_fit_rmse"] < 1e-10


def test_stage_profiles_include_experimental_metrics_with_structural_coverage():
    base = np.column_stack(
        [np.arange(7, dtype=float), np.zeros(7), np.ones(7)]
    )
    trajectory = np.stack([base, base + 0.25], axis=0)

    frame = reduce_stage_profiles(
        trajectory,
        np.array([0.1, 0.2]),
        representation="last_s32",
        metadata={"record_id": "q:r"},
        base_common=np.zeros_like(base),
    )

    assert set(EXTENDED_PROFILE_COLUMNS).issubset(frame.columns)
    early = frame.loc[(frame["stage"] == 0) & (frame["layer_index"] < 4)]
    assert early["v5_rolling_path_length"].isna().all()
    assert early["profile_coverage_count_v5_rolling_path_length"].eq(0).all()
    late = frame.loc[(frame["stage"] == 0) & (frame["layer_index"] >= 4)]
    assert late["v5_rolling_path_length"].notna().all()
    assert all(
        METRIC_REGISTRY[metric]["status"] == "experimental"
        for metric in EXTENDED_PROFILE_COLUMNS
    )
