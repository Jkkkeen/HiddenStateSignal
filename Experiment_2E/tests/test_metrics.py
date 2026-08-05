import numpy as np

from experiment_2e.metrics import (
    energy_entropy,
    energy_entropy_rows,
    h1_movement,
    h2_path,
    h3_turning,
    h4_angular_velocity,
    h5_er_dynamics,
    h6_directional_er,
    h8_token_entropy_stage,
    h8_token_entropy_all_stages,
    resolve_anchor_layers,
    spectral_effective_rank,
    v1_layer_update_norm,
    v2_raw_state_angle,
    v3_state_angle_demean,
    v4_layer_update_turning,
    v5_vertical_path,
    v6_raw_activation_entropy,
    v7_entropies,
    v8_vertical_er,
)


def test_anchor_resolution_is_runtime_based():
    assert resolve_anchor_layers(37) == (3, 12, 24, 36)


def test_path_geometry_line_and_return():
    line = h2_path(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))
    assert np.isclose(line["straightness"], 1.0)
    assert np.isclose(line["log_detour"], 0.0)
    out_back = h2_path(np.array([[1.0, 0.0], [-1.0, 0.0]]))
    assert out_back["straightness"] < 1e-10
    assert np.isfinite(out_back["log_detour"])


def test_turn_and_angular_velocity_keep_original_positions():
    points = np.vstack(
        [
            np.zeros(2),
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],  # zero displacement creates a two-angle gap
            [2.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
        ]
    )
    progress = np.linspace(0.0, 1.0, len(points))
    angles = h3_turning(points, progress, 1)
    velocity = h4_angular_velocity(points, progress, 3)
    assert angles["excluded_zero_norm"] == 2
    assert angles["n_valid"] == 1
    assert velocity["n_valid"] == 1
    assert velocity["excluded_omega_gap"] == 3


def test_h1_rms_is_euclidean_norm_over_sqrt_dimension():
    points = np.array([[1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    result = h1_movement(points, np.array([0.0, 0.3, 0.6]), 1)
    assert np.isclose(result["median_rms_movement"], 2.0 / np.sqrt(2))


def test_directional_er_is_one_for_one_direction_and_two_for_orthogonal():
    one = h6_directional_er(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), np.array([0, 1, 1.0]), 3)
    two = h6_directional_er(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), np.array([0, 1, 1.0]), 3)
    assert np.isclose(one["directional_ER"], 1.0)
    assert np.isclose(two["directional_ER"], 2.0)


def test_centered_er_is_translation_invariant():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    translated = points + np.array([100.0, -50.0])
    progress = np.linspace(0.0, 1.0, len(points))
    left = h5_er_dynamics(points, progress, 3)
    right = h5_er_dynamics(translated, progress, 3)
    assert np.isclose(left["centered_ERV"], right["centered_ERV"])


def test_vertical_families_return_finite_scalars():
    layers = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    common = layers.mean(axis=0)
    sigma = np.full_like(layers, 0.5)
    outputs = [
        v1_layer_update_norm(layers),
        v2_raw_state_angle(layers),
        v3_state_angle_demean(layers, common),
        v4_layer_update_turning(layers),
        v5_vertical_path(layers),
        v6_raw_activation_entropy(layers),
        v7_entropies(layers, common, sigma),
        v8_vertical_er(layers),
    ]
    for output in outputs:
        assert output["coverage_ok"] is True
        assert all(np.isfinite(value) for key, value in output.items() if key != "coverage_ok" and isinstance(value, (int, float)))


def test_h8_has_three_distinct_entropy_sequences_and_final_stage():
    token_hidden = np.zeros((4, 2, 3), dtype=float)
    token_hidden[:, 0, :] = np.arange(4)[:, None] * np.array([1.0, 0.0, 0.0])
    token_hidden[:, 1, :] = np.arange(4)[:, None] * np.array([0.0, 1.0, 0.0])
    out = h8_token_entropy_stage(
        token_hidden,
        endpoints=np.array([1, 2, 3, 4]),
        progress=np.array([0.25, 0.5, 0.75, 1.0]),
        stage=3,
        layer=1,
    )
    assert out["coverage_ok"] is True
    assert "median_token_time_diff_coordinate_entropy" in out
    assert "median_token_state_coordinate_entropy" in out
    assert "median_token_state_raw_energy_entropy" in out
    rows = h8_token_entropy_all_stages(
        token_hidden,
        endpoints=np.array([1, 2, 3, 4]),
        progress=np.array([0.25, 0.5, 0.75, 1.0]),
        layers=(1,),
    )
    assert len(rows) == 4
    assert rows[3]["median_token_time_diff_coordinate_entropy"] == out[
        "median_token_time_diff_coordinate_entropy"
    ]


def test_entropy_bounds_and_spectral_rank():
    assert np.isclose(energy_entropy(np.ones(8), center=False), 1.0)
    assert np.isclose(energy_entropy(np.array([1.0, 0, 0, 0]), center=False), 0.0)
    assert 1.0 <= spectral_effective_rank(np.eye(3)) <= 3.0


def test_vectorized_entropy_matches_scalar_definition():
    rng = np.random.default_rng(4)
    states = rng.normal(size=(7, 11)).astype(np.float32)
    for center in (False, True):
        expected = np.asarray([energy_entropy(row, center=center) for row in states])
        actual = energy_entropy_rows(states, center=center, zero_value=0.0)
        assert np.allclose(actual, expected, atol=1e-7)
