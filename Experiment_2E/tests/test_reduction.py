import numpy as np

from experiment_2e.reduction import (
    pool_token_hidden,
    reduce_h8,
    reduce_horizontal,
    reduce_vertical,
    trajectory_endpoints,
)


def _token_hidden(tokens: int = 256, layers: int = 26, dimension: int = 8) -> np.ndarray:
    time = np.arange(tokens, dtype=float)[:, None, None]
    depth = np.arange(layers, dtype=float)[None, :, None]
    coords = np.arange(1, dimension + 1, dtype=float)[None, None, :]
    return np.sin(time / (coords + 3.0)) + depth / (coords + 10.0)


def test_endpoints_include_terminal_once_and_pool_both_representations():
    endpoints = trajectory_endpoints(257)
    assert endpoints[-1] == 257
    assert len(endpoints) == len(set(endpoints.tolist()))
    hidden = _token_hidden(tokens=257)
    pooled = pool_token_hidden(hidden, endpoints)
    assert set(pooled) == {"mean_w128_s32", "last_s32"}
    assert pooled["mean_w128_s32"].shape == pooled["last_s32"].shape
    np.testing.assert_allclose(pooled["last_s32"][-1], hidden[-1])


def test_reduction_emits_all_primary_families_and_failed_coverage_rows():
    hidden = _token_hidden()
    endpoints = trajectory_endpoints(len(hidden))
    progress = endpoints / len(hidden)
    pooled = pool_token_hidden(hidden, endpoints)["mean_w128_s32"]
    metadata = {"run_id": "r", "checkpoint": "base", "question_id": "q", "rollout_id": "x"}
    common = pooled.mean(axis=0)
    sigma = np.full_like(common, 0.5)
    horizontal = reduce_horizontal(pooled, progress, representation="mean_w128_s32", metadata=metadata)
    vertical = reduce_vertical(
        pooled,
        progress,
        representation="mean_w128_s32",
        metadata=metadata,
        base_common=common,
        coordinate_common=common,
        coordinate_sigma=sigma,
    )
    assert {row["family_id"] for row in horizontal} == {f"H{i}" for i in range(1, 8)}
    assert {row["family_id"] for row in vertical} == {f"V{i}" for i in range(1, 9)}
    assert all("coverage" in row and "value" in row for row in horizontal + vertical)
    assert any(not row["coverage"] and np.isnan(row["value"]) for row in horizontal + vertical)


def test_h8_reduction_is_token_representation_only():
    hidden = _token_hidden(tokens=160)
    endpoints = trajectory_endpoints(len(hidden))
    rows = reduce_h8(
        hidden,
        endpoints,
        endpoints / len(hidden),
        metadata={"run_id": "r", "checkpoint": "base", "question_id": "q", "rollout_id": "x"},
    )
    assert rows
    assert {row["representation"] for row in rows} == {"token"}
    assert {row["family_id"] for row in rows} == {"H8"}
    primary = [row for row in rows if row["is_primary_metric"]]
    assert len(primary) == hidden.shape[1] * 4
