from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment0_hidden_dynamics import (
    aggregate_token_dynamics,
    build_span_direction_records,
    coordinate_energy_entropy,
    pool_span_vectors,
    prototype_shrinkage,
    score_cross_rollout_queries,
)


def test_coordinate_energy_entropy_distinguishes_concentrated_and_uniform_energy() -> None:
    concentrated = np.array([[3.0, -1.0, -1.0, -1.0]])
    spread = np.array([[1.0, -1.0, 1.0, -1.0]])

    h_concentrated, n_concentrated = coordinate_energy_entropy(concentrated)
    h_spread, n_spread = coordinate_energy_entropy(spread)

    assert h_concentrated[0] < h_spread[0]
    assert n_concentrated[0] < n_spread[0]
    assert 0.0 <= h_concentrated[0] <= 1.0
    assert 0.0 <= h_spread[0] <= 1.0


def test_coordinate_energy_entropy_zero_update_is_finite() -> None:
    entropy, effective = coordinate_energy_entropy(np.zeros((2, 4)))

    np.testing.assert_allclose(entropy, 0.0)
    np.testing.assert_allclose(effective, 1.0)


def test_prototype_shrinkage_detects_cancellation() -> None:
    aligned = np.array([[1.0, 0.0], [1.0, 0.0]])
    cancelling = np.array([[1.0, 0.0], [-1.0, 0.0]])

    assert prototype_shrinkage(aligned) == pytest.approx(1.0)
    assert prototype_shrinkage(cancelling) == pytest.approx(0.0)


def test_pool_and_span_direction_records_use_pooled_displacements() -> None:
    hidden = np.zeros((3, 8, 2), dtype=np.float32)
    hidden[:, :, 0] = np.arange(8, dtype=np.float32)
    hidden[1, :, 1] = 1.0
    hidden[2, :, 1] = 3.0

    pooled, starts, ends, progress = pool_span_vectors(
        hidden, window=4, stride=2, mode="mean"
    )
    horizontal, vertical = build_span_direction_records(
        pooled,
        layers=np.array([0, 1, 2]),
        starts=starts,
        ends=ends,
        relative_progress=progress,
        rollout_id=7,
        is_correct=True,
        representation="mean_w4_s2",
        progress_bins=2,
    )

    assert pooled.shape == (3, 3, 2)
    assert len(horizontal) == 2 * 3
    assert horizontal["displacement_norm"].min() == pytest.approx(2.0)
    assert horizontal.dropna(subset=["span_turn_cos"])["span_turn_cos"].min() == pytest.approx(1.0)
    assert set(vertical["layer"]) == {1, 2}
    assert vertical.loc[vertical["layer"] == 2, "span_layer_turn_cos"].notna().all()


def _synthetic_cross_records() -> pd.DataFrame:
    rows = []
    labels = {0: True, 1: True, 2: True, 3: False, 4: False, 5: False}
    for rollout_id, correct in labels.items():
        direction = 1.0 if correct else -1.0
        for span_id, rho in enumerate((0.42, 0.48)):
            rows.append(
                {
                    "question_id": "q1",
                    "rollout_id": rollout_id,
                    "is_correct": correct,
                    "representation": "mean_w128_s64",
                    "span_id": span_id,
                    "span_start": span_id * 64,
                    "span_end": span_id * 64 + 128,
                    "relative_progress": rho,
                    "progress_bin": 4,
                    "layer": 1,
                    "displacement": np.array([direction, 0.01 * span_id]),
                    "displacement_norm": float(np.hypot(direction, 0.01 * span_id)),
                    "span_turn_cos": np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_cross_rollout_scores_separated_paths_and_balanced_subsets() -> None:
    scores, diagnostics = score_cross_rollout_queries(_synthetic_cross_records())

    assert len(scores) == 12
    assert scores.loc[scores["is_correct"], "cross_set_direction"].min() > 1.9
    assert scores.loc[~scores["is_correct"], "cross_set_direction"].max() < -1.9
    assert scores["balanced_subset_count"].eq(3).all()
    assert diagnostics["n_positive_references"].eq(2).all()
    assert diagnostics["n_negative_references"].eq(2).all()
    assert diagnostics["prototype_valid_020"].all()


def test_cross_rollout_selects_one_nearest_span_per_reference_rollout() -> None:
    records = _synthetic_cross_records()
    scores, diagnostics = score_cross_rollout_queries(records)
    query = scores[(scores["rollout_id"] == 0) & (scores["span_id"] == 0)].iloc[0]
    subset = diagnostics[
        (diagnostics["rollout_id"] == 0)
        & (diagnostics["span_id"] == 0)
        & (diagnostics["subset_id"] == 0)
    ].iloc[0]

    assert query["reference_rollout_count_pos"] == 2
    assert query["reference_rollout_count_neg"] == 2
    assert len(subset["positive_reference_ids"].split(",")) == 2
    assert len(subset["negative_reference_ids"].split(",")) == 2


def test_aggregate_token_dynamics_emits_horizontal_vertical_and_entropy() -> None:
    hidden = np.zeros((3, 6, 4), dtype=np.float32)
    for layer in range(3):
        for token in range(6):
            hidden[layer, token] = np.array(
                [token + layer, token - layer, layer, -layer], dtype=np.float32
            )

    frame = aggregate_token_dynamics(hidden, progress_bins=3)

    assert set(frame["layer"]) == {0, 1, 2}
    assert frame.loc[frame["layer"] == 0, "vertical_norm_mean"].isna().all()
    assert frame.loc[frame["layer"] >= 1, "vertical_norm_mean"].notna().all()
    assert frame.loc[frame["layer"] >= 1, "coordinate_entropy_mean"].between(0, 1).all()
    assert frame["horizontal_token_count"].sum() == 3 * 5
