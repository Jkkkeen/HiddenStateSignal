from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_experiment0_hidden_dynamics_qwen3vl import (
    aggregate_span_features,
    question_stem,
    reduce_rollout_hidden,
    valid_completed_question,
)


def _hidden_states() -> tuple[torch.Tensor, ...]:
    states = []
    for layer in range(4):
        values = torch.zeros((1, 12, 4), dtype=torch.float32)
        for token in range(12):
            values[0, token] = torch.tensor(
                [token + layer, token - layer, 2 * layer, -layer], dtype=torch.float32
            )
        states.append(values)
    return tuple(states)


def test_reduce_hidden_emits_all_layers_and_span_representations() -> None:
    reduced = reduce_rollout_hidden(
        _hidden_states(),
        segment_start=0,
        segment_end=12,
        progress_bins=3,
        span_specs=((4, 2), (6, 3)),
        endpoint_spec=(4, 2),
        question_id="q1",
        rollout_id=0,
        is_correct=True,
        think_length=12,
    )

    assert set(reduced.token_features["layer"]) == {0, 1, 2, 3}
    assert set(reduced.token_features["progress_bin"]) == {0, 1, 2}
    assert {
        "horizontal_norm_mean",
        "vertical_norm_p90",
        "coordinate_entropy_mean",
        "token_turn_cos_mean",
    }.issubset(reduced.token_features.columns)
    assert set(reduced.span_horizontal["representation"]) == {
        "mean_w4_s2",
        "mean_w6_s3",
        "last_w4_s2",
    }
    assert set(reduced.span_horizontal["layer"]) == {0, 1, 2, 3}
    assert set(reduced.audit_vectors) == {"mean_w4_s2", "mean_w6_s3", "last_w4_s2"}
    assert reduced.audit_vectors["mean_w4_s2"].dtype == np.float16


def test_aggregate_span_features_has_one_row_per_rollout_bin_layer_representation() -> None:
    reduced = reduce_rollout_hidden(
        _hidden_states(),
        segment_start=0,
        segment_end=12,
        progress_bins=3,
        span_specs=((4, 2),),
        endpoint_spec=(4, 2),
        question_id="q1",
        rollout_id=0,
        is_correct=True,
        think_length=12,
    )

    frame = aggregate_span_features(reduced.span_horizontal, reduced.span_vertical)

    keys = ["question_id", "rollout_id", "representation", "progress_bin", "layer"]
    assert not frame.duplicated(keys).any()
    assert {
        "span_turn_cos_mean",
        "span_layer_turn_cos_mean",
        "span_turn_cos_split_a",
        "span_turn_cos_split_b",
        "span_count",
    }.issubset(frame.columns)


def test_non_audit_reduction_keeps_vectors_only_for_primary_cross_rollout_geometry() -> None:
    reduced = reduce_rollout_hidden(
        _hidden_states(),
        segment_start=0,
        segment_end=12,
        progress_bins=3,
        span_specs=((4, 2), (6, 3)),
        endpoint_spec=(4, 2),
        primary_spec=(6, 3),
        question_id="q1",
        rollout_id=0,
        is_correct=True,
        think_length=12,
        retain_audit_vectors=False,
    )

    assert reduced.audit_vectors == {}
    assert reduced.audit_metadata == {}
    primary = reduced.span_horizontal["representation"] == "mean_w6_s3"
    assert reduced.span_horizontal.loc[primary, "displacement"].notna().all()
    assert reduced.span_horizontal.loc[~primary, "displacement"].isna().all()


def test_short_rollout_skips_unavailable_sensitivity_span_spec() -> None:
    reduced = reduce_rollout_hidden(
        _hidden_states(),
        segment_start=0,
        segment_end=12,
        progress_bins=3,
        span_specs=((4, 2), (10, 5)),
        endpoint_spec=(4, 2),
        primary_spec=(4, 2),
        question_id="q1",
        rollout_id=0,
        is_correct=True,
        think_length=12,
    )

    assert set(reduced.span_horizontal["representation"]) == {
        "mean_w4_s2",
        "last_w4_s2",
    }


def test_question_complete_requires_marker_and_all_parquet_schemas(tmp_path: Path) -> None:
    stem = question_stem("q1")
    (tmp_path / "bin_features").mkdir()
    (tmp_path / "span_features").mkdir()
    (tmp_path / "prototype_diagnostics").mkdir()
    (tmp_path / "pairwise_geometry").mkdir()
    (tmp_path / "completed").mkdir()
    marker = tmp_path / "completed" / f"{stem}.complete.json"
    marker.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    assert valid_completed_question(tmp_path, stem) is False

    pd.DataFrame(
        [{"question_id": "q1", "rollout_id": 0, "progress_bin": 0, "layer": 0}]
    ).to_parquet(tmp_path / "bin_features" / f"{stem}.parquet", index=False)
    pd.DataFrame(
        [
            {
                "question_id": "q1",
                "rollout_id": 0,
                "representation": "mean_w4_s2",
                "progress_bin": 0,
                "layer": 0,
            }
        ]
    ).to_parquet(tmp_path / "span_features" / f"{stem}.parquet", index=False)
    pd.DataFrame(
        [
            {
                "question_id": "q1",
                "rollout_id": 0,
                "representation": "mean_w4_s2",
                "progress_bin": 0,
                "layer": 0,
                "kappa_pos": 1.0,
                "kappa_neg": 1.0,
            }
        ]
    ).to_parquet(tmp_path / "prototype_diagnostics" / f"{stem}.parquet", index=False)

    assert valid_completed_question(tmp_path, stem) is False

    pd.DataFrame(
        [
            {
                "question_id": "q1",
                "rollout_id": 0,
                "representation": "mean_w4_s2",
                "progress_bin": 0,
                "layer": 0,
                "span_id": 1,
                "reference_rollout_id": 1,
                "reference_norm": 1.0,
                "cosine_similarity": 0.5,
            }
        ]
    ).to_parquet(tmp_path / "pairwise_geometry" / f"{stem}.parquet", index=False)

    assert valid_completed_question(tmp_path, stem) is True


def test_question_stem_is_stable_and_safe() -> None:
    assert question_stem("some/question:id") == question_stem("some/question:id")
    assert "/" not in question_stem("some/question:id")
    assert ":" not in question_stem("some/question:id")
