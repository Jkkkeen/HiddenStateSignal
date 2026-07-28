from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_long_success_trajectory import (
    PreparedQuery,
    amplitude_score,
    interaction_from_cells,
    main,
    score_prepared_queries,
)


def test_four_cell_interaction_positive_for_separated_paths() -> None:
    cells = {"A_pp": 0.9, "A_pn": 0.2, "A_np": 0.1, "A_nn": 0.8}

    assert interaction_from_cells(cells) == pytest.approx(1.4)


def test_amplitude_suppresses_tiny_noisy_displacement() -> None:
    assert amplitude_score(norm=0.01, s_pos=0.9, s_neg=0.1) == pytest.approx(0.008)


def test_balanced_loo_scores_separated_observed_paths() -> None:
    labels = {0: True, 1: True, 2: True, 3: False, 4: False, 5: False}
    prepared = []
    for rollout_id, correct in labels.items():
        similarities = {}
        for reference_id, reference_correct in labels.items():
            if reference_id == rollout_id:
                continue
            similarities[reference_id] = 0.9 if correct == reference_correct else 0.1
        prepared.append(
            PreparedQuery(
                rollout_id=rollout_id,
                progress_bin=4,
                relative_progress=0.45,
                span_start=256,
                displacement_norm=2.0,
                think_length=3000,
                similarities=similarities,
            )
        )

    span, rollout_bin, rollout, interaction, coverage = score_prepared_queries(
        "q1", prepared, labels
    )

    assert interaction["I_angle"] == pytest.approx(1.6)
    assert interaction["I_amp"] == pytest.approx(3.2)
    assert interaction["pairwise_auc_angle"] == pytest.approx(1.0)
    assert interaction["pairwise_auc_theta"] == pytest.approx(1.0)
    assert interaction["pairwise_auc_amp"] == pytest.approx(1.0)
    assert span.loc[span["is_correct"], "d_theta_deg"].min() > 0
    assert span.loc[~span["is_correct"], "d_theta_deg"].max() < 0
    assert len(span) == 6
    assert len(rollout_bin) == 6
    assert len(rollout) == 6
    assert not coverage.empty


def test_balanced_loo_excludes_requested_reference() -> None:
    labels = {0: True, 1: True, 2: True, 3: False, 4: False, 5: False}
    prepared = []
    for rollout_id in labels:
        prepared.append(
            PreparedQuery(
                rollout_id=rollout_id,
                progress_bin=1,
                relative_progress=0.15,
                span_start=64,
                displacement_norm=1.0,
                think_length=1000,
                similarities={rid: 0.5 for rid in labels if rid != rollout_id},
            )
        )

    _, _, rollout, _, coverage = score_prepared_queries(
        "q1", prepared, labels, excluded_rollout=0
    )

    assert 0 not in set(rollout["rollout_id"])
    assert 0 not in set(coverage["reference_id"])


def test_analyzer_writes_end_to_end_artifacts(tmp_path: Path, monkeypatch) -> None:
    shard_path = tmp_path / "question_q1.npz"
    last_parts = []
    rollout_ids = []
    labels = []
    starts = []
    ends = []
    progress = []
    lengths = []
    for rollout_id in range(6):
        correct = rollout_id < 3
        direction = 1.0 if correct else -1.0
        for span_id in range(4):
            point = np.array([direction * span_id, 0.1 * rollout_id, 0.0], dtype=np.float16)
            last_parts.append(np.stack([point, point], axis=0))
            rollout_ids.append(rollout_id)
            labels.append(correct)
            starts.append(span_id * 64)
            ends.append(span_id * 64 + 128)
            progress.append((span_id + 1) / 5)
            lengths.append(320)
    span_last = np.stack(last_parts, axis=0)
    np.savez_compressed(
        shard_path,
        question_id=np.asarray("q1"),
        layers=np.asarray([24, 36]),
        span_last=span_last,
        span_mean=span_last,
        rollout_id=np.asarray(rollout_ids),
        is_correct=np.asarray(labels),
        span_start=np.asarray(starts),
        span_end=np.asarray(ends),
        relative_progress=np.asarray(progress),
        think_length=np.asarray(lengths),
    )
    output_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_long_success_trajectory.py",
            "--input-glob",
            str(shard_path),
            "--output-dir",
            str(output_dir),
            "--progress-bins",
            "2",
            "--bootstrap",
            "10",
            "--permutations",
            "2",
            "--run-label",
            "Discovery Expansion",
        ],
    )

    main()

    assert (output_dir / "LONG_EXPERIMENT_1_RESULTS.md").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "question_interactions.csv").is_file()
    assert (output_dir / "span_scores.parquet").is_file()
    report = (output_dir / "LONG_EXPERIMENT_1_RESULTS.md").read_text(encoding="utf-8")
    assert report.startswith("# Long Experiment 1 Discovery Expansion Results")
