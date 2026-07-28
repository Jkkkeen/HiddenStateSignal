from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from extract_experiment04_qwen3vl import save_rollout_cache  # noqa: E402
from reduce_experiment04 import ParquetAppender, compute_calibration, reduce_cache  # noqa: E402


def _cache(path: Path, offset: float, rollout_id: int) -> Path:
    chunks, reps, layers, dim = 4, 3, 4, 3
    vectors = np.zeros((chunks, reps, layers, dim), dtype=np.float16)
    for chunk in range(chunks):
        for rep in range(reps):
            for layer in range(layers):
                vectors[chunk, rep, layer] = [offset + chunk, rep, layer]
    reduced = {
        "full_vectors": vectors,
        "partial_vectors": vectors[:1],
        "full_bounds": np.asarray([[0, 4], [4, 8], [8, 12], [12, 16]], dtype=np.int32),
        "partial_bounds": np.asarray([[16, 18]], dtype=np.int32),
        "token_entropy_full": np.ones((4, 4, 3, 4), dtype=np.float32),
        "token_entropy_partial": np.ones((1, 4, 3, 4), dtype=np.float32),
    }
    save_rollout_cache(
        path,
        reduced,
        metadata={
            "question_id": "q",
            "rollout_id": rollout_id,
            "is_correct": rollout_id == 0,
            "think_length": 18,
            "response_length": 18,
            "chunk_size": 4,
        },
    )
    return path


def _partial_only_cache(path: Path, rollout_id: int) -> Path:
    reps, layers, dim = 3, 4, 3
    reduced = {
        "full_vectors": np.empty((0, reps, layers, dim), dtype=np.float16),
        "partial_vectors": np.zeros((1, reps, layers, dim), dtype=np.float16),
        "full_bounds": np.empty((0, 2), dtype=np.int32),
        "partial_bounds": np.asarray([[0, 7]], dtype=np.int32),
        "token_entropy_full": np.empty((0, layers, 3, 4), dtype=np.float32),
        "token_entropy_partial": np.ones((1, layers, 3, 4), dtype=np.float32),
    }
    save_rollout_cache(
        path,
        reduced,
        metadata={
            "question_id": "partial",
            "rollout_id": rollout_id,
            "is_correct": False,
            "think_length": 7,
            "response_length": 7,
            "chunk_size": 256,
        },
    )
    return path


def test_calibration_is_rollout_equal_and_has_vertical_thresholds(tmp_path: Path) -> None:
    paths = [_cache(tmp_path / "a.npz", 0.0, 0), _cache(tmp_path / "b.npz", 10.0, 1)]
    calibration = compute_calibration(paths)

    assert calibration["common"].shape == (3, 4, 3)
    assert calibration["vertical_scale"].shape == (3, 3)
    assert calibration["vertical_threshold"].shape == (3, 3)
    assert np.allclose(
        calibration["vertical_threshold"],
        np.maximum(1e-8, 1e-4 * calibration["vertical_scale"]),
    )


def test_partial_only_rollout_is_skipped_for_calibration(tmp_path: Path) -> None:
    full = _cache(tmp_path / "full.npz", 0.0, 0)
    partial = _partial_only_cache(tmp_path / "partial.npz", 1)

    calibration = compute_calibration([full, partial])

    assert calibration["calibration_rollout_count"].item() == 1
    assert calibration["partial_only_rollout_count"].item() == 1
    frames = reduce_cache(partial, calibration)
    assert frames["horizontal"].empty
    assert not frames["vertical"].empty
    assert frames["vertical"]["is_partial"].all()


def test_parquet_appender_ignores_empty_frames(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    writer = ParquetAppender(path)
    writer.append(pd.DataFrame({"value": [1.0]}))
    writer.append(pd.DataFrame())
    writer.close()

    assert pd.read_parquet(path)["value"].tolist() == [1.0]


def test_reduce_cache_emits_full_horizontal_and_partial_vertical_only(tmp_path: Path) -> None:
    path = _cache(tmp_path / "a.npz", 0.0, 0)
    calibration = compute_calibration([path])
    frames = reduce_cache(path, calibration)

    horizontal = frames["horizontal"]
    vertical = frames["vertical"]
    summary = frames["summary"]
    assert len(horizontal) == 4 * 4 * 3
    assert not horizontal["is_partial"].any()
    assert vertical["is_partial"].any()
    assert set(vertical["layer"]) == {0, 1, 2, 3}
    assert "turn_angle_mean" not in horizontal.columns
    assert {"turn_angle_std", "turn_angle_late_std"} <= set(summary.columns)
    assert "step_norm_scaled" in vertical.columns
    assert horizontal["state_angle_demean"].notna().any()
