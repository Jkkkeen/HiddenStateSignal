from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from extract_experiment04_qwen3vl import save_rollout_cache  # noqa: E402
from reduce_experiment04 import compute_calibration, reduce_cache  # noqa: E402


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
