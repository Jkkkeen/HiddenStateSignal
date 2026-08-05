from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .common import sha256_file, write_json_atomic
from .metrics import base_layer_common


REPRESENTATIONS = ("mean_w128_s32", "last_s32")


def cache_name(rollout_id: str) -> str:
    from .common import sha256_text

    return f"{sha256_text(rollout_id)[:20]}.npz"


def write_pooled_cache(
    path: Path,
    *,
    pooled: dict[str, np.ndarray],
    endpoints: np.ndarray,
    progress: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            mean_w128_s32=np.asarray(pooled["mean_w128_s32"], dtype=np.float16),
            last_s32=np.asarray(pooled["last_s32"], dtype=np.float16),
            endpoints=np.asarray(endpoints, dtype=np.int32),
            progress=np.asarray(progress, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
    os.replace(temporary, path)


def load_pooled_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "mean_w128_s32": data["mean_w128_s32"].astype(np.float32),
            "last_s32": data["last_s32"].astype(np.float32),
            "endpoints": data["endpoints"].astype(int),
            "progress": data["progress"].astype(float),
            "metadata": json.loads(str(data["metadata_json"])),
        }


def fit_base_calibrators(cache_paths: list[Path], output_path: Path) -> dict[str, Any]:
    if not cache_paths:
        raise ValueError("base calibrators require at least one pooled cache")
    trajectories: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATIONS}
    chunk_sum: dict[str, np.ndarray] = {}
    chunk_square_sum: dict[str, np.ndarray] = {}
    chunk_count = {name: 0 for name in REPRESENTATIONS}
    layer_shape = None
    rollout_ids = []
    for path in sorted(cache_paths):
        cache = load_pooled_cache(path)
        rollout_ids.append(cache["metadata"]["rollout_id"])
        for representation in REPRESENTATIONS:
            trajectory = cache[representation].astype(np.float64)
            if trajectory.ndim != 3:
                raise ValueError(f"invalid pooled trajectory in {path}: {trajectory.shape}")
            layer_shape = trajectory.shape[1:]
            trajectories[representation].append(trajectory)
            chunk_sum.setdefault(representation, np.zeros(layer_shape, dtype=np.float64))
            chunk_square_sum.setdefault(representation, np.zeros(layer_shape, dtype=np.float64))
            chunk_sum[representation] += trajectory.sum(axis=0)
            chunk_square_sum[representation] += (trajectory * trajectory).sum(axis=0)
            chunk_count[representation] += trajectory.shape[0]

    arrays: dict[str, np.ndarray] = {}
    for representation in REPRESENTATIONS:
        common = base_layer_common(trajectories[representation]).astype(np.float32)
        mean = chunk_sum[representation] / chunk_count[representation]
        variance = np.maximum(chunk_square_sum[representation] / chunk_count[representation] - mean * mean, 0.0)
        arrays[f"common__{representation}"] = common
        arrays[f"coordinate_mean__{representation}"] = mean.astype(np.float32)
        arrays[f"coordinate_sigma__{representation}"] = np.sqrt(variance).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, output_path)
    audit = {
        "n_rollouts": len(cache_paths),
        "n_unique_rollouts": len(set(rollout_ids)),
        "representations": list(REPRESENTATIONS),
        "layer_shape": list(layer_shape or ()),
        "chunk_count": chunk_count,
        "calibrator_sha256": sha256_file(output_path),
        "label_blind": True,
        "common_weighting": "rollout-equal after within-rollout chunk mean",
        "coordinate_weighting": "chunk-equal across all base rollouts",
    }
    write_json_atomic(output_path.with_suffix(".audit.json"), audit)
    return audit


def load_calibrators(path: Path, representation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    with np.load(path, allow_pickle=False) as data:
        return (
            data[f"common__{representation}"].astype(np.float32),
            data[f"coordinate_mean__{representation}"].astype(np.float32),
            data[f"coordinate_sigma__{representation}"].astype(np.float32),
        )
