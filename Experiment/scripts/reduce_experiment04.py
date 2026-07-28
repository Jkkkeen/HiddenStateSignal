#!/usr/bin/env python3
"""Two-pass scalar reduction for Experiment 04 chunk-vector caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment04_metrics import (
    EPS,
    REPRESENTATIONS,
    coordinate_entropy,
    late_window,
    path_metrics,
    state_angles,
)
from extract_experiment04_qwen3vl import (
    AGGREGATES,
    ENTROPY_FAMILIES,
    cache_is_complete,
    question_stem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce Experiment 04 caches.")
    parser.add_argument("--input", required=True, help="Frozen manifest JSONL.")
    parser.add_argument("--extraction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("discovery", "confirm"), required=True)
    parser.add_argument("--calibration", help="Required for confirm mode.")
    parser.add_argument("--rolling", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("manifest contains no rows")
    return rows


def cache_paths(rows: list[dict[str, Any]], extraction_dir: Path) -> list[Path]:
    paths = []
    for row in rows:
        path = (
            extraction_dir
            / "cache"
            / question_stem(str(row["question_id"]))
            / f"rollout_{int(row['rollout_id'])}.npz"
        )
        if not cache_is_complete(path, expected_layers=37):
            raise FileNotFoundError(f"missing or incomplete rollout cache: {path}")
        paths.append(path)
    return paths


def compute_calibration(paths: list[Path]) -> dict[str, np.ndarray]:
    common_sum: np.ndarray | None = None
    rollout_vertical_medians = []
    count = 0
    partial_only = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as cached:
            vectors = cached["full_vectors"].astype(np.float32)
        if vectors.ndim != 4:
            raise ValueError(f"cache has malformed full chunks: {path}")
        if vectors.shape[0] == 0:
            # A rollout shorter than one chunk is valid terminal-partial data,
            # but cannot contribute to full-chunk calibration statistics.
            partial_only += 1
            continue
        if common_sum is None:
            common_sum = vectors.mean(axis=0).astype(np.float64)
        else:
            common_sum += vectors.mean(axis=0).astype(np.float64)
        update_norm = np.linalg.norm(np.diff(vectors, axis=2), axis=-1)
        rollout_vertical_medians.append(np.median(update_norm, axis=0))
        count += 1
    if common_sum is None or count == 0:
        raise ValueError("no caches supplied for calibration")
    common = (common_sum / count).astype(np.float32)
    vertical_scale = np.median(np.stack(rollout_vertical_medians), axis=0).astype(np.float32)
    vertical_threshold = np.maximum(1e-8, 1e-4 * vertical_scale).astype(np.float32)
    return {
        "common": common,
        "vertical_scale": vertical_scale,
        "vertical_threshold": vertical_threshold,
        "calibration_rollout_count": np.asarray(count, dtype=np.int64),
        "partial_only_rollout_count": np.asarray(partial_only, dtype=np.int64),
    }


def _metadata(cached: Any) -> dict[str, Any]:
    return {
        "question_id": str(cached["question_id"].item()),
        "rollout_id": int(cached["rollout_id"].item()),
        "is_correct": bool(cached["is_correct"].item()),
        "think_length": int(cached["think_length"].item()),
        "response_length": int(cached["response_length"].item()),
    }


def _local_arrays(
    points: np.ndarray,
    *,
    common: np.ndarray,
    rolling: int,
) -> dict[str, np.ndarray | float]:
    count = points.shape[0]
    geometry = path_metrics(points, rolling=rolling)
    arrays = {
        "pooled_raw_entropy": coordinate_entropy(points),
        "step_norm": np.full(count, np.nan),
        "state_angle_raw": np.full(count, np.nan),
        "state_angle_demean": np.full(count, np.nan),
        "diff_entropy": np.full(count, np.nan),
        "turn_cos": np.full(count, np.nan),
        "turn_angle": np.full(count, np.nan),
        "rolling_path_length": np.full(count, np.nan),
        "rolling_net_displacement": np.full(count, np.nan),
        "rolling_straightness": np.full(count, np.nan),
        "rolling_log_detour": np.full(count, np.nan),
    }
    arrays["step_norm"][1:] = geometry["step_norm"]
    arrays["state_angle_raw"][1:] = state_angles(points)
    arrays["state_angle_demean"][1:] = state_angles(points, common=common)
    arrays["diff_entropy"][1:] = coordinate_entropy(geometry["displacement"])
    if count >= 3:
        arrays["turn_cos"][2:] = geometry["turn_cos"]
        arrays["turn_angle"][2:] = geometry["turn_angle"]
    rolling_count = np.asarray(geometry["rolling_straightness"]).size
    if rolling_count:
        start = rolling
        arrays["rolling_path_length"][start : start + rolling_count] = geometry[
            "rolling_path_length"
        ]
        arrays["rolling_net_displacement"][start : start + rolling_count] = geometry[
            "rolling_net_displacement"
        ]
        arrays["rolling_straightness"][start : start + rolling_count] = geometry[
            "rolling_straightness"
        ]
        arrays["rolling_log_detour"][start : start + rolling_count] = geometry[
            "rolling_log_detour"
        ]
    arrays.update(
        {
            "path_length": float(geometry["path_length"]),
            "net_displacement": float(geometry["net_displacement"]),
            "straightness": float(geometry["straightness"]),
            "log_detour": float(geometry["log_detour"]),
            "direction_consistency": float(geometry["direction_consistency"]),
            "turn_angle_std": float(geometry["turn_angle_std"]),
            "turn_angle_late_std": float(geometry["turn_angle_late_std"]),
        }
    )
    return arrays


def _token_columns(values: np.ndarray, chunk_id: int, layer: int) -> dict[str, float]:
    columns = {}
    for family_id, family in enumerate(ENTROPY_FAMILIES):
        for aggregate_id, aggregate in enumerate(AGGREGATES):
            columns[f"token_{family}_entropy_{aggregate}"] = float(
                values[chunk_id, layer, family_id, aggregate_id]
            )
    return columns


def _summary_row(
    metadata: dict[str, Any],
    representation: str,
    direction: str,
    geometry: dict[str, Any],
    *,
    layer: int = -1,
    chunk_id: int = -1,
    is_partial: bool = False,
) -> dict[str, Any]:
    finite_steps = np.asarray(geometry["step_norm"], dtype=np.float64)
    finite_steps = finite_steps[np.isfinite(finite_steps)]
    return {
        **metadata,
        "representation": representation,
        "direction": direction,
        "layer": layer,
        "chunk_id": chunk_id,
        "is_partial": is_partial,
        "step_norm_mean": float(np.mean(finite_steps)) if finite_steps.size else np.nan,
        "step_norm_median": float(np.median(finite_steps)) if finite_steps.size else np.nan,
        "path_length": geometry["path_length"],
        "net_displacement": geometry["net_displacement"],
        "straightness": geometry["straightness"],
        "log_detour": geometry["log_detour"],
        "direction_consistency": geometry["direction_consistency"],
        "turn_angle_std": geometry["turn_angle_std"],
        "turn_angle_late_std": geometry["turn_angle_late_std"],
    }


def _masked_vertical_geometry(
    points: np.ndarray,
    threshold: np.ndarray,
    *,
    rolling: int,
) -> dict[str, Any]:
    geometry = path_metrics(points, rolling=rolling)
    turn_cos = np.asarray(geometry["turn_cos"], dtype=np.float64).copy()
    if turn_cos.size:
        norms = np.asarray(geometry["step_norm"])
        valid = (norms[:-1] > threshold[:-1]) & (norms[1:] > threshold[1:])
        turn_cos[~valid] = np.nan
    turn_angle = np.arccos(turn_cos)
    finite = turn_angle[np.isfinite(turn_angle)]
    late = late_window(finite)
    geometry["turn_cos"] = turn_cos
    geometry["turn_angle"] = turn_angle
    geometry["direction_consistency"] = (
        float(np.nanmean(turn_cos)) if np.any(np.isfinite(turn_cos)) else np.nan
    )
    geometry["turn_angle_std"] = float(np.std(finite, ddof=1)) if finite.size >= 2 else np.nan
    geometry["turn_angle_late_std"] = (
        float(np.std(late, ddof=1)) if late.size >= 2 else np.nan
    )
    return geometry


def reduce_cache(path: Path, calibration: dict[str, np.ndarray], rolling: int = 4) -> dict[str, pd.DataFrame]:
    with np.load(path, allow_pickle=False) as cached:
        full_vectors = cached["full_vectors"].astype(np.float32)
        partial_vectors = cached["partial_vectors"].astype(np.float32)
        full_bounds = cached["full_bounds"].astype(np.int32)
        partial_bounds = cached["partial_bounds"].astype(np.int32)
        token_full = cached["token_entropy_full"].astype(np.float32)
        token_partial = cached["token_entropy_partial"].astype(np.float32)
        metadata = _metadata(cached)

    common = calibration["common"]
    scale = calibration["vertical_scale"]
    threshold = calibration["vertical_threshold"]
    horizontal_rows = []
    vertical_rows = []
    summary_rows = []
    full_count = full_vectors.shape[0]

    for rep_id, representation in enumerate(REPRESENTATIONS):
        for layer in range(full_vectors.shape[2]):
            points = full_vectors[:, rep_id, layer]
            local = _local_arrays(points, common=common[rep_id, layer], rolling=rolling)
            summary_rows.append(
                _summary_row(metadata, representation, "horizontal", local, layer=layer)
            )
            for chunk_id in range(full_count):
                row = {
                    **metadata,
                    "representation": representation,
                    "layer": layer,
                    "chunk_id": chunk_id,
                    "chunk_start": int(full_bounds[chunk_id, 0]),
                    "chunk_end": int(full_bounds[chunk_id, 1]),
                    "relative_progress": float((chunk_id + 0.5) / full_count),
                    "end_aligned_chunk": int(chunk_id - full_count + 1),
                    "is_partial": False,
                }
                for name, values in local.items():
                    if isinstance(values, np.ndarray) and values.shape == (full_count,):
                        row[name] = float(values[chunk_id])
                row.update(_token_columns(token_full, chunk_id, layer))
                horizontal_rows.append(row)

        for is_partial, vectors, bounds, token_values in (
            (False, full_vectors, full_bounds, token_full),
            (True, partial_vectors, partial_bounds, token_partial),
        ):
            for chunk_id in range(vectors.shape[0]):
                points = vectors[chunk_id, rep_id]
                geometry = _masked_vertical_geometry(points, threshold[rep_id], rolling=rolling)
                raw_angle = state_angles(points)
                demean_angle = state_angles(points - common[rep_id])
                raw_entropy = coordinate_entropy(points)
                diff_entropy = coordinate_entropy(geometry["displacement"])
                summary_rows.append(
                    _summary_row(
                        metadata,
                        representation,
                        "vertical",
                        geometry,
                        chunk_id=chunk_id,
                        is_partial=is_partial,
                    )
                )
                for layer in range(points.shape[0]):
                    row = {
                        **metadata,
                        "representation": representation,
                        "layer": layer,
                        "chunk_id": chunk_id,
                        "chunk_start": int(bounds[chunk_id, 0]),
                        "chunk_end": int(bounds[chunk_id, 1]),
                        "relative_progress": float((chunk_id + 0.5) / max(full_count, 1))
                        if not is_partial
                        else 1.0,
                        "end_aligned_chunk": int(chunk_id - full_count + 1)
                        if not is_partial
                        else 1,
                        "is_partial": is_partial,
                        "pooled_raw_entropy": float(raw_entropy[layer]),
                        "step_norm": float(geometry["step_norm"][layer - 1])
                        if layer >= 1
                        else np.nan,
                        "step_norm_scaled": float(
                            geometry["step_norm"][layer - 1] / (scale[rep_id, layer - 1] + EPS)
                        )
                        if layer >= 1
                        else np.nan,
                        "state_angle_raw": float(raw_angle[layer - 1]) if layer >= 1 else np.nan,
                        "state_angle_demean": float(demean_angle[layer - 1])
                        if layer >= 1
                        else np.nan,
                        "diff_entropy": float(diff_entropy[layer - 1]) if layer >= 1 else np.nan,
                        "turn_cos": float(geometry["turn_cos"][layer - 2])
                        if layer >= 2
                        else np.nan,
                        "turn_angle": float(geometry["turn_angle"][layer - 2])
                        if layer >= 2
                        else np.nan,
                        "rolling_path_length": np.nan,
                        "rolling_net_displacement": np.nan,
                        "rolling_straightness": np.nan,
                        "rolling_log_detour": np.nan,
                    }
                    rolling_id = layer - rolling
                    if rolling_id >= 0 and rolling_id < len(geometry["rolling_straightness"]):
                        row["rolling_path_length"] = float(geometry["rolling_path_length"][rolling_id])
                        row["rolling_net_displacement"] = float(
                            geometry["rolling_net_displacement"][rolling_id]
                        )
                        row["rolling_straightness"] = float(
                            geometry["rolling_straightness"][rolling_id]
                        )
                        row["rolling_log_detour"] = float(geometry["rolling_log_detour"][rolling_id])
                    row.update(_token_columns(token_values, chunk_id, layer))
                    vertical_rows.append(row)

    return {
        "horizontal": pd.DataFrame(horizontal_rows),
        "vertical": pd.DataFrame(vertical_rows),
        "summary": pd.DataFrame(summary_rows),
    }


class ParquetAppender:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: Any | None = None

    def append(self, frame: pd.DataFrame) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def save_calibration(path: Path, calibration: dict[str, np.ndarray], manifest_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        **calibration,
        source_role=np.asarray("discovery"),
        manifest_sha256=np.asarray(manifest_sha256),
    )
    temporary.replace(path)


def load_calibration(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["source_role"].item()) != "discovery":
            raise ValueError("calibration must originate from discovery")
        return {
            "common": cached["common"].astype(np.float32),
            "vertical_scale": cached["vertical_scale"].astype(np.float32),
            "vertical_threshold": cached["vertical_threshold"].astype(np.float32),
        }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.input)
    rows = load_manifest(manifest_path)
    roles = {str(row.get("experiment04_split", "")) for row in rows}
    if roles != {args.mode}:
        raise ValueError(f"manifest role mismatch: expected {args.mode}, got {roles}")
    paths = cache_paths(rows, Path(args.extraction_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256_file(manifest_path)
    calibration_path = output_dir / "calibration.npz"
    if args.mode == "discovery":
        calibration = compute_calibration(paths)
        save_calibration(calibration_path, calibration, manifest_sha)
    else:
        if not args.calibration:
            raise ValueError("--calibration is required in confirm mode")
        calibration = load_calibration(Path(args.calibration))

    writers = {
        name: ParquetAppender(output_dir / f"{name}_metrics.parquet")
        for name in ("horizontal", "vertical", "summary")
    }
    try:
        for index, path in enumerate(paths, start=1):
            frames = reduce_cache(path, calibration, rolling=args.rolling)
            for name, frame in frames.items():
                writers[name].append(frame)
            print(f"reduced {index}/{len(paths)} {path}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    audit = {
        "mode": args.mode,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "rollout_caches": len(paths),
        "representations": list(REPRESENTATIONS),
        "hidden_state_indexes": 37,
        "rolling_displacements": args.rolling,
    }
    if args.mode == "discovery":
        audit.update(
            {
                "calibration_rollouts": int(calibration["calibration_rollout_count"].item()),
                "partial_only_rollouts": int(
                    calibration["partial_only_rollout_count"].item()
                ),
            }
        )
    (output_dir / "reduction_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
