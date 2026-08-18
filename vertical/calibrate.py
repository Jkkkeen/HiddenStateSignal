from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pooling import pool_response_states, trajectory_endpoints
from .profiles import select_decoder_backbone
from .schema import VerticalRecord


SIGMA_FLOOR = 1e-4
Z_CLIP = 8.0


@dataclass(frozen=True)
class BaseCalibrator:
    common: np.ndarray
    coordinate_mean: np.ndarray
    coordinate_sigma: np.ndarray
    model_family: str
    representation: str
    input_sha256: str
    n_records: int
    n_chunks: int

    def validate(self) -> None:
        common = np.asarray(self.common)
        if common.ndim != 2 or common.shape != np.asarray(self.coordinate_mean).shape:
            raise ValueError("calibrator arrays must share shape (layer, dim)")
        if common.shape != np.asarray(self.coordinate_sigma).shape:
            raise ValueError("calibrator arrays must share shape (layer, dim)")
        if not all(
            np.isfinite(array).all()
            for array in (self.common, self.coordinate_mean, self.coordinate_sigma)
        ):
            raise ValueError("calibrator arrays must be finite")
        if np.any(np.asarray(self.coordinate_sigma) < 0):
            raise ValueError("coordinate_sigma must be nonnegative")
        if not self.model_family or not self.representation:
            raise ValueError("calibrator identity must be nonempty")
        if len(self.input_sha256) != 64:
            raise ValueError("calibrator input_sha256 must be a SHA256 hex digest")
        if self.n_records < 1 or self.n_chunks < 1:
            raise ValueError("calibrator requires positive record and chunk counts")


def trajectory_for_record(record: VerticalRecord, representation: str) -> np.ndarray:
    record.validate()
    if representation in record.payloads:
        payload = record.payloads[representation]
        if payload.kind != "pooled":
            raise ValueError("named representation payload must be pooled")
        values = np.asarray(payload.values, dtype=np.float32)
        return select_decoder_backbone(
            values,
            num_decoder_layers=record.metadata.num_decoder_layers,
            layer_kind=payload.layer_kind,
        )
    raw = record.payloads.get("raw")
    if raw is None or raw.kind != "raw":
        raise ValueError(f"record does not provide representation: {representation}")
    states = np.asarray(raw.values)
    if raw.response_start is not None and raw.response_stop is not None:
        states = states[raw.response_start : raw.response_stop]
    endpoints = trajectory_endpoints(record.metadata.response_token_count)
    pooled = pool_response_states(states, endpoints)
    if representation not in pooled:
        raise ValueError(f"unknown pooled representation: {representation}")
    return select_decoder_backbone(
        pooled[representation],
        num_decoder_layers=record.metadata.num_decoder_layers,
        layer_kind=raw.layer_kind,
    )


def fit_base_calibrator(
    records: Iterable[VerticalRecord],
    representation: str,
) -> BaseCalibrator:
    materialized = list(records)
    if not materialized:
        raise ValueError("Base calibrator requires at least one record")
    if any(record.metadata.condition != "base" for record in materialized):
        raise ValueError("Base calibrator accepts Base records only")
    families = {record.metadata.model_family for record in materialized}
    if len(families) != 1:
        raise ValueError("Base calibrator records must share one model_family")

    family = next(iter(families))
    common_sum: np.ndarray | None = None
    coordinate_sum: np.ndarray | None = None
    coordinate_square_sum: np.ndarray | None = None
    chunk_count = 0
    digest = hashlib.sha256()
    expected_shape: tuple[int, int] | None = None
    for record in materialized:
        trajectory = trajectory_for_record(record, representation).astype(np.float64)
        if trajectory.ndim != 3 or trajectory.shape[0] < 1:
            raise ValueError("calibration trajectory must have shape (chunk, layer, dim)")
        shape = (int(trajectory.shape[1]), int(trajectory.shape[2]))
        if expected_shape is None:
            expected_shape = shape
            common_sum = np.zeros(shape, dtype=np.float64)
            coordinate_sum = np.zeros(shape, dtype=np.float64)
            coordinate_square_sum = np.zeros(shape, dtype=np.float64)
        elif shape != expected_shape:
            raise ValueError("Base calibrator trajectories must share layer and hidden dimensions")
        assert common_sum is not None
        assert coordinate_sum is not None
        assert coordinate_square_sum is not None
        common_sum += trajectory.mean(axis=0)
        coordinate_sum += trajectory.sum(axis=0)
        coordinate_square_sum += np.square(trajectory).sum(axis=0)
        chunk_count += int(trajectory.shape[0])
        digest.update(record.metadata.record_id.encode("utf-8"))
        digest.update(record.metadata.source_path.encode("utf-8"))
        digest.update(np.asarray(trajectory, dtype=np.float32).tobytes())

    assert common_sum is not None
    assert coordinate_sum is not None
    assert coordinate_square_sum is not None
    common = common_sum / len(materialized)
    mean = coordinate_sum / chunk_count
    variance = np.maximum(coordinate_square_sum / chunk_count - mean * mean, 0.0)
    calibrator = BaseCalibrator(
        common=common.astype(np.float32),
        coordinate_mean=mean.astype(np.float32),
        coordinate_sigma=np.sqrt(variance).astype(np.float32),
        model_family=family,
        representation=representation,
        input_sha256=digest.hexdigest(),
        n_records=len(materialized),
        n_chunks=chunk_count,
    )
    calibrator.validate()
    return calibrator


def save_calibrator(calibrator: BaseCalibrator, path: Path) -> None:
    calibrator.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    metadata = {
        "model_family": calibrator.model_family,
        "representation": calibrator.representation,
        "input_sha256": calibrator.input_sha256,
        "n_records": calibrator.n_records,
        "n_chunks": calibrator.n_chunks,
    }
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            common=np.asarray(calibrator.common, dtype=np.float32),
            coordinate_mean=np.asarray(calibrator.coordinate_mean, dtype=np.float32),
            coordinate_sigma=np.asarray(calibrator.coordinate_sigma, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(temporary, path)


def load_calibrator(path: Path, representation: str) -> BaseCalibrator:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).reshape(()).item()))
        calibrator = BaseCalibrator(
            common=data["common"].astype(np.float32),
            coordinate_mean=data["coordinate_mean"].astype(np.float32),
            coordinate_sigma=data["coordinate_sigma"].astype(np.float32),
            model_family=str(metadata["model_family"]),
            representation=str(metadata["representation"]),
            input_sha256=str(metadata["input_sha256"]),
            n_records=int(metadata["n_records"]),
            n_chunks=int(metadata["n_chunks"]),
        )
    calibrator.validate()
    if calibrator.representation != representation:
        raise ValueError(
            f"calibrator representation mismatch: {calibrator.representation} != {representation}"
        )
    return calibrator


def standardize_layers(
    layers: np.ndarray,
    calibrator: BaseCalibrator,
    *,
    model_family: str,
    representation: str,
    sigma_floor: float = SIGMA_FLOOR,
    z_clip: float = Z_CLIP,
) -> np.ndarray:
    calibrator.validate()
    if model_family != calibrator.model_family:
        raise ValueError(
            f"calibrator model_family mismatch: {model_family} != {calibrator.model_family}"
        )
    if representation != calibrator.representation:
        raise ValueError(
            "calibrator representation mismatch: "
            f"{representation} != {calibrator.representation}"
        )
    values = np.asarray(layers, dtype=np.float32)
    if values.shape[-2:] != calibrator.coordinate_mean.shape:
        raise ValueError("layer values do not match calibrator shape")
    if sigma_floor <= 0 or z_clip <= 0:
        raise ValueError("sigma_floor and z_clip must be positive")
    denominator = np.maximum(calibrator.coordinate_sigma, sigma_floor)
    return np.clip(
        (values - calibrator.coordinate_mean) / denominator,
        -z_clip,
        z_clip,
    )
