from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .calibrate import BaseCalibrator, trajectory_and_progress_for_record
from .profiles import PROFILE_COLUMNS, reduce_stage_profiles
from .schema import RecordMetadata, VerticalRecord


OPTIONAL_PROFILE_FIELDS = (
    "reward",
    "format_correct",
    "difficulty",
    "policy_entropy",
    "token_logprob_mean",
    "hidden_norm_mean",
    "trajectory_point_count",
    "rollout_slot",
    "response_hash",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


@dataclass(frozen=True)
class ProfileAudit:
    passed: bool
    expected_rows: int
    actual_rows: int
    n_records: int
    representations: tuple[str, ...]
    finite_coverage: Mapping[str, float]
    structural_nan_checks: Mapping[str, bool]
    gates: Mapping[str, bool]
    input_sha256: str
    output_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "expected_rows": self.expected_rows,
            "actual_rows": self.actual_rows,
            "n_records": self.n_records,
            "representations": list(self.representations),
            "finite_coverage": dict(self.finite_coverage),
            "structural_nan_checks": dict(self.structural_nan_checks),
            "gates": dict(self.gates),
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
        }


def calibrator_key(model_family: str, representation: str) -> str:
    return f"{model_family}::{representation}"


def _metadata_row(metadata: RecordMetadata) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_id": metadata.record_id,
        "model_family": metadata.model_family,
        "model_name": metadata.model_name,
        "condition": metadata.condition,
        "question_id": metadata.question_id,
        "rollout_id": metadata.rollout_id,
        "is_correct": metadata.is_correct,
        "response_token_count": metadata.response_token_count,
        "num_decoder_layers": metadata.num_decoder_layers,
        "hidden_state_count": metadata.hidden_state_count,
        "hidden_dimension": metadata.hidden_dimension,
        "source_path": metadata.source_path,
        "checkpoint": metadata.checkpoint,
        "global_step": metadata.global_step,
        "training_progress": metadata.training_progress,
    }
    row.update({name: metadata.extras.get(name) for name in OPTIONAL_PROFILE_FIELDS})
    return row


def _profile_structural_checks(frame: pd.DataFrame) -> dict[str, bool]:
    checks = {
        "v1_raw_layer0_nan": frame.loc[
            frame["layer_index"] == 0, "v1_raw_update_norm"
        ].isna().all(),
        "v1_relative_layer0_nan": frame.loc[
            frame["layer_index"] == 0, "v1_relative_update_norm"
        ].isna().all(),
        "v2_layer0_nan": frame.loc[
            frame["layer_index"] == 0, "v2_raw_state_angle"
        ].isna().all(),
        "v3_layer0_nan": frame.loc[
            frame["layer_index"] == 0, "v3_demean_state_angle"
        ].isna().all(),
        "v4_layers0_1_nan": frame.loc[
            frame["layer_index"] < 2, "v4_layer_update_turning_angle"
        ].isna().all(),
        "v7_layer0_nan": frame.loc[
            frame["layer_index"] == 0, "v7_layer_difference_entropy"
        ].isna().all(),
        "v5_rolling_first3_nan": frame.loc[
            frame["layer_index"] < 4, "v5_rolling_path_length"
        ].isna().all(),
    }
    return {name: bool(value) for name, value in checks.items()}


def _finite_coverage(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in PROFILE_COLUMNS:
        coverage_column = f"profile_coverage_count_{metric}"
        possible = frame[coverage_column].gt(0)
        result[metric] = (
            float(frame.loc[possible, metric].notna().mean()) if possible.any() else 0.0
        )
    return result


def write_profile_partition(
    records: Iterable[VerticalRecord],
    calibrators: Mapping[str, BaseCalibrator],
    output_path: Path,
) -> ProfileAudit:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    expected_rows = 0
    record_count = 0
    seen_ids: set[tuple[Any, ...]] = set()
    representations: set[str] = set()
    digest = hashlib.sha256()
    try:
        for record in records:
            record.validate()
            identity = (
                record.metadata.model_family,
                record.metadata.model_name,
                record.metadata.condition,
                record.metadata.checkpoint,
                record.metadata.global_step,
                record.metadata.record_id,
            )
            if identity in seen_ids:
                raise ValueError(f"duplicate profile record: {identity}")
            seen_ids.add(identity)
            record_count += 1
            family_prefix = f"{record.metadata.model_family}::"
            family_calibrators = {
                key.removeprefix(family_prefix): value
                for key, value in calibrators.items()
                if key.startswith(family_prefix)
            }
            if not family_calibrators:
                raise ValueError(
                    f"missing calibrator for model_family={record.metadata.model_family}"
                )
            digest.update(record.metadata.record_id.encode("utf-8"))
            digest.update(record.metadata.source_path.encode("utf-8"))
            for representation, calibrator in sorted(family_calibrators.items()):
                if calibrator.model_family != record.metadata.model_family:
                    raise ValueError("calibrator mapping key does not match calibrator family")
                if calibrator.representation != representation:
                    raise ValueError(
                        "calibrator mapping key does not match calibrator representation"
                    )
                trajectory, progress = trajectory_and_progress_for_record(
                    record,
                    representation,
                )
                if trajectory.shape[1:] != calibrator.common.shape:
                    raise ValueError("record trajectory does not match calibrator shape")
                frame = reduce_stage_profiles(
                    trajectory,
                    progress,
                    representation=representation,
                    metadata=_metadata_row(record.metadata),
                    base_common=calibrator.common,
                )
                expected_rows += len(frame)
                representations.add(representation)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema)
                elif table.schema != writer.schema:
                    table = table.cast(writer.schema)
                writer.write_table(table)
        if writer is None or expected_rows == 0:
            raise ValueError("profile partition requires at least one record")
        writer.close()
        writer = None
        os.replace(temporary, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise

    actual_rows = int(pq.ParquetFile(output_path).metadata.num_rows)
    frame = pd.read_parquet(output_path)
    unique_rows = not frame.duplicated(
        [
            "model_family",
            "model_name",
            "condition",
            "checkpoint",
            "global_step",
            "record_id",
            "representation",
            "stage",
            "layer_index",
        ]
    ).any()
    structural = _profile_structural_checks(frame)
    gates = {
        "row_count": actual_rows == expected_rows,
        "unique_rows": bool(unique_rows),
        "structural_nan_checks": all(structural.values()),
        "output_nonempty": output_path.stat().st_size > 0,
    }
    audit = ProfileAudit(
        passed=bool(all(gates.values())),
        expected_rows=expected_rows,
        actual_rows=actual_rows,
        n_records=record_count,
        representations=tuple(sorted(representations)),
        finite_coverage=_finite_coverage(frame),
        structural_nan_checks=structural,
        gates=gates,
        input_sha256=digest.hexdigest(),
        output_sha256=sha256_file(output_path),
    )
    write_json_atomic(audit.to_dict(), output_path.with_suffix(".audit.json"))
    if not audit.passed:
        raise RuntimeError(f"profile partition audit failed: {audit.to_dict()}")
    return audit


def load_completed_partition(path: Path, audit_path: Path) -> pd.DataFrame:
    path = Path(path)
    audit_path = Path(audit_path)
    if not path.is_file() or not audit_path.is_file():
        raise ValueError("completed partition requires both Parquet and audit files")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("completed partition audit is unreadable") from exc
    if not audit.get("passed"):
        raise ValueError("completed partition audit did not pass")
    frame = pd.read_parquet(path)
    if "actual_rows" in audit and int(audit["actual_rows"]) != len(frame):
        raise ValueError("completed partition audit row count does not match")
    if "output_sha256" in audit and audit["output_sha256"] != sha256_file(path):
        raise ValueError("completed partition audit hash does not match")
    return frame
