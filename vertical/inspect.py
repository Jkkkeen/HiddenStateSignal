from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import AdapterInspection, get_adapter
from .config import DatasetConfig, VerticalConfig


METADATA_FIELDS = (
    "record_id",
    "model_family",
    "model_name",
    "condition",
    "question_id",
    "rollout_id",
    "is_correct",
    "response_token_count",
    "num_decoder_layers",
    "hidden_state_count",
    "hidden_dimension",
    "source_path",
)


def _dataset_for_source(source: Path, config: VerticalConfig) -> DatasetConfig:
    source = Path(source)
    matches = [dataset for dataset in config.datasets if Path(dataset.source) == source]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one configured dataset for source={source}")
    return matches[0]


def _manifest_hash(files: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        path = Path(name)
        digest.update(path.as_posix().encode("utf-8"))
        if path.is_file():
            digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def _increment(mapping: dict[str, int], key: str) -> None:
    mapping[key] = mapping.get(key, 0) + 1


def inspect_source(
    source: Path,
    config: VerticalConfig,
    sample_limit: int = 16,
) -> AdapterInspection:
    if sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    source = Path(source)
    try:
        dataset = _dataset_for_source(source, config)
    except ValueError as exc:
        return AdapterInspection(adapter="unknown", input_mode="unknown", errors=[str(exc)])
    adapter = get_adapter(dataset.adapter)
    try:
        report = adapter.inspect(source, config, sample_limit)
        records = []
        for index, record in enumerate(adapter.iter_records(source, config)):
            if index >= sample_limit:
                break
            records.append(record)
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        return AdapterInspection(
            adapter=dataset.adapter,
            input_mode=dataset.input_mode,
            errors=[f"{type(exc).__name__}: {exc}"],
        )

    report.sampled_records = len(records)
    report.input_manifest_sha256 = _manifest_hash(report.files)
    if not records:
        report.errors.append("adapter produced no sampled records")
        return report

    identifiers = [record.metadata.record_id for record in records]
    report.duplicate_record_ids = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if report.duplicate_record_ids:
        report.errors.append(
            f"duplicate sampled record IDs: {report.duplicate_record_ids}"
        )

    coverage = Counter[str]()
    mtp_positions: set[int] = set()
    for record in records:
        metadata = record.metadata
        for name in METADATA_FIELDS:
            value: Any = getattr(metadata, name)
            present = value is not None and (not isinstance(value, str) or bool(value.strip()))
            coverage[name] += int(present)
        extra_start = metadata.num_decoder_layers + 1
        if metadata.hidden_state_count > extra_start:
            mtp_positions.update(range(extra_start, metadata.hidden_state_count))
        for payload in record.payloads.values():
            if payload.kind == "raw":
                boundary = (
                    "explicit_slice"
                    if payload.response_start is not None and payload.response_stop is not None
                    else "response_only"
                )
            elif payload.endpoints is not None and payload.progress is not None:
                boundary = "endpoints_and_progress"
            elif payload.endpoints is not None:
                boundary = "endpoints"
            else:
                boundary = "progress"
            _increment(report.response_boundary_status, boundary)
            if payload.layer_kind is not None:
                kinds = np.asarray(payload.layer_kind, dtype=str)
                mtp_positions.update(
                    int(index)
                    for index, value in enumerate(kinds)
                    if value.lower() not in {"embedding", "decoder"}
                )
    report.metadata_coverage = {
        name: float(coverage[name] / len(records)) for name in METADATA_FIELDS
    }
    report.mtp_candidate_positions = sorted(mtp_positions)
    return report


def write_input_audit(report: AdapterInspection, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)

