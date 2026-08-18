from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vertical.config import DatasetConfig, VerticalConfig
from vertical.schema import HiddenPayload, RecordMetadata, VerticalRecord


@dataclass
class AdapterInspection:
    adapter: str
    input_mode: str
    files: list[str] = field(default_factory=list)
    sample_shapes: list[tuple[int, ...]] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata_coverage: dict[str, float] = field(default_factory=dict)
    finite_rates: list[float] = field(default_factory=list)
    record_count: int = 0

    @property
    def passed(self) -> bool:
        return not self.errors


class VerticalAdapter(Protocol):
    def inspect(
        self,
        source: Path,
        config: VerticalConfig,
        sample_limit: int,
    ) -> AdapterInspection: ...

    def iter_records(
        self,
        source: Path,
        config: VerticalConfig,
    ) -> Iterator[VerticalRecord]: ...


def resolve_dataset(
    config: VerticalConfig,
    source: Path,
    *,
    adapter: str,
) -> DatasetConfig:
    source = Path(source)
    matches = [
        dataset
        for dataset in config.datasets
        if dataset.adapter == adapter and Path(dataset.source) == source
    ]
    if not matches:
        matches = [dataset for dataset in config.datasets if dataset.adapter == adapter]
    if len(matches) != 1:
        raise ValueError(
            f"expected one dataset for adapter={adapter} source={source}, found {len(matches)}"
        )
    return matches[0]


def axis_order(dataset: DatasetConfig) -> tuple[str, str, str]:
    raw = dataset.options.get("axis_order")
    if isinstance(raw, str):
        axes = tuple(item.strip() for item in raw.split(","))
    elif isinstance(raw, (list, tuple)):
        axes = tuple(str(item).strip() for item in raw)
    else:
        raise ValueError("axis_order must be an explicit comma-separated string or list")
    if len(axes) != 3 or len(set(axes)) != 3:
        raise ValueError("axis_order must contain three unique axes")
    expected = {"layer", "dim", "token" if dataset.input_mode == "raw" else "endpoint"}
    if set(axes) != expected:
        raise ValueError(f"axis_order must contain exactly {sorted(expected)}")
    return axes  # type: ignore[return-value]


def normalize_axes(values: np.ndarray, dataset: DatasetConfig) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError(f"hidden tensor must be 3D, found shape={values.shape}")
    axes = axis_order(dataset)
    leading = "token" if dataset.input_mode == "raw" else "endpoint"
    permutation = (axes.index(leading), axes.index("layer"), axes.index("dim"))
    return np.transpose(values, permutation)


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)


def build_metadata(
    dataset: DatasetConfig,
    source_path: Path,
    raw_metadata: Mapping[str, Any],
    values: np.ndarray,
) -> RecordMetadata:
    question_id = str(raw_metadata.get("question_id", "")).strip()
    rollout_id = str(raw_metadata.get("rollout_id", "")).strip()
    if not question_id or not rollout_id:
        raise ValueError("source metadata requires question_id and rollout_id")
    response_token_count = raw_metadata.get("response_token_count")
    if response_token_count is None:
        start = raw_metadata.get("response_start")
        stop = raw_metadata.get("response_stop")
        if start is not None and stop is not None:
            response_token_count = int(stop) - int(start)
        elif dataset.input_mode == "raw" and bool(dataset.options.get("response_only", False)):
            response_token_count = values.shape[0]
        else:
            raise ValueError(
                "response_token_count is required unless response_only is explicitly true"
            )
    decoder_layers = dataset.decoder_layers
    if decoder_layers is None:
        decoder_layers = _optional_int(raw_metadata.get("num_decoder_layers"))
    if decoder_layers is None:
        raise ValueError("num_decoder_layers is required when config uses auto")
    hidden_dimension = dataset.hidden_dimension or values.shape[2]
    is_correct_raw = raw_metadata.get("is_correct")
    is_correct = None if is_correct_raw is None else bool(is_correct_raw)
    canonical = {
        "record_id",
        "question_id",
        "rollout_id",
        "is_correct",
        "response_token_count",
        "num_decoder_layers",
        "checkpoint",
        "global_step",
        "training_progress",
        "response_start",
        "response_stop",
        "layer_kind",
    }
    metadata = RecordMetadata(
        record_id=str(raw_metadata.get("record_id") or f"{question_id}:{rollout_id}"),
        model_family=dataset.model_family,
        model_name=dataset.model_name,
        condition=dataset.condition,
        question_id=question_id,
        rollout_id=rollout_id,
        is_correct=is_correct,
        response_token_count=int(response_token_count),
        num_decoder_layers=int(decoder_layers),
        hidden_state_count=int(values.shape[1]),
        hidden_dimension=int(hidden_dimension),
        source_path=str(source_path),
        checkpoint=(
            None if raw_metadata.get("checkpoint") is None else str(raw_metadata["checkpoint"])
        ),
        global_step=_optional_int(raw_metadata.get("global_step")),
        training_progress=(
            None
            if raw_metadata.get("training_progress") is None
            else float(raw_metadata["training_progress"])
        ),
        extras={key: value for key, value in raw_metadata.items() if key not in canonical},
    )
    metadata.validate()
    return metadata


def layer_kind(raw_metadata: Mapping[str, Any]) -> np.ndarray | None:
    value = raw_metadata.get("layer_kind")
    if value is None:
        return None
    return np.asarray(value, dtype=str)


def build_raw_record(
    dataset: DatasetConfig,
    source_path: Path,
    raw_metadata: Mapping[str, Any],
    physical_values: np.ndarray,
) -> VerticalRecord:
    values = normalize_axes(physical_values, dataset)
    metadata = build_metadata(dataset, source_path, raw_metadata, values)
    payload = HiddenPayload(
        kind="raw",
        values=values,
        response_start=_optional_int(raw_metadata.get("response_start")),
        response_stop=_optional_int(raw_metadata.get("response_stop")),
        layer_kind=layer_kind(raw_metadata),
    )
    record = VerticalRecord(metadata=metadata, payloads={"raw": payload})
    record.validate()
    return record


def build_pooled_record(
    dataset: DatasetConfig,
    source_path: Path,
    raw_metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    config: VerticalConfig,
) -> VerticalRecord:
    available = [name for name in config.representations if name in arrays]
    if not available:
        raise ValueError("pooled source contains none of the configured representations")
    normalized = {name: normalize_axes(arrays[name], dataset) for name in available}
    metadata = build_metadata(dataset, source_path, raw_metadata, normalized[available[0]])
    endpoints_key = str(dataset.options.get("endpoints_key", "endpoints"))
    progress_key = str(dataset.options.get("progress_key", "progress"))
    endpoints = arrays.get(endpoints_key)
    progress = arrays.get(progress_key)
    payloads = {
        name: HiddenPayload(
            kind="pooled",
            values=values,
            representation=name,
            endpoints=None if endpoints is None else np.asarray(endpoints, dtype=int),
            progress=None if progress is None else np.asarray(progress, dtype=float),
            layer_kind=layer_kind(raw_metadata),
        )
        for name, values in normalized.items()
    }
    record = VerticalRecord(metadata=metadata, payloads=payloads)
    record.validate()
    return record


def inspect_records(
    adapter_name: str,
    input_mode: str,
    files: list[Path],
    records: Iterator[VerticalRecord],
    sample_limit: int,
) -> AdapterInspection:
    report = AdapterInspection(
        adapter=adapter_name,
        input_mode=input_mode,
        files=[str(path) for path in files],
        record_count=len(files),
    )
    for index, record in enumerate(records):
        if index >= sample_limit:
            break
        for name, payload in record.payloads.items():
            values = np.asarray(payload.values)
            report.keys.append(name)
            report.sample_shapes.append(tuple(int(item) for item in values.shape))
            report.dtypes.append(str(values.dtype))
            report.finite_rates.append(float(np.isfinite(values).mean()))
    report.keys = sorted(set(report.keys))
    report.dtypes = sorted(set(report.dtypes))
    return report

