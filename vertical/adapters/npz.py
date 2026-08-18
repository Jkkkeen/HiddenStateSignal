from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from vertical.config import VerticalConfig
from vertical.schema import VerticalRecord

from .base import (
    AdapterInspection,
    build_pooled_record,
    build_raw_record,
    inspect_records,
    resolve_dataset,
)


def _npz_files(source: Path) -> list[Path]:
    source = Path(source)
    files = [source] if source.is_file() else sorted(source.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no NPZ files under {source}")
    return files


def _metadata(data: Any, key: str) -> dict[str, Any]:
    if key not in data.files:
        raise ValueError(f"NPZ source is missing metadata key: {key}")
    raw = data[key]
    if np.asarray(raw).size != 1:
        raise ValueError("metadata JSON array must contain one scalar value")
    value = np.asarray(raw).reshape(()).item()
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise ValueError("metadata JSON must decode to a mapping")
    return decoded


class NpzAdapter:
    name = "npz"

    def iter_records(self, source: Path, config: VerticalConfig) -> Iterator[VerticalRecord]:
        dataset = resolve_dataset(config, source, adapter=self.name)
        metadata_key = str(dataset.options.get("metadata_json_key", "metadata_json"))
        tensor_key = str(dataset.options.get("tensor_key", "hidden_states"))
        for path in _npz_files(source):
            with np.load(path, allow_pickle=False) as data:
                raw_metadata = _metadata(data, metadata_key)
                if dataset.input_mode == "raw":
                    if tensor_key not in data.files:
                        raise ValueError(f"NPZ source is missing tensor key: {tensor_key}")
                    yield build_raw_record(dataset, path, raw_metadata, data[tensor_key])
                else:
                    arrays = {key: data[key] for key in data.files if key != metadata_key}
                    yield build_pooled_record(dataset, path, raw_metadata, arrays, config)

    def inspect(
        self,
        source: Path,
        config: VerticalConfig,
        sample_limit: int,
    ) -> AdapterInspection:
        dataset = resolve_dataset(config, source, adapter=self.name)
        files = _npz_files(source)
        return inspect_records(
            self.name,
            dataset.input_mode,
            files,
            self.iter_records(source, config),
            sample_limit,
        )

