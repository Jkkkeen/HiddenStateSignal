from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from vertical.config import DatasetConfig, VerticalConfig
from vertical.schema import VerticalRecord

from .base import (
    AdapterInspection,
    build_pooled_record,
    build_raw_record,
    inspect_records,
    resolve_dataset,
)


def read_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported manifest format: {path.suffix}")


def manifest_path(source: Path, dataset: DatasetConfig) -> Path:
    raw = dataset.options.get("manifest")
    if not raw:
        raise ValueError("numpy_directory adapter requires manifest")
    path = Path(str(raw))
    return path if path.is_absolute() else Path(source) / path


class NumpyDirectoryAdapter:
    name = "numpy_directory"

    def iter_records(self, source: Path, config: VerticalConfig) -> Iterator[VerticalRecord]:
        source = Path(source)
        dataset = resolve_dataset(config, source, adapter=self.name)
        tensor_path_column = str(dataset.options.get("tensor_path_column", ""))
        if not tensor_path_column:
            raise ValueError("numpy_directory adapter requires tensor_path_column")
        manifest = read_manifest(manifest_path(source, dataset))
        if tensor_path_column not in manifest:
            raise ValueError(f"manifest is missing tensor path column: {tensor_path_column}")
        tensor_key = str(dataset.options.get("tensor_key", "hidden_states"))
        endpoints_key = str(dataset.options.get("endpoints_key", "endpoints"))
        progress_key = str(dataset.options.get("progress_key", "progress"))
        for row in manifest.to_dict(orient="records"):
            relative = Path(str(row[tensor_path_column]))
            path = relative if relative.is_absolute() else source / relative
            if not path.is_file():
                raise FileNotFoundError(f"manifest tensor path does not exist: {path}")
            if path.suffix.lower() == ".npy":
                if dataset.input_mode != "raw":
                    raise ValueError("pooled numpy_directory records must use NPZ files")
                values = np.load(path, allow_pickle=False, mmap_mode="r")
                yield build_raw_record(dataset, path, row, values)
                continue
            if path.suffix.lower() != ".npz":
                raise ValueError(f"unsupported tensor file format: {path.suffix}")
            with np.load(path, allow_pickle=False) as data:
                if dataset.input_mode == "raw":
                    if tensor_key not in data.files:
                        raise ValueError(f"NPZ source is missing tensor key: {tensor_key}")
                    yield build_raw_record(dataset, path, row, data[tensor_key])
                else:
                    keys = set(config.representations) | {endpoints_key, progress_key}
                    arrays = {key: data[key] for key in data.files if key in keys}
                    yield build_pooled_record(dataset, path, row, arrays, config)

    def inspect(
        self,
        source: Path,
        config: VerticalConfig,
        sample_limit: int,
    ) -> AdapterInspection:
        dataset = resolve_dataset(config, source, adapter=self.name)
        manifest = read_manifest(manifest_path(source, dataset))
        column = str(dataset.options.get("tensor_path_column", ""))
        files = [Path(source) / Path(str(value)) for value in manifest[column].tolist()]
        return inspect_records(
            self.name,
            dataset.input_mode,
            files,
            self.iter_records(source, config),
            sample_limit,
        )

