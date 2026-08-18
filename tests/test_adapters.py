import json
from pathlib import Path

import numpy as np
import pytest

from vertical.adapters import get_adapter
from vertical.config import DatasetConfig, VerticalConfig


def dataset_config(source: Path, *, adapter: str = "npz", input_mode: str = "raw", **options):
    return DatasetConfig(
        name="sample",
        input_mode=input_mode,
        adapter=adapter,
        source=source,
        model_family="qwen3",
        model_name="Qwen3-8B-Base",
        condition="base",
        decoder_layers=4,
        hidden_dimension=8,
        options=options,
    )


def vertical_config(dataset: DatasetConfig, representations=("last_s32",)):
    return VerticalConfig(
        run_id="adapter_test",
        output_root=Path("runs/adapter_test"),
        datasets=(dataset,),
        representations=representations,
    )


def metadata(**overrides):
    values = {
        "record_id": "q1:r0",
        "question_id": "q1",
        "rollout_id": "r0",
        "is_correct": True,
        "response_token_count": 6,
        "num_decoder_layers": 4,
    }
    values.update(overrides)
    return values


def test_npz_adapter_normalizes_declared_raw_axes(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    physical = np.arange(5 * 6 * 8, dtype=np.float32).reshape(5, 6, 8)
    np.savez(
        source / "record.npz",
        hidden_states=physical,
        metadata_json=np.asarray(json.dumps(metadata())),
    )
    dataset = dataset_config(
        source,
        tensor_key="hidden_states",
        metadata_json_key="metadata_json",
        axis_order="layer,token,dim",
    )

    records = list(get_adapter("npz").iter_records(source, vertical_config(dataset)))

    assert len(records) == 1
    assert records[0].payloads["raw"].values.shape == (6, 5, 8)
    assert np.array_equal(records[0].payloads["raw"].values[:, 0, :], physical[0])
    records[0].validate()


def test_npz_adapter_loads_pooled_representations(tmp_path):
    source = tmp_path / "pooled"
    source.mkdir()
    values = np.arange(3 * 5 * 8, dtype=np.float32).reshape(3, 5, 8)
    np.savez(
        source / "record.npz",
        last_s32=values,
        endpoints=np.array([2, 4, 6], dtype=np.int32),
        progress=np.array([1 / 3, 2 / 3, 1.0], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata())),
    )
    dataset = dataset_config(
        source,
        input_mode="pooled",
        metadata_json_key="metadata_json",
        axis_order="endpoint,layer,dim",
        endpoints_key="endpoints",
        progress_key="progress",
    )

    record = next(get_adapter("npz").iter_records(source, vertical_config(dataset)))

    assert set(record.payloads) == {"last_s32"}
    assert record.payloads["last_s32"].values.shape == (3, 5, 8)
    record.validate()


def test_numpy_directory_adapter_reads_manifest_and_npy(tmp_path):
    source = tmp_path / "directory"
    source.mkdir()
    np.save(source / "r0.npy", np.zeros((6, 5, 8), dtype=np.float32))
    row = metadata(hidden_path="r0.npy")
    (source / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = dataset_config(
        source,
        adapter="numpy_directory",
        manifest="manifest.jsonl",
        tensor_path_column="hidden_path",
        axis_order="token,layer,dim",
    )

    record = next(
        get_adapter("numpy_directory").iter_records(source, vertical_config(dataset))
    )

    assert record.metadata.source_path.endswith("r0.npy")
    assert record.payloads["raw"].values.shape == (6, 5, 8)


def test_custom_adapter_requires_explicit_axis_order(tmp_path):
    source = tmp_path / "custom"
    source.mkdir()
    dataset = dataset_config(
        source,
        adapter="custom_template",
        manifest="manifest.jsonl",
        tensor_path_column="hidden_path",
    )

    with pytest.raises(ValueError, match="axis_order"):
        list(get_adapter("custom_template").iter_records(source, vertical_config(dataset)))


def test_adapter_registry_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown adapter"):
        get_adapter("unknown")

