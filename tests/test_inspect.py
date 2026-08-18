import json
from pathlib import Path

import numpy as np

from vertical.config import DatasetConfig, VerticalConfig
from vertical.inspect import inspect_source, write_input_audit


def make_config(source: Path, **options):
    dataset = DatasetConfig(
        name="qwen_base",
        input_mode="raw",
        adapter="npz",
        source=source,
        model_family="qwen3",
        model_name="Qwen3-8B-Base",
        condition="base",
        decoder_layers=4,
        hidden_dimension=8,
        options={
            "tensor_key": "hidden_states",
            "metadata_json_key": "metadata_json",
            "axis_order": "token,layer,dim",
            **options,
        },
    )
    return VerticalConfig(
        run_id="inspect_test",
        output_root=Path("runs/inspect_test"),
        datasets=(dataset,),
        representations=("last_s32",),
    )


def write_npz(source: Path, name: str, metadata: dict, *, layers: int = 5):
    np.savez(
        source / name,
        hidden_states=np.zeros((6, layers, 8), dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def base_metadata(**overrides):
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


def test_inspector_reports_shape_metadata_and_response_coverage(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    write_npz(source, "record.npz", base_metadata())

    report = inspect_source(source, make_config(source), sample_limit=4)

    assert report.errors == []
    assert report.sample_shapes == [(6, 5, 8)]
    assert report.metadata_coverage["question_id"] == 1.0
    assert report.response_boundary_status == {"response_only": 1}
    assert report.sampled_records == 1
    assert len(report.input_manifest_sha256) == 64


def test_inspector_returns_error_for_missing_response_boundary(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    metadata = base_metadata()
    metadata.pop("response_token_count")
    write_npz(source, "record.npz", metadata)

    report = inspect_source(source, make_config(source), sample_limit=4)

    assert not report.passed
    assert any("response_token_count" in error for error in report.errors)


def test_inspector_rejects_duplicate_sampled_record_ids(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    write_npz(source, "a.npz", base_metadata())
    write_npz(source, "b.npz", base_metadata())

    report = inspect_source(source, make_config(source), sample_limit=4)

    assert report.duplicate_record_ids == ["q1:r0"]
    assert any("duplicate" in error for error in report.errors)


def test_inspector_reports_nondecoder_layer_candidates(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    write_npz(
        source,
        "record.npz",
        base_metadata(layer_kind=["embedding", "decoder", "decoder", "decoder", "decoder", "mtp"]),
        layers=6,
    )

    report = inspect_source(source, make_config(source), sample_limit=4)

    assert report.mtp_candidate_positions == [5]


def test_input_audit_write_is_json_and_atomic(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    write_npz(source, "record.npz", base_metadata())
    report = inspect_source(source, make_config(source), sample_limit=4)
    output = tmp_path / "audit" / "input_audit.json"

    write_input_audit(report, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["sampled_records"] == 1
    assert not output.with_suffix(".json.tmp").exists()

