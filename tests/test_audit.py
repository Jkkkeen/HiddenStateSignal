import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vertical.audit import load_completed_partition, write_profile_partition
from vertical.calibrate import fit_base_calibrator
from vertical.schema import HiddenPayload, RecordMetadata, VerticalRecord


def make_record(record_id: str, *, offset: float = 0.0, correct: bool = False):
    values = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5) + offset + 1
    question_id, rollout_id = record_id.split(":")
    metadata = RecordMetadata(
        record_id=record_id,
        model_family="qwen3",
        model_name="Qwen3-Test",
        condition="base",
        question_id=question_id,
        rollout_id=rollout_id,
        is_correct=correct,
        response_token_count=8,
        num_decoder_layers=2,
        hidden_state_count=3,
        hidden_dimension=5,
        source_path=f"{record_id}.npz",
        extras={"reward": float(correct)},
    )
    payload = HiddenPayload(
        kind="pooled",
        values=values,
        representation="last_s32",
        endpoints=np.array([2, 4, 6, 8]),
        progress=np.array([0.25, 0.5, 0.75, 1.0]),
    )
    return VerticalRecord(metadata=metadata, payloads={"last_s32": payload})


def test_profile_partition_has_expected_unique_rows_and_audit(tmp_path):
    records = [make_record("q1:r0"), make_record("q2:r0", offset=2, correct=True)]
    calibrator = fit_base_calibrator(records, "last_s32")
    path = tmp_path / "profiles" / "qwen_base.parquet"

    audit = write_profile_partition(
        records,
        {"qwen3::last_s32": calibrator},
        path,
    )

    frame = pd.read_parquet(path)
    assert audit.passed
    assert audit.expected_rows == audit.actual_rows == 2 * 4 * 3
    assert frame.groupby(
        ["record_id", "representation", "stage", "layer_index"]
    ).size().max() == 1
    assert frame.loc[frame["layer_index"] == 0, "v1_raw_update_norm"].isna().all()
    assert frame.loc[frame["layer_index"] < 2, "v4_layer_update_turning_angle"].isna().all()
    assert (path.with_suffix(".audit.json")).is_file()
    assert not path.with_suffix(".parquet.tmp").exists()


def test_completed_partition_requires_passing_audit(tmp_path):
    path = tmp_path / "profiles.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
    audit_path = tmp_path / "profiles.audit.json"
    audit_path.write_text(json.dumps({"passed": False}), encoding="utf-8")

    with pytest.raises(ValueError, match="audit"):
        load_completed_partition(path, audit_path)


def test_missing_audit_is_not_treated_as_complete(tmp_path):
    path = tmp_path / "profiles.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="audit"):
        load_completed_partition(path, tmp_path / "missing.json")


def test_wrong_family_calibrator_is_rejected(tmp_path):
    records = [make_record("q1:r0")]
    calibrator = fit_base_calibrator(records, "last_s32")

    with pytest.raises(ValueError, match="calibrator"):
        write_profile_partition(
            records,
            {"mimo::last_s32": calibrator},
            tmp_path / "profiles.parquet",
        )

