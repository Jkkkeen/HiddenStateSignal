from dataclasses import replace

import numpy as np
import pytest

from vertical.schema import HiddenPayload, RecordMetadata, VerticalRecord


def make_metadata(**overrides):
    values = {
        "record_id": "q1:r0",
        "model_family": "qwen3",
        "model_name": "Qwen3-8B-Base",
        "condition": "base",
        "question_id": "q1",
        "rollout_id": "r0",
        "is_correct": True,
        "response_token_count": 4,
        "num_decoder_layers": 4,
        "hidden_state_count": 5,
        "hidden_dimension": 8,
        "source_path": "sample.npz",
    }
    values.update(overrides)
    return RecordMetadata(**values)


def test_raw_payload_accepts_response_only_tensor():
    payload = HiddenPayload(
        kind="raw",
        values=np.zeros((4, 5, 8), dtype=np.float32),
    )
    record = VerticalRecord(metadata=make_metadata(), payloads={"raw": payload})

    record.validate()


def test_raw_payload_accepts_explicit_response_slice():
    payload = HiddenPayload(
        kind="raw",
        values=np.zeros((9, 5, 8), dtype=np.float32),
        response_start=3,
        response_stop=7,
    )
    record = VerticalRecord(metadata=make_metadata(), payloads={"raw": payload})

    record.validate()


def test_invalid_layer_count_fails_closed():
    payload = HiddenPayload(
        kind="raw",
        values=np.zeros((4, 4, 8), dtype=np.float32),
    )
    record = VerticalRecord(metadata=make_metadata(), payloads={"raw": payload})

    with pytest.raises(ValueError, match="hidden_state_count"):
        record.validate()


def test_pooled_payload_requires_aligned_progress_or_endpoints():
    payload = HiddenPayload(
        kind="pooled",
        values=np.zeros((3, 5, 8), dtype=np.float32),
        representation="last_s32",
    )
    record = VerticalRecord(metadata=make_metadata(), payloads={"last_s32": payload})

    with pytest.raises(ValueError, match="endpoints or progress"):
        record.validate()


def test_nonfinite_payload_is_rejected_in_strict_mode():
    values = np.zeros((4, 5, 8), dtype=np.float32)
    values[0, 0, 0] = np.nan
    record = VerticalRecord(
        metadata=make_metadata(),
        payloads={"raw": HiddenPayload(kind="raw", values=values)},
    )

    with pytest.raises(ValueError, match="non-finite"):
        record.validate(strict_finite=True)


def test_metadata_rejects_nonpositive_dimensions():
    metadata = replace(make_metadata(), hidden_dimension=0)

    with pytest.raises(ValueError, match="hidden_dimension"):
        metadata.validate()

