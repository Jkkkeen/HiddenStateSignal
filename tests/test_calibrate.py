from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vertical.calibrate import (
    BaseCalibrator,
    fit_base_calibrator,
    load_calibrator,
    save_calibrator,
    standardize_layers,
)
from vertical.schema import HiddenPayload, RecordMetadata, VerticalRecord


def make_record(
    record_id: str,
    values: np.ndarray,
    *,
    family: str = "mimo",
    condition: str = "base",
    is_correct: bool = False,
):
    values = np.asarray(values, dtype=np.float32)
    metadata = RecordMetadata(
        record_id=record_id,
        model_family=family,
        model_name=f"{family}-model",
        condition=condition,
        question_id=record_id.split(":")[0],
        rollout_id=record_id.split(":")[1],
        is_correct=is_correct,
        response_token_count=8,
        num_decoder_layers=values.shape[1] - 1,
        hidden_state_count=values.shape[1],
        hidden_dimension=values.shape[2],
        source_path=f"{record_id}.npz",
    )
    payload = HiddenPayload(
        kind="pooled",
        values=values,
        representation="last_s32",
        endpoints=np.linspace(8 / values.shape[0], 8, values.shape[0], dtype=int),
        progress=np.linspace(1 / values.shape[0], 1.0, values.shape[0]),
    )
    return VerticalRecord(metadata=metadata, payloads={"last_s32": payload})


def test_base_calibrator_is_label_blind_and_uses_frozen_weighting():
    first_values = np.ones((2, 3, 4), dtype=np.float32)
    second_values = np.full((4, 3, 4), 3.0, dtype=np.float32)
    records = [
        make_record("q1:r0", first_values, is_correct=False),
        make_record("q2:r0", second_values, is_correct=True),
    ]

    first = fit_base_calibrator(records, "last_s32")
    permuted = [
        replace(record, metadata=replace(record.metadata, is_correct=not record.metadata.is_correct))
        for record in records
    ]
    second = fit_base_calibrator(permuted, "last_s32")

    assert np.allclose(first.common, 2.0)
    assert np.allclose(first.coordinate_mean, 7 / 3)
    assert np.allclose(first.common, second.common)
    assert np.allclose(first.coordinate_sigma, second.coordinate_sigma)
    assert first.n_records == 2
    assert first.n_chunks == 6


def test_sft_only_calibration_is_rejected():
    record = make_record("q1:r0", np.ones((2, 3, 4)), condition="sft")

    with pytest.raises(ValueError, match="Base records only"):
        fit_base_calibrator([record], "last_s32")


def test_standardization_checks_family_representation_and_shape():
    calibrator = BaseCalibrator(
        common=np.zeros((3, 4), dtype=np.float32),
        coordinate_mean=np.ones((3, 4), dtype=np.float32),
        coordinate_sigma=np.full((3, 4), 2.0, dtype=np.float32),
        model_family="mimo",
        representation="last_s32",
        input_sha256="a" * 64,
        n_records=2,
        n_chunks=4,
    )

    standardized = standardize_layers(
        np.full((3, 4), 3.0),
        calibrator,
        model_family="mimo",
        representation="last_s32",
    )
    assert np.allclose(standardized, 1.0)

    with pytest.raises(ValueError, match="model_family"):
        standardize_layers(
            np.zeros((3, 4)),
            calibrator,
            model_family="qwen3",
            representation="last_s32",
        )


def test_calibrator_round_trip_is_atomic(tmp_path):
    record = make_record("q1:r0", np.arange(2 * 3 * 4).reshape(2, 3, 4))
    calibrator = fit_base_calibrator([record], "last_s32")
    path = tmp_path / "calibrators" / "mimo_last.npz"

    save_calibrator(calibrator, path)
    loaded = load_calibrator(path, "last_s32")

    assert np.allclose(loaded.common, calibrator.common)
    assert np.allclose(loaded.coordinate_sigma, calibrator.coordinate_sigma)
    assert loaded.model_family == "mimo"
    assert not path.with_suffix(".npz.tmp").exists()


def test_raw_record_is_pooled_before_calibration():
    values = np.arange(8 * 3 * 4, dtype=np.float32).reshape(8, 3, 4)
    metadata = RecordMetadata(
        record_id="q1:r0",
        model_family="qwen3",
        model_name="Qwen3-8B-Base",
        condition="base",
        question_id="q1",
        rollout_id="r0",
        is_correct=True,
        response_token_count=8,
        num_decoder_layers=2,
        hidden_state_count=3,
        hidden_dimension=4,
        source_path="raw.npz",
    )
    record = VerticalRecord(
        metadata=metadata,
        payloads={"raw": HiddenPayload(kind="raw", values=values)},
    )

    calibrator = fit_base_calibrator([record], "mean_w128_s32")

    assert calibrator.n_chunks == 1
    assert np.allclose(calibrator.common, values.mean(axis=0))
