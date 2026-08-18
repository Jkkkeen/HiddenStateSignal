import json

import numpy as np
import pandas as pd
import pytest
import torch

from experiment_2e.calibrators import write_pooled_cache
from experiment_2e.q3_layer_extract import (
    atomic_parquet,
    checkpoint_audit,
    parameter_sample_sha256,
    reduce_checkpoint_caches,
    validate_trained_parameter_sample,
)


def _metadata(rollout_id, slot):
    return {
        "run_id": "run",
        "model": "Qwen3-test",
        "checkpoint": "step000",
        "global_step": 0,
        "training_progress": 0.0,
        "question_id": f"q{slot}",
        "rollout_id": rollout_id,
        "rollout_slot": slot,
        "is_correct": bool(slot),
        "answer_reward": float(bool(slot)),
        "format_correct": True,
        "parse_correct": True,
        "difficulty": 1.0,
        "prompt_hash": f"hash{slot}",
        "prompt_token_count": 2,
        "response_token_count": 8,
        "response_length": 8,
        "trajectory_point_count": 4,
        "n_hidden_states": 4,
    }


def _controls(metadata):
    return [
        {
            **metadata,
            "stage": stage,
            "response_token_count": 8,
            "stage_token_count": 2,
            "trajectory_point_count": 4,
            "policy_entropy": 1.0 + stage,
            "token_logprob_mean": -1.0 - stage,
            "hidden_norm_mean": 2.0 + stage,
        }
        for stage in range(4)
    ]


def _write_cache(root, rollout_id, slot):
    rng = np.random.default_rng(slot + 10)
    mean = rng.normal(size=(4, 4, 6)).astype(np.float32) + 1.0
    last = rng.normal(size=(4, 4, 6)).astype(np.float32) + 1.0
    cache_path = root / "cache" / f"cache{slot}.npz"
    metadata = _metadata(rollout_id, slot)
    write_pooled_cache(
        cache_path,
        pooled={"mean_w128_s32": mean, "last_s32": last},
        endpoints=np.array([2, 4, 6, 8]),
        progress=np.array([0.25, 0.50, 0.75, 1.0]),
        metadata=metadata,
    )
    controls_dir = root / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    (controls_dir / f"{cache_path.stem}.json").write_text(
        json.dumps({"rollout_id": rollout_id, "controls": _controls(metadata)}),
        encoding="utf-8",
    )
    return cache_path


def _write_calibrator(path):
    arrays = {}
    for representation in ("mean_w128_s32", "last_s32"):
        arrays[f"common__{representation}"] = np.zeros((4, 6), dtype=np.float32)
        arrays[f"coordinate_mean__{representation}"] = np.zeros((4, 6), dtype=np.float32)
        arrays[f"coordinate_sigma__{representation}"] = np.ones((4, 6), dtype=np.float32)
    np.savez(path, **arrays)
    path.with_suffix(".audit.json").write_text(
        json.dumps({"label_blind": True}),
        encoding="utf-8",
    )


def test_atomic_parquet_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "rows.parquet"
    atomic_parquet(pd.DataFrame({"x": [1, 2]}), path)
    assert pd.read_parquet(path)["x"].tolist() == [1, 2]
    assert not path.with_suffix(".parquet.tmp").exists()


def test_parameter_sample_changes_and_trained_identity_cannot_equal_base(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    before = parameter_sample_sha256(model)
    with torch.no_grad():
        model[0].weight[0, 0] += 1.0
    after = parameter_sample_sha256(model)
    assert before != after

    identity = tmp_path / "base.json"
    identity.write_text(json.dumps({"parameter_sample_sha256": after}), encoding="utf-8")
    with pytest.raises(ValueError, match="matches base"):
        validate_trained_parameter_sample(after, identity, global_step=25)


def test_reduce_caches_and_checkpoint_audit(tmp_path):
    cache_paths = [_write_cache(tmp_path, f"r{slot}", slot) for slot in range(2)]
    calibrator = tmp_path / "base_calibrators.npz"
    _write_calibrator(calibrator)

    profiles, aggregate, controls = reduce_checkpoint_caches(
        cache_paths,
        controls_dir=tmp_path / "controls",
        calibrator_path=calibrator,
    )

    assert len(profiles) == 2 * 2 * 4 * 4
    assert set(profiles["representation"]) == {"mean_w128_s32", "last_s32"}
    assert set(aggregate["family_id"]) == {f"V{index}" for index in range(1, 10)}
    assert len(controls) == 2 * 4
    audit = checkpoint_audit(
        profiles=profiles,
        controls=controls,
        expected_rollouts=2,
        expected_hidden_states=4,
        output_root=tmp_path,
        identity_passed=True,
        calibrator_path=calibrator,
    )
    assert audit["passed"] is True
    assert audit["n_profile_rows"] == 2 * 2 * 4 * 4


def test_checkpoint_audit_rejects_missing_layer(tmp_path):
    cache_paths = [_write_cache(tmp_path, "r0", 0)]
    calibrator = tmp_path / "base_calibrators.npz"
    _write_calibrator(calibrator)
    profiles, _, controls = reduce_checkpoint_caches(
        cache_paths,
        controls_dir=tmp_path / "controls",
        calibrator_path=calibrator,
    )
    profiles = profiles.loc[profiles["layer_index"] != 3]
    audit = checkpoint_audit(
        profiles=profiles,
        controls=controls,
        expected_rollouts=1,
        expected_hidden_states=4,
        output_root=tmp_path,
        identity_passed=True,
        calibrator_path=calibrator,
    )
    assert audit["passed"] is False
    assert audit["gates"]["profile_row_count"] is False
