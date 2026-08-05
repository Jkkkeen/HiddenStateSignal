import json
from pathlib import Path

import pandas as pd
import pytest

from experiment_2e.formal_manifest import (
    REPRESENTATIVES,
    build_seed_manifest,
    checkpoint_schedule,
    freeze_formal_run,
)


def _eval_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"extra_info": [{"question_id": f"q{index}"} for index in range(n)]}
    )


def test_checkpoint_schedule_requires_five_equal_intervals():
    assert [row["global_step"] for row in checkpoint_schedule(100)] == [0, 20, 40, 60, 80, 100]
    with pytest.raises(ValueError, match="multiple of 5"):
        checkpoint_schedule(101)


def test_roll8_seed_manifest_is_complete_unique_and_deterministic():
    left = build_seed_manifest(_eval_frame(3), base_seed=7)
    right = build_seed_manifest(_eval_frame(3), base_seed=7)
    assert len(left) == 3 * 6 * 8
    assert set(left["rollout_slot"]) == set(range(8))
    assert not left.duplicated(["question_id", "checkpoint", "rollout_slot"]).any()
    pd.testing.assert_frame_equal(left, right)


def test_freeze_formal_run_writes_exact_rollout_count_and_hashes(tmp_path: Path):
    train_file = tmp_path / "train.parquet"
    eval_file = tmp_path / "eval.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(train_file, index=False)
    _eval_frame(4).to_parquet(eval_file, index=False)
    length_spec = tmp_path / "length_spec.json"
    length_spec.write_text(json.dumps({"window": 128, "stride": 32}), encoding="utf-8")

    output = tmp_path / "manifest"
    result = freeze_formal_run(
        eval_file=eval_file,
        train_file=train_file,
        source_length_spec=length_spec,
        output_dir=output,
        run_id="test",
        total_training_steps=100,
        base_seed=9,
        expected_eval_questions=4,
    )
    assert result["n_evaluation_rollouts"] == 4 * 6 * 8
    assert result["save_frequency"] == 20
    seeds = pd.read_parquet(output / "rollout_seed_manifest.parquet")
    assert len(seeds) == 4 * 6 * 8
    frozen = json.loads((output / "frozen_analysis_spec.json").read_text(encoding="utf-8"))
    assert frozen["rollouts_per_question"] == 8
    assert frozen["n_primary_training_stage_tests"] == 37
    assert len(frozen["representatives"]) == len(REPRESENTATIVES) == 16
