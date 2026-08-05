import json

import pandas as pd
from experiment_2e.formal_manifest import CHECKPOINT_NAMES
from experiment_2e.rollout_audit import audit_rollout_dir
from experiment_2e.rollouts import audit_rollout_records, expected_rollout_keys, rollout_id


def test_roll8_expected_keys_and_ids():
    expected = expected_rollout_keys(["q1"], checkpoints=("base",), rollouts_per_question=8)
    assert len(expected) == 8
    assert rollout_id("base", "q1", 7) == "base/q1/r07"


def test_rollout_audit_rejects_missing_and_duplicate_keys():
    expected = expected_rollout_keys(["q1"], checkpoints=("base",), rollouts_per_question=2)
    frame = pd.DataFrame(
        [
            {"checkpoint": "base", "question_id": "q1", "rollout_slot": 0, "response": "a", "truncated": False},
            {"checkpoint": "base", "question_id": "q1", "rollout_slot": 0, "response": "b", "truncated": False},
        ]
    )
    audit = audit_rollout_records(frame, expected_keys=expected)
    assert audit["passed"] is False
    assert audit["n_duplicate_keys"] == 1
    assert audit["n_missing_keys"] == 1


def test_rollout_audit_accepts_complete_roll8_keys():
    expected = expected_rollout_keys(["q1"], checkpoints=("base",), rollouts_per_question=8)
    frame = pd.DataFrame(
        [
            {
                "checkpoint": "base",
                "question_id": "q1",
                "rollout_slot": slot,
                "response": str(slot),
                "truncated": False,
                "answer_reward": slot % 2,
            }
            for slot in range(8)
        ]
    )
    audit = audit_rollout_records(frame, expected_keys=expected)
    assert audit["passed"] is True
    assert audit["n_records"] == 8


def test_directory_audit_checks_frozen_rollout_seeds(tmp_path):
    eval_file = tmp_path / "eval.parquet"
    pd.DataFrame({"extra_info": [{"question_id": "q1"}]}).to_parquet(eval_file, index=False)
    rows = []
    seeds = []
    for checkpoint in CHECKPOINT_NAMES:
        for slot in range(8):
            seed = 1000 + slot
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "question_id": "q1",
                    "rollout_slot": slot,
                    "rollout_seed": seed,
                    "response": "answer",
                    "truncated": False,
                    "answer_reward": 1.0,
                }
            )
            seeds.append(
                {
                    "checkpoint": checkpoint,
                    "question_id": "q1",
                    "rollout_slot": slot,
                    "seed": seed,
                }
            )
    part = tmp_path / "rollouts" / "base" / "parts" / "part_00000.jsonl"
    part.parent.mkdir(parents=True)
    part.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    seed_file = tmp_path / "seeds.parquet"
    pd.DataFrame(seeds).to_parquet(seed_file, index=False)
    audit = audit_rollout_dir(
        rollout_dir=tmp_path / "rollouts",
        eval_file=eval_file,
        seed_manifest_path=seed_file,
    )
    assert audit["passed"] is True
    assert audit["seed_manifest_checked"] is True
    assert audit["n_seed_mismatches"] == 0

    seed_frame = pd.read_parquet(seed_file)
    seed_frame.loc[0, "seed"] += 1
    seed_frame.to_parquet(seed_file, index=False)
    audit = audit_rollout_dir(
        rollout_dir=tmp_path / "rollouts",
        eval_file=eval_file,
        seed_manifest_path=seed_file,
    )
    assert audit["passed"] is False
    assert audit["n_seed_mismatches"] == 1
