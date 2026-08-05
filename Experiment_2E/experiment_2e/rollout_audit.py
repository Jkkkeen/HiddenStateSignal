from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import write_json_atomic
from .formal_manifest import CHECKPOINT_NAMES
from .rollouts import audit_rollout_records, expected_rollout_keys


def load_rollouts(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("**/parts/part_*.jsonl"))
    if not files:
        raise ValueError(f"No rollout parts found under {path}")
    rows = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return pd.DataFrame(rows)


def audit_rollout_dir(
    *,
    rollout_dir: Path,
    eval_file: Path,
    rollouts_per_question: int = 8,
    seed_manifest_path: Path | None = None,
) -> dict:
    eval_frame = pd.read_parquet(eval_file)
    question_ids = [str(value["question_id"]) for value in eval_frame["extra_info"].tolist()]
    records = load_rollouts(rollout_dir)
    records["question_id"] = records["question_id"].astype(str)
    expected = expected_rollout_keys(
        question_ids,
        checkpoints=CHECKPOINT_NAMES,
        rollouts_per_question=rollouts_per_question,
    )
    audit = audit_rollout_records(records, expected_keys=expected)
    audit["seed_manifest_checked"] = seed_manifest_path is not None
    audit["n_seed_mismatches"] = 0
    if seed_manifest_path is not None:
        seeds = pd.read_parquet(seed_manifest_path).rename(columns={"seed": "frozen_seed"})
        seeds["question_id"] = seeds["question_id"].astype(str)
        comparison = records.merge(
            seeds[["checkpoint", "question_id", "rollout_slot", "frozen_seed"]],
            on=["checkpoint", "question_id", "rollout_slot"],
            how="left",
            validate="one_to_one",
        )
        mismatch = comparison["frozen_seed"].isna() | (
            comparison["rollout_seed"].astype("Int64") != comparison["frozen_seed"].astype("Int64")
        )
        audit["n_seed_mismatches"] = int(mismatch.sum())
        audit["passed"] = bool(audit["passed"] and not mismatch.any())
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the complete six-checkpoint roll8 cohort.")
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--rollouts-per-question", type=int, default=8)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_rollout_dir(
        rollout_dir=args.rollout_dir,
        eval_file=args.eval_file,
        rollouts_per_question=args.rollouts_per_question,
        seed_manifest_path=args.seed_manifest,
    )
    write_json_atomic(args.output, audit)
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit("Rollout audit failed")


if __name__ == "__main__":
    main()
