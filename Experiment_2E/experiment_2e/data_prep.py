from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .common import sha256_file, sha256_text, write_json_atomic, write_jsonl
from .math_reward import extract_last_boxed


DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
INSTRUCTION = r"Let's think step by step and output the final answer within \boxed{}."
CHECKPOINT_NAMES = ("base", "20pct", "40pct", "60pct", "80pct", "final")


def parse_level(value: Any) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def normalized_problem(problem: str) -> str:
    return " ".join((problem or "").split())


def build_verl_records(
    frame: pd.DataFrame,
    *,
    split: str,
    levels: set[int] | None = None,
    data_source: str = DATA_SOURCE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_index, row in frame.reset_index(drop=False).iterrows():
        original_index = row.get("index", source_index)
        problem = normalized_problem(str(row.get("problem", "")))
        level = parse_level(row.get("level"))
        subject = str(row.get("type", row.get("subject", "unknown")))
        solution = str(row.get("solution", ""))
        question_id = f"math:{split}:{original_index}"

        reason: str | None = None
        if not problem:
            reason = "empty_problem"
        elif levels is not None and level not in levels:
            reason = "level_not_selected"
        gold = extract_last_boxed(solution)
        if reason is None and gold is None:
            reason = "gold_unparseable"
        if reason is not None:
            rejected.append({"question_id": question_id, "reason": reason, "level": level, "subject": subject})
            continue

        prompt_hash = sha256_text(problem)
        prompt_text = f"{problem} {INSTRUCTION}"
        records.append(
            {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info": {
                    "split": split,
                    "source_index": int(original_index),
                    "question_id": question_id,
                    "subject": subject,
                    "level": level,
                    "prompt_hash": prompt_hash,
                    "problem": problem,
                },
            }
        )
    return records, rejected


def records_manifest(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        extra = record["extra_info"]
        rows.append(
            {
                "question_id": extra["question_id"],
                "source_split": extra["split"],
                "source_index": extra["source_index"],
                "subject": extra["subject"],
                "level": extra["level"],
                "gold_answer": record["reward_model"]["ground_truth"],
                "prompt_hash": extra["prompt_hash"],
            }
        )
    return pd.DataFrame(rows)


def stratified_sample(frame: pd.DataFrame, *, n: int, strata: tuple[str, ...], seed: int) -> pd.DataFrame:
    if n > len(frame):
        raise ValueError(f"Requested {n} rows from a frame with only {len(frame)} rows")
    if n == len(frame):
        return frame.sample(frac=1, random_state=seed).reset_index(drop=True)

    grouped = list(frame.groupby(list(strata), dropna=False, sort=True))
    quotas = []
    for key, group in grouped:
        exact = n * len(group) / len(frame)
        floor = min(len(group), math.floor(exact))
        quotas.append({"key": key, "group": group, "quota": floor, "remainder": exact - floor})
    remaining = n - sum(item["quota"] for item in quotas)
    while remaining:
        eligible = [item for item in quotas if item["quota"] < len(item["group"])]
        if not eligible:
            raise RuntimeError("Unable to allocate exact stratified sample size")
        eligible.sort(key=lambda item: (-item["remainder"], str(item["key"])))
        for item in eligible:
            if remaining == 0:
                break
            item["quota"] += 1
            item["remainder"] = -1.0
            remaining -= 1

    sampled = []
    for offset, item in enumerate(quotas):
        if item["quota"]:
            sampled.append(item["group"].sample(n=item["quota"], random_state=seed + offset))
    return pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def stable_seed(base_seed: int, question_id: str, checkpoint: str, rollout_slot: int) -> int:
    digest = sha256_text(f"{base_seed}|{question_id}|{checkpoint}|{rollout_slot}")
    return int(digest[:8], 16) & 0x7FFFFFFF


def prepare_dataset(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: Path,
    *,
    eval_size: int = 256,
    seed: int = 20260805,
    eval_rollouts_per_checkpoint: int = 8,
) -> dict[str, Any]:
    if eval_rollouts_per_checkpoint < 1:
        raise ValueError("eval_rollouts_per_checkpoint must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records, train_rejected = build_verl_records(train_frame, split="train", levels={3, 4, 5})
    test_records, test_rejected = build_verl_records(test_frame, split="test", levels=None)
    if len(test_records) < eval_size:
        raise ValueError(f"Only {len(test_records)} valid test records; need {eval_size}")

    test_table = pd.DataFrame(
        [
            {"record_index": index, **record["extra_info"]}
            for index, record in enumerate(test_records)
        ]
    )
    selected = stratified_sample(test_table, n=eval_size, strata=("level", "subject"), seed=seed)
    eval_records = [test_records[int(index)] for index in selected["record_index"]]

    train_hashes = {record["extra_info"]["prompt_hash"] for record in train_records}
    eval_hashes = {record["extra_info"]["prompt_hash"] for record in eval_records}
    leakage = sorted(train_hashes & eval_hashes)
    if leakage:
        raise ValueError(f"Train/eval prompt leakage detected for {len(leakage)} normalized problems")

    train_path = output_dir / "train_level3_5.parquet"
    eval_path = output_dir / "eval_256.parquet"
    pd.DataFrame(train_records).to_parquet(train_path, index=False)
    pd.DataFrame(eval_records).to_parquet(eval_path, index=False)
    train_manifest = records_manifest(train_records)
    eval_manifest = records_manifest(eval_records)
    train_manifest.to_csv(output_dir / "train_manifest.csv", index=False)
    eval_manifest.to_csv(output_dir / "eval_manifest.csv", index=False)
    write_jsonl(output_dir / "rejected_train.jsonl", train_rejected)
    write_jsonl(output_dir / "rejected_test.jsonl", test_rejected)

    seed_rows = []
    for record in eval_records:
        question_id = record["extra_info"]["question_id"]
        for checkpoint in CHECKPOINT_NAMES:
            for rollout_slot in range(eval_rollouts_per_checkpoint):
                seed_rows.append(
                    {
                        "question_id": question_id,
                        "checkpoint": checkpoint,
                        "rollout_slot": rollout_slot,
                        "seed": stable_seed(seed, question_id, checkpoint, rollout_slot),
                    }
                )
    pd.DataFrame(seed_rows).to_csv(output_dir / "eval_seed_manifest.csv", index=False)

    audit = {
        "data_source": DATA_SOURCE,
        "seed": seed,
        "eval_rollouts_per_checkpoint": eval_rollouts_per_checkpoint,
        "train_levels": [3, 4, 5],
        "n_train": len(train_records),
        "n_eval": len(eval_records),
        "n_train_rejected": len(train_rejected),
        "n_test_rejected": len(test_rejected),
        "train_eval_prompt_hash_overlap": 0,
        "files": {
            train_path.name: sha256_file(train_path),
            eval_path.name: sha256_file(eval_path),
            "train_manifest.csv": sha256_file(output_dir / "train_manifest.csv"),
            "eval_manifest.csv": sha256_file(output_dir / "eval_manifest.csv"),
            "eval_seed_manifest.csv": sha256_file(output_dir / "eval_seed_manifest.csv"),
        },
    }
    write_json_atomic(output_dir / "data_audit.json", audit)
    return audit


def _load_dataset(dataset_name: str, local_dataset_path: str | None):
    from datasets import load_dataset, load_from_disk

    if local_dataset_path and (Path(local_dataset_path) / "dataset_dict.json").exists():
        return load_from_disk(local_dataset_path)
    return load_dataset(local_dataset_path or dataset_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frozen MATH train/evaluation Parquet files for Experiment 2E.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DATA_SOURCE)
    parser.add_argument("--local-dataset-path")
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--eval-rollouts-per-checkpoint", type=int, default=8)
    args = parser.parse_args()
    dataset = _load_dataset(args.dataset_name, args.local_dataset_path)
    audit = prepare_dataset(
        dataset["train"].to_pandas(),
        dataset["test"].to_pandas(),
        args.output_dir,
        eval_size=args.eval_size,
        seed=args.seed,
        eval_rollouts_per_checkpoint=args.eval_rollouts_per_checkpoint,
    )
    print(audit)


if __name__ == "__main__":
    main()
