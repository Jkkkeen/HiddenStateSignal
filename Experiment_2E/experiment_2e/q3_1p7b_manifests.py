from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


INSTRUCTION = (
    "Please reason step by step. Put only the final answer inside "
    r"\boxed{}, and do not write anything after the boxed answer."
)


def _mapping(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, dict):
        raise TypeError(f"expected mapping, got {type(value).__name__}")
    return dict(value)


def _normalize_question(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _order(seed: int, question_hash: str) -> str:
    return _hash(f"{seed}:{question_hash}")


def _canonicalize(frame: pd.DataFrame, *, source: str, instruction: str = INSTRUCTION) -> pd.DataFrame:
    rows = []
    for index, row in frame.iterrows():
        extra = _mapping(row["extra_info"])
        reward = _mapping(row["reward_model"])
        question = _normalize_question(extra.get("question", row.get("question", "")))
        if not question:
            continue
        difficulty = float(extra.get("difficulty", row.get("level", extra.get("level", np.nan))))
        question_hash = _hash(question.casefold())
        question_id = f"{source}:{question_hash[:20]}"
        extra.update(
            {
                "question_id": question_id,
                "question": question,
                "difficulty": difficulty,
                "prompt_hash": question_hash,
                "source_row_index": int(index),
                "source_dataset": source,
            }
        )
        rows.append(
            {
                "data_source": source,
                "prompt": [{"role": "user", "content": f"{question}\n\n{instruction}"}],
                "ability": "math",
                "reward_model": {"ground_truth": str(reward["ground_truth"]), "style": "rule"},
                "extra_info": extra,
            }
        )
    output = pd.DataFrame(rows)
    prompt_hashes = output["extra_info"].map(lambda value: value["prompt_hash"])
    output = output.loc[~prompt_hashes.duplicated(keep="first")]
    return output.reset_index(drop=True)


def _write_split(frame: pd.DataFrame, path: Path, split: str) -> None:
    work = frame.copy()
    work["extra_info"] = work["extra_info"].map(lambda value: {**value, "experiment_split": split})
    work.to_parquet(path, index=False)


def prepare(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    deepmath = _canonicalize(pd.read_parquet(args.deepmath), source="deepmath103k")
    levels = deepmath["extra_info"].map(lambda value: float(value["difficulty"]))
    deepmath = deepmath.loc[(levels >= 3.0) & (levels <= 6.0)].copy()
    deepmath["_order"] = deepmath["extra_info"].map(
        lambda value: _order(args.seed, value["prompt_hash"])
    )
    deepmath = deepmath.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    required = args.calibration_size + args.heldout_size + args.smoke_size
    if len(deepmath) <= required:
        raise ValueError("DeepMath level 3-6 pool is too small")
    calibration = deepmath.iloc[: args.calibration_size]
    heldout = deepmath.iloc[args.calibration_size : args.calibration_size + args.heldout_size]
    smoke = deepmath.iloc[
        args.calibration_size + args.heldout_size : required
    ]
    train = deepmath.iloc[required:]

    simplerl = _canonicalize(pd.read_parquet(args.simplerl), source="simplerl_level3_5")
    simplerl["_order"] = simplerl["extra_info"].map(
        lambda value: _order(args.seed + 1, value["prompt_hash"])
    )
    simplerl = simplerl.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    simplerl_calibration = simplerl.iloc[: args.calibration_size]

    paths = {
        "deepmath_calibration": args.output_dir / "deepmath_calibration_256.parquet",
        "deepmath_heldout": args.output_dir / "deepmath_heldout_256.parquet",
        "deepmath_smoke": args.output_dir / f"deepmath_smoke_{args.smoke_size}.parquet",
        "deepmath_train": args.output_dir / "deepmath_train_candidates.parquet",
        "simplerl_calibration": args.output_dir / "simplerl_calibration_256.parquet",
    }
    _write_split(calibration, paths["deepmath_calibration"], "calibration")
    _write_split(heldout, paths["deepmath_heldout"], "heldout")
    _write_split(smoke, paths["deepmath_smoke"], "smoke")
    _write_split(train, paths["deepmath_train"], "train")
    _write_split(simplerl_calibration, paths["simplerl_calibration"], "calibration")

    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    }
    split_hashes = {
        name: set(pd.read_parquet(path)["extra_info"].map(lambda value: value["prompt_hash"]))
        for name, path in paths.items()
        if name.startswith("deepmath_")
    }
    leakage = {
        f"{left}__{right}": len(split_hashes[left] & split_hashes[right])
        for i, left in enumerate(split_hashes)
        for right in list(split_hashes)[i + 1 :]
    }
    audit = {
        "seed": args.seed,
        "instruction": INSTRUCTION,
        "deepmath_level_range": [3.0, 6.0],
        "counts": {name: int(len(pd.read_parquet(path))) for name, path in paths.items()},
        "sha256": hashes,
        "normalized_exact_hash_leakage": leakage,
        "passed": all(value == 0 for value in leakage.values()),
    }
    (args.output_dir / "manifest_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["passed"]:
        raise SystemExit("manifest leakage audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepmath", type=Path, required=True)
    parser.add_argument("--simplerl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--heldout-size", type=int, default=256)
    parser.add_argument("--smoke-size", type=int, default=128)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
