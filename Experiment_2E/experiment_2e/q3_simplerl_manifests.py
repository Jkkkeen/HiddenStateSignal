from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .q3_1p7b_manifests import _canonicalize, _order, _write_split
from .q3_prompt_calibration import STRICT_INSTRUCTION


def prepare(
    train_source: Path,
    test_source: Path,
    output_dir: Path,
    *,
    seed: int,
    calibration_size: int,
    heldout_size: int,
    smoke_size: int,
    smoke_val_size: int = 16,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_pool = _canonicalize(
        pd.read_parquet(train_source), source="simplerl_level3_5", instruction=STRICT_INSTRUCTION
    )
    train_pool["_order"] = train_pool["extra_info"].map(
        lambda value: _order(seed + 1, value["prompt_hash"])
    )
    train_pool = train_pool.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    required = calibration_size + smoke_size
    if len(train_pool) <= required:
        raise ValueError("SimpleRL train pool is too small")
    calibration = train_pool.iloc[:calibration_size]
    smoke = train_pool.iloc[calibration_size:required]
    train = train_pool.iloc[required:]

    heldout_pool = _canonicalize(
        pd.read_parquet(test_source), source="simplerl_level3_5_test", instruction=STRICT_INSTRUCTION
    )
    train_hashes = set(train_pool["extra_info"].map(lambda value: value["prompt_hash"]))
    heldout_pool = heldout_pool.loc[
        ~heldout_pool["extra_info"].map(lambda value: value["prompt_hash"]).isin(train_hashes)
    ].copy()
    heldout_pool["_order"] = heldout_pool["extra_info"].map(
        lambda value: _order(seed + 2, value["prompt_hash"])
    )
    heldout_pool = heldout_pool.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    if len(heldout_pool) < heldout_size:
        raise ValueError("SimpleRL held-out pool is too small after de-duplication")
    heldout = heldout_pool.iloc[:heldout_size]

    paths = {
        "simplerl_calibration": output_dir / "simplerl_calibration_256.parquet",
        "simplerl_smoke": output_dir / f"simplerl_smoke_{smoke_size}.parquet",
        "simplerl_train": output_dir / "simplerl_train_candidates.parquet",
        "simplerl_heldout": output_dir / "simplerl_heldout_256.parquet",
    }
    _write_split(calibration, paths["simplerl_calibration"], "calibration")
    _write_split(smoke, paths["simplerl_smoke"], "smoke")
    _write_split(train, paths["simplerl_train"], "train")
    _write_split(heldout, paths["simplerl_heldout"], "heldout")
    smoke_val_path = output_dir / f"simplerl_smoke_val_{smoke_val_size}.parquet"
    _write_split(calibration.iloc[:smoke_val_size], smoke_val_path, "smoke_validation")

    split_hashes = {
        name: set(pd.read_parquet(path)["extra_info"].map(lambda value: value["prompt_hash"]))
        for name, path in paths.items()
    }
    leakage = {
        f"{left}__{right}": len(split_hashes[left] & split_hashes[right])
        for index, left in enumerate(split_hashes)
        for right in list(split_hashes)[index + 1 :]
    }
    payload = {
        "seed": seed,
        "source_train": str(train_source),
        "source_test": str(test_source),
        "prompt_protocol": "strict_boxed_v2",
        "instruction": STRICT_INSTRUCTION,
        "counts": {name: len(pd.read_parquet(path)) for name, path in paths.items()},
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "normalized_exact_hash_leakage": leakage,
        "passed": all(value == 0 for value in leakage.values()),
        "smoke_validation": {
            "path": str(smoke_val_path),
            "size": smoke_val_size,
            "sha256": hashlib.sha256(smoke_val_path.read_bytes()).hexdigest(),
            "source": "permanently excluded calibration subset",
        },
    }
    (output_dir / "manifest_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not payload["passed"]:
        raise SystemExit("SimpleRL manifest leakage audit failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--heldout-size", type=int, default=256)
    parser.add_argument("--smoke-size", type=int, default=128)
    parser.add_argument("--smoke-val-size", type=int, default=16)
    args = parser.parse_args()
    payload = prepare(
        args.train_source,
        args.test_source,
        args.output_dir,
        seed=args.seed,
        calibration_size=args.calibration_size,
        heldout_size=args.heldout_size,
        smoke_size=args.smoke_size,
        smoke_val_size=args.smoke_val_size,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
