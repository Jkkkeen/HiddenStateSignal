from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import sha256_file, write_json_atomic
from .math_reward import score_response


EQUIVALENCE_CASES = (
    (r"\frac{1}{2}", r"Thus \boxed{0.5}.", True),
    (r"\sqrt{4}", r"Thus \boxed{2}.", True),
    ("4", r"Thus \boxed{5}.", False),
    ("4", "Thus 4.", False),
)


def audit_reward(data_file: Path, sample_size: int = 256) -> dict:
    frame = pd.read_parquet(data_file).head(sample_size)
    roundtrip = []
    for _, row in frame.iterrows():
        reward_model = row["reward_model"]
        if hasattr(reward_model, "tolist"):
            reward_model = reward_model.tolist()
        gold = dict(reward_model)["ground_truth"]
        result = score_response(rf"Final answer: \boxed{{{gold}}}.", gold)
        roundtrip.append(result["answer_reward"])
    cases = []
    for gold, response, expected in EQUIVALENCE_CASES:
        result = score_response(response, gold)
        cases.append({"gold": gold, "response": response, "expected": expected, **result})
    return {
        "data_file": str(data_file),
        "data_file_sha256": sha256_file(data_file),
        "n_gold_roundtrip": len(roundtrip),
        "gold_roundtrip_success_rate": sum(roundtrip) / len(roundtrip) if roundtrip else None,
        "equivalence_cases": cases,
        "all_equivalence_cases_pass": all(bool(case["answer_reward"]) == case["expected"] for case in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen MATH reward implementation.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=256)
    args = parser.parse_args()
    audit = audit_reward(args.data_file, args.sample_size)
    write_json_atomic(args.output, audit)
    print(json.dumps(audit, indent=2))
    if audit["gold_roundtrip_success_rate"] != 1.0 or not audit["all_equivalence_cases_pass"]:
        raise SystemExit("Reward audit failed")


if __name__ == "__main__":
    main()
