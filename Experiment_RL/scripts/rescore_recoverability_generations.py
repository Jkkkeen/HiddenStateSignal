#!/usr/bin/env python3
"""Re-score saved recoverability generations with the strict answer parser."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from run_recoverability_revision_vllm import (
    extract_revision_choice,
    load_jsonl,
    summarize_prefixes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--run-summary", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.generations))
    for row in records:
        row["permissive_prediction"] = row.get("prediction")
        prediction = extract_revision_choice(row.get("response"))
        row["prediction"] = prediction
        row["correct"] = bool(prediction is not None and prediction == row.get("answer"))
        row["strict_parse_valid"] = prediction is not None

    prefixes = summarize_prefixes(records)
    old_summary = (
        json.loads(Path(args.run_summary).read_text(encoding="utf-8"))
        if args.run_summary and Path(args.run_summary).exists()
        else {}
    )
    invalid = sum(row["prediction"] is None for row in records)
    nontruncated = [row for row in records if not row.get("truncated_by_length")]
    nontruncated_invalid = sum(row["prediction"] is None for row in nontruncated)
    exhausted_without_answer = sum(
        bool(row.get("truncated_by_length")) and row.get("prediction") is None
        for row in records
    )
    old_summary.update(
        {
            "scoring": "strict_explicit_final_answer",
            "total_generations": len(records),
            "prefixes": len(prefixes),
            "mean_recovery_rate": (
                sum(float(row["recovery_rate"]) for row in prefixes) / len(prefixes)
                if prefixes
                else 0.0
            ),
            "any_recovery_prefixes": sum(bool(row["any_recovery"]) for row in prefixes),
            "all_recovered_prefixes": sum(bool(row["all_recovered"]) for row in prefixes),
            "zero_recovery_prefixes": sum(float(row["recovery_rate"]) == 0.0 for row in prefixes),
            "invalid_generations": invalid,
            "invalid_rate": invalid / len(records) if records else 0.0,
            "nontruncated_invalid_generations": nontruncated_invalid,
            "nontruncated_invalid_rate": (
                nontruncated_invalid / len(nontruncated) if nontruncated else 0.0
            ),
            "budget_exhausted_without_answer": exhausted_without_answer,
            "prediction_counts": dict(
                Counter(str(row.get("prediction") or "INVALID") for row in records)
            ),
        }
    )

    with (output_dir / "revision_generations_strict.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "recoverability_scores_strict.jsonl").open("w", encoding="utf-8") as handle:
        for row in prefixes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(prefixes).to_csv(output_dir / "recoverability_scores_strict.csv", index=False)
    (output_dir / "run_summary_strict.json").write_text(
        json.dumps(old_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(old_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
