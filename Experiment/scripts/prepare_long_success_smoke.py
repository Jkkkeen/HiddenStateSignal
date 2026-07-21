#!/usr/bin/env python3
"""Freeze Experiment 1 long-response discovery/confirmatory splits and smoke rows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from long_success_trajectory_common import (
    eligible_question_summaries,
    is_clean_complete_row,
    select_stratified_questions,
    split_primary_questions,
)


@dataclass(frozen=True)
class ManifestResult:
    summaries: pd.DataFrame
    primary_questions: list[str]
    secondary_questions: list[str]
    discovery_questions: list[str]
    confirmatory_questions: list[str]
    smoke_questions: list[str]
    smoke_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed long success-trajectory smoke rows.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-questions", type=int, default=32)
    parser.add_argument("--discovery-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_source_line", line_number)
            rows.append(row)
    return rows


def build_manifest(
    rows: list[dict[str, Any]],
    smoke_questions: int,
    discovery_fraction: float,
    seed: int,
) -> ManifestResult:
    summaries = eligible_question_summaries(rows)
    primary = summaries.loc[summaries["is_primary"], "question_id"].astype(str).tolist()
    secondary = summaries.loc[summaries["is_secondary"], "question_id"].astype(str).tolist()
    discovery, confirmatory = split_primary_questions(summaries, discovery_fraction, seed)
    smoke = select_stratified_questions(summaries, discovery, smoke_questions, seed)
    smoke_set = set(smoke)
    selected_rows = [
        row
        for row in rows
        if is_clean_complete_row(row) and str(row.get("question_id", "")) in smoke_set
    ]
    selected_rows.sort(key=lambda row: (str(row.get("question_id", "")), int(row.get("rollout_id", -1))))
    return ManifestResult(
        summaries=summaries,
        primary_questions=sorted(primary),
        secondary_questions=sorted(secondary),
        discovery_questions=discovery,
        confirmatory_questions=confirmatory,
        smoke_questions=smoke,
        smoke_rows=selected_rows,
    )


def write_manifest(result: ManifestResult, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_smoke{len(result.smoke_questions)}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in result.smoke_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries = result.summaries.copy()
    split_by_question = {qid: "discovery" for qid in result.discovery_questions}
    split_by_question.update({qid: "confirmatory" for qid in result.confirmatory_questions})
    summaries["split"] = summaries["question_id"].map(split_by_question).fillna("secondary_or_ineligible")
    summaries["in_smoke"] = summaries["question_id"].isin(result.smoke_questions)
    summaries.to_csv(output_dir / "question_splits.csv", index=False)

    audit = {
        "input": str(args.input),
        "seed": int(args.seed),
        "discovery_fraction": float(args.discovery_fraction),
        "clean_complete_rows": int(sum(is_clean_complete_row(row) for row in result.smoke_rows)),
        "primary_questions": len(result.primary_questions),
        "secondary_questions": len(result.secondary_questions),
        "discovery_questions": len(result.discovery_questions),
        "confirmatory_questions": len(result.confirmatory_questions),
        "smoke_questions": len(result.smoke_questions),
        "smoke_rollouts": len(result.smoke_rows),
        "manifest": manifest_path.name,
    }
    (output_dir / "ELIGIBILITY_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Experiment 1 Eligibility Audit",
        "",
        f"- Primary 3+3 questions: {audit['primary_questions']}",
        f"- Secondary 2+2 questions: {audit['secondary_questions']}",
        f"- Discovery questions: {audit['discovery_questions']}",
        f"- Locked confirmatory questions: {audit['confirmatory_questions']}",
        f"- Smoke questions: {audit['smoke_questions']}",
        f"- Smoke fixed rollouts: {audit['smoke_rollouts']}",
        "- New generation: no",
        "",
    ]
    (output_dir / "ELIGIBILITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    result = build_manifest(
        rows,
        smoke_questions=args.smoke_questions,
        discovery_fraction=args.discovery_fraction,
        seed=args.seed,
    )
    write_manifest(result, Path(args.output_dir), args)


if __name__ == "__main__":
    main()
