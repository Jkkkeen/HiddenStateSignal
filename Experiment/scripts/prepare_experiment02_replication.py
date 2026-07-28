#!/usr/bin/env python3
"""Freeze the Experiment 02 replication and smoke manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from long_success_trajectory_common import (
    eligible_question_summaries,
    is_clean_complete_row,
    select_stratified_questions,
)


@dataclass(frozen=True)
class ReplicationManifest:
    summaries: pd.DataFrame
    selected_questions: list[str]
    strict_questions: list[str]
    smoke_questions: list[str]
    selected_rows: list[dict[str, Any]]
    smoke_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Experiment 02 cohort.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--exclude-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question-limit", type=int, default=96)
    parser.add_argument("--smoke-questions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_source_line", source_line)
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_replication_manifest(
    rows: list[dict[str, Any]],
    *,
    excluded_question_ids: set[str],
    question_limit: int,
    smoke_questions: int,
    seed: int,
) -> ReplicationManifest:
    summaries = eligible_question_summaries(rows)
    eligible = summaries[
        (summaries["n_correct"] >= 2)
        & (summaries["n_wrong"] >= 2)
        & ~summaries["question_id"].astype(str).isin(excluded_question_ids)
    ].copy()
    if len(eligible) < question_limit:
        raise ValueError(
            f"only {len(eligible)} eligible new questions; need {question_limit}"
        )
    candidates = eligible["question_id"].astype(str).tolist()
    selected = select_stratified_questions(summaries, candidates, question_limit, seed)
    smoke = select_stratified_questions(
        summaries,
        selected,
        min(smoke_questions, len(selected)),
        seed + 1,
    )
    selected_set = set(selected)
    smoke_set = set(smoke)
    selected_rows = [
        row
        for row in rows
        if is_clean_complete_row(row)
        and str(row.get("question_id", "")) in selected_set
    ]
    selected_rows.sort(
        key=lambda row: (
            str(row.get("question_id", "")),
            int(row.get("rollout_id", -1)),
        )
    )
    strict = eligible.loc[
        eligible["question_id"].astype(str).isin(selected_set)
        & (eligible["n_correct"] >= 3)
        & (eligible["n_wrong"] >= 3),
        "question_id",
    ].astype(str).tolist()
    return ReplicationManifest(
        summaries=summaries,
        selected_questions=sorted(selected),
        strict_questions=sorted(strict),
        smoke_questions=sorted(smoke),
        selected_rows=selected_rows,
        smoke_rows=[
            row for row in selected_rows if str(row.get("question_id", "")) in smoke_set
        ],
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(
    result: ReplicationManifest,
    *,
    output_dir: Path,
    input_path: Path,
    exclude_path: Path,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formal_path = output_dir / "manifest_replication96.jsonl"
    smoke_path = output_dir / f"manifest_smoke{len(result.smoke_questions)}.jsonl"
    _write_jsonl(formal_path, result.selected_rows)
    _write_jsonl(smoke_path, result.smoke_rows)

    summaries = result.summaries.copy()
    summaries["in_replication"] = summaries["question_id"].astype(str).isin(
        result.selected_questions
    )
    summaries["in_smoke"] = summaries["question_id"].astype(str).isin(
        result.smoke_questions
    )
    summaries["in_strict_3plus3"] = summaries["question_id"].astype(str).isin(
        result.strict_questions
    )
    summaries.to_csv(output_dir / "question_splits.csv", index=False)
    labels = pd.DataFrame(result.selected_rows).groupby("question_id")["is_correct"].agg(
        n_rollouts="size",
        n_correct="sum",
    )
    labels["n_wrong"] = labels["n_rollouts"] - labels["n_correct"]
    if (labels[["n_correct", "n_wrong"]] < 2).any().any():
        raise RuntimeError("frozen manifest violates 2+2 eligibility")

    audit = {
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "exclude_manifest": str(exclude_path),
        "exclude_manifest_sha256": _sha256(exclude_path),
        "seed": int(seed),
        "eligible_new_questions": int(
            (
                (summaries["n_correct"] >= 2)
                & (summaries["n_wrong"] >= 2)
                & ~summaries["question_id"].astype(str).isin(
                    set(load_excluded_ids(exclude_path))
                )
            ).sum()
        ),
        "replication_questions": len(result.selected_questions),
        "replication_rollouts": len(result.selected_rows),
        "strict_3plus3_questions": len(result.strict_questions),
        "smoke_questions": len(result.smoke_questions),
        "smoke_rollouts": len(result.smoke_rows),
        "formal_manifest": formal_path.name,
        "smoke_manifest": smoke_path.name,
        "new_generation": False,
    }
    (output_dir / "ELIGIBILITY_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def load_excluded_ids(path: Path) -> set[str]:
    return {
        str(row.get("question_id", ""))
        for row in load_jsonl(path)
        if str(row.get("question_id", ""))
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    exclude_path = Path(args.exclude_manifest)
    rows = load_jsonl(input_path)
    excluded = load_excluded_ids(exclude_path)
    result = build_replication_manifest(
        rows,
        excluded_question_ids=excluded,
        question_limit=args.question_limit,
        smoke_questions=args.smoke_questions,
        seed=args.seed,
    )
    write_manifest(
        result,
        output_dir=Path(args.output_dir),
        input_path=input_path,
        exclude_path=exclude_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
