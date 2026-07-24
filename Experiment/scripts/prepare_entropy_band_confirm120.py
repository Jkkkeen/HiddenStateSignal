#!/usr/bin/env python3
"""Freeze entropy-band acquisition candidates and select the formal cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


MCQ_ANSWERS = {"A", "B", "C", "D"}
QUERY_FIELDS = ("query_wo", "query", "question", "problem", "prompt")


@dataclass(frozen=True)
class FormalCohort:
    selected_questions: list[str]
    strict_questions: list[str]
    selected_rows: list[dict[str, Any]]
    summaries: pd.DataFrame
    target_questions: int

    @property
    def ready(self) -> bool:
        return len(self.selected_questions) >= self.target_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Freeze the entropy-blind candidate order.")
    freeze.add_argument("--mathverse-json", required=True)
    freeze.add_argument("--exclude-rollouts", required=True)
    freeze.add_argument("--output-dir", required=True)
    freeze.add_argument("--candidate-count", type=int, default=800)
    freeze.add_argument("--primary-count", type=int, default=600)
    freeze.add_argument("--reserve-batch", type=int, default=50)
    freeze.add_argument("--seed", type=int, default=20260725)

    extend = subparsers.add_parser(
        "extend", help="Freeze a prefix-validated candidate extension."
    )
    extend.add_argument("--mathverse-json", required=True)
    extend.add_argument("--exclude-rollouts", required=True)
    extend.add_argument("--existing-candidate-ids", required=True)
    extend.add_argument("--output-dir", required=True)
    extend.add_argument("--extension-count", type=int, default=500)
    extend.add_argument("--batch-size", type=int, default=50)
    extend.add_argument("--seed", type=int, default=20260725)

    select = subparsers.add_parser("select", help="Select the first eligible frozen candidates.")
    select.add_argument("--labeled-rollouts", required=True)
    select.add_argument("--candidate-ids", required=True)
    select.add_argument("--output-dir", required=True)
    select.add_argument("--target-questions", type=int, default=120)
    select.add_argument("--min-correct", type=int, default=2)
    select.add_argument("--min-wrong", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "records"):
            if isinstance(data.get(key), list):
                return data[key]
        if all(isinstance(value, dict) for value in data.values()):
            return list(data.values())
    raise ValueError(f"unsupported MathVerse JSON structure: {path}")


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


def question_id(record: dict[str, Any], index: int) -> str:
    for field in ("question_id", "sample_index", "id", "uid", "problem_id"):
        value = record.get(field)
        if value is not None and value != "":
            return str(value)
    return f"mathverse_{index:06d}"


def _has_query(record: dict[str, Any]) -> bool:
    return any(str(record.get(field, "")).strip() for field in QUERY_FIELDS)


def ordered_candidate_ids(
    records: Iterable[dict[str, Any]],
    *,
    excluded_question_ids: set[str],
    seed: int,
) -> list[str]:
    unique: set[str] = set()
    for index, record in enumerate(records):
        qid = question_id(record, index)
        answer = str(record.get("answer", "")).strip().upper()
        if qid in excluded_question_ids or answer not in MCQ_ANSWERS or not _has_query(record):
            continue
        unique.add(qid)
    return sorted(
        unique,
        key=lambda qid: (
            hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest(),
            qid,
        ),
    )


def freeze_candidates(
    records: Iterable[dict[str, Any]],
    *,
    excluded_question_ids: set[str],
    candidate_count: int,
    seed: int,
) -> list[str]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    ordered = ordered_candidate_ids(
        records,
        excluded_question_ids=excluded_question_ids,
        seed=seed,
    )
    if len(ordered) < candidate_count:
        raise ValueError(f"only {len(ordered)} unused MCQ candidates; need {candidate_count}")
    return ordered[:candidate_count]


def freeze_candidate_extension(
    records: Iterable[dict[str, Any]],
    *,
    excluded_question_ids: set[str],
    existing_candidate_ids: list[str],
    extension_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not existing_candidate_ids:
        raise ValueError("existing candidate IDs must not be empty")
    if extension_count <= 0:
        raise ValueError("extension_count must be positive")
    ordered = ordered_candidate_ids(
        records,
        excluded_question_ids=excluded_question_ids,
        seed=seed,
    )
    prefix_size = len(existing_candidate_ids)
    required = prefix_size + extension_count
    if len(ordered) < required:
        raise ValueError(f"only {len(ordered)} unused MCQ candidates; need {required}")
    if ordered[:prefix_size] != existing_candidate_ids:
        raise ValueError("existing candidate prefix does not match deterministic order")
    combined = ordered[:required]
    return combined, combined[prefix_size:]


def is_clean_complete(row: dict[str, Any]) -> bool:
    answer = str(row.get("answer", "")).strip().upper()
    prediction = str(row.get("pred_answer", "")).strip().upper()
    return bool(
        not row.get("truncated", False)
        and row.get("has_think_close", False)
        and answer in MCQ_ANSWERS
        and prediction in MCQ_ANSWERS
    )


def select_formal_cohort(
    rows: Iterable[dict[str, Any]],
    candidate_order: list[str],
    *,
    target_questions: int,
    min_correct: int = 2,
    min_wrong: int = 2,
) -> FormalCohort:
    if target_questions <= 0 or min_correct <= 0 or min_wrong <= 0:
        raise ValueError("target and eligibility counts must be positive")
    order_index = {qid: index for index, qid in enumerate(candidate_order)}
    clean = [
        dict(row)
        for row in rows
        if str(row.get("question_id", "")) in order_index and is_clean_complete(row)
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in clean:
        grouped.setdefault(str(row["question_id"]), []).append(row)

    summary_rows = []
    eligible = []
    strict = []
    for qid in candidate_order:
        group = grouped.get(qid, [])
        labels = [bool(row.get("is_correct")) for row in group]
        n_correct = int(sum(labels))
        n_wrong = int(len(labels) - n_correct)
        is_eligible = n_correct >= min_correct and n_wrong >= min_wrong
        is_strict = n_correct >= 3 and n_wrong >= 3
        summary_rows.append(
            {
                "question_id": qid,
                "candidate_rank": order_index[qid],
                "n_clean": len(group),
                "n_correct": n_correct,
                "n_wrong": n_wrong,
                "eligible_2plus2": is_eligible,
                "eligible_3plus3": is_strict,
            }
        )
        if is_eligible and len(eligible) < target_questions:
            eligible.append(qid)
            if is_strict:
                strict.append(qid)

    selected_set = set(eligible)
    selected_rows = [row for row in clean if str(row["question_id"]) in selected_set]
    selected_rows.sort(
        key=lambda row: (
            order_index[str(row["question_id"])],
            int(row.get("rollout_id", -1)),
        )
    )
    return FormalCohort(
        selected_questions=eligible,
        strict_questions=strict,
        selected_rows=selected_rows,
        summaries=pd.DataFrame(summary_rows),
        target_questions=target_questions,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_ids(path: Path, ids: list[str]) -> None:
    _atomic_text(path, "".join(f"{qid}\n" for qid in ids))


def write_frozen_ids(path: Path, ids: list[str]) -> None:
    content = "".join(f"{qid}\n" for qid in ids)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"frozen candidate file differs: {path}")
        return
    _atomic_text(path, content)


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        ids = [line.strip() for line in handle if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate candidate IDs in {path}")
    return ids


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def run_freeze(args: argparse.Namespace) -> None:
    data_path = Path(args.mathverse_json)
    exclude_path = Path(args.exclude_rollouts)
    output_dir = Path(args.output_dir)
    excluded = {
        str(row.get("question_id", ""))
        for row in load_jsonl(exclude_path)
        if str(row.get("question_id", ""))
    }
    candidates = freeze_candidates(
        load_json(data_path),
        excluded_question_ids=excluded,
        candidate_count=args.candidate_count,
        seed=args.seed,
    )
    if not 0 < args.primary_count <= len(candidates):
        raise ValueError("primary_count must be within the frozen candidate count")
    if args.reserve_batch <= 0:
        raise ValueError("reserve_batch must be positive")
    write_ids(output_dir / "candidate_ids_all800.txt", candidates)
    write_ids(output_dir / "candidate_ids_primary600.txt", candidates[: args.primary_count])
    reserve = candidates[args.primary_count :]
    reserve_files = []
    for batch_index, start in enumerate(range(0, len(reserve), args.reserve_batch), start=1):
        path = output_dir / f"candidate_ids_reserve50_{batch_index:02d}.txt"
        write_ids(path, reserve[start : start + args.reserve_batch])
        reserve_files.append(path.name)
    audit = {
        "mathverse_json": str(data_path),
        "mathverse_sha256": sha256_file(data_path),
        "exclude_rollouts": str(exclude_path),
        "exclude_sha256": sha256_file(exclude_path),
        "excluded_prior_questions": len(excluded),
        "seed": args.seed,
        "candidate_count": len(candidates),
        "primary_count": args.primary_count,
        "reserve_count": len(reserve),
        "reserve_batch": args.reserve_batch,
        "reserve_files": reserve_files,
        "entropy_used_for_selection": False,
    }
    _atomic_json(output_dir / "candidate_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def run_extend(args: argparse.Namespace) -> None:
    data_path = Path(args.mathverse_json)
    exclude_path = Path(args.exclude_rollouts)
    existing_path = Path(args.existing_candidate_ids)
    output_dir = Path(args.output_dir)
    excluded = {
        str(row.get("question_id", ""))
        for row in load_jsonl(exclude_path)
        if str(row.get("question_id", ""))
    }
    existing = read_ids(existing_path)
    combined, extension = freeze_candidate_extension(
        load_json(data_path),
        excluded_question_ids=excluded,
        existing_candidate_ids=existing,
        extension_count=args.extension_count,
        seed=args.seed,
    )
    if args.batch_size <= 0 or args.extension_count % args.batch_size:
        raise ValueError("extension_count must be divisible by batch_size")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"candidate_ids_all{len(combined)}.txt"
    extension_path = output_dir / f"candidate_ids_extension{len(extension)}.txt"
    write_frozen_ids(combined_path, combined)
    write_frozen_ids(extension_path, extension)
    batch_files = []
    for batch_index, start in enumerate(
        range(0, len(extension), args.batch_size), start=1
    ):
        path = output_dir / (
            f"candidate_ids_extension{args.batch_size}_{batch_index:02d}.txt"
        )
        write_frozen_ids(path, extension[start : start + args.batch_size])
        batch_files.append({"name": path.name, "sha256": sha256_file(path)})
    audit = {
        "mathverse_json": str(data_path),
        "mathverse_sha256": sha256_file(data_path),
        "exclude_rollouts": str(exclude_path),
        "exclude_sha256": sha256_file(exclude_path),
        "existing_candidate_ids": str(existing_path),
        "existing_candidate_sha256": sha256_file(existing_path),
        "existing_count": len(existing),
        "extension_count": len(extension),
        "combined_count": len(combined),
        "extension_rank_start_one_based": len(existing) + 1,
        "extension_rank_end_one_based": len(combined),
        "combined_sha256": sha256_file(combined_path),
        "extension_sha256": sha256_file(extension_path),
        "batch_size": args.batch_size,
        "batch_files": batch_files,
        "seed": args.seed,
        "entropy_used_for_selection": False,
    }
    audit_path = output_dir / "candidate_extension_audit.json"
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    if audit_path.exists() and audit_path.read_text(encoding="utf-8") != audit_text:
        raise ValueError(f"frozen extension audit differs: {audit_path}")
    if not audit_path.exists():
        _atomic_text(audit_path, audit_text)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def run_select(args: argparse.Namespace) -> None:
    labeled_path = Path(args.labeled_rollouts)
    candidate_path = Path(args.candidate_ids)
    output_dir = Path(args.output_dir)
    result = select_formal_cohort(
        load_jsonl(labeled_path),
        read_ids(candidate_path),
        target_questions=args.target_questions,
        min_correct=args.min_correct,
        min_wrong=args.min_wrong,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "question_eligibility.csv"
    temporary = summary_path.with_suffix(".tmp.csv")
    result.summaries.to_csv(temporary, index=False)
    os.replace(temporary, summary_path)
    manifest_name = "manifest_confirm120.jsonl" if result.ready else "manifest_partial.jsonl"
    _atomic_jsonl(output_dir / manifest_name, result.selected_rows)
    status = {
        "ready": result.ready,
        "target_questions": args.target_questions,
        "selected_questions": len(result.selected_questions),
        "selected_rollouts": len(result.selected_rows),
        "strict_3plus3_questions": len(result.strict_questions),
        "formal_manifest": manifest_name,
        "candidate_ids": str(candidate_path),
        "candidate_ids_sha256": sha256_file(candidate_path),
        "labeled_rollouts": str(labeled_path),
        "labeled_rollouts_sha256": sha256_file(labeled_path),
        "entropy_used_for_selection": False,
    }
    _atomic_json(output_dir / "eligibility_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        run_freeze(args)
    elif args.command == "extend":
        run_extend(args)
    else:
        run_select(args)


if __name__ == "__main__":
    main()
