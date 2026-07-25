#!/usr/bin/env python3
"""Freeze the Experiment 04 eligible cohort and blind question split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Experiment 04 manifests.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--expected-rollouts", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_source_line", source_line)
            question_id = str(row.get("question_id", ""))
            if not question_id:
                raise ValueError(f"missing question_id at source line {source_line}")
            if "rollout_id" not in row or "is_correct" not in row:
                raise ValueError(f"missing rollout_id/is_correct at source line {source_line}")
            row["question_id"] = question_id
            row["rollout_id"] = int(row["rollout_id"])
            row["is_correct"] = bool(row["is_correct"])
            rows.append(row)
    if not rows:
        raise ValueError("input contains no rollout rows")
    return rows


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _order_key(question_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()


def _think_length(row: dict[str, Any]) -> int:
    for key in ("think_token_count", "response_token_count", "output_token_count"):
        value = row.get(key)
        if value is not None:
            return int(value)
    return len(str(row.get("response", "")))


def prepare(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int,
    expected_rollouts: int | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir)
    rows = _load_jsonl(input_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)

    duplicate_pairs = []
    for question_id, question_rows in grouped.items():
        rollout_ids = [int(row["rollout_id"]) for row in question_rows]
        if len(rollout_ids) != len(set(rollout_ids)):
            duplicate_pairs.append(question_id)
    if duplicate_pairs:
        raise ValueError(f"duplicate rollout IDs for questions: {duplicate_pairs[:5]}")

    eligible = []
    strict = []
    non_expected = []
    counts: dict[str, dict[str, int]] = {}
    for question_id, question_rows in grouped.items():
        correct = sum(bool(row["is_correct"]) for row in question_rows)
        wrong = len(question_rows) - correct
        counts[question_id] = {
            "rollouts": len(question_rows),
            "correct": correct,
            "wrong": wrong,
        }
        if correct >= 2 and wrong >= 2:
            eligible.append(question_id)
            if correct >= 3 and wrong >= 3:
                strict.append(question_id)
            if expected_rollouts is not None and len(question_rows) != expected_rollouts:
                non_expected.append(question_id)
    if expected_rollouts is not None and non_expected:
        raise ValueError(
            f"eligible questions with rollout count != {expected_rollouts}: {non_expected[:5]}"
        )
    if len(eligible) < 2:
        raise ValueError("at least two eligible questions are required")

    ordered = sorted(eligible, key=lambda question_id: _order_key(question_id, seed))
    discovery_ids = ordered[: len(ordered) // 2]
    confirm_ids = ordered[len(ordered) // 2 :]
    discovery_set = set(discovery_ids)
    confirm_set = set(confirm_ids)
    if discovery_set & confirm_set:
        raise AssertionError("discovery and confirm questions overlap")

    def selected_rows(question_ids: list[str], role: str) -> list[dict[str, Any]]:
        selected = []
        for question_id in question_ids:
            for row in sorted(grouped[question_id], key=lambda value: value["rollout_id"]):
                copy = dict(row)
                copy["experiment04_split"] = role
                copy["experiment04_seed"] = seed
                selected.append(copy)
        return selected

    discovery_rows = selected_rows(discovery_ids, "discovery")
    confirm_rows = selected_rows(confirm_ids, "confirm")
    smoke_id = max(
        discovery_ids,
        key=lambda question_id: max(_think_length(row) for row in grouped[question_id]),
    )
    smoke_rows = [row for row in discovery_rows if row["question_id"] == smoke_id]

    _atomic_jsonl(output_dir / "manifest_discovery.jsonl", discovery_rows)
    _atomic_jsonl(output_dir / "manifest_confirm.jsonl", confirm_rows)
    _atomic_jsonl(output_dir / "manifest_smoke.jsonl", smoke_rows)
    _atomic_json(
        output_dir / "question_counts.json",
        {question_id: counts[question_id] for question_id in ordered},
    )

    ordered_digest = hashlib.sha256(("\n".join(ordered) + "\n").encode("utf-8")).hexdigest()
    summary: dict[str, Any] = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "seed": seed,
        "input_rows": len(rows),
        "input_questions": len(grouped),
        "eligible_questions": len(ordered),
        "strict_3plus3_questions": len(strict),
        "discovery_questions": len(discovery_ids),
        "discovery_rollouts": len(discovery_rows),
        "confirm_questions": len(confirm_ids),
        "confirm_rollouts": len(confirm_rows),
        "question_overlap": len(discovery_set & confirm_set),
        "smoke_question_id": smoke_id,
        "smoke_rollouts": len(smoke_rows),
        "smoke_max_think_tokens": max(_think_length(row) for row in smoke_rows),
        "ordered_question_ids": ordered,
        "ordered_question_ids_sha256": ordered_digest,
    }
    _atomic_json(output_dir / "split_manifest.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = prepare(
        Path(args.input),
        Path(args.output_dir),
        seed=args.seed,
        expected_rollouts=args.expected_rollouts,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
