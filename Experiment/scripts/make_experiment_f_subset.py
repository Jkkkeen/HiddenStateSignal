#!/usr/bin/env python3
"""Create a fixed question-id subset for Experiment F checkpoint sweeps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select fixed eval question ids for Experiment F.")
    parser.add_argument("--input", default="data/rollouts_mathverse_full_cot_labeled_mcq_mixed.jsonl")
    parser.add_argument("--output", default="data/experiment_f/question_ids.txt")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require-mixed", action="store_true", default=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_question_ids(
    rows: list[dict[str, Any]],
    limit: int,
    seed: int,
    require_mixed: bool = True,
) -> list[str]:
    labels: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row.get("question_id") is None:
            continue
        labels[str(row["question_id"])].append(bool(row.get("is_correct", False)))

    candidates = []
    for qid, ys in labels.items():
        has_true = any(ys)
        has_false = any(not y for y in ys)
        if require_mixed and not (has_true and has_false):
            continue
        candidates.append(qid)

    candidates = sorted(candidates)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    selected = [candidates[int(i)] for i in order]
    if limit is not None and limit >= 0:
        selected = selected[:limit]
    return selected


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    selected = select_question_ids(rows, args.limit, args.seed, args.require_mixed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for qid in selected:
            f.write(f"{qid}\n")
    print(f"input rows: {len(rows)}")
    print(f"selected questions: {len(selected)}")
    print(f"saved: {out}")
    print("first ids:", selected[:10])


if __name__ == "__main__":
    main()

