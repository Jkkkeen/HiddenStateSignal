#!/usr/bin/env python3
"""Label and filter Experiment F rollouts to the fixed question-id subset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label/filter Experiment F rollouts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question-ids", default="data/experiment_f/question_ids.txt")
    parser.add_argument("--mcq-only", action="store_true", default=True)
    return parser.parse_args()


def load_question_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def extract_choice(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip()
    patterns = [
        r"(?:answer)\s*(?:is|:)?\s*\**\s*([ABCD])\b",
        r"\b([ABCD])\s*(?:is\s+the\s+answer|is\s+correct)\b",
        r"(?:option|choice)\s*\**\s*([ABCD])\b",
        r"\*\*([ABCD])\*\*",
        r"\b([ABCD])\s*[:\.\)]",
        r"\(([ABCD])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, t, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    tail = t[-300:]
    matches = re.findall(r"\b([ABCD])\b", tail)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"\b([ABCD])\b", t)
    if matches:
        return matches[-1].upper()
    return None


def label_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer", "")).strip().upper()
    pred = extract_choice(str(row.get("response", "")))
    out = dict(row)
    out["answer"] = answer
    out["pred_answer"] = pred
    out["is_correct"] = bool(pred is not None and answer and pred == answer)
    out["response_length"] = len(str(row.get("response", "")).split())
    return out


def main() -> None:
    args = parse_args()
    keep_ids = load_question_ids(Path(args.question_ids))
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    pred_counter: Counter[str | None] = Counter()
    answer_counter: Counter[str] = Counter()
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            qid = str(row.get("question_id", ""))
            if qid not in keep_ids:
                continue
            labeled = label_row(row)
            answer = labeled["answer"]
            if args.mcq_only and answer not in {"A", "B", "C", "D"}:
                continue
            kept += 1
            pred_counter[labeled["pred_answer"]] += 1
            answer_counter[answer] += 1
            dst.write(json.dumps(labeled, ensure_ascii=False) + "\n")

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"question ids: {len(keep_ids)}")
    print(f"input rows: {total}")
    print(f"kept rows: {kept}")
    print("pred_counter:", dict(pred_counter))
    print("answer_counter:", dict(answer_counter))


if __name__ == "__main__":
    main()

