#!/usr/bin/env python3
"""Inspect answer-after-think response segments before response-stage forward."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from response_segment_steps import answer_after_think, split_response_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect response-after-think segments.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="angle_response_inspect")
    parser.add_argument("--examples", type=int, default=30)
    parser.add_argument("--min-step-chars", type=int, default=12)
    parser.add_argument("--short-answer-chars", type=int, default=40)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile(values: list[int], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def inspect_rows(
    rows: list[dict[str, Any]],
    min_step_chars: int,
    short_answer_chars: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for row in rows:
        segment, start, end, status = answer_after_think(str(row.get("response", "")))
        steps, mode = split_response_steps(
            segment,
            min_chars=min_step_chars,
            short_answer_chars=short_answer_chars,
        )
        record = {
            "question_id": str(row.get("question_id", "")),
            "rollout_id": int(row.get("rollout_id", -1)),
            "is_correct": bool(row.get("is_correct", False)),
            "answer": row.get("answer"),
            "pred_answer": row.get("pred_answer"),
            "segment_status": status,
            "step_mode": mode,
            "answer_chars": len(segment),
            "answer_steps": len(steps),
            "segment_start_char": start,
            "segment_end_char": end,
        }
        records.append(record)
        examples.append({**record, "segment": segment, "steps": steps})
    return pd.DataFrame(records), examples


def choose_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    groups = [
        sorted(examples, key=lambda x: x["answer_chars"])[: max(1, limit // 5)],
        [x for x in examples if x["step_mode"] == "block"][: max(1, limit // 3)],
        [x for x in examples if x["step_mode"] == "sentence"][: max(1, limit // 3)],
        sorted(examples, key=lambda x: x["answer_chars"], reverse=True)[: max(1, limit // 4)],
    ]
    picked: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for group in groups:
        for item in group:
            key = (item["question_id"], item["rollout_id"])
            if key in seen:
                continue
            seen.add(key)
            picked.append(item)
            if len(picked) >= limit:
                return picked
    return picked


def write_report(output_dir: Path, df: pd.DataFrame, examples: list[dict[str, Any]]) -> None:
    report = output_dir / "RESPONSE_SEGMENT_INSPECT.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Response Segment Inspect\n\n")
        f.write("This report inspects `</think>`-after answer segments before response-stage basin analysis.\n\n")
        f.write("## Summary\n\n")
        f.write(f"- rows: {len(df)}\n")
        f.write(f"- questions: {df['question_id'].nunique() if not df.empty else 0}\n")
        f.write(f"- empty segments: {int((df['answer_chars'] == 0).sum())}\n")
        f.write(f"- <=1 step segments: {int((df['answer_steps'] <= 1).sum())}\n")
        f.write(f"- >=3 step segments: {int((df['answer_steps'] >= 3).sum())}\n\n")

        f.write("## Length Percentiles\n\n")
        f.write("| field | p50 | p75 | p90 | p95 | max |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for col in ["answer_chars", "answer_steps"]:
            vals = df[col].astype(int).tolist()
            f.write(
                f"| {col} | {percentile(vals, 50):.1f} | {percentile(vals, 75):.1f} | "
                f"{percentile(vals, 90):.1f} | {percentile(vals, 95):.1f} | {max(vals) if vals else 0} |\n"
            )

        f.write("\n## Segment Status\n\n")
        f.write("| status | count |\n|---|---:|\n")
        for status, count in Counter(df["segment_status"]).most_common():
            f.write(f"| {status} | {count} |\n")

        f.write("\n## Step Mode\n\n")
        f.write("| mode | count |\n|---|---:|\n")
        for mode, count in Counter(df["step_mode"]).most_common():
            f.write(f"| {mode} | {count} |\n")

        f.write("\n## Examples\n\n")
        for item in examples:
            f.write(
                f"### qid={item['question_id']} rid={item['rollout_id']} "
                f"correct={item['is_correct']} answer={item['answer']} pred={item['pred_answer']}\n\n"
            )
            f.write(
                f"- chars: {item['answer_chars']}\n"
                f"- steps: {item['answer_steps']}\n"
                f"- mode: {item['step_mode']}\n\n"
            )
            for idx, step in enumerate(item["steps"][:8]):
                heading = step.get("heading") or f"step {idx}"
                text = str(step.get("text", "")).replace("\n", " ")
                f.write(f"**{idx}. {heading}**  \n{text[:500]}\n\n")
            if len(item["steps"]) > 8:
                f.write(f"... {len(item['steps']) - 8} more steps omitted.\n\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    df, examples = inspect_rows(rows, args.min_step_chars, args.short_answer_chars)
    picked = choose_examples(examples, args.examples)
    df.to_csv(output_dir / "response_segment_summary.csv", index=False)
    write_report(output_dir, df, picked)

    print(f"input: {input_path}")
    print(f"rows: {len(df)}")
    print(f"questions: {df['question_id'].nunique() if not df.empty else 0}")
    print("segment_status:", dict(Counter(df["segment_status"])))
    print("step_mode:", dict(Counter(df["step_mode"])))
    print(
        "answer_chars p50/p90/max:",
        percentile(df["answer_chars"].astype(int).tolist(), 50),
        percentile(df["answer_chars"].astype(int).tolist(), 90),
        int(df["answer_chars"].max()) if not df.empty else 0,
    )
    print(
        "answer_steps p50/p90/max:",
        percentile(df["answer_steps"].astype(int).tolist(), 50),
        percentile(df["answer_steps"].astype(int).tolist(), 90),
        int(df["answer_steps"].max()) if not df.empty else 0,
    )
    print(f"saved: {output_dir / 'RESPONSE_SEGMENT_INSPECT.md'}")


if __name__ == "__main__":
    main()
