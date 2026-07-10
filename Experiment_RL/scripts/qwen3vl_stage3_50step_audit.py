#!/usr/bin/env python3
"""Audit matched A1/C2 Qwen3-VL 50-step offline answer eval results."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


VALID_OPTIONS = ("A", "B", "C", "D")
LOG_METRICS = (
    "c2_frac_groups_with_nonzero_b_gain_std",
    "c2_frac_groups_where_b_gain_changes_ranking",
    "c2_all_same_answer_groups",
    "c2_frac_all_same_answer_groups_with_b_gain_ranking",
    "c2_within_group_corr_answer_b_gain_mean",
    "c2_1/actor_option_probe_failed_mean",
    "c2_1/actor_option_gain_raw_mean",
    "c2_1/actor_option_gain_raw_std",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "critic/score/mean",
    "actor/grad_norm",
    "timing_s/gen",
    "timing_s/adv",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _qid(row: dict[str, Any], fallback: int) -> str:
    value = row.get("question_id")
    return str(value) if value is not None and value != "" else f"row_{fallback}"


def _pred(row: dict[str, Any]) -> str:
    value = row.get("prediction")
    return str(value) if value in VALID_OPTIONS else "INVALID"


def paired_outcome_counts(
    a1_rows: list[dict[str, Any]], c2_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    a1_by_id = {_qid(row, index): row for index, row in enumerate(a1_rows)}
    c2_by_id = {_qid(row, index): row for index, row in enumerate(c2_rows)}
    common_ids = [qid for qid in a1_by_id if qid in c2_by_id]

    outcome_counts = Counter(
        {
            "both_wrong": 0,
            "a1_wrong_c2_right": 0,
            "a1_right_c2_wrong": 0,
            "both_right": 0,
        }
    )
    prediction_delta = Counter()
    truncation_pairs = Counter()
    length_deltas: list[int] = []
    flips: list[dict[str, Any]] = []

    for qid in common_ids:
        a1 = a1_by_id[qid]
        c2 = c2_by_id[qid]
        a1_correct = bool(a1.get("correct"))
        c2_correct = bool(c2.get("correct"))
        if a1_correct and c2_correct:
            outcome = "both_right"
        elif a1_correct and not c2_correct:
            outcome = "a1_right_c2_wrong"
        elif not a1_correct and c2_correct:
            outcome = "a1_wrong_c2_right"
        else:
            outcome = "both_wrong"
        outcome_counts[outcome] += 1

        a1_pred = _pred(a1)
        c2_pred = _pred(c2)
        prediction_delta[c2_pred] += 1
        prediction_delta[a1_pred] -= 1
        a1_tokens = int(a1.get("output_token_count") or 0)
        c2_tokens = int(c2.get("output_token_count") or 0)
        length_delta = c2_tokens - a1_tokens
        length_deltas.append(length_delta)
        trunc_key = f"{bool(a1.get('truncated_by_length'))}->{bool(c2.get('truncated_by_length'))}"
        truncation_pairs[trunc_key] += 1

        if outcome in {"a1_wrong_c2_right", "a1_right_c2_wrong"}:
            flips.append(
                {
                    "question_id": qid,
                    "answer": a1.get("answer", c2.get("answer")),
                    "a1_prediction": a1_pred,
                    "c2_prediction": c2_pred,
                    "a1_correct": a1_correct,
                    "c2_correct": c2_correct,
                    "a1_tokens": a1_tokens,
                    "c2_tokens": c2_tokens,
                    "token_delta": length_delta,
                    "a1_truncated": bool(a1.get("truncated_by_length")),
                    "c2_truncated": bool(c2.get("truncated_by_length")),
                }
            )

    return {
        "num_common": len(common_ids),
        "outcome_counts": dict(outcome_counts),
        "prediction_delta": dict(sorted(prediction_delta.items())),
        "truncation_pairs": dict(sorted(truncation_pairs.items())),
        "length_delta_mean": float(mean(length_deltas)) if length_deltas else 0.0,
        "length_delta_min": min(length_deltas) if length_deltas else 0,
        "length_delta_max": max(length_deltas) if length_deltas else 0,
        "flips": flips,
    }


def _metric_values(text: str, metric: str) -> list[float]:
    pattern = re.compile(rf"(?<![\w/]){re.escape(metric)}:np\.float64\(([-+0-9.eE]+)\)|(?<![\w/]){re.escape(metric)}:([-+0-9.eE]+)")
    values: list[float] = []
    for match in pattern.finditer(text):
        raw = match.group(1) or match.group(2)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "last": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "last": float(values[-1]),
    }


def parse_c2_training_log(log_text: str) -> dict[str, Any]:
    steps = [int(value) for value in re.findall(r"training/global_step:(\d+)", log_text)]
    if not steps:
        steps = [int(value) for value in re.findall(r"\bstep:(\d+)", log_text)]
    summary: dict[str, Any] = {
        "num_logged_steps": len(steps),
        "latest_step": max(steps) if steps else None,
    }
    for metric in LOG_METRICS:
        summary[metric] = _summarize_values(_metric_values(log_text, metric))
    return summary


def summarize_eval_pair(a1_dir: Path, c2_dir: Path, c2_log_text: str) -> dict[str, Any]:
    a1_summary = read_json(a1_dir / "summary.json")
    c2_summary = read_json(c2_dir / "summary.json")
    a1_rows = read_jsonl(a1_dir / "generations.jsonl")
    c2_rows = read_jsonl(c2_dir / "generations.jsonl")
    paired = paired_outcome_counts(a1_rows, c2_rows)
    return {
        "headline": {
            "a1_accuracy": a1_summary.get("accuracy"),
            "c2_accuracy": c2_summary.get("accuracy"),
            "accuracy_delta": round(
                float(c2_summary.get("accuracy", 0.0)) - float(a1_summary.get("accuracy", 0.0)),
                12,
            ),
            "a1_balanced_accuracy": a1_summary.get("balanced_accuracy"),
            "c2_balanced_accuracy": c2_summary.get("balanced_accuracy"),
            "balanced_accuracy_delta": round(
                float(c2_summary.get("balanced_accuracy", 0.0))
                - float(a1_summary.get("balanced_accuracy", 0.0)),
                12,
            ),
            "a1_invalid_predictions": a1_summary.get("invalid_predictions"),
            "c2_invalid_predictions": c2_summary.get("invalid_predictions"),
            "a1_output_tokens_mean": a1_summary.get("output_tokens_mean"),
            "c2_output_tokens_mean": c2_summary.get("output_tokens_mean"),
            "a1_truncated_count": a1_summary.get("truncated_count"),
            "c2_truncated_count": c2_summary.get("truncated_count"),
        },
        "a1_summary": a1_summary,
        "c2_summary": c2_summary,
        "paired": paired,
        "c2_training_log": parse_c2_training_log(c2_log_text),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    h = result["headline"]
    paired = result["paired"]
    c2_log = result["c2_training_log"]
    outcome = paired["outcome_counts"]
    pred_delta = paired["prediction_delta"]
    lines = [
        "# Qwen3-VL A1/C2 50-Step Audit",
        "",
        "## Headline",
        "",
        "| Metric | A1 | C2 | Delta |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {_fmt(h['a1_accuracy'])} | {_fmt(h['c2_accuracy'])} | {_fmt(h['accuracy_delta'])} |",
        f"| Balanced accuracy | {_fmt(h['a1_balanced_accuracy'])} | {_fmt(h['c2_balanced_accuracy'])} | {_fmt(h['balanced_accuracy_delta'])} |",
        f"| Invalid predictions | {_fmt(h['a1_invalid_predictions'])} | {_fmt(h['c2_invalid_predictions'])} | |",
        f"| Mean output tokens | {_fmt(h['a1_output_tokens_mean'])} | {_fmt(h['c2_output_tokens_mean'])} | |",
        f"| Truncated count | {_fmt(h['a1_truncated_count'])} | {_fmt(h['c2_truncated_count'])} | |",
        "",
        "## Paired Outcome",
        "",
        "| Outcome | Count |",
        "|---|---:|",
        f"| both right | {outcome.get('both_right', 0)} |",
        f"| both wrong | {outcome.get('both_wrong', 0)} |",
        f"| A1 wrong, C2 right | {outcome.get('a1_wrong_c2_right', 0)} |",
        f"| A1 right, C2 wrong | {outcome.get('a1_right_c2_wrong', 0)} |",
        "",
        "## Prediction Delta",
        "",
        "| Prediction | C2 - A1 |",
        "|---|---:|",
    ]
    for key in sorted(pred_delta):
        lines.append(f"| {key} | {pred_delta[key]} |")

    lines.extend(
        [
            "",
            "## C2 Training Diagnostics",
            "",
            "| Metric | Mean | Min | Max | Last | Count |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in LOG_METRICS:
        item = c2_log.get(metric, {})
        lines.append(
            f"| {metric} | {_fmt(item.get('mean'))} | {_fmt(item.get('min'))} | "
            f"{_fmt(item.get('max'))} | {_fmt(item.get('last'))} | {_fmt(item.get('count'))} |"
        )

    lines.extend(
        [
            "",
            "## Flip Examples",
            "",
            "| qid | answer | A1 pred | C2 pred | A1 ok | C2 ok | A1 tok | C2 tok | delta |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired["flips"][:40]:
        lines.append(
            f"| {row['question_id']} | {row['answer']} | {row['a1_prediction']} | {row['c2_prediction']} | "
            f"{row['a1_correct']} | {row['c2_correct']} | {row['a1_tokens']} | {row['c2_tokens']} | {row['token_delta']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-dir", required=True, type=Path)
    parser.add_argument("--c2-dir", required=True, type=Path)
    parser.add_argument("--c2-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-name", default="A1_C2_50STEP_AUDIT.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c2_log_text = args.c2_log.read_text(errors="ignore", encoding="utf-8")
    result = summarize_eval_pair(args.a1_dir, args.c2_dir, c2_log_text)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / args.report_name, result)
    print(json.dumps(result["headline"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir / args.report_name}")


if __name__ == "__main__":
    main()
