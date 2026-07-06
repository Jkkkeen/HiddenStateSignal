#!/usr/bin/env python3
"""Inspect Qwen3-VL Thinking long-CoT smoke rollouts.

Step 1 of ER_long.md only needs behavior-level diagnostics:
think length distribution, truncation rate, answer parsing, and mixed question
count.  No hidden states are loaded here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LENGTH_BINS = [
    ("<2k", 0, 2000),
    ("2-4k", 2000, 4000),
    ("4-8k", 4000, 8000),
    ("8k+", 8000, math.inf),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect long-CoT Thinking rollout smoke data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="long_cot_smoke")
    parser.add_argument("--labeled-output", default="")
    parser.add_argument("--model", default="", help="Optional model/tokenizer path for exact token counts.")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def rough_token_count(text: str | None) -> int:
    if not text:
        return 0
    # This is only a fallback for old rollout files without vLLM token counts.
    return len(str(text).split())


def load_tokenizer(model: str, hf_endpoint: str, hf_home: str, local_files_only: bool) -> Any | None:
    if not model:
        return None
    os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_home) / "hub")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
    except Exception:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model, local_files_only=local_files_only)
        return getattr(processor, "tokenizer", None)


def token_count(text: str | None, tokenizer: Any | None) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return rough_token_count(text)
    return int(len(tokenizer.encode(str(text), add_special_tokens=False)))


def segment_thinking_text(response: str | None) -> dict[str, Any]:
    text = str(response or "")
    open_tag = "<think>"
    close_tag = "</think>"
    open_idx = text.find(open_tag)
    close_idx = text.find(close_tag)
    has_open = open_idx >= 0
    has_close = close_idx >= 0

    if has_open and has_close and close_idx > open_idx:
        think_text = text[open_idx + len(open_tag) : close_idx]
        answer_text = text[close_idx + len(close_tag) :]
        status = "ok" if answer_text.strip() else "missing_answer_after_think"
    elif has_open and not has_close:
        think_text = text[open_idx + len(open_tag) :]
        answer_text = ""
        status = "missing_think_close"
    elif has_close and not has_open:
        think_text = text[:close_idx]
        answer_text = text[close_idx + len(close_tag) :]
        status = "implicit_think_close_only"
    else:
        think_text = ""
        answer_text = text
        status = "no_think_tags"

    return {
        "has_think_open": has_open,
        "has_think_close": has_close,
        "think_text": think_text,
        "answer_text": answer_text,
        "segment_status": status,
    }


def extract_choice(text: str | None) -> str | None:
    if not text:
        return None
    segment = segment_thinking_text(text)
    candidates = [segment["answer_text"], text]
    patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\**\s*([ABCD])\b",
        r"\b([ABCD])\s*(?:is\s+the\s+answer|is\s+correct)\b",
        r"(?:option|choice)\s*\**\s*([ABCD])\b",
        r"\*\*([ABCD])\*\*",
        r"\b([ABCD])\s*[:\.\)]",
        r"\(([ABCD])\)",
    ]
    for raw in candidates:
        t = str(raw or "").strip()
        if not t:
            continue
        for pattern in patterns:
            matches = re.findall(pattern, t, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        tail = t[-500:]
        matches = re.findall(r"\b([ABCD])\b", tail)
        if matches:
            return matches[-1].upper()
    return None


def length_bin(value: int | float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "unknown"
    v = float(value)
    for name, low, high in LENGTH_BINS:
        if low <= v < high:
            return name
    return "unknown"


def label_and_segment_row(row: dict[str, Any], tokenizer: Any | None = None) -> dict[str, Any]:
    out = dict(row)
    response = str(row.get("response", ""))
    segment = segment_thinking_text(response)
    answer = str(row.get("answer", "")).strip().upper()
    pred = extract_choice(response)

    output_token_count = row.get("output_token_count")
    if output_token_count is None:
        output_token_count = token_count(response, tokenizer)
    think_token_count = token_count(segment["think_text"], tokenizer)
    answer_token_count = token_count(segment["answer_text"], tokenizer)

    finish_reason = row.get("finish_reason")
    truncated_by_length = bool(row.get("truncated_by_length", False))
    if finish_reason == "length":
        truncated_by_length = True
    if output_token_count is not None and row.get("max_tokens") is not None:
        try:
            truncated_by_length = truncated_by_length or int(output_token_count) >= int(row["max_tokens"])
        except (TypeError, ValueError):
            pass

    truncated = bool(
        truncated_by_length
        or segment["segment_status"] in {"missing_think_close", "missing_answer_after_think"}
    )

    out.update(
        {
            "answer": answer,
            "pred_answer": pred,
            "is_correct": bool(pred is not None and answer in {"A", "B", "C", "D"} and pred == answer),
            "parsed_answer": pred is not None,
            "response_token_count": int(output_token_count or 0),
            "think_token_count": int(think_token_count),
            "answer_token_count": int(answer_token_count),
            "think_length_bin": length_bin(think_token_count),
            "truncated": truncated,
            **{k: v for k, v in segment.items() if not k.endswith("_text")},
        }
    )
    return out


def mixed_question_count(rows: list[dict[str, Any]], clean_only: bool) -> int:
    labels: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if clean_only and row.get("truncated"):
            continue
        labels[str(row.get("question_id", ""))].append(bool(row.get("is_correct", False)))
    return sum(1 for ys in labels.values() if any(ys) and any(not y for y in ys))


def safe_percentile(values: list[float], pct: float) -> float:
    values = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.percentile(values, pct)) if values else float("nan")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    questions = len({str(row.get("question_id", "")) for row in rows})
    clean = [row for row in rows if not row.get("truncated")]
    think_lengths = [float(row.get("think_token_count", 0)) for row in rows]
    response_lengths = [float(row.get("response_token_count", 0)) for row in rows]
    answer_lengths = [float(row.get("answer_token_count", 0)) for row in rows]
    return {
        "rollouts": n,
        "questions": questions,
        "clean_rollouts": len(clean),
        "truncated_rollouts": n - len(clean),
        "truncation_rate": float((n - len(clean)) / n) if n else float("nan"),
        "parsed_rate": float(sum(bool(row.get("parsed_answer")) for row in rows) / n) if n else float("nan"),
        "accuracy_full": float(sum(bool(row.get("is_correct")) for row in rows) / n) if n else float("nan"),
        "accuracy_clean": (
            float(sum(bool(row.get("is_correct")) for row in clean) / len(clean)) if clean else float("nan")
        ),
        "mixed_questions_full": mixed_question_count(rows, clean_only=False),
        "mixed_questions_clean": mixed_question_count(rows, clean_only=True),
        "mean_response_tokens": float(np.mean(response_lengths)) if response_lengths else float("nan"),
        "p50_response_tokens": safe_percentile(response_lengths, 50),
        "p90_response_tokens": safe_percentile(response_lengths, 90),
        "p95_response_tokens": safe_percentile(response_lengths, 95),
        "max_response_tokens": float(np.max(response_lengths)) if response_lengths else float("nan"),
        "mean_think_tokens": float(np.mean(think_lengths)) if think_lengths else float("nan"),
        "p50_think_tokens": safe_percentile(think_lengths, 50),
        "p90_think_tokens": safe_percentile(think_lengths, 90),
        "p95_think_tokens": safe_percentile(think_lengths, 95),
        "max_think_tokens": float(np.max(think_lengths)) if think_lengths else float("nan"),
        "mean_answer_tokens": float(np.mean(answer_lengths)) if answer_lengths else float("nan"),
        "has_think_close_rate": float(sum(bool(row.get("has_think_close")) for row in rows) / n)
        if n
        else float("nan"),
    }


def bin_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    data = []
    for name, _, _ in LENGTH_BINS:
        subset = [row for row in rows if row.get("think_length_bin") == name]
        if not subset:
            data.append(
                {
                    "think_length_bin": name,
                    "rollouts": 0,
                    "questions": 0,
                    "truncation_rate": np.nan,
                    "accuracy_full": np.nan,
                    "mixed_questions_clean": 0,
                }
            )
            continue
        data.append(
            {
                "think_length_bin": name,
                "rollouts": len(subset),
                "questions": len({str(row.get("question_id", "")) for row in subset}),
                "truncation_rate": float(sum(bool(row.get("truncated")) for row in subset) / len(subset)),
                "accuracy_full": float(sum(bool(row.get("is_correct")) for row in subset) / len(subset)),
                "mixed_questions_clean": mixed_question_count(subset, clean_only=True),
            }
        )
    return pd.DataFrame(data)


def write_report(out_dir: Path, summary: dict[str, Any], bins: pd.DataFrame, counters: dict[str, Counter]) -> Path:
    report = out_dir / "LONG_SMOKE_REPORT.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Long-CoT Thinking Smoke Report\n\n")
        f.write("Step 1 measures length distribution, truncation rate, and mixed-correctness availability.\n\n")
        f.write("## Summary\n\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"- {key}: {value:.4f}\n")
            else:
                f.write(f"- {key}: {value}\n")
        f.write("\n## Think Length Bins\n\n")
        f.write("| bin | rollouts | questions | truncation rate | accuracy | mixed clean questions |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for _, row in bins.iterrows():
            f.write(
                f"| {row['think_length_bin']} | {int(row['rollouts'])} | {int(row['questions'])} | "
                f"{row['truncation_rate']:.4f} | {row['accuracy_full']:.4f} | "
                f"{int(row['mixed_questions_clean'])} |\n"
            )
        f.write("\n## Counters\n\n")
        for name, counter in counters.items():
            f.write(f"### {name}\n\n")
            for key, value in counter.most_common(20):
                f.write(f"- {key}: {value}\n")
            f.write("\n")
    return report


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = Path(args.labeled_output) if args.labeled_output else out_dir / "rollouts_labeled.jsonl"

    tokenizer = load_tokenizer(args.model, args.hf_endpoint, args.hf_home, args.local_files_only)
    rows = [label_and_segment_row(row, tokenizer=tokenizer) for row in load_jsonl(input_path)]
    summary = summarize_rows(rows)
    bins = bin_summary(rows)
    counters = {
        "finish_reason": Counter(str(row.get("finish_reason")) for row in rows),
        "segment_status": Counter(str(row.get("segment_status")) for row in rows),
        "pred_answer": Counter(str(row.get("pred_answer")) for row in rows),
    }

    with labeled_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "long_smoke_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    bins.to_csv(out_dir / "long_smoke_length_bins.csv", index=False)
    report = write_report(out_dir, summary, bins, counters)

    print(f"input: {input_path}")
    print(f"labeled: {labeled_path}")
    print(f"summary: {out_dir / 'long_smoke_summary.json'}")
    print(f"bins: {out_dir / 'long_smoke_length_bins.csv'}")
    print(f"report: {report}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
