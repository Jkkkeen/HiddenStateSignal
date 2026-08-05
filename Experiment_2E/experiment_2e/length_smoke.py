from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import sha256_file, write_json_atomic


def trajectory_point_count(token_count: int, *, window: int = 128, stride: int = 64) -> int:
    """Count regular window endpoints plus the final response endpoint."""
    if token_count <= 0:
        return 0
    if token_count < window:
        return 1
    regular = ((token_count - window) // stride) + 1
    last_regular_endpoint = window + (regular - 1) * stride
    return regular + int(last_regular_endpoint < token_count)


def analyze_length_records(
    frame: pd.DataFrame,
    *,
    window: int = 128,
    primary_stride: int = 64,
    fallback_stride: int = 32,
    minimum_median_points: int = 12,
    max_new_tokens: int = 1536,
) -> dict[str, Any]:
    required = {"question_id", "rollout_slot", "token_count", "finish_reason", "answer_reward"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing length-smoke columns: {missing}")
    work = frame.copy()
    finish = work["finish_reason"].fillna("").astype(str).str.lower()
    if "truncated" in work.columns:
        work["is_truncated"] = work["truncated"].astype(bool) | finish.eq("length")
    else:
        work["is_truncated"] = finish.eq("length")
    valid = work.loc[~work["is_truncated"]].copy()
    if valid.empty:
        raise ValueError("No non-truncated rollout is available for stride freezing")
    valid["points_stride64"] = valid["token_count"].map(
        lambda value: trajectory_point_count(int(value), window=window, stride=primary_stride)
    )
    valid["points_stride32"] = valid["token_count"].map(
        lambda value: trajectory_point_count(int(value), window=window, stride=fallback_stride)
    )
    median_points = float(valid["points_stride64"].median())
    frozen_stride = primary_stride if median_points >= minimum_median_points else fallback_stride

    per_question = work.groupby("question_id")["answer_reward"].agg(
        lambda values: bool((values > 0.5).any() and (values <= 0.5).any())
    )
    lengths = valid["token_count"].astype(float)
    return {
        "n_rollouts": int(len(work)),
        "n_questions": int(work["question_id"].nunique()),
        "n_non_truncated": int(len(valid)),
        "truncation_rate": float(work["is_truncated"].mean()),
        "boxed_format_rate": float(work["format_reward"].mean()) if "format_reward" in work else None,
        "answer_accuracy": float(work["answer_reward"].mean()),
        "eval_mixed_question_rate_n4": float(per_question.mean()),
        "token_count": {
            "median": float(lengths.median()),
            "p25": float(lengths.quantile(0.25)),
            "p75": float(lengths.quantile(0.75)),
            "p90": float(lengths.quantile(0.90)),
            "max": int(lengths.max()),
        },
        "median_trajectory_points_stride64": median_points,
        "median_trajectory_points_stride32": float(valid["points_stride32"].median()),
        "frozen_window": window,
        "frozen_stride": frozen_stride,
        "max_new_tokens": max_new_tokens,
        "stride_rule": (
            f"stride={primary_stride} iff non-truncated median trajectory points >= {minimum_median_points}; "
            f"otherwise stride={fallback_stride}"
        ),
        "truncation_warning": bool(work["is_truncated"].mean() > 0.05),
        "notes": [
            "The mixed rate here uses four evaluation rollouts per question and is diagnostic only.",
            "The >=20% hard-subset decision is made from group-size-8 GRPO smoke, not this n=4 diagnostic.",
        ],
    }


def load_generation_parts(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("parts/part_*.jsonl")) if path.is_dir() else [path]
    rows = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError(f"No generation records found under {path}")
    return pd.DataFrame(rows)


def freeze_length_smoke(input_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_generation_parts(input_path)
    summary = analyze_length_records(frame)
    input_files = sorted(input_path.glob("parts/part_*.jsonl")) if input_path.is_dir() else [input_path]
    summary["input_sha256"] = {str(file): sha256_file(file) for file in input_files}
    summary_path = output_dir / "length_smoke_summary.json"
    write_json_atomic(summary_path, summary)
    frozen = {
        "schema_version": 1,
        "decision_source": "length-only smoke; no hidden-state metric read",
        "window": summary["frozen_window"],
        "stride": summary["frozen_stride"],
        "representations": ["mean", "last"],
        "max_new_tokens": summary["max_new_tokens"],
        "primary_representation": f"mean_w{summary['frozen_window']}_s{summary['frozen_stride']}",
        "sensitivity_representation": f"last_s{summary['frozen_stride']}",
        "length_smoke_summary_sha256": sha256_file(summary_path),
    }
    frozen_path = output_dir / "frozen_analysis_spec.json"
    write_json_atomic(frozen_path, frozen)
    (output_dir / "frozen_analysis_spec.sha256").write_text(
        f"{sha256_file(frozen_path)}  {frozen_path.name}\n", encoding="ascii"
    )
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze length-only generations and freeze the trajectory stride.")
    parser.add_argument("--input", type=Path, required=True, help="Generation directory or JSONL file")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_length_smoke(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
