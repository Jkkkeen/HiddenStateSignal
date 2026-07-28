#!/usr/bin/env python3
"""Build a deterministic discovery manifest for recoverability experiments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_CANDIDATES = (
    ROOT / "Experiment" / "scripts",
    Path(__file__).resolve().parent,
)
for script_dir in SCRIPT_CANDIDATES:
    if (script_dir / "run_option_logit_trimmed_conclusion_qwen3vl.py").exists():
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        break

from run_option_logit_trimmed_conclusion_qwen3vl import trim_final_conclusion
from run_semantic_step_basin_qwen3vl import think_text_and_char_span


SELECTION_FEATURE = "trimmed_margin_mean"
FROZEN_FEATURES = (
    "trimmed_margin_mean",
    "trimmed_margin_max",
    "trimmed_entropy",
    "prompt_only_margin_mean",
    "prompt_only_margin_max",
    "think_final_margin_mean",
    "think_final_margin_max",
    "think_early_to_final_gain_mean",
    "think_early_to_final_gain_max",
    "think_late_margin_drop_mean",
    "think_late_margin_drop_max",
    "think_entropy_delta",
)

ANSWER_CLAIM_PATTERNS = (
    re.compile(
        r"(?i)\b(?:final\s+answer|the\s+answer|answer)\s*"
        r"(?:is|(?:should|would|must)\s+be|:)?\s*\**\s*([ABCD])\b"
    ),
    re.compile(r"(?i)\b(?:option|choice)\s+([ABCD])\s+(?:is\s+)?(?:the\s+)?(?:correct|answer)\b"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-rollouts", required=True)
    parser.add_argument("--trimmed-features", required=True)
    parser.add_argument("--trajectory-features", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--question-count", type=int, default=16)
    parser.add_argument("--errors-per-question", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_source_line", source_line)
            rows.append(row)
    return rows


def explicit_answer_claims(text: str) -> list[str]:
    claims: list[str] = []
    for pattern in ANSWER_CLAIM_PATTERNS:
        claims.extend(match.upper() for match in pattern.findall(str(text or "")))
    return claims


def evenly_spaced_rows(frame: pd.DataFrame, count: int, feature: str) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("count must be positive")
    ordered = frame.sort_values([feature, "rollout_id"], kind="stable").reset_index(drop=True)
    if len(ordered) < count:
        raise ValueError(f"need {count} rows, found {len(ordered)}")
    indices = np.rint(np.linspace(0, len(ordered) - 1, num=count)).astype(int)
    selected = ordered.iloc[indices].copy().reset_index(drop=True)
    selected["signal_rank"] = np.arange(len(selected), dtype=int)
    selected["signal_quantile"] = (
        selected["signal_rank"] / max(len(selected) - 1, 1)
    )
    return selected


def _merge_features(
    trimmed: pd.DataFrame,
    trajectory: pd.DataFrame | None,
) -> pd.DataFrame:
    keys = ["question_id", "rollout_id"]
    features = trimmed.copy()
    features["question_id"] = features["question_id"].astype(str)
    features["rollout_id"] = features["rollout_id"].astype(int)
    if trajectory is None or trajectory.empty:
        return features
    trajectory = trajectory.copy()
    trajectory["question_id"] = trajectory["question_id"].astype(str)
    trajectory["rollout_id"] = trajectory["rollout_id"].astype(int)
    added = [col for col in FROZEN_FEATURES if col in trajectory and col not in features]
    return features.merge(trajectory[keys + added], on=keys, how="left", validate="one_to_one")


def build_manifest(
    raw_path: Path,
    trimmed_features_path: Path,
    trajectory_features_path: Path | None,
    output_path: Path,
    summary_path: Path,
    question_count: int,
    errors_per_question: int,
    seed: int,
) -> dict[str, Any]:
    raw_rows = load_jsonl(raw_path)
    raw_by_key = {
        (str(row.get("question_id", "")), int(row.get("rollout_id", -1))): row
        for row in raw_rows
    }
    trimmed = pd.read_parquet(trimmed_features_path)
    trajectory = (
        pd.read_parquet(trajectory_features_path)
        if trajectory_features_path is not None and trajectory_features_path.exists()
        else None
    )
    features = _merge_features(trimmed, trajectory)
    candidates = features[features["is_correct"].astype(bool) == False].copy()  # noqa: E712
    candidates = candidates[candidates[SELECTION_FEATURE].notna()]

    audited_rows: list[dict[str, Any]] = []
    missing_raw: list[tuple[str, int]] = []
    for _, feature_row in candidates.iterrows():
        question_id = str(feature_row["question_id"])
        rollout_id = int(feature_row["rollout_id"])
        raw = raw_by_key.get((question_id, rollout_id))
        if raw is None:
            missing_raw.append((question_id, rollout_id))
            continue
        response = str(raw.get("response", ""))
        think_text, _, _, think_status = think_text_and_char_span(response)
        revision_prefix, trim_status = trim_final_conclusion(think_text)
        answer_claims = explicit_answer_claims(revision_prefix)
        answer = str(raw.get("answer", "")).strip().upper()
        prediction = str(raw.get("pred_answer", "")).strip().upper()
        audited = feature_row.to_dict()
        audited.update(
            {
                "_revision_prefix": revision_prefix,
                "_think_status": think_status,
                "_trim_status_recomputed": trim_status,
                "_answer_claims": answer_claims,
                "_gold_answer_claim_flag": answer in answer_claims,
                "_original_prediction_claim_flag": prediction in answer_claims,
            }
        )
        audited_rows.append(audited)
    if missing_raw:
        raise ValueError(f"missing raw rows for keys: {missing_raw[:5]}")
    audited_candidates = pd.DataFrame(audited_rows)
    clean_candidates = audited_candidates[
        audited_candidates["_gold_answer_claim_flag"].astype(bool) == False  # noqa: E712
    ].copy()

    eligible_questions = sorted(
        str(question_id)
        for question_id, group in clean_candidates.groupby("question_id")
        if len(group) >= errors_per_question
    )
    if len(eligible_questions) < question_count:
        raise ValueError(
            f"need {question_count} eligible questions, found {len(eligible_questions)}"
        )
    rng = np.random.default_rng(seed)
    chosen_questions = sorted(
        rng.choice(eligible_questions, size=question_count, replace=False).tolist()
    )

    records: list[dict[str, Any]] = []
    for question_id in chosen_questions:
        group = clean_candidates[clean_candidates["question_id"] == question_id]
        selected = evenly_spaced_rows(group, errors_per_question, SELECTION_FEATURE)
        for _, feature_row in selected.iterrows():
            rollout_id = int(feature_row["rollout_id"])
            key = (question_id, rollout_id)
            raw = raw_by_key.get(key)
            if raw is None:
                raise ValueError(f"raw row disappeared for key {key}")
            response = str(raw.get("response", ""))
            revision_prefix = str(feature_row["_revision_prefix"])
            answer_claims = list(feature_row["_answer_claims"])
            frozen = {
                feature: (
                    None if feature not in feature_row or pd.isna(feature_row[feature]) else float(feature_row[feature])
                )
                for feature in FROZEN_FEATURES
            }
            records.append(
                {
                    "experiment": "recoverability_v1",
                    "split_role": "discovery_smoke",
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "source_line": int(raw.get("_source_line", raw.get("source_line", -1))),
                    "answer": str(raw.get("answer", "")).strip().upper(),
                    "pred_answer": str(raw.get("pred_answer", "")).strip().upper(),
                    "is_correct": bool(raw.get("is_correct", False)),
                    "prompt": raw.get("prompt", ""),
                    "image_path": raw.get("image_path", ""),
                    "original_response": response,
                    "revision_prefix": revision_prefix,
                    "think_status": str(feature_row["_think_status"]),
                    "trim_status_recomputed": str(feature_row["_trim_status_recomputed"]),
                    "answer_claims_in_prefix": answer_claims,
                    "answer_leakage_flag": bool(feature_row["_gold_answer_claim_flag"]),
                    "original_prediction_claim_flag": bool(
                        feature_row["_original_prediction_claim_flag"]
                    ),
                    "signal_rank": int(feature_row["signal_rank"]),
                    "signal_quantile": float(feature_row["signal_quantile"]),
                    "selection_feature": SELECTION_FEATURE,
                    **frozen,
                }
            )

    records.sort(key=lambda row: (row["question_id"], row["signal_rank"], row["rollout_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "protocol_version": "recoverability_v1",
        "split_role": "discovery_smoke",
        "source_rollouts": str(raw_path),
        "source_trimmed_features": str(trimmed_features_path),
        "source_trajectory_features": str(trajectory_features_path or ""),
        "seed": int(seed),
        "selection_feature": SELECTION_FEATURE,
        "frozen_features": list(FROZEN_FEATURES),
        "wrong_candidates": int(len(audited_candidates)),
        "candidate_explicit_answer_claim_count": int(
            audited_candidates["_answer_claims"].map(bool).sum()
        ),
        "candidate_gold_answer_claim_count": int(
            audited_candidates["_gold_answer_claim_flag"].sum()
        ),
        "candidate_original_prediction_claim_count": int(
            audited_candidates["_original_prediction_claim_flag"].sum()
        ),
        "leakage_free_candidates": int(len(clean_candidates)),
        "eligible_questions": len(eligible_questions),
        "selected_questions": len({row["question_id"] for row in records}),
        "selected_rollouts": len(records),
        "errors_per_question": int(errors_per_question),
        "answer_leakage_count": int(sum(row["answer_leakage_flag"] for row in records)),
        "selected_original_prediction_claim_count": int(
            sum(row["original_prediction_claim_flag"] for row in records)
        ),
        "chosen_question_ids": chosen_questions,
        "untouched_holdout": False,
        "holdout_note": (
            "All smoke500 questions participated in prior feature discovery. "
            "A true holdout requires newly sampled questions."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_manifest(
        raw_path=Path(args.raw_rollouts),
        trimmed_features_path=Path(args.trimmed_features),
        trajectory_features_path=Path(args.trajectory_features) if args.trajectory_features else None,
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        question_count=args.question_count,
        errors_per_question=args.errors_per_question,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
