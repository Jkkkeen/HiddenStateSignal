#!/usr/bin/env python3
"""Compare letter-token and option-content predictors of recoverability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from analyze_recoverability import analyze_feature
    from recoverability_candidate_scoring import VALID_OPTIONS, choice_features
except ModuleNotFoundError:
    from scripts.analyze_recoverability import analyze_feature
    from scripts.recoverability_candidate_scoring import VALID_OPTIONS, choice_features


FEATURE_SPECS: list[dict[str, str]] = [
    {"name": "letter_gold_support", "column": "letter_gold_margin", "direction": "pos", "hypothesis": "higher gold support"},
    {"name": "letter_low_gold_margin", "column": "letter_gold_margin", "direction": "neg", "hypothesis": "lower gold margin is more editable"},
    {"name": "letter_gold_mean_support", "column": "letter_gold_margin_mean", "direction": "pos", "hypothesis": "higher gold support versus mean wrong option"},
    {"name": "letter_low_gold_mean_margin", "column": "letter_gold_margin_mean", "direction": "neg", "hypothesis": "lower mean gold margin is more editable"},
    {"name": "letter_commitment", "column": "letter_commitment_gap", "direction": "neg", "hypothesis": "weaker wrong-answer commitment"},
    {"name": "letter_top2_uncertainty", "column": "letter_top2_gap", "direction": "neg", "hypothesis": "smaller top-two gap"},
    {"name": "letter_entropy", "column": "letter_entropy", "direction": "pos", "hypothesis": "higher option entropy"},
    {"name": "content_gold_support", "column": "content_gold_margin", "direction": "pos", "hypothesis": "higher semantic gold support"},
    {"name": "content_low_gold_margin", "column": "content_gold_margin", "direction": "neg", "hypothesis": "lower semantic gold margin is more editable"},
    {"name": "content_gold_mean_support", "column": "content_gold_margin_mean", "direction": "pos", "hypothesis": "higher semantic gold support versus mean wrong option"},
    {"name": "content_low_gold_mean_margin", "column": "content_gold_margin_mean", "direction": "neg", "hypothesis": "lower semantic mean gold margin is more editable"},
    {"name": "content_commitment", "column": "content_commitment_gap", "direction": "neg", "hypothesis": "weaker semantic wrong-answer commitment"},
    {"name": "content_top2_uncertainty", "column": "content_top2_gap", "direction": "neg", "hypothesis": "smaller semantic top-two gap"},
    {"name": "content_entropy", "column": "content_entropy", "direction": "pos", "hypothesis": "higher semantic option entropy"},
    {"name": "calibrated_content_gold_support", "column": "content_calibrated_gold_margin", "direction": "pos", "hypothesis": "reasoning adds gold support"},
    {"name": "calibrated_content_low_gold_margin", "column": "content_calibrated_gold_margin", "direction": "neg", "hypothesis": "smaller calibrated gold margin is more editable"},
    {"name": "calibrated_content_gold_mean_support", "column": "content_calibrated_gold_margin_mean", "direction": "pos", "hypothesis": "reasoning adds gold support versus mean wrong option"},
    {"name": "calibrated_content_low_gold_mean_margin", "column": "content_calibrated_gold_margin_mean", "direction": "neg", "hypothesis": "smaller calibrated mean gold margin is more editable"},
    {"name": "calibrated_content_commitment", "column": "content_calibrated_commitment_gap", "direction": "neg", "hypothesis": "weaker calibrated wrong-answer commitment"},
    {"name": "calibrated_content_top2_uncertainty", "column": "content_calibrated_top2_gap", "direction": "neg", "hypothesis": "smaller calibrated top-two gap"},
    {"name": "calibrated_content_entropy", "column": "content_calibrated_entropy", "direction": "pos", "hypothesis": "higher calibrated option entropy"},
]

PAIRED_COMPARISONS: list[dict[str, str]] = [
    {"name": "content_low_mean_minus_letter_low_mean", "left": "content_low_gold_mean_margin", "right": "letter_low_gold_mean_margin"},
    {"name": "calibrated_entropy_minus_letter_top2", "left": "calibrated_content_entropy", "right": "letter_top2_uncertainty"},
    {"name": "calibrated_top2_minus_letter_top2", "left": "calibrated_content_top2_uncertainty", "right": "letter_top2_uncertainty"},
    {"name": "letter_top2_minus_letter_low_mean", "left": "letter_top2_uncertainty", "right": "letter_low_gold_mean_margin"},
    {"name": "calibrated_entropy_minus_content_low_mean", "left": "calibrated_content_entropy", "right": "content_low_gold_mean_margin"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-scores", required=True)
    parser.add_argument("--letter-probes", required=True)
    parser.add_argument("--content-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["question_id"] = normalized["question_id"].astype(str)
    normalized["rollout_id"] = normalized["rollout_id"].astype(int)
    return normalized


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def build_analysis_frame(
    recovery: pd.DataFrame,
    probes: pd.DataFrame,
    content: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["question_id", "rollout_id"]
    recovery = _normalize_keys(recovery)
    probes = _normalize_keys(probes)
    content = _normalize_keys(content)
    if recovery.duplicated(keys).any() or probes.duplicated(keys).any() or content.duplicated(keys).any():
        raise ValueError("all inputs must be unique by question_id/rollout_id")

    probe_columns = keys + [f"logit_{label}" for label in VALID_OPTIONS]
    content_columns = keys + [column for column in content if column.startswith("content_")]
    merged = recovery.merge(
        probes[probe_columns], on=keys, how="left", validate="one_to_one"
    ).merge(
        content[content_columns], on=keys, how="left", validate="one_to_one"
    )
    required = [f"logit_{label}" for label in VALID_OPTIONS] + [
        "content_option_labels",
        "content_gold_margin",
        "content_commitment_gap",
        "content_top2_gap",
        "content_entropy",
        "content_calibrated_gold_margin",
        "content_calibrated_commitment_gap",
        "content_calibrated_top2_gap",
        "content_calibrated_entropy",
    ]
    missing = [column for column in required if column not in merged or merged[column].isna().any()]
    if missing:
        raise ValueError(f"missing merged candidate scores: {missing}")

    selected_column = "original_prediction" if "original_prediction" in merged else "pred_answer"
    letter_rows: list[dict[str, float | str]] = []
    for _, row in merged.iterrows():
        scores = {label: float(row[f"logit_{label}"]) for label in VALID_OPTIONS}
        features = choice_features(
            scores,
            str(row["answer"]).strip().upper(),
            str(row[selected_column]).strip().upper(),
        )
        letter_rows.append(
            {
                "letter_gold_margin": float(features["gold_margin"]),
                "letter_gold_margin_mean": float(features["gold_margin_mean"]),
                "letter_commitment_gap": float(features["commitment_gap"]),
                "letter_top2_gap": float(features["top2_gap"]),
                "letter_entropy": float(features["entropy"]),
                "letter_top_option": str(features["top_option"]),
            }
        )
    letters = pd.DataFrame(letter_rows, index=merged.index)
    merged = pd.concat([merged, letters], axis=1)

    content_mean_rows: list[dict[str, float]] = []
    for _, row in merged.iterrows():
        labels = tuple(str(row["content_option_labels"]).strip())
        reasoning_scores = {
            label: float(row[f"content_score_{label}"]) for label in labels
        }
        calibrated_scores = {
            label: float(row[f"content_calibrated_score_{label}"]) for label in labels
        }
        gold = str(row["answer"]).strip().upper()
        selected = str(row[selected_column]).strip().upper()
        reasoning_features = choice_features(reasoning_scores, gold, selected)
        calibrated_features = choice_features(calibrated_scores, gold, selected)
        content_mean_rows.append(
            {
                "content_gold_margin_mean": float(reasoning_features["gold_margin_mean"]),
                "content_calibrated_gold_margin_mean": float(
                    calibrated_features["gold_margin_mean"]
                ),
            }
        )
    merged = pd.concat(
        [merged, pd.DataFrame(content_mean_rows, index=merged.index)], axis=1
    )
    merged["recovery_rate"] = merged["recovery_rate"].astype(float)
    merged["any_recovery"] = merged["any_recovery"].map(_normalize_bool)
    return merged


def analyze_candidate_features(
    frame: pd.DataFrame,
    specs: list[dict[str, str]] = FEATURE_SPECS,
    n_boot: int = 5000,
    seed: int = 20260711,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    per_question_frames: list[pd.DataFrame] = []
    for index, spec in enumerate(specs):
        column = spec["column"]
        if column not in frame:
            continue
        summary, per_question = analyze_feature(
            frame,
            feature=column,
            direction=spec["direction"],
            n_boot=n_boot,
            seed=seed + index * 17,
        )
        summary.update(
            {
                "feature": spec["name"],
                "source_column": column,
                "hypothesis": spec["hypothesis"],
            }
        )
        per_question = per_question.copy()
        per_question["feature"] = spec["name"]
        per_question["source_column"] = column
        per_question["direction"] = spec["direction"]
        summaries.append(summary)
        per_question_frames.append(per_question)
    summary_frame = pd.DataFrame(summaries)
    per_question_frame = (
        pd.concat(per_question_frames, ignore_index=True)
        if per_question_frames
        else pd.DataFrame()
    )
    return summary_frame, per_question_frame


def paired_feature_comparisons(
    per_question: pd.DataFrame,
    comparisons: list[dict[str, str]] = PAIRED_COMPARISONS,
    n_boot: int = 5000,
    seed: int = 20260711,
) -> pd.DataFrame:
    pivot = per_question.pivot(
        index="question_id", columns="feature", values="within_pairwise_auc"
    )
    rows: list[dict[str, Any]] = []
    for index, comparison in enumerate(comparisons):
        left = comparison["left"]
        right = comparison["right"]
        if left not in pivot or right not in pivot:
            continue
        paired = pivot[[left, right]].dropna()
        differences = (paired[left] - paired[right]).to_numpy(dtype=float)
        if differences.size and n_boot > 0:
            rng = np.random.default_rng(seed + index * 31)
            samples = rng.choice(
                differences,
                size=(n_boot, differences.size),
                replace=True,
            ).mean(axis=1)
            ci_low = float(np.quantile(samples, 0.025))
            ci_high = float(np.quantile(samples, 0.975))
        else:
            ci_low = float("nan")
            ci_high = float("nan")
        rows.append(
            {
                "comparison": comparison["name"],
                "left_feature": left,
                "right_feature": right,
                "valid_questions": int(differences.size),
                "mean_auc_delta": (
                    float(differences.mean()) if differences.size else float("nan")
                ),
                "auc_delta_ci_low": ci_low,
                "auc_delta_ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not np.isfinite(number) else f"{number:.4f}"


def write_report(
    path: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    ranked = summary.sort_values("mean_within_pairwise_auc", ascending=False)
    lines = [
        "# Recoverability Candidate-Scoring Discovery",
        "",
        "This reuses the existing 64 prefixes and strict revision outcomes. No new revisions were generated. All feature comparisons are exploratory and use question-level bootstrap intervals.",
        "",
        "## Data",
        "",
        f"- Prefixes: {len(frame)}",
        f"- Questions: {frame['question_id'].nunique()}",
        f"- Questions with recovery variation: {frame.groupby('question_id')['recovery_rate'].nunique().gt(1).sum()}",
        "",
        "## Within-Question Results",
        "",
        "| feature | dir | valid q | pair AUC | 95% CI | mean corr | top-bottom delta | hypothesis |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            f"| `{row['feature']}` | {row['direction']} | {int(row['valid_pairwise_questions'])} | "
            f"{_fmt(row['mean_within_pairwise_auc'])} | [{_fmt(row['pairwise_auc_ci_low'])}, "
            f"{_fmt(row['pairwise_auc_ci_high'])}] | {_fmt(row['mean_within_corr'])} | "
            f"{_fmt(row['top_bottom_recovery_delta'])} | {row['hypothesis']} |"
        )
    lines.extend(
        [
            "",
            "## Paired AUC Differences",
            "",
            "Positive values favor the left feature. Intervals resample the same valid questions.",
            "",
            "| comparison | valid q | mean delta | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in paired.iterrows():
        lines.append(
            f"| `{row['comparison']}` | {int(row['valid_questions'])} | "
            f"{_fmt(row['mean_auc_delta'])} | [{_fmt(row['auc_delta_ci_low'])}, "
            f"{_fmt(row['auc_delta_ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Positive and negative orientations of the same gold-margin column are complementary, not independent discoveries.",
            "- This is discovery data previously used for option-logit feature selection; an untouched holdout is required for confirmation.",
            "- With only a small number of questions showing within-question recovery variation, confidence intervals are the primary uncertainty summary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    recovery = pd.read_csv(args.recovery_scores)
    probes = pd.read_parquet(args.letter_probes)
    content = pd.read_csv(args.content_scores)
    frame = build_analysis_frame(recovery, probes, content)
    summary, per_question = analyze_candidate_features(
        frame,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    paired = paired_feature_comparisons(
        per_question,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "candidate_recoverability_rows.csv", index=False)
    summary.to_csv(output_dir / "candidate_feature_summary.csv", index=False)
    per_question.to_csv(output_dir / "candidate_per_question.csv", index=False)
    paired.to_csv(output_dir / "candidate_paired_comparisons.csv", index=False)
    write_report(
        output_dir / "RECOVERABILITY_CANDIDATE_SCORING.md",
        frame,
        summary,
        paired,
    )
    print(summary.sort_values("mean_within_pairwise_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
