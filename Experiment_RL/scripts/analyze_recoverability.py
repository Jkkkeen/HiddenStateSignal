#!/usr/bin/env python3
"""Analyze whether frozen option-logit features predict revision recoverability."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


FEATURE_DIRECTIONS = {
    "trimmed_margin_mean": "pos",
    "trimmed_margin_max": "pos",
    "trimmed_relative_margin_mean": "pos",
    "trimmed_relative_margin_max": "pos",
    "prompt_only_margin_mean": "pos",
    "prompt_only_margin_max": "pos",
    "think_final_margin_mean": "pos",
    "think_final_margin_max": "pos",
    "think_early_to_final_gain_mean": "pos",
    "think_early_to_final_gain_max": "pos",
    "think_late_margin_drop_mean": "neg",
    "think_late_margin_drop_max": "neg",
    "trimmed_entropy": "neg",
    "think_entropy_delta": "neg",
    "exact_c2_gain_mean": "pos",
    "exact_c2_gain_max": "pos",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--run-summary", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def pairwise_concordance(feature: Sequence[float], outcome: Sequence[float]) -> float:
    x = np.asarray(feature, dtype=float)
    y = np.asarray(outcome, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    x = x[usable]
    y = y[usable]
    wins = 0.0
    pairs = 0
    for left in range(len(y)):
        for right in range(left + 1, len(y)):
            if y[left] == y[right]:
                continue
            high, low = (left, right) if y[left] > y[right] else (right, left)
            pairs += 1
            if x[high] > x[low]:
                wins += 1.0
            elif x[high] == x[low]:
                wins += 0.5
    return float(wins / pairs) if pairs else float("nan")


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    usable = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(usable) < 2 or usable["x"].nunique() < 2 or usable["y"].nunique() < 2:
        return float("nan")
    if method == "spearman":
        x_values = usable["x"].rank(method="average").tolist()
        y_values = usable["y"].rank(method="average").tolist()
    else:
        x_values = usable["x"].astype(float).tolist()
        y_values = usable["y"].astype(float).tolist()
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    numerator = sum(a * b for a, b in zip(x_centered, y_centered, strict=True))
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _bootstrap_mean(values: Sequence[float], n_boot: int, seed: int) -> tuple[float, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if clean.size == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(n_boot, clean.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def analyze_feature(
    scores: pd.DataFrame,
    feature: str,
    direction: str,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sign = -1.0 if direction == "neg" else 1.0
    question_rows: list[dict[str, Any]] = []
    top_values: list[float] = []
    bottom_values: list[float] = []
    for question_id, group in scores.groupby("question_id", sort=True):
        group = group[group[feature].notna()].copy()
        if group.empty:
            continue
        oriented = group[feature].astype(float) * sign
        recovery = group["recovery_rate"].astype(float)
        any_recovery = group["any_recovery"].astype(bool).astype(float)
        question_rows.append(
            {
                "question_id": str(question_id),
                "feature": feature,
                "n_rollouts": int(len(group)),
                "within_corr": _corr(oriented, recovery, "pearson"),
                "within_spearman": _corr(oriented, recovery, "spearman"),
                "within_pairwise_auc": pairwise_concordance(oriented, recovery),
                "within_any_recovery_auc": pairwise_concordance(oriented, any_recovery),
            }
        )
        top_values.append(float(recovery.loc[oriented.idxmax()]))
        bottom_values.append(float(recovery.loc[oriented.idxmin()]))

    per_question = pd.DataFrame(question_rows)
    pairwise = per_question["within_pairwise_auc"].dropna() if not per_question.empty else pd.Series(dtype=float)
    any_auc = per_question["within_any_recovery_auc"].dropna() if not per_question.empty else pd.Series(dtype=float)
    corr = per_question["within_corr"].dropna() if not per_question.empty else pd.Series(dtype=float)
    spearman = per_question["within_spearman"].dropna() if not per_question.empty else pd.Series(dtype=float)
    ci_low, ci_high = _bootstrap_mean(pairwise, n_boot, seed)
    top_mean = float(np.mean(top_values)) if top_values else float("nan")
    bottom_mean = float(np.mean(bottom_values)) if bottom_values else float("nan")
    result = {
        "feature": feature,
        "direction": direction,
        "questions": int(scores["question_id"].nunique()),
        "valid_pairwise_questions": int(len(pairwise)),
        "valid_corr_questions": int(len(corr)),
        "mean_within_pairwise_auc": float(pairwise.mean()) if len(pairwise) else float("nan"),
        "pairwise_auc_ci_low": ci_low,
        "pairwise_auc_ci_high": ci_high,
        "mean_within_any_recovery_auc": float(any_auc.mean()) if len(any_auc) else float("nan"),
        "mean_within_corr": float(corr.mean()) if len(corr) else float("nan"),
        "median_within_corr": float(corr.median()) if len(corr) else float("nan"),
        "mean_within_spearman": float(spearman.mean()) if len(spearman) else float("nan"),
        "top_recovery_mean": top_mean,
        "bottom_recovery_mean": bottom_mean,
        "top_bottom_recovery_delta": top_mean - bottom_mean,
        "top_bottom_recovery_ratio": (
            top_mean / bottom_mean if np.isfinite(bottom_mean) and bottom_mean > 0 else float("nan")
        ),
    }
    return result, per_question


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_report(
    path: Path,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    run_summary: dict[str, Any],
) -> None:
    primary = summary[summary["feature"] == "trimmed_margin_mean"]
    anchored = scores[scores.get("original_prediction_claim_flag", False).astype(bool)] if "original_prediction_claim_flag" in scores else pd.DataFrame()
    nonanchored = scores[~scores.get("original_prediction_claim_flag", False).astype(bool)] if "original_prediction_claim_flag" in scores else scores
    lines = [
        "# Recoverability Discovery Analysis",
        "",
        "This is a discovery smoke on questions previously used for option-logit feature discovery. It is not an untouched confirmation.",
        "",
        "## Generation Quality",
        "",
        f"- Prefixes: {len(scores)}",
        f"- Questions: {scores['question_id'].nunique()}",
        f"- Mean recovery rate: {_fmt(scores['recovery_rate'].mean())}",
        f"- Prefixes with any recovery: {int(scores['any_recovery'].astype(bool).sum())}",
        f"- Non-truncated invalid rate: {_fmt(run_summary.get('nontruncated_invalid_rate', run_summary.get('invalid_rate', float('nan'))))}",
        f"- Budget exhausted without explicit answer: {run_summary.get('budget_exhausted_without_answer', 'unknown')}",
        f"- Truncated generations: {run_summary.get('truncated_generations', 'unknown')}",
        f"- Anchored-error prefixes: {len(anchored)}; mean recovery: {_fmt(anchored['recovery_rate'].mean() if len(anchored) else float('nan'))}",
        f"- Non-anchored prefixes: {len(nonanchored)}; mean recovery: {_fmt(nonanchored['recovery_rate'].mean() if len(nonanchored) else float('nan'))}",
        "",
        "## Frozen Feature Results",
        "",
        "| feature | dir | valid q | pair AUC | 95% CI | mean corr | top recovery | bottom recovery | delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['feature']}` | {row['direction']} | {int(row['valid_pairwise_questions'])} | "
            f"{_fmt(row['mean_within_pairwise_auc'])} | "
            f"[{_fmt(row['pairwise_auc_ci_low'])}, {_fmt(row['pairwise_auc_ci_high'])}] | "
            f"{_fmt(row['mean_within_corr'])} | {_fmt(row['top_recovery_mean'])} | "
            f"{_fmt(row['bottom_recovery_mean'])} | {_fmt(row['top_bottom_recovery_delta'])} |"
        )
    lines.extend(["", "## Frozen Gate", ""])
    if primary.empty:
        lines.append("`trimmed_margin_mean` is missing; the primary gate cannot be evaluated.")
    else:
        row = primary.iloc[0]
        passed = bool(
            row["mean_within_pairwise_auc"] > 0.65
            and row["pairwise_auc_ci_low"] > 0.50
            and row["top_bottom_recovery_delta"] > 0
        )
        lines.append(f"- Discovery gate passed: `{str(passed).lower()}`")
        lines.append(f"- Pairwise AUC > 0.65: `{bool(row['mean_within_pairwise_auc'] > 0.65)}`")
        lines.append(f"- Bootstrap lower bound > 0.50: `{bool(row['pairwise_auc_ci_low'] > 0.50)}`")
        lines.append(f"- Top-bottom recovery delta > 0: `{bool(row['top_bottom_recovery_delta'] > 0)}`")
    lines.extend(
        [
            "",
            "Exact C2 gain is reported only if exact online probe fractions were recomputed. Nearby trajectory-grid gains are diagnostics, not substitutes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    scores["question_id"] = scores["question_id"].astype(str)
    for suffix in ("mean", "max"):
        trimmed = f"trimmed_margin_{suffix}"
        prompt = f"prompt_only_margin_{suffix}"
        relative = f"trimmed_relative_margin_{suffix}"
        if relative not in scores and trimmed in scores and prompt in scores:
            scores[relative] = scores[trimmed].astype(float) - scores[prompt].astype(float)
    run_summary = (
        json.loads(Path(args.run_summary).read_text(encoding="utf-8"))
        if args.run_summary and Path(args.run_summary).exists()
        else {}
    )

    summaries: list[dict[str, Any]] = []
    per_question_frames: list[pd.DataFrame] = []
    for feature, direction in FEATURE_DIRECTIONS.items():
        if feature not in scores or scores[feature].notna().sum() == 0:
            continue
        summary, per_question = analyze_feature(
            scores,
            feature=feature,
            direction=direction,
            n_boot=args.bootstrap,
            seed=args.seed + len(summaries) * 17,
        )
        summaries.append(summary)
        per_question_frames.append(per_question)
    summary_df = pd.DataFrame(summaries)
    per_question_df = (
        pd.concat(per_question_frames, ignore_index=True)
        if per_question_frames
        else pd.DataFrame()
    )
    summary_df.to_csv(output_dir / "recoverability_feature_summary.csv", index=False)
    per_question_df.to_csv(output_dir / "recoverability_per_question.csv", index=False)
    write_report(
        output_dir / "RECOVERABILITY_DISCOVERY_ANALYSIS.md",
        scores,
        summary_df,
        run_summary,
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
