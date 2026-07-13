#!/usr/bin/env python3
"""Aggregate RL03 atomic scores into probe, gain, AUC, and smoke-gate outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


GROUP_KEYS = [
    "question_id",
    "rollout_id",
    "is_correct",
    "gold_letter",
    "probe_id",
    "probe_kind",
    "frac",
    "interface_id",
    "representation",
    "prompt_mode",
]

ROLLOUT_KEYS = [
    "question_id",
    "rollout_id",
    "is_correct",
    "gold_letter",
    "interface_id",
    "representation",
    "prompt_mode",
]


def _ensure_prompt_mode(frame: pd.DataFrame) -> pd.DataFrame:
    if "prompt_mode" in frame.columns:
        return frame
    result = frame.copy()
    result["prompt_mode"] = "unspecified"
    return result


def _surface_column(frame: pd.DataFrame) -> pd.Series:
    if "surface_id" in frame:
        surface = frame["surface_id"].astype("string")
    else:
        surface = pd.Series(pd.NA, index=frame.index, dtype="string")
    return surface.fillna(frame["target_id"].astype("string"))


def _surface_gains(
    group: pd.DataFrame,
    start_kind: str,
    start_frac: float | None,
    end_kind: str,
    end_frac: float | None,
) -> list[float]:
    gold = group[group["is_gold_target"].astype(bool)].copy()
    gold["_surface"] = _surface_column(gold)

    def select(kind: str, frac: float | None) -> pd.DataFrame:
        selected = gold[gold["probe_kind"] == kind]
        if frac is not None:
            selected = selected[np.isclose(selected["frac"].astype(float), frac)]
        return selected[["_surface", "score_mean"]].drop_duplicates("_surface")

    start = select(start_kind, start_frac).rename(columns={"score_mean": "start"})
    end = select(end_kind, end_frac).rename(columns={"score_mean": "end"})
    paired = start.merge(end, on="_surface", how="inner").sort_values("_surface", kind="stable")
    return (paired["end"].astype(float) - paired["start"].astype(float)).tolist()


def _canonical_surface_gain(
    group: pd.DataFrame,
    start_kind: str,
    start_frac: float | None,
    end_kind: str,
    end_frac: float | None,
) -> float | None:
    gold = group[group["is_gold_target"].astype(bool)].copy()
    gold["_surface"] = _surface_column(gold)
    canonical = gold[gold["_surface"] == "gold_surface_0"]
    if canonical.empty:
        return None

    def value(kind: str, frac: float | None) -> float | None:
        selected = canonical[canonical["probe_kind"] == kind]
        if frac is not None:
            selected = selected[np.isclose(selected["frac"].astype(float), frac)]
        return float(selected.iloc[0]["score_mean"]) if len(selected) == 1 else None

    start = value(start_kind, start_frac)
    end = value(end_kind, end_frac)
    return None if start is None or end is None else end - start


def build_probe_features(scores: pd.DataFrame) -> pd.DataFrame:
    scores = _ensure_prompt_mode(scores)
    rows = []
    for key, group in scores.groupby(GROUP_KEYS, dropna=False, sort=False):
        row = dict(zip(GROUP_KEYS, key, strict=True))
        gold = group[group["is_gold_target"].astype(bool)]
        wrong = group[~group["is_gold_target"].astype(bool)]
        if gold.empty:
            continue
        gold_level = float(gold["score_mean"].median())
        row["gold_level"] = gold_level
        row["gold_target_count"] = int(len(gold))
        row["wrong_target_count"] = int(len(wrong))
        row["gold_margin"] = (
            gold_level - float(wrong["score_mean"].max()) if not wrong.empty else np.nan
        )
        row["target_margin"] = row["gold_margin"]
        row["content_margin"] = (
            row["gold_margin"] if str(row["representation"]) == "content_sequence" else np.nan
        )
        row["letter_margin"] = (
            row["gold_margin"] if str(row["representation"]).startswith("letter_") else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_rollout_features(scores: pd.DataFrame) -> pd.DataFrame:
    scores = _ensure_prompt_mode(scores)
    rows: list[dict[str, object]] = []
    for key, group in scores.groupby(ROLLOUT_KEYS, dropna=False, sort=False):
        row = dict(zip(ROLLOUT_KEYS, key, strict=True))
        probes = build_probe_features(group)
        for frac, suffix in (
            (0.0, "0"),
            (0.05, "5"),
            (0.25, "25"),
            (0.50, "50"),
            (0.90, "90"),
            (1.0, "100"),
        ):
            match = probes[
                probes["probe_kind"].isin(["prompt", "think"])
                & np.isclose(probes["frac"].astype(float), frac)
            ]
            if not match.empty:
                row[f"gold_level_{suffix}"] = float(match.iloc[0]["gold_level"])
                row[f"gold_margin_{suffix}"] = float(match.iloc[0]["gold_margin"])
        trimmed = probes[probes["probe_kind"] == "trimmed"]
        if not trimmed.empty:
            row["gold_level_trimmed"] = float(trimmed.iloc[0]["gold_level"])
            row["gold_margin_trimmed"] = float(trimmed.iloc[0]["gold_margin"])

        for name, end_kind, end_frac in (
            ("gold_gain_25_90", "think", 0.90),
            ("gold_gain_25_trimmed", "trimmed", None),
        ):
            gains = _surface_gains(group, "think", 0.25, end_kind, end_frac)
            if gains:
                row[name] = float(np.median(gains))
                row[f"{name}_mean_surface"] = float(np.mean(gains))
                row[f"{name}_minimum_surface"] = float(np.min(gains))
                row[f"{name}_surface_std"] = float(np.std(gains, ddof=0))
                canonical = _canonical_surface_gain(group, "think", 0.25, end_kind, end_frac)
                if canonical is not None:
                    row[f"{name}_canonical"] = float(canonical)
        if "gold_level_90" in row and "gold_level_0" in row:
            row["gold_relative_level_90"] = float(row["gold_level_90"] - row["gold_level_0"])
        if "gold_margin_90" in row and "gold_margin_25" in row:
            row["gold_margin_gain_25_90"] = float(row["gold_margin_90"] - row["gold_margin_25"])
            if str(row["representation"]) == "content_sequence":
                row["content_margin_gain_25_90"] = row["gold_margin_gain_25_90"]
            if str(row["representation"]).startswith("letter_"):
                row["letter_margin_gain_25_90"] = row["gold_margin_gain_25_90"]
        if (
            str(row["representation"]).startswith("letter_")
            and "gold_margin_5" in row
            and "gold_margin_100" in row
        ):
            row["letter_margin_gain_5_100"] = float(
                row["gold_margin_100"] - row["gold_margin_5"]
            )

        slopes: list[float] = []
        gold = group[group["is_gold_target"].astype(bool)].copy()
        gold = gold[gold["probe_kind"].isin(["prompt", "think"])]
        gold["frac"] = pd.to_numeric(gold["frac"], errors="coerce")
        gold = gold[np.isfinite(gold["frac"])]
        gold["_surface"] = _surface_column(gold)
        for _, surface in gold.groupby("_surface", sort=True):
            surface = surface.sort_values("frac").drop_duplicates("frac", keep="last")
            if len(surface) >= 2 and surface["frac"].nunique() >= 2:
                x_values = surface["frac"].astype(float).tolist()
                y_values = surface["score_mean"].astype(float).tolist()
                x_mean = sum(x_values) / len(x_values)
                y_mean = sum(y_values) / len(y_values)
                denominator = sum((value - x_mean) ** 2 for value in x_values)
                numerator = sum(
                    (x_value - x_mean) * (y_value - y_mean)
                    for x_value, y_value in zip(x_values, y_values, strict=True)
                )
                slopes.append(float(numerator / denominator))
        if slopes:
            row["gold_trajectory_slope"] = float(np.median(slopes))

        rollout_steps = build_step_features(group)
        if not rollout_steps.empty:
            gains = rollout_steps["step_gain"].astype(float)
            row["gold_positive_gain_rate"] = float((gains > 0).mean())
            row["gold_monotonicity_violation_rate"] = float((gains < 0).mean())
            row["gold_step_gain_vector"] = json.dumps(gains.tolist())
            row["gold_step_gain_rate_vector"] = json.dumps(
                rollout_steps["step_gain_rate"].astype(float).tolist()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_step_features(scores: pd.DataFrame) -> pd.DataFrame:
    scores = _ensure_prompt_mode(scores)
    if "score_mean" not in scores or "is_gold_target" not in scores:
        return _build_steps_from_probe_levels(scores)

    base = ROLLOUT_KEYS
    rows = []
    for key, group in scores.groupby(base, dropna=False, sort=False):
        probe_levels = build_probe_features(group)
        ordered = probe_levels[probe_levels["probe_kind"].isin(["prompt", "think"])].sort_values("frac")
        ordered = ordered.drop_duplicates("frac", keep="last")
        for (_, left), (_, right) in zip(ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows()):
            span = float(right["frac"] - left["frac"])
            if span <= 0:
                continue
            gains = _surface_gains(
                group,
                str(left["probe_kind"]),
                float(left["frac"]),
                str(right["probe_kind"]),
                float(right["frac"]),
            )
            if not gains:
                continue
            level_gain = float(np.median(gains))
            margin_gain = (
                float(right["gold_margin"] - left["gold_margin"])
                if pd.notna(right["gold_margin"]) and pd.notna(left["gold_margin"])
                else np.nan
            )
            rows.append(
                {
                    **dict(zip(base, key, strict=True)),
                    "frac_start": float(left["frac"]),
                    "frac_end": float(right["frac"]),
                    "frac_mid": float((left["frac"] + right["frac"]) / 2.0),
                    "probe_id_start": str(left["probe_id"]),
                    "probe_id_end": str(right["probe_id"]),
                    "step_gain": level_gain,
                    "step_gain_rate": level_gain / span,
                    "gold_step_gain": level_gain,
                    "gold_step_gain_rate": level_gain / span,
                    "level_gain": level_gain,
                    "level_gain_rate": level_gain / span,
                    "margin_gain": margin_gain,
                    "margin_gain_rate": margin_gain / span if pd.notna(margin_gain) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_steps_from_probe_levels(probes: pd.DataFrame) -> pd.DataFrame:
    base = ROLLOUT_KEYS
    rows = []
    for key, group in probes.groupby(base, dropna=False, sort=False):
        ordered = group[group["probe_kind"].isin(["prompt", "think"])].sort_values("frac")
        ordered = ordered.drop_duplicates("frac", keep="last")
        for (_, left), (_, right) in zip(ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows()):
            span = float(right["frac"] - left["frac"])
            if span <= 0:
                continue
            gain = float(right["gold_level"] - left["gold_level"])
            rows.append(
                {
                    **dict(zip(base, key, strict=True)),
                    "frac_start": float(left["frac"]),
                    "frac_end": float(right["frac"]),
                    "frac_mid": float((left["frac"] + right["frac"]) / 2.0),
                    "step_gain": gain,
                    "step_gain_rate": gain / span,
                    "level_gain": gain,
                    "level_gain_rate": gain / span,
                    "margin_gain": float("nan"),
                    "margin_gain_rate": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def question_equal_auc(frame: pd.DataFrame, feature: str) -> dict[str, object]:
    per_question: list[dict[str, object]] = []
    usable = frame.copy()
    usable[feature] = pd.to_numeric(usable[feature], errors="coerce")
    usable = usable[np.isfinite(usable[feature])]
    for question_id, group in usable.groupby("question_id", sort=True, dropna=False):
        positive = group.loc[group["is_correct"].astype(bool), feature].to_numpy(dtype=float)
        negative = group.loc[~group["is_correct"].astype(bool), feature].to_numpy(dtype=float)
        if positive.size == 0 or negative.size == 0:
            continue
        comparisons = positive[:, None] - negative[None, :]
        auc = float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)
        per_question.append(
            {
                "question_id": question_id,
                "auc": auc,
                "positive_rollouts": int(positive.size),
                "negative_rollouts": int(negative.size),
                "pairs": int(comparisons.size),
            }
        )
    per_question_frame = pd.DataFrame(
        per_question,
        columns=["question_id", "auc", "positive_rollouts", "negative_rollouts", "pairs"],
    )
    return {
        "mean_auc": (
            float(per_question_frame["auc"].mean())
            if not per_question_frame.empty
            else float("nan")
        ),
        "mixed_questions": int(len(per_question_frame)),
        "per_question": per_question_frame,
    }


def within_question_auc(frame: pd.DataFrame, feature: str) -> float:
    return float(question_equal_auc(frame, feature)["mean_auc"])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    usable = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(usable) < 2 or usable["x"].nunique() < 2 or usable["y"].nunique() < 2:
        return float("nan")
    x_ranks = usable["x"].rank(method="average").astype(float).tolist()
    y_ranks = usable["y"].rank(method="average").astype(float).tolist()
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    x_centered = [value - x_mean for value in x_ranks]
    y_centered = [value - y_mean for value in y_ranks]
    numerator = sum(
        left * right for left, right in zip(x_centered, y_centered, strict=True)
    )
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


def gain_level_diagnostics(
    frame: pd.DataFrame,
    early_feature: str,
    late_feature: str,
    gain_feature: str,
    near_zero_threshold: float = 1e-12,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    centered_parts: list[pd.DataFrame] = []
    for question_id, group in frame.groupby("question_id", sort=True, dropna=False):
        usable = group[[early_feature, late_feature, gain_feature]].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if usable.empty:
            continue
        early_variance = float(usable[early_feature].var(ddof=0))
        final_variance = float(usable[late_feature].var(ddof=0))
        ratio = early_variance / final_variance if final_variance > near_zero_threshold else float("nan")
        rank_correlation = _spearman(usable[late_feature], usable[gain_feature])
        rows.append(
            {
                "question_id": question_id,
                "early_variance": early_variance,
                "final_variance": final_variance,
                "early_final_variance_ratio": ratio,
                "late_gain_rank_correlation": rank_correlation,
            }
        )
        centered_parts.append(
            pd.DataFrame(
                {
                    "late": usable[late_feature] - usable[late_feature].mean(),
                    "gain": usable[gain_feature] - usable[gain_feature].mean(),
                }
            )
        )
    per_question = pd.DataFrame(rows)
    correlations = (
        per_question["late_gain_rank_correlation"].dropna()
        if not per_question.empty
        else pd.Series(dtype=float)
    )
    ratios = (
        per_question["early_final_variance_ratio"].dropna()
        if not per_question.empty
        else pd.Series(dtype=float)
    )
    centered = pd.concat(centered_parts, ignore_index=True) if centered_parts else pd.DataFrame()
    centered_correlation = (
        _spearman(centered["late"], centered["gain"])
        if not centered.empty
        else float("nan")
    )
    positive = int((correlations > 0).sum())
    negative = int((correlations < 0).sum())
    return {
        "valid_variance_questions": int(len(ratios)),
        "median_early_final_variance_ratio": float(ratios.median()) if len(ratios) else float("nan"),
        "mean_early_final_variance_ratio": float(ratios.mean()) if len(ratios) else float("nan"),
        "near_zero_early_variance_fraction": (
            float((per_question["early_variance"] <= near_zero_threshold).mean())
            if not per_question.empty
            else float("nan")
        ),
        "centered_late_gain_rank_correlation": centered_correlation,
        "mean_within_question_rank_correlation": (
            float(correlations.mean()) if len(correlations) else float("nan")
        ),
        "median_within_question_rank_correlation": (
            float(correlations.median()) if len(correlations) else float("nan")
        ),
        "positive_correlation_fraction": (
            float(positive / len(correlations)) if len(correlations) else float("nan")
        ),
        "positive_correlation_questions": positive,
        "negative_correlation_questions": negative,
        "zero_correlation_questions": int((correlations == 0).sum()),
    }


def _logistic_predictions(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    test_features: pd.DataFrame,
) -> np.ndarray:
    train = train_features.to_numpy(dtype=float)
    test = test_features.to_numpy(dtype=float)
    means = train.mean(axis=0)
    scales = train.std(axis=0, ddof=0)
    scales[scales <= 1e-12] = 1.0
    train = (train - means) / scales
    test = (test - means) / scales
    train_design = np.column_stack([np.ones(len(train)), train])
    test_design = np.column_stack([np.ones(len(test)), test])
    labels = train_labels.to_numpy(dtype=float)
    coefficients = np.zeros(train_design.shape[1], dtype=float)
    for _ in range(500):
        logits = np.clip(train_design @ coefficients, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = train_design.T @ (probabilities - labels) / len(train_design)
        gradient[1:] += 1e-4 * coefficients[1:]
        coefficients -= 0.2 * gradient
    test_logits = np.clip(test_design @ coefficients, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-test_logits))


def cross_fitted_level_gain_comparison(
    frame: pd.DataFrame,
    level_feature: str,
    gain_feature: str,
    folds: int = 5,
    seed: int = 20260713,
) -> dict[str, object]:
    columns = ["question_id", "is_correct", level_feature, gain_feature]
    usable = frame[columns].copy()
    usable[level_feature] = pd.to_numeric(usable[level_feature], errors="coerce")
    usable[gain_feature] = pd.to_numeric(usable[gain_feature], errors="coerce")
    usable = usable.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    questions = np.asarray(sorted(usable["question_id"].unique(), key=str), dtype=object)
    if len(questions) < 2:
        return {
            "metric": "held_out_question_equal_pairwise_auc",
            "folds": 0,
            "questions": int(len(questions)),
            "level_only_score": float("nan"),
            "level_gain_score": float("nan"),
            "improvement": float("nan"),
        }
    fold_count = min(max(int(folds), 2), len(questions))
    shuffled = np.random.default_rng(seed).permutation(questions)
    fold_by_question = {
        question_id: index % fold_count for index, question_id in enumerate(shuffled)
    }
    predictions = usable[["question_id", "is_correct"]].copy()
    predictions["level_only_prediction"] = np.nan
    predictions["level_gain_prediction"] = np.nan

    for fold_index in range(fold_count):
        test_mask = usable["question_id"].map(fold_by_question).eq(fold_index)
        train = usable[~test_mask]
        test = usable[test_mask]
        for feature_names, output_name in (
            ([level_feature], "level_only_prediction"),
            ([level_feature, gain_feature], "level_gain_prediction"),
        ):
            labels = train["is_correct"].astype(bool).astype(int)
            if labels.nunique() < 2:
                predicted = np.full(len(test), float(labels.mean()) if len(labels) else 0.5)
            else:
                predicted = _logistic_predictions(
                    train[feature_names],
                    labels,
                    test[feature_names],
                )
            predictions.loc[test_mask, output_name] = predicted

    level_score = float(question_equal_auc(predictions, "level_only_prediction")["mean_auc"])
    combined_score = float(question_equal_auc(predictions, "level_gain_prediction")["mean_auc"])
    return {
        "metric": "held_out_question_equal_pairwise_auc",
        "folds": int(fold_count),
        "questions": int(len(questions)),
        "level_only_score": level_score,
        "level_gain_score": combined_score,
        "improvement": combined_score - level_score,
    }


def _question_equal_table(
    question_values: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_index, (key, group) in enumerate(
        question_values.groupby(group_columns, sort=True, dropna=False)
    ):
        values = pd.to_numeric(group[value_column], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        keys = key if isinstance(key, tuple) else (key,)
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            bootstrap_samples=bootstrap_samples,
            seed=seed + group_index,
        )
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "estimate": float(values.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "questions": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def build_dense_tables(
    probes: pd.DataFrame,
    steps: pd.DataFrame,
    bootstrap_samples: int = 10_000,
    seed: int = 20260713,
) -> dict[str, pd.DataFrame]:
    cell_columns = [
        column
        for column in ("interface_id", "representation", "prompt_mode")
        if column in probes.columns
    ]
    numeric_probes = probes[probes["probe_kind"].isin(["prompt", "think"])].copy()
    numeric_probes["frac"] = pd.to_numeric(numeric_probes["frac"], errors="coerce")
    numeric_probes = numeric_probes[np.isfinite(numeric_probes["frac"])]
    center_keys = ["question_id", *cell_columns, "frac"]
    numeric_probes["centered_gold_level"] = numeric_probes["gold_level"] - numeric_probes.groupby(
        center_keys, dropna=False
    )["gold_level"].transform("mean")

    t1_parts: list[pd.DataFrame] = []
    for panel_index, (panel, value_column) in enumerate(
        (("raw", "gold_level"), ("within_question_centered", "centered_gold_level"))
    ):
        per_question = (
            numeric_probes.groupby(
                ["question_id", *cell_columns, "frac", "is_correct"],
                as_index=False,
                dropna=False,
            )[value_column]
            .mean()
        )
        table = _question_equal_table(
            per_question,
            [*cell_columns, "frac", "is_correct"],
            value_column,
            bootstrap_samples,
            seed + panel_index * 10_000,
        )
        table.insert(0, "panel", panel)
        t1_parts.append(table)
    t1 = pd.concat(t1_parts, ignore_index=True) if t1_parts else pd.DataFrame()

    step_cell_columns = [
        column
        for column in ("interface_id", "representation", "prompt_mode")
        if column in steps.columns
    ]
    interval_columns = ["frac_start", "frac_end", "frac_mid"]
    per_question_rate = (
        steps.groupby(
            ["question_id", *step_cell_columns, *interval_columns, "is_correct"],
            as_index=False,
            dropna=False,
        )["step_gain_rate"]
        .mean()
    )
    t2 = _question_equal_table(
        per_question_rate,
        [*step_cell_columns, *interval_columns, "is_correct"],
        "step_gain_rate",
        bootstrap_samples,
        seed + 20_000,
    )

    per_question_gain = (
        steps.groupby(
            ["question_id", *step_cell_columns, *interval_columns, "is_correct"],
            as_index=False,
            dropna=False,
        )["step_gain"]
        .mean()
    )
    gain_pivot = per_question_gain.pivot_table(
        index=["question_id", *step_cell_columns, *interval_columns],
        columns="is_correct",
        values="step_gain",
        aggfunc="mean",
    ).reset_index()
    if False in gain_pivot and True in gain_pivot:
        gain_pivot["correct_minus_incorrect_step_gain"] = gain_pivot[True] - gain_pivot[False]
        t3 = _question_equal_table(
            gain_pivot,
            [*step_cell_columns, *interval_columns],
            "correct_minus_incorrect_step_gain",
            bootstrap_samples,
            seed + 30_000,
        )
    else:
        t3 = pd.DataFrame()

    t4_rows: list[dict[str, object]] = []
    t4_group_columns = [*step_cell_columns, *interval_columns]
    for group_index, (key, group) in enumerate(
        steps.groupby(t4_group_columns, sort=True, dropna=False)
    ):
        result = question_equal_auc(group, "step_gain")
        per_question_auc = result["per_question"]
        keys = key if isinstance(key, tuple) else (key,)
        ci_low, ci_high = bootstrap_mean_ci(
            per_question_auc["auc"] if not per_question_auc.empty else [],
            bootstrap_samples,
            seed + 40_000 + group_index,
        )
        t4_rows.append(
            {
                **dict(zip(t4_group_columns, keys, strict=True)),
                "estimate": result["mean_auc"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "questions": result["mixed_questions"],
            }
        )
    t4 = pd.DataFrame(t4_rows)
    return {"T1": t1, "T2": t2, "T3": t3, "T4": t4}


def bootstrap_mean_ci(
    values: list[float] | np.ndarray | pd.Series,
    bootstrap_samples: int = 10_000,
    seed: int = 20260713,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0 or bootstrap_samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(int(bootstrap_samples), clean.size), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_auc_bootstrap(
    frame: pd.DataFrame,
    gain_feature: str,
    level_feature: str,
    bootstrap_samples: int = 10_000,
    seed: int = 20260713,
) -> dict[str, object]:
    gain = question_equal_auc(frame, gain_feature)["per_question"]
    level = question_equal_auc(frame, level_feature)["per_question"]
    paired = gain[["question_id", "auc"]].merge(
        level[["question_id", "auc"]],
        on="question_id",
        suffixes=("_gain", "_level"),
        how="inner",
    )
    paired["delta_auc"] = paired["auc_gain"] - paired["auc_level"]
    ci_low, ci_high = bootstrap_mean_ci(
        paired["delta_auc"],
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "valid_questions": int(len(paired)),
        "mean_gain_auc": float(paired["auc_gain"].mean()) if len(paired) else float("nan"),
        "mean_level_auc": float(paired["auc_level"].mean()) if len(paired) else float("nan"),
        "mean_delta_auc": float(paired["delta_auc"].mean()) if len(paired) else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _finite_metric(metrics: dict[str, object], name: str) -> float | None:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def assign_stage_a_gate(metrics: dict[str, object]) -> dict[str, object]:
    if metrics.get("pipeline_pass") is False:
        return {
            "label": "PIPELINE-FAIL",
            "provisional": False,
            "passed": False,
            "missing_controls": [],
        }

    i0_auc = _finite_metric(metrics, "i0_letter_gain_auc")
    if i0_auc is not None and i0_auc < 0.70:
        return {
            "label": "PIPELINE-FAIL",
            "provisional": False,
            "passed": False,
            "missing_controls": [],
        }

    gain_auc = _finite_metric(metrics, "gain_auc")
    gain_ci_low = _finite_metric(metrics, "gain_auc_ci_low")
    gain_ci_high = _finite_metric(metrics, "gain_auc_ci_high")
    level_auc = _finite_metric(metrics, "level_auc")
    delta_auc = _finite_metric(metrics, "paired_delta_auc")
    delta_ci_low = _finite_metric(metrics, "paired_delta_ci_low")
    delta_ci_high = _finite_metric(metrics, "paired_delta_ci_high")
    crossfit = _finite_metric(metrics, "crossfit_improvement")

    interface_auc = _finite_metric(metrics, "legacy_interface_letter_gain_auc")
    i0_interface_auc = _finite_metric(metrics, "i0_interface_letter_gain_auc")
    if i0_interface_auc is None:
        i0_interface_auc = i0_auc
    neutral_auc = _finite_metric(metrics, "neutral_letter_gain_auc")
    if (
        i0_interface_auc is not None
        and interface_auc is not None
        and i0_interface_auc >= 0.70
        and interface_auc < 0.65
    ):
        return {"label": "INTERFACE-FAIL", "provisional": False, "passed": False, "missing_controls": []}

    content_auc = _finite_metric(metrics, "content_margin_gain_auc")
    matched_representation_fail = metrics.get("representation_fail")
    if matched_representation_fail is True:
        return {"label": "REPRESENTATION-FAIL", "provisional": False, "passed": False, "missing_controls": []}
    if (
        matched_representation_fail is None
        and neutral_auc is not None
        and content_auc is not None
        and neutral_auc >= 0.70
        and content_auc < 0.60
    ):
        return {"label": "REPRESENTATION-FAIL", "provisional": False, "passed": False, "missing_controls": []}

    delta_includes_zero = (
        delta_ci_low is not None
        and delta_ci_high is not None
        and delta_ci_low <= 0.0 <= delta_ci_high
    )
    if gain_auc is not None and gain_auc >= 0.70 and (
        delta_includes_zero or (crossfit is not None and crossfit <= 0.0)
    ):
        return {"label": "LEVEL-EQUIVALENT", "provisional": False, "passed": False, "missing_controls": []}

    control_names = (
        "positive_correlation_fraction",
        "i1_i2_auc_difference",
        "trimmed_auc",
        "label_permutation_auc_change",
        "paired_delta_auc",
        "paired_delta_ci_low",
        "crossfit_improvement",
    )
    missing_controls = [name for name in control_names if _finite_metric(metrics, name) is None]
    if metrics.get("pipeline_pass") is None:
        missing_controls.insert(0, "pipeline_completion")
    pass_core = gain_auc is not None and gain_auc >= 0.70 and gain_ci_low is not None and gain_ci_low > 0.60
    if pass_core and missing_controls:
        return {
            "label": "PASS-PROCESS",
            "provisional": True,
            "passed": False,
            "missing_controls": missing_controls,
        }
    if pass_core and not missing_controls:
        passed = bool(
            float(metrics["positive_correlation_fraction"]) >= 0.65
            and abs(float(metrics["i1_i2_auc_difference"])) <= 0.03
            and float(metrics["trimmed_auc"]) >= 0.65
            and abs(float(metrics["label_permutation_auc_change"])) <= 0.02
            and delta_auc is not None
            and delta_auc > 0.0
            and delta_ci_low is not None
            and delta_ci_low > 0.0
            and crossfit is not None
            and crossfit > 0.0
        )
        if passed:
            return {"label": "PASS-PROCESS", "provisional": False, "passed": True, "missing_controls": []}

    contrastive_auc = _finite_metric(metrics, "contrastive_gain_auc")
    if contrastive_auc is not None and contrastive_auc >= 0.70 and (gain_auc is None or gain_auc < 0.65):
        return {"label": "CONTRASTIVE-ONLY", "provisional": False, "passed": False, "missing_controls": []}

    if (
        gain_auc is not None
        and 0.65 <= gain_auc < 0.70
        or gain_ci_low is not None
        and 0.50 < gain_ci_low <= 0.60
    ):
        return {"label": "BORDERLINE-PROCESS", "provisional": False, "passed": False, "missing_controls": []}

    gain_ci_includes_chance = (
        gain_ci_low is not None
        and gain_ci_high is not None
        and gain_ci_low <= 0.50 <= gain_ci_high
    )
    if level_auc is not None and level_auc >= 0.75 and (
        gain_auc is None or gain_auc < 0.65 or gain_ci_includes_chance
    ):
        return {"label": "VERIFIER-ONLY", "provisional": False, "passed": False, "missing_controls": []}

    return {
        "label": "NO-SIGNAL",
        "provisional": bool(metrics.get("pipeline_pass") is None),
        "passed": False,
        "missing_controls": missing_controls,
    }


def smoke_gate(scores: pd.DataFrame, steps: pd.DataFrame, expected_requests: int) -> dict[str, object]:
    completion = len(scores) / expected_requests if expected_requests else 0.0
    finite = float(np.isfinite(scores["score_mean"].astype(float)).mean()) if len(scores) else 0.0
    variation = False
    varying_groups = 0
    variance_groups = 0
    if not steps.empty:
        variance_keys = [
            column
            for column in (
                "question_id",
                "interface_id",
                "representation",
                "prompt_mode",
                "frac_start",
                "frac_end",
            )
            if column in steps.columns
        ]
        per_cell_interval = steps.groupby(variance_keys, dropna=False)[
            "level_gain"
        ].nunique(dropna=True)
        variance_groups = int(len(per_cell_interval))
        varying_groups = int((per_cell_interval > 1).sum())
        variation = varying_groups > 0
    passed = completion >= 0.99 and finite >= 0.99 and variation
    return {
        "expected_requests": int(expected_requests),
        "completed_requests": int(len(scores)),
        "completion_rate": float(completion),
        "finite_score_rate": finite,
        "nonzero_within_question_gain_variance": variation,
        "varying_cell_interval_groups": varying_groups,
        "cell_interval_groups": variance_groups,
        "varying_cell_interval_fraction": (
            float(varying_groups / variance_groups) if variance_groups else 0.0
        ),
        "passed": bool(passed),
    }


def _sign_test_pvalue(positive: int, negative: int) -> float:
    trials = int(positive + negative)
    if trials == 0:
        return float("nan")
    tail = min(int(positive), int(negative))
    probability = sum(math.comb(trials, count) for count in range(tail + 1)) / (2**trials)
    return float(min(1.0, 2.0 * probability))


def summarize_feature(
    frame: pd.DataFrame,
    feature: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    auc_result = question_equal_auc(frame, feature)
    per_question_auc = auc_result["per_question"]
    ci_low, ci_high = bootstrap_mean_ci(
        per_question_auc["auc"] if not per_question_auc.empty else [],
        bootstrap_samples,
        seed,
    )
    correlations: list[float] = []
    for _, group in frame.groupby("question_id", sort=True, dropna=False):
        correlation = _spearman(
            pd.to_numeric(group[feature], errors="coerce"),
            group["is_correct"].astype(bool).astype(float),
        )
        if np.isfinite(correlation):
            correlations.append(correlation)
    positive = sum(value > 0 for value in correlations)
    negative = sum(value < 0 for value in correlations)
    return {
        "feature": feature,
        "rows": int(pd.to_numeric(frame[feature], errors="coerce").notna().sum()),
        "mixed_questions": int(auc_result["mixed_questions"]),
        "mean_auc": float(auc_result["mean_auc"]),
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "mean_within_question_correlation": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
        "median_within_question_correlation": (
            float(np.median(correlations)) if correlations else float("nan")
        ),
        "positive_correlation_fraction": (
            float(positive / len(correlations)) if correlations else float("nan")
        ),
        "positive_correlation_questions": int(positive),
        "negative_correlation_questions": int(negative),
        "sign_test_pvalue": _sign_test_pvalue(positive, negative),
    }


def _cell_records(
    rollouts: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    feature_names = (
        "gold_level_25",
        "gold_level_90",
        "gold_level_trimmed",
        "gold_relative_level_90",
        "gold_gain_25_90",
        "gold_gain_25_trimmed",
        "gold_trajectory_slope",
        "gold_positive_gain_rate",
        "gold_monotonicity_violation_rate",
        "gold_margin_90",
        "gold_margin_gain_25_90",
        "content_margin_gain_25_90",
        "letter_margin_gain_25_90",
        "letter_margin_gain_5_100",
    )
    metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    crossfit_rows: list[dict[str, object]] = []
    for cell_index, (key, group) in enumerate(
        rollouts.groupby(
            ["interface_id", "representation", "prompt_mode"],
            sort=True,
            dropna=False,
        )
    ):
        interface_id, representation, prompt_mode = key
        cell = {
            "interface_id": interface_id,
            "representation": representation,
            "prompt_mode": prompt_mode,
        }
        for feature_index, feature in enumerate(feature_names):
            if feature not in group or pd.to_numeric(group[feature], errors="coerce").notna().sum() == 0:
                continue
            metric_rows.append(
                {
                    **cell,
                    **summarize_feature(
                        group,
                        feature,
                        bootstrap_samples,
                        seed + cell_index * 1000 + feature_index,
                    ),
                }
            )
        matched = ["gold_level_25", "gold_level_90", "gold_gain_25_90"]
        if all(column in group for column in matched):
            diagnostic_rows.append(
                {
                    **cell,
                    **gain_level_diagnostics(
                        group,
                        "gold_level_25",
                        "gold_level_90",
                        "gold_gain_25_90",
                    ),
                }
            )
            paired = paired_auc_bootstrap(
                group,
                "gold_gain_25_90",
                "gold_level_90",
                bootstrap_samples,
                seed + 50_000 + cell_index,
            )
            paired_rows.append({**cell, **paired})
            crossfit = cross_fitted_level_gain_comparison(
                group,
                "gold_level_90",
                "gold_gain_25_90",
                seed=seed + 60_000 + cell_index,
            )
            crossfit_rows.append({**cell, **crossfit})
    return metric_rows, diagnostic_rows, paired_rows, crossfit_rows


def _matching_record(
    records: list[dict[str, object]],
    interface_id: object,
    representation: object,
    feature: str | None = None,
    prompt_mode: object | None = None,
) -> dict[str, object] | None:
    for record in records:
        if record.get("interface_id") != interface_id or record.get("representation") != representation:
            continue
        if prompt_mode is not None and record.get("prompt_mode") != prompt_mode:
            continue
        if feature is None or record.get("feature") == feature:
            return record
    return None


def _best_metric(
    records: list[dict[str, object]],
    feature: str,
    representations: tuple[str, ...],
    interfaces: tuple[str, ...] = ("I0", "I1", "I2"),
    prompt_modes: tuple[str, ...] | None = None,
) -> dict[str, object] | None:
    candidates = [
        record
        for record in records
        if record.get("feature") == feature
        and record.get("representation") in representations
        and record.get("interface_id") in interfaces
        and (prompt_modes is None or record.get("prompt_mode") in prompt_modes)
        and _finite_metric(record, "mean_auc") is not None
    ]
    return max(candidates, key=lambda record: float(record["mean_auc"])) if candidates else None


def _gate_metrics(
    pipeline: dict[str, object],
    metrics: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    paired: list[dict[str, object]],
    crossfit: list[dict[str, object]],
) -> dict[str, object]:
    primary = _best_metric(
        metrics,
        "gold_gain_25_90",
        ("content_sequence",),
        ("I1", "I2"),
        ("neutralized_letter_instruction", "unspecified"),
    )
    if primary is None:
        primary = _best_metric(
            metrics,
            "gold_gain_25_90",
            ("letter_first", "letter_sequence", "content_sequence"),
        )
    pipeline_pass: bool | None
    if not bool(pipeline["passed"]):
        pipeline_pass = False
    elif bool(pipeline.get("completion_known", False)):
        pipeline_pass = True
    else:
        pipeline_pass = None
    gate_metrics: dict[str, object] = {"pipeline_pass": pipeline_pass}
    if primary is not None:
        gate_metrics.update(
            {
                "gain_auc": primary["mean_auc"],
                "gain_auc_ci_low": primary["auc_ci_low"],
                "gain_auc_ci_high": primary["auc_ci_high"],
                "positive_correlation_fraction": primary["positive_correlation_fraction"],
            }
        )
        interface_id = primary["interface_id"]
        representation = primary["representation"]
        prompt_mode = primary.get("prompt_mode")
        level = _matching_record(
            metrics, interface_id, representation, "gold_level_90", prompt_mode
        )
        trimmed = _matching_record(
            metrics, interface_id, representation, "gold_gain_25_trimmed", prompt_mode
        )
        paired_record = _matching_record(
            paired, interface_id, representation, prompt_mode=prompt_mode
        )
        crossfit_record = _matching_record(
            crossfit, interface_id, representation, prompt_mode=prompt_mode
        )
        if level:
            gate_metrics["level_auc"] = level["mean_auc"]
        if trimmed:
            gate_metrics["trimmed_auc"] = trimmed["mean_auc"]
        if paired_record:
            gate_metrics.update(
                {
                    "paired_delta_auc": paired_record["mean_delta_auc"],
                    "paired_delta_ci_low": paired_record["ci_low"],
                    "paired_delta_ci_high": paired_record["ci_high"],
                }
            )
        if crossfit_record:
            gate_metrics["crossfit_improvement"] = crossfit_record["improvement"]

    i0 = _best_metric(
        metrics,
        "letter_margin_gain_5_100",
        ("letter_first",),
        ("I0",),
        ("legacy_letter_instruction", "unspecified"),
    )
    neutral = _best_metric(
        metrics,
        "letter_margin_gain_25_90",
        ("letter_first", "letter_sequence"),
        ("I1", "I2"),
        ("neutralized_letter_instruction", "unspecified"),
    )
    legacy_interface = _best_metric(
        metrics,
        "letter_margin_gain_25_90",
        ("letter_first",),
        ("I1", "I2"),
        ("legacy_letter_instruction",),
    )
    i0_interface = _best_metric(
        metrics,
        "letter_margin_gain_25_90",
        ("letter_first",),
        ("I0",),
        ("legacy_letter_instruction", "unspecified"),
    )
    content = _best_metric(
        metrics,
        "gold_gain_25_90",
        ("content_sequence",),
        ("I1", "I2"),
        ("neutralized_letter_instruction", "unspecified"),
    )
    content_margin = _best_metric(
        metrics,
        "content_margin_gain_25_90",
        ("content_sequence",),
        ("I1", "I2"),
        ("neutralized_letter_instruction", "unspecified"),
    )
    contrastive = _best_metric(
        metrics,
        "gold_margin_gain_25_90",
        ("content_sequence",),
        ("I1", "I2"),
        ("neutralized_letter_instruction", "unspecified"),
    )
    if i0:
        gate_metrics["i0_letter_gain_auc"] = i0["mean_auc"]
    if neutral:
        gate_metrics["neutral_letter_gain_auc"] = neutral["mean_auc"]
    if legacy_interface:
        gate_metrics["legacy_interface_letter_gain_auc"] = legacy_interface["mean_auc"]
    if i0_interface:
        gate_metrics["i0_interface_letter_gain_auc"] = i0_interface["mean_auc"]
    if content:
        gate_metrics["gold_content_gain_auc"] = content["mean_auc"]
    if content_margin:
        gate_metrics["content_margin_gain_auc"] = content_margin["mean_auc"]
    representation_comparisons: list[dict[str, object]] = []
    for interface_id in ("I1", "I2"):
        letter_record = _matching_record(
            metrics,
            interface_id,
            "letter_first",
            "letter_margin_gain_25_90",
            "neutralized_letter_instruction",
        ) or _matching_record(
            metrics,
            interface_id,
            "letter_first",
            "letter_margin_gain_25_90",
            "unspecified",
        )
        content_record = _matching_record(
            metrics,
            interface_id,
            "content_sequence",
            "content_margin_gain_25_90",
            "neutralized_letter_instruction",
        ) or _matching_record(
            metrics,
            interface_id,
            "content_sequence",
            "content_margin_gain_25_90",
            "unspecified",
        )
        if letter_record is None or content_record is None:
            continue
        letter_auc = _finite_metric(letter_record, "mean_auc")
        content_auc = _finite_metric(content_record, "mean_auc")
        if letter_auc is None or content_auc is None:
            continue
        representation_comparisons.append(
            {
                "interface_id": interface_id,
                "letter_first_auc": letter_auc,
                "content_margin_auc": content_auc,
            }
        )
    if representation_comparisons:
        gate_metrics["representation_comparisons"] = representation_comparisons
        gate_metrics["representation_fail"] = any(
            float(row["letter_first_auc"]) >= 0.70
            and float(row["content_margin_auc"]) < 0.60
            for row in representation_comparisons
        )
    if contrastive:
        gate_metrics["contrastive_gain_auc"] = contrastive["mean_auc"]
    i1 = _matching_record(
        metrics,
        "I1",
        "content_sequence",
        "gold_gain_25_90",
        "neutralized_letter_instruction",
    ) or _matching_record(metrics, "I1", "content_sequence", "gold_gain_25_90", "unspecified")
    i2 = _matching_record(
        metrics,
        "I2",
        "content_sequence",
        "gold_gain_25_90",
        "neutralized_letter_instruction",
    ) or _matching_record(metrics, "I2", "content_sequence", "gold_gain_25_90", "unspecified")
    if i1 and i2:
        gate_metrics["i1_i2_auc_difference"] = abs(float(i1["mean_auc"]) - float(i2["mean_auc"]))
    return gate_metrics


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _plot_dense_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> list[str]:
    if "torch" in sys.modules:
        code = """
import json
import sys
from pathlib import Path
import pandas as pd
from scripts.analyze_rl03_mcq_audit import _plot_dense_tables

output_dir = Path(sys.argv[1])
tables = {
    "T1": pd.read_csv(output_dir / "T1_level_trajectory.csv"),
    "T2": pd.read_csv(output_dir / "T2_adjacent_gain_trajectory.csv"),
    "T3": pd.read_csv(output_dir / "T3_gain_separation.csv"),
    "T4": pd.read_csv(output_dir / "T4_interval_discrimination.csv"),
}
print(json.dumps(_plot_dense_tables(tables, output_dir)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, str(output_dir)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return []
        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        return json.loads(output_lines[-1]) if output_lines else []

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: list[str] = []
    colors = {False: "#D55E00", True: "#0072B2"}

    t1 = tables["T1"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for axis, panel in zip(axes, ("raw", "within_question_centered"), strict=True):
        panel_rows = t1[t1["panel"] == panel]
        for key, group in panel_rows.groupby(
            ["interface_id", "representation", "prompt_mode", "is_correct"]
        ):
            interface_id, representation, prompt_mode, is_correct = key
            group = group.sort_values("frac")
            label = (
                f"{interface_id} {representation} {prompt_mode} "
                f"{'correct' if is_correct else 'incorrect'}"
            )
            axis.plot(group["frac"], group["estimate"], marker="o", label=label, color=colors[bool(is_correct)])
            axis.fill_between(group["frac"], group["ci_low"], group["ci_high"], alpha=0.15, color=colors[bool(is_correct)])
        axis.set_title("Raw" if panel == "raw" else "Within-question centered")
        axis.set_xlabel("Thinking progress")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Gold level")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    name = "T1_level_trajectory.png"
    figure.savefig(output_dir / name, dpi=180)
    plt.close(figure)
    written.append(name)

    plot_specs = (
        ("T2", "T2_adjacent_gain_trajectory.png", "Step gain rate", None),
        ("T3", "T3_gain_separation.png", "Correct - incorrect step gain", None),
        ("T4", "T4_interval_discrimination.png", "Within-question AUC", 0.5),
    )
    for table_key, name, ylabel, reference in plot_specs:
        table = tables[table_key]
        figure, axis = plt.subplots(figsize=(6.5, 4))
        grouping = ["interface_id", "representation", "prompt_mode"]
        if table_key == "T2":
            grouping.append("is_correct")
        for key, group in table.groupby(grouping, dropna=False):
            values = key if isinstance(key, tuple) else (key,)
            group = group.sort_values("frac_mid")
            is_correct = values[-1] if table_key == "T2" else None
            label = " ".join(str(item) for item in values)
            color = colors[bool(is_correct)] if is_correct is not None else None
            axis.plot(group["frac_mid"], group["estimate"], marker="o", label=label, color=color)
            axis.fill_between(group["frac_mid"], group["ci_low"], group["ci_high"], alpha=0.15, color=color)
        if reference is not None:
            axis.axhline(reference, color="#555555", linestyle="--", linewidth=1)
        axis.set_xlabel("Interval midpoint")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / name, dpi=180)
        plt.close(figure)
        written.append(name)
    return written


def _write_report(path: Path, summary: dict[str, object]) -> None:
    gate = summary["gate"]
    pipeline = summary["pipeline"]
    lines = [
        "# RL03 Stage A Audit Report",
        "",
        f"- Gate: **{gate['label']}**",
        f"- Provisional: `{str(gate['provisional']).lower()}`",
        f"- Process gate passed: `{str(gate['passed']).lower()}`",
        f"- Questions: {summary['questions']}",
        f"- Rollouts: {summary['rollouts']}",
        f"- Atomic scores: {summary['atomic_scores']}",
        f"- Pipeline completion: {pipeline['completion_rate']:.4f}",
        f"- Finite score rate: {pipeline['finite_score_rate']:.4f}",
        "- Cross-fitted comparison metric: `held_out_question_equal_pairwise_auc`",
        "",
        "## Feature Metrics",
        "",
        "| interface | prompt mode | representation | feature | mixed q | AUC | 95% CI | positive corr |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["metrics"]:
        ci = f"[{row['auc_ci_low']:.4f}, {row['auc_ci_high']:.4f}]"
        lines.append(
            f"| {row['interface_id']} | {row['prompt_mode']} | {row['representation']} | "
            f"`{row['feature']}` | "
            f"{row['mixed_questions']} | {row['mean_auc']:.4f} | {ci} | "
            f"{row['positive_correlation_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gain vs Level Diagnostics",
            "",
            "| interface | prompt mode | representation | median early/final var | centered rank corr | positive rank frac | paired AUC delta | paired 95% CI | cross-fit improvement |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["gain_level_diagnostics"]:
        paired = _matching_record(
            summary["paired_gain_minus_level"],
            row["interface_id"],
            row["representation"],
            prompt_mode=row.get("prompt_mode"),
        )
        crossfit = _matching_record(
            summary["cross_fitted_comparisons"],
            row["interface_id"],
            row["representation"],
            prompt_mode=row.get("prompt_mode"),
        )
        paired_delta = float(paired["mean_delta_auc"]) if paired else float("nan")
        paired_ci = (
            f"[{float(paired['ci_low']):.4f}, {float(paired['ci_high']):.4f}]"
            if paired
            else "n/a"
        )
        crossfit_improvement = float(crossfit["improvement"]) if crossfit else float("nan")
        lines.append(
            f"| {row['interface_id']} | {row['prompt_mode']} | {row['representation']} | "
            f"{float(row['median_early_final_variance_ratio']):.4f} | "
            f"{float(row['centered_late_gain_rank_correlation']):.4f} | "
            f"{float(row['positive_correlation_fraction']):.4f} | "
            f"{paired_delta:.4f} | {paired_ci} | {crossfit_improvement:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gate Notes",
            "",
            "Missing Stage A controls keep a process candidate provisional; they never count as a completed PASS-PROCESS decision.",
        ]
    )
    if gate["missing_controls"]:
        lines.append("Missing controls: " + ", ".join(f"`{name}`" for name in gate["missing_controls"]) + ".")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    scores_path: Path,
    output_dir: Path,
    bootstrap_samples: int = 10_000,
    seed: int = 20260713,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = _ensure_prompt_mode(pd.read_json(scores_path, lines=True))
    required = set(GROUP_KEYS) | {
        "target_id",
        "is_gold_target",
        "target_text",
        "score_mean",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"score rows are missing required columns: {', '.join(missing)}")
    scores["question_id"] = scores["question_id"].astype(str)
    probes = build_probe_features(scores)
    rollouts = build_rollout_features(scores)
    steps = build_step_features(scores)

    probes.to_parquet(output_dir / "rollout_probe_metrics.parquet", index=False)
    probes.to_csv(output_dir / "rollout_probe_metrics.csv", index=False)
    rollouts.to_csv(output_dir / "rollout_features.csv", index=False)
    steps.to_csv(output_dir / "dense_step_metrics.csv", index=False)

    tables = build_dense_tables(probes, steps, bootstrap_samples, seed)
    table_names = {
        "T1": "T1_level_trajectory.csv",
        "T2": "T2_adjacent_gain_trajectory.csv",
        "T3": "T3_gain_separation.csv",
        "T4": "T4_interval_discrimination.csv",
    }
    for key, name in table_names.items():
        tables[key].to_csv(output_dir / name, index=False)

    metric_rows, diagnostic_rows, paired_rows, crossfit_rows = _cell_records(
        rollouts,
        bootstrap_samples,
        seed,
    )
    run_summary_path = Path(scores_path).parent / "run_summary.json"
    run_summary: dict[str, object] = {}
    if run_summary_path.is_file():
        loaded = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            run_summary = loaded
    expected_requests = int(run_summary.get("requests", len(scores)))
    pipeline = smoke_gate(scores, steps, expected_requests=expected_requests)
    pipeline["completion_known"] = "requests" in run_summary
    score_failures = int(run_summary.get("score_failures", 0))
    pipeline["score_failures"] = score_failures
    pipeline["score_failure_rate"] = (
        float(score_failures / expected_requests) if expected_requests else float("nan")
    )
    if expected_requests and score_failures / expected_requests > 0.01:
        pipeline["passed"] = False
    gate_inputs = _gate_metrics(
        pipeline,
        metric_rows,
        diagnostic_rows,
        paired_rows,
        crossfit_rows,
    )
    gate = assign_stage_a_gate(gate_inputs)
    plot_files = _plot_dense_tables(tables, output_dir)
    summary = {
        "protocol": "RL03 Stage A",
        "scores_path": str(scores_path),
        "seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples),
        "atomic_scores": int(len(scores)),
        "questions": int(scores["question_id"].nunique()),
        "rollouts": int(scores[["question_id", "rollout_id"]].drop_duplicates().shape[0]),
        "pipeline": pipeline,
        "metrics": metric_rows,
        "gain_level_diagnostics": diagnostic_rows,
        "paired_gain_minus_level": paired_rows,
        "cross_fitted_metric": "held_out_question_equal_pairwise_auc",
        "cross_fitted_comparisons": crossfit_rows,
        "gate_inputs": gate_inputs,
        "gate": gate,
        "plot_files": plot_files,
    }
    safe_summary = _json_safe(summary)
    (output_dir / "metric_summary.json").write_text(
        json.dumps(safe_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "STAGE_A_AUDIT_REPORT.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        args.scores,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(_json_safe(summary["gate"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
