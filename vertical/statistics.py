from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PROFILE_KEYS = [
    "model_family",
    "model_name",
    "condition",
    "checkpoint",
    "global_step",
    "training_progress",
    "stage",
    "representation",
    "layer_index",
    "relative_depth",
]


def _valid_metric_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in frame:
        raise ValueError(f"missing profile metric: {metric}")
    missing = sorted(set(PROFILE_KEYS + ["question_id", "rollout_id"]) - set(frame.columns))
    if missing:
        raise ValueError(f"profile frame is missing columns: {missing}")
    result = frame.loc[np.isfinite(pd.to_numeric(frame[metric], errors="coerce"))].copy()
    coverage = f"profile_coverage_count_{metric}"
    if coverage in result:
        result = result.loc[result[coverage] > 0]
    return result


def summarize_policy_profiles(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = _valid_metric_rows(frame, metric)
    question = (
        data.groupby(PROFILE_KEYS + ["question_id"], dropna=False, as_index=False)
        .agg(profile_mean=(metric, "mean"), n_rollouts=("rollout_id", "nunique"))
    )
    if question.empty:
        return pd.DataFrame(
            columns=PROFILE_KEYS
            + ["profile_mean", "profile_std", "n_questions", "n_rollouts"]
        )
    return (
        question.groupby(PROFILE_KEYS, dropna=False, as_index=False)
        .agg(
            profile_mean=("profile_mean", "mean"),
            profile_std=("profile_mean", lambda values: float(np.std(values, ddof=0))),
            n_questions=("question_id", "nunique"),
            n_rollouts=("n_rollouts", "sum"),
        )
        .sort_values(
            ["model_family", "condition", "stage", "representation", "layer_index"]
        )
        .reset_index(drop=True)
    )


def summarize_outcome_profiles(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = _valid_metric_rows(frame, metric)
    if "is_correct" not in data:
        raise ValueError("profile frame is missing is_correct")
    data = data.loc[data["is_correct"].notna()].copy()
    question_label = (
        data.groupby(
            PROFILE_KEYS + ["question_id", "is_correct"],
            dropna=False,
            as_index=False,
        )
        .agg(profile_mean=(metric, "mean"), n_rollouts=("rollout_id", "nunique"))
    )
    output_columns = PROFILE_KEYS + [
        "correct_mean",
        "wrong_mean",
        "correct_minus_wrong",
        "n_mixed_questions",
        "n_rollouts",
    ]
    if question_label.empty:
        return pd.DataFrame(columns=output_columns)
    question_label["outcome"] = np.where(
        question_label["is_correct"].astype(bool),
        "correct",
        "wrong",
    )
    values = question_label.pivot(
        index=PROFILE_KEYS + ["question_id"],
        columns="outcome",
        values="profile_mean",
    )
    counts = question_label.pivot(
        index=PROFILE_KEYS + ["question_id"],
        columns="outcome",
        values="n_rollouts",
    )
    if "wrong" not in values.columns or "correct" not in values.columns:
        return pd.DataFrame(columns=output_columns)
    mixed = values.loc[:, ["wrong", "correct"]].dropna()
    if mixed.empty:
        return pd.DataFrame(columns=output_columns)
    mixed_counts = counts.reindex(mixed.index).fillna(0).sum(axis=1)
    mixed = mixed.assign(
        correct_minus_wrong=mixed["correct"] - mixed["wrong"],
        n_rollouts=mixed_counts,
    ).reset_index()
    return (
        mixed.groupby(PROFILE_KEYS, dropna=False, as_index=False)
        .agg(
            correct_mean=("correct", "mean"),
            wrong_mean=("wrong", "mean"),
            correct_minus_wrong=("correct_minus_wrong", "mean"),
            n_mixed_questions=("question_id", "nunique"),
            n_rollouts=("n_rollouts", "sum"),
        )
        .sort_values(
            ["model_family", "condition", "stage", "representation", "layer_index"]
        )
        .reset_index(drop=True)
    )


def within_question_auc(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    required = {"question_id", "is_correct", score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"AUROC frame is missing columns: {missing}")
    data = frame.loc[frame["is_correct"].notna()].copy()
    data[score_column] = pd.to_numeric(data[score_column], errors="coerce")
    data = data.loc[np.isfinite(data[score_column])]
    rows: list[dict[str, Any]] = []
    for question_id, group in data.groupby("question_id", sort=True):
        labels = group["is_correct"].astype(bool).to_numpy()
        positive = int(labels.sum())
        negative = int((~labels).sum())
        if positive == 0 or negative == 0:
            continue
        scores = group[score_column].to_numpy(float)
        ranks = rankdata(scores, method="average")
        auc = (
            float(ranks[labels].sum()) - positive * (positive + 1) / 2
        ) / (positive * negative)
        rows.append(
            {
                "question_id": question_id,
                "auc": float(auc),
                "positive_count": positive,
                "negative_count": negative,
                "pair_count": positive * negative,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "question_id",
            "auc",
            "positive_count",
            "negative_count",
            "pair_count",
        ],
    )


def summarize_layer_auc(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = _valid_metric_rows(frame, metric)
    rows: list[dict[str, Any]] = []
    for keys, group in data.groupby(PROFILE_KEYS, dropna=False, sort=False):
        table = within_question_auc(group, score_column=metric)
        base = dict(zip(PROFILE_KEYS, keys, strict=True))
        rows.append(
            {
                **base,
                "question_equal_auc": float(table["auc"].mean())
                if len(table)
                else np.nan,
                "pair_weighted_auc": float(
                    np.average(table["auc"], weights=table["pair_count"])
                )
                if len(table)
                else np.nan,
                "n_mixed_questions": int(len(table)),
                "n_pairs": int(table["pair_count"].sum()) if len(table) else 0,
                "unsupported_reason": "" if len(table) else "no mixed-outcome questions",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=PROFILE_KEYS
            + [
                "question_equal_auc",
                "pair_weighted_auc",
                "n_mixed_questions",
                "n_pairs",
                "unsupported_reason",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["model_family", "condition", "stage", "representation", "layer_index"]
    ).reset_index(drop=True)

