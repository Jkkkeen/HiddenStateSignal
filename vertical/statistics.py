from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t


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


def question_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if "question_id" not in frame:
        raise ValueError("question bootstrap requires question_id")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    question_ids = pd.Index(frame["question_id"].dropna().unique())
    if question_ids.empty:
        raise ValueError("question bootstrap requires at least one question")
    groups = {
        question_id: group.copy()
        for question_id, group in frame.loc[
            frame["question_id"].notna()
        ].groupby("question_id", sort=False)
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for bootstrap_id in range(n_boot):
        selected = rng.integers(0, len(question_ids), size=len(question_ids))
        sampled = []
        for draw_id, question_index in enumerate(selected):
            original = question_ids[int(question_index)]
            part = groups[original].copy()
            part["source_question_id"] = part["question_id"]
            part["question_id"] = f"bootstrap_{draw_id}"
            sampled.append(part)
        value = float(statistic(pd.concat(sampled, ignore_index=True)))
        rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "estimate": value,
                "n_questions": int(len(question_ids)),
                "seed": int(seed),
                "resampling_unit": "question_id",
            }
        )
    return pd.DataFrame(rows)


def _two_sample_t(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    statistics = np.full(values.shape[1], np.nan, dtype=float)
    for layer in range(values.shape[1]):
        column = values[:, layer]
        positive = column[labels & np.isfinite(column)]
        negative = column[(~labels) & np.isfinite(column)]
        if len(positive) < 2 or len(negative) < 2:
            continue
        difference = float(positive.mean() - negative.mean())
        variance = float(
            positive.var(ddof=1) / len(positive)
            + negative.var(ddof=1) / len(negative)
        )
        if not np.isfinite(variance):
            continue
        standard_error = np.sqrt(max(variance, 1e-24))
        statistics[layer] = difference / standard_error
    return statistics


def _one_sample_t(values: np.ndarray) -> np.ndarray:
    statistics = np.full(values.shape[1], np.nan, dtype=float)
    for layer in range(values.shape[1]):
        column = values[:, layer]
        finite = column[np.isfinite(column)]
        if len(finite) < 2:
            continue
        standard_error = float(finite.std(ddof=1) / np.sqrt(len(finite)))
        statistics[layer] = float(finite.mean()) / max(standard_error, 1e-12)
    return statistics


def _one_dimensional_clusters(
    statistics: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    values = np.asarray(statistics, dtype=float)
    selected = np.isfinite(values) & (np.abs(values) >= threshold)
    clusters: list[dict[str, Any]] = []
    index = 0
    while index < len(values):
        if not selected[index]:
            index += 1
            continue
        sign = 1 if values[index] > 0 else -1
        start = index
        index += 1
        while (
            index < len(values)
            and selected[index]
            and (1 if values[index] > 0 else -1) == sign
        ):
            index += 1
        end = index - 1
        clusters.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "sign": int(sign),
                "mass": float(np.abs(values[start : end + 1]).sum()),
                "cell_count": int(end - start + 1),
            }
        )
    return clusters


def cluster_permutation(
    values: np.ndarray,
    labels: np.ndarray,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    arrays = np.asarray(values, dtype=float)
    outcomes = np.asarray(labels)
    if arrays.ndim != 2:
        raise ValueError("cluster values must have shape (question, layer)")
    if outcomes.ndim != 1 or len(outcomes) != len(arrays):
        raise ValueError("cluster labels must align with question rows")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if not np.isin(outcomes, [False, True, 0, 1]).all():
        raise ValueError("cluster labels must be binary")
    outcomes = outcomes.astype(bool)
    if outcomes.sum() < 2 or (~outcomes).sum() < 2:
        raise ValueError("cluster permutation requires at least two questions per label")
    threshold = 2.0
    observed = _one_dimensional_clusters(
        _two_sample_t(arrays, outcomes),
        threshold,
    )
    rng = np.random.default_rng(seed)
    null_maximum = np.zeros(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        permuted = rng.permutation(outcomes)
        clusters = _one_dimensional_clusters(
            _two_sample_t(arrays, permuted),
            threshold,
        )
        null_maximum[permutation] = max(
            (cluster["mass"] for cluster in clusters),
            default=0.0,
        )
    columns = [
        "cluster_id",
        "start_index",
        "end_index",
        "sign",
        "mass",
        "cell_count",
        "p_value",
        "n_questions",
        "n_permutations",
        "seed",
        "threshold",
        "correction_method",
    ]
    rows = []
    for cluster_id, cluster in enumerate(observed):
        rows.append(
            {
                "cluster_id": cluster_id,
                **cluster,
                "p_value": float(
                    (1 + np.sum(null_maximum >= cluster["mass"]))
                    / (n_permutations + 1)
                ),
                "n_questions": int(len(arrays)),
                "n_permutations": int(n_permutations),
                "seed": int(seed),
                "threshold": threshold,
                "correction_method": "max_cluster_mass",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def cluster_signflip(
    question_effects: np.ndarray,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    values = np.asarray(question_effects, dtype=float)
    if values.ndim != 2:
        raise ValueError("cluster effects must have shape (question, layer)")
    if len(values) < 2:
        raise ValueError("cluster sign-flip requires at least two questions")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    threshold = 2.0
    observed = _one_dimensional_clusters(_one_sample_t(values), threshold)
    rng = np.random.default_rng(seed)
    null_maximum = np.zeros(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=(len(values), 1))
        clusters = _one_dimensional_clusters(
            _one_sample_t(values * signs),
            threshold,
        )
        null_maximum[permutation] = max(
            (cluster["mass"] for cluster in clusters),
            default=0.0,
        )
    rows = []
    for cluster_id, cluster in enumerate(observed):
        rows.append(
            {
                "cluster_id": cluster_id,
                **cluster,
                "p_value": float(
                    (1 + np.sum(null_maximum >= cluster["mass"]))
                    / (n_permutations + 1)
                ),
                "n_questions": int(len(values)),
                "n_permutations": int(n_permutations),
                "seed": int(seed),
                "threshold": threshold,
                "correction_method": "question_sign_flip_max_cluster_mass",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "cluster_id",
            "start_index",
            "end_index",
            "sign",
            "mass",
            "cell_count",
            "p_value",
            "n_questions",
            "n_permutations",
            "seed",
            "threshold",
            "correction_method",
        ],
    )


def _unsupported_regression(
    *,
    reason: str,
    controls_used: Sequence[str],
    controls_omitted: Sequence[str],
    n_observations: int,
    n_questions: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "term": "is_correct",
                "estimate": np.nan,
                "standard_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "n_observations": int(n_observations),
                "n_questions": int(n_questions),
                "controls_used": ",".join(controls_used),
                "controls_omitted": ",".join(controls_omitted),
                "coverage_ok": False,
                "unsupported_reason": reason,
                "standard_error_method": "question_cluster_robust",
            }
        ]
    )


def fit_nuisance_controlled_effects(
    frame: pd.DataFrame,
    score: str,
    controls: Sequence[str],
) -> pd.DataFrame:
    required = {"question_id", "is_correct", score}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"nuisance frame is missing columns: {missing}")
    requested = list(dict.fromkeys(str(control) for control in controls))
    used: list[str] = []
    omitted: list[str] = []
    numeric_controls: dict[str, pd.Series] = {}
    for control in requested:
        if control not in frame:
            omitted.append(control)
            continue
        numeric = pd.to_numeric(frame[control], errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if len(finite) < 2 or float(finite.std(ddof=0)) <= 1e-12:
            omitted.append(control)
            continue
        used.append(control)
        numeric_controls[control] = numeric

    data = pd.DataFrame(
        {
            "question_id": frame["question_id"],
            "is_correct": pd.to_numeric(frame["is_correct"], errors="coerce"),
            "score": pd.to_numeric(frame[score], errors="coerce"),
            **numeric_controls,
        }
    )
    finite = data["question_id"].notna()
    for column in ["is_correct", "score", *used]:
        finite &= np.isfinite(data[column])
    data = data.loc[finite].reset_index(drop=True)
    n_questions = int(data["question_id"].nunique())
    if data["is_correct"].nunique() < 2:
        return _unsupported_regression(
            reason="is_correct has no variation",
            controls_used=used,
            controls_omitted=omitted,
            n_observations=len(data),
            n_questions=n_questions,
        )
    terms = ["intercept", "is_correct", *used]
    design = np.column_stack(
        [
            np.ones(len(data), dtype=float),
            data["is_correct"].to_numpy(float),
            *(data[control].to_numpy(float) for control in used),
        ]
    )
    outcome = data["score"].to_numpy(float)
    if len(data) <= len(terms) or n_questions < 2:
        return _unsupported_regression(
            reason="insufficient complete observations or questions",
            controls_used=used,
            controls_omitted=omitted,
            n_observations=len(data),
            n_questions=n_questions,
        )
    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank < len(terms):
        return _unsupported_regression(
            reason="design matrix is rank deficient",
            controls_used=used,
            controls_omitted=omitted,
            n_observations=len(data),
            n_questions=n_questions,
        )
    residuals = outcome - design @ coefficients
    bread = np.linalg.inv(design.T @ design)
    meat = np.zeros((len(terms), len(terms)), dtype=float)
    for _, indices in data.groupby("question_id", sort=False).groups.items():
        locations = np.asarray(list(indices), dtype=int)
        score_vector = design[locations].T @ residuals[locations]
        meat += np.outer(score_vector, score_vector)
    correction = (
        n_questions / (n_questions - 1)
        * (len(data) - 1)
        / (len(data) - len(terms))
    )
    covariance = correction * bread @ meat @ bread
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2.0 * student_t.sf(np.abs(t_statistics), df=n_questions - 1)
    return pd.DataFrame(
        [
            {
                "term": term,
                "estimate": float(coefficients[index]),
                "standard_error": float(standard_errors[index]),
                "t_stat": float(t_statistics[index]),
                "p_value": float(p_values[index]),
                "n_observations": int(len(data)),
                "n_questions": n_questions,
                "controls_used": ",".join(used),
                "controls_omitted": ",".join(omitted),
                "coverage_ok": True,
                "unsupported_reason": "",
                "standard_error_method": "question_cluster_robust",
            }
            for index, term in enumerate(terms)
        ]
    )


def paired_condition_delta(
    frame: pd.DataFrame,
    base_condition: str,
    target_condition: str,
    keys: Sequence[str],
) -> pd.DataFrame:
    if "condition" not in frame:
        raise ValueError("paired condition frame requires condition")
    pair_keys = list(dict.fromkeys(str(key) for key in keys))
    if not pair_keys or "condition" in pair_keys:
        raise ValueError("paired condition keys must be nonempty and exclude condition")
    missing = sorted(set(pair_keys) - set(frame.columns))
    if missing:
        raise ValueError(f"paired condition frame is missing keys: {missing}")
    base = frame.loc[frame["condition"] == base_condition].copy()
    target = frame.loc[frame["condition"] == target_condition].copy()
    if base.empty or target.empty:
        raise ValueError("paired condition delta requires both conditions")
    if base.duplicated(pair_keys).any() or target.duplicated(pair_keys).any():
        raise ValueError("paired condition keys must be one-to-one within each condition")
    common_numeric = [
        column
        for column in frame.columns
        if column not in set(pair_keys + ["condition"])
        and pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]
    if not common_numeric:
        raise ValueError("paired condition delta requires at least one numeric value column")
    base_values = base[pair_keys + common_numeric]
    target_values = target[pair_keys + common_numeric]
    outer = base_values[pair_keys].merge(
        target_values[pair_keys],
        on=pair_keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    paired = base_values.merge(
        target_values,
        on=pair_keys,
        how="inner",
        suffixes=("_base", "_target"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("paired condition keys have no aligned rows")
    for column in common_numeric:
        paired[f"{column}_delta"] = (
            paired[f"{column}_target"] - paired[f"{column}_base"]
        )
    unpaired_base = int((outer["_merge"] == "left_only").sum())
    unpaired_target = int((outer["_merge"] == "right_only").sum())
    paired["base_condition"] = base_condition
    paired["target_condition"] = target_condition
    paired["pairing_status"] = (
        "complete" if unpaired_base == 0 and unpaired_target == 0 else "partial"
    )
    paired["n_pairs"] = int(len(paired))
    paired["n_unpaired_base"] = unpaired_base
    paired["n_unpaired_target"] = unpaired_target
    return paired.sort_values(pair_keys).reset_index(drop=True)


def standardize_within_family_base(
    frame: pd.DataFrame,
    score: str,
    *,
    base_condition: str = "base",
) -> pd.DataFrame:
    if score not in frame:
        raise ValueError(f"missing score column: {score}")
    candidate_keys = ["model_family", "representation", "stage", "layer_index"]
    keys = [key for key in candidate_keys if key in frame]
    if "model_family" not in keys or "condition" not in frame:
        raise ValueError("Base standardization requires model_family and condition")
    data = frame.copy()
    data[score] = pd.to_numeric(data[score], errors="coerce")
    base = data.loc[
        (data["condition"] == base_condition) & np.isfinite(data[score])
    ]
    if base.empty:
        raise ValueError(f"missing Base condition={base_condition}")
    mean_name = f"base_mean_{score}"
    std_name = f"base_std_{score}"
    standardized_name = f"base_standardized_{score}"
    calibration = (
        base.groupby(keys, dropna=False, as_index=False)[score]
        .agg(["mean", lambda values: float(np.std(values, ddof=0))])
        .reset_index()
        .rename(columns={"mean": mean_name, "<lambda_0>": std_name})
    )
    merged = data.merge(calibration, on=keys, how="left", validate="many_to_one")
    denominator = merged[std_name].where(merged[std_name] > 1e-12)
    merged[standardized_name] = (merged[score] - merged[mean_name]) / denominator
    merged["standardization_ok"] = np.isfinite(merged[standardized_name])
    merged["standardization_scope"] = "within_model_family_base"
    return merged
