from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .formal_manifest import REPRESENTATIVES


CELL_KEYS = [
    "axis",
    "family_id",
    "family",
    "representation",
    "anchor_layer",
    "aggregation_mode",
    "metric",
    "stage",
]
TEST_KEYS = CELL_KEYS[:-1]


def select_primary_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the frozen 37-test family without inspecting metric values."""
    required = {
        "family_id",
        "metric",
        "representation",
        "anchor_layer",
        "aggregation_mode",
        "is_primary_metric",
        "coverage",
        "truncated",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"metric frame is missing columns: {sorted(missing)}")
    selected = frame.loc[
        frame["is_primary_metric"].astype(bool)
        & frame["coverage"].astype(bool)
        & ~frame["truncated"].astype(bool)
    ].copy()
    selected = selected.loc[
        (selected["family_id"] == "H8")
        | (selected["representation"] == "mean_w128_s32")
    ]
    selected = selected.loc[
        selected.apply(lambda row: REPRESENTATIVES.get(row["family_id"]) == row["metric"], axis=1)
    ]
    selected = selected.loc[
        ~((selected["family_id"] == "H2") & (selected["aggregation_mode"] != "local"))
    ]
    if (selected["family_id"] == "H8").any():
        h8 = selected["family_id"] == "H8"
        final_layer = pd.to_numeric(selected.loc[h8, "anchor_layer"]).max()
        selected = selected.loc[~h8 | (pd.to_numeric(selected["anchor_layer"], errors="coerce") == final_layer)]
    return selected.reset_index(drop=True)


def fit_base_standardizers(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame.loc[frame["checkpoint"] == "base"].copy()
    if base.empty:
        raise ValueError("base checkpoint is required for standardization")
    rows = []
    for keys, group in base.groupby(CELL_KEYS, dropna=False, sort=False):
        values = group["value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        rows.append(
            {
                **dict(zip(CELL_KEYS, keys, strict=True)),
                "base_mean": float(values.mean()),
                "base_std": float(values.std(ddof=0)),
                "base_n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def apply_base_standardizers(frame: pd.DataFrame, calibrators: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(calibrators, on=CELL_KEYS, how="left", validate="many_to_one")
    denominator = merged["base_std"].where(merged["base_std"] > 1e-12)
    merged["z_value"] = (merged["value"] - merged["base_mean"]) / denominator
    merged["standardization_ok"] = np.isfinite(merged["z_value"])
    return merged


def question_bootstrap_mean_ci(
    question_values: pd.Series,
    *,
    n_bootstrap: int = 4000,
    seed: int = 20260805,
) -> tuple[float, float]:
    values = question_values.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    boot = values[draws].mean(axis=1)
    return tuple(float(value) for value in np.quantile(boot, [0.025, 0.975]))


def summarize_question_equal(
    frame: pd.DataFrame,
    *,
    value_column: str = "z_value",
    n_bootstrap: int = 4000,
    seed: int = 20260805,
) -> pd.DataFrame:
    group_keys = TEST_KEYS + ["checkpoint", "training_progress", "stage"]
    question = (
        frame.groupby(group_keys + ["question_id"], dropna=False, as_index=False)[value_column]
        .mean()
    )
    rows = []
    for index, (keys, group) in enumerate(question.groupby(group_keys, dropna=False, sort=False)):
        values = group[value_column].dropna()
        low, high = question_bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            seed=seed + index,
        )
        rows.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "question_equal_mean": float(values.mean()) if len(values) else np.nan,
                "question_equal_median": float(values.median()) if len(values) else np.nan,
                "ci_low": low,
                "ci_high": high,
                "n_questions": int(group["question_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def auc_from_scores(labels: Iterable[bool], scores: Iterable[float]) -> float:
    labels_ = np.asarray(list(labels), dtype=bool)
    scores_ = np.asarray(list(scores), dtype=float)
    valid = np.isfinite(scores_)
    labels_, scores_ = labels_[valid], scores_[valid]
    positive, negative = scores_[labels_], scores_[~labels_]
    if not len(positive) or not len(negative):
        return np.nan
    comparison = positive[:, None] - negative[None, :]
    return float((np.sum(comparison > 0) + 0.5 * np.sum(comparison == 0)) / comparison.size)


def within_question_auc_table(frame: pd.DataFrame, *, score_column: str = "z_value") -> pd.DataFrame:
    rows = []
    for question_id, group in frame.groupby("question_id", sort=False):
        n_positive = int(group["is_correct"].astype(bool).sum())
        n_negative = int((~group["is_correct"].astype(bool)).sum())
        if not n_positive or not n_negative:
            continue
        rows.append(
            {
                "question_id": question_id,
                "auc": auc_from_scores(group["is_correct"], group[score_column]),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "pair_count": n_positive * n_negative,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_auc_aggregates(
    table: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if table.empty:
        return (np.nan, np.nan), (np.nan, np.nan)
    auc = table["auc"].to_numpy(dtype=float)
    weights = table["pair_count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(table), size=(n_bootstrap, len(table)))
    sampled_auc = auc[draws]
    sampled_weight = weights[draws]
    question_equal = sampled_auc.mean(axis=1)
    pair_weighted = (sampled_auc * sampled_weight).sum(axis=1) / sampled_weight.sum(axis=1)
    return (
        tuple(float(value) for value in np.quantile(question_equal, [0.025, 0.975])),
        tuple(float(value) for value in np.quantile(pair_weighted, [0.025, 0.975])),
    )


def summarize_outcome_auc(
    frame: pd.DataFrame,
    *,
    score_column: str = "z_value",
    n_bootstrap: int = 4000,
    seed: int = 20260805,
) -> pd.DataFrame:
    group_keys = TEST_KEYS + ["checkpoint", "training_progress", "stage"]
    rows = []
    for index, (keys, group) in enumerate(frame.groupby(group_keys, dropna=False, sort=False)):
        table = within_question_auc_table(group, score_column=score_column)
        if table.empty:
            continue
        qe_ci, pw_ci = _bootstrap_auc_aggregates(
            table,
            n_bootstrap=n_bootstrap,
            seed=seed + index,
        )
        weights = table["pair_count"].to_numpy(dtype=float)
        centered_auc = table["auc"].to_numpy(dtype=float) - 0.5
        try:
            outcome_p = float(stats.wilcoxon(centered_auc, alternative="two-sided").pvalue)
        except ValueError:
            outcome_p = 1.0
        rows.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "question_equal_auc": float(table["auc"].mean()),
                "question_equal_ci_low": qe_ci[0],
                "question_equal_ci_high": qe_ci[1],
                "pair_weighted_auc": float(np.average(table["auc"], weights=weights)),
                "pair_weighted_ci_low": pw_ci[0],
                "pair_weighted_ci_high": pw_ci[1],
                "question_equal_p": outcome_p,
                "n_mixed_questions": int(len(table)),
                "n_pairs": int(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def grouped_oof_length_increment(
    frame: pd.DataFrame,
    *,
    metric_column: str = "z_value",
    n_splits: int = 5,
    seed: int = 20260805,
) -> dict[str, Any]:
    """Compare grouped-OOF length/policy-entropy baseline with baseline + hidden metric."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    columns = ["question_id", "is_correct", "response_length", "policy_entropy", metric_column]
    data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n_questions = data["question_id"].nunique()
    folds = min(n_splits, n_questions)
    if folds < 2 or data["is_correct"].nunique() < 2:
        return {"fit_ok": False, "reason": "insufficient questions or outcome classes"}
    data["log_response_length"] = np.log1p(data["response_length"].astype(float))
    baseline_columns = ["log_response_length", "policy_entropy"]
    extended_columns = baseline_columns + [metric_column]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    baseline_score = np.full(len(data), np.nan)
    extended_score = np.full(len(data), np.nan)
    y = data["is_correct"].astype(int).to_numpy()
    groups = data["question_id"].to_numpy()
    for train, test in splitter.split(data, y, groups):
        if np.unique(y[train]).size < 2:
            return {"fit_ok": False, "reason": "a training fold has one outcome class"}
        for features, target in (
            (baseline_columns, baseline_score),
            (extended_columns, extended_score),
        ):
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=seed),
            )
            model.fit(data.iloc[train][features], y[train])
            target[test] = model.predict_proba(data.iloc[test][features])[:, 1]
    scored = data[["question_id", "is_correct"]].copy()
    scored["baseline_score"] = baseline_score
    scored["extended_score"] = extended_score
    baseline = within_question_auc_table(scored, score_column="baseline_score")
    extended = within_question_auc_table(scored, score_column="extended_score")
    joined = baseline[["question_id", "auc", "pair_count"]].merge(
        extended[["question_id", "auc"]],
        on="question_id",
        suffixes=("_baseline", "_extended"),
        validate="one_to_one",
    )
    if joined.empty:
        return {"fit_ok": False, "reason": "no mixed-label question in OOF predictions"}
    weights = joined["pair_count"].to_numpy(dtype=float)
    return {
        "fit_ok": True,
        "n_rows": int(len(data)),
        "n_questions": int(n_questions),
        "n_mixed_questions": int(len(joined)),
        "question_equal_baseline_auc": float(joined["auc_baseline"].mean()),
        "question_equal_extended_auc": float(joined["auc_extended"].mean()),
        "question_equal_delta_auc": float((joined["auc_extended"] - joined["auc_baseline"]).mean()),
        "pair_weighted_baseline_auc": float(np.average(joined["auc_baseline"], weights=weights)),
        "pair_weighted_extended_auc": float(np.average(joined["auc_extended"], weights=weights)),
        "pair_weighted_delta_auc": float(
            np.average(joined["auc_extended"] - joined["auc_baseline"], weights=weights)
        ),
    }


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if not len(valid_positions):
        return adjusted
    valid = values[valid_positions]
    order = np.argsort(valid)
    ranked = valid[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.minimum(q, 1.0)
    adjusted[valid_positions] = restored
    return adjusted


def spearman_by_stage(frame: pd.DataFrame, *, value_column: str = "z_value") -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(TEST_KEYS + ["stage"], dropna=False, sort=False):
        question_checkpoint = (
            group.groupby(["question_id", "training_progress"], as_index=False)[value_column].mean()
        )
        result = stats.spearmanr(
            question_checkpoint["training_progress"],
            question_checkpoint[value_column],
            nan_policy="omit",
        )
        rows.append(
            {
                **dict(zip(TEST_KEYS + ["stage"], keys, strict=True)),
                "spearman_rho": float(result.statistic),
                "spearman_p": float(result.pvalue),
                "n_question_checkpoints": int(len(question_checkpoint)),
            }
        )
    return pd.DataFrame(rows)


def fit_training_stage_interaction(frame: pd.DataFrame, *, value_column: str = "z_value") -> dict[str, Any]:
    """Fit the preregistered random-intercept model, with clustered OLS fallback."""
    import statsmodels.formula.api as smf

    columns = [
        value_column,
        "training_progress",
        "stage",
        "response_length",
        "policy_entropy",
        "is_correct",
        "level",
        "question_id",
    ]
    data = frame[columns].dropna().copy()
    data["log_response_length"] = np.log1p(data["response_length"].astype(float))
    data["stage"] = data["stage"].astype("category")
    data["is_correct"] = data["is_correct"].astype(int)
    if data["training_progress"].nunique() < 2 or data["stage"].nunique() < 2:
        return {"fit_ok": False, "reason": "insufficient progress or stage variation"}
    full_formula = (
        f"{value_column} ~ training_progress * C(stage) + log_response_length + "
        "policy_entropy + is_correct + C(level)"
    )
    reduced_formula = (
        f"{value_column} ~ training_progress + C(stage) + log_response_length + "
        "policy_entropy + is_correct + C(level)"
    )
    try:
        full = smf.mixedlm(full_formula, data, groups=data["question_id"]).fit(
            reml=False, method="lbfgs", disp=False
        )
        reduced = smf.mixedlm(reduced_formula, data, groups=data["question_id"]).fit(
            reml=False, method="lbfgs", disp=False
        )
        degrees = max(int(len(full.fe_params) - len(reduced.fe_params)), 1)
        likelihood_ratio = max(0.0, 2.0 * (full.llf - reduced.llf))
        return {
            "fit_ok": True,
            "method": "mixedlm_random_question_intercept",
            "interaction_statistic": likelihood_ratio,
            "interaction_df": degrees,
            "interaction_p": float(stats.chi2.sf(likelihood_ratio, degrees)),
            "n_rows": int(len(data)),
            "n_questions": int(data["question_id"].nunique()),
        }
    except (ValueError, np.linalg.LinAlgError):
        fixed_formula = full_formula + " + C(question_id)"
        fit = smf.ols(fixed_formula, data).fit(
            cov_type="cluster", cov_kwds={"groups": data["question_id"]}
        )
        interaction_names = [name for name in fit.params.index if "training_progress:C(stage)" in name]
        if not interaction_names:
            return {"fit_ok": False, "reason": "interaction terms absent after fallback"}
        hypothesis = ", ".join(f"{name} = 0" for name in interaction_names)
        test = fit.wald_test(hypothesis, scalar=True)
        return {
            "fit_ok": True,
            "method": "question_fixed_effect_clustered_ols",
            "interaction_statistic": float(test.statistic),
            "interaction_df": len(interaction_names),
            "interaction_p": float(test.pvalue),
            "n_rows": int(len(data)),
            "n_questions": int(data["question_id"].nunique()),
        }
