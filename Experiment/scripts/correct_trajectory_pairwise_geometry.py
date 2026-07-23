#!/usr/bin/env python3
"""Pure metrics for correct-trajectory pairwise hidden geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12
METRICS = ("angle_rad", "angle_deg", "delta_vec", "delta_rel", "delta_amp")
PRIMARY_METRICS = ("angle_rad", "delta_rel", "delta_amp")


def compute_pair_metrics(
    frame: pd.DataFrame, eps: float = EPS
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recover pairwise distances from the two norms and their saved cosine."""
    result = frame.copy()
    left = result["displacement_norm"].to_numpy(dtype=np.float64)
    right = result["reference_norm"].to_numpy(dtype=np.float64)
    cosine = result["cosine_similarity"].to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(left)
        & np.isfinite(right)
        & np.isfinite(cosine)
        & (left > eps)
        & (right > eps)
    )
    result = result.loc[valid].copy()
    left = left[valid]
    right = right[valid]
    cosine = np.clip(cosine[valid], -1.0, 1.0)
    result["angle_rad"] = np.arccos(cosine)
    result["angle_deg"] = np.degrees(result["angle_rad"])
    radicand = np.maximum(
        left * left + right * right - 2.0 * left * right * cosine,
        0.0,
    )
    result["delta_vec"] = np.sqrt(radicand)
    result["delta_rel"] = result["delta_vec"].to_numpy() / (left + right + eps)
    result["delta_amp"] = np.abs(np.log(left + eps) - np.log(right + eps))
    return result, {
        "input_rows": int(len(frame)),
        "excluded_rows": int((~valid).sum()),
        "valid_rows": int(valid.sum()),
    }


def aggregate_directed_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every directed rollout pair one observation per layer and progress bin."""
    keys = [
        "question_id",
        "layer",
        "progress_bin",
        "rollout_id",
        "reference_rollout_id",
        "is_correct",
        "reference_is_correct",
    ]
    return (
        frame.groupby(keys, as_index=False, observed=True)[list(METRICS)]
        .median()
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )


def symmetrize_pairs(directed: pd.DataFrame) -> pd.DataFrame:
    """Average i->j and j->i into one canonical undirected rollout pair."""
    view = directed.copy()
    view["pair_lo"] = view[["rollout_id", "reference_rollout_id"]].min(axis=1).astype(int)
    view["pair_hi"] = view[["rollout_id", "reference_rollout_id"]].max(axis=1).astype(int)
    view["pair_type"] = np.where(
        view["is_correct"] & view["reference_is_correct"],
        "++",
        np.where(
            ~view["is_correct"] & ~view["reference_is_correct"],
            "--",
            "+-",
        ),
    )
    keys = [
        "question_id",
        "layer",
        "progress_bin",
        "pair_lo",
        "pair_hi",
        "pair_type",
    ]
    return (
        view.groupby(keys, as_index=False, observed=True)[list(METRICS)]
        .mean()
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )


def pair_type_means(pairs: pd.DataFrame, whole_trajectory: bool) -> pd.DataFrame:
    """Average pairs inside each question so every question has equal weight."""
    view = pairs.copy()
    if whole_trajectory:
        pair_keys = ["question_id", "layer", "pair_lo", "pair_hi", "pair_type"]
        view = view.groupby(pair_keys, as_index=False, observed=True)[list(METRICS)].mean()
        group_keys = ["question_id", "layer", "pair_type"]
    else:
        group_keys = ["question_id", "layer", "progress_bin", "pair_type"]
    return (
        view.groupby(group_keys, as_index=False, observed=True)[list(METRICS)]
        .mean()
        .sort_values(group_keys, kind="stable")
        .reset_index(drop=True)
    )


def question_contrasts(
    means: pd.DataFrame,
    metrics: Iterable[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """Calculate correct-correct minus each control within every question."""
    id_cols = [
        column
        for column in ("question_id", "layer", "progress_bin")
        if column in means.columns
    ]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        pivot = means.pivot_table(
            index=id_cols,
            columns="pair_type",
            values=metric,
            aggfunc="first",
        )
        for contrast, control in (("pp_minus_mm", "--"), ("pp_minus_pm", "+-")):
            if "++" not in pivot.columns or control not in pivot.columns:
                continue
            values = (pivot["++"] - pivot[control]).dropna()
            for key, value in values.items():
                key_tuple = key if isinstance(key, tuple) else (key,)
                rows.append(
                    {
                        **dict(zip(id_cols, key_tuple)),
                        "metric": str(metric),
                        "contrast": contrast,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def directed_four_cell_interactions(
    directed: pd.DataFrame,
    metrics: Iterable[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """Retain the directed ++/+-/-+/-- interaction before symmetrization."""
    view = directed.copy()
    view["cell"] = np.select(
        [
            view["is_correct"] & view["reference_is_correct"],
            view["is_correct"] & ~view["reference_is_correct"],
            ~view["is_correct"] & view["reference_is_correct"],
        ],
        ["++", "+-", "-+"],
        default="--",
    )
    base_keys = ["question_id", "layer", "progress_bin"]
    cells = view.groupby(base_keys + ["cell"], as_index=False, observed=True)[
        list(metrics)
    ].mean()
    rows: list[dict[str, object]] = []
    for metric in metrics:
        pivot = cells.pivot_table(
            index=base_keys,
            columns="cell",
            values=metric,
            aggfunc="first",
        )
        if not {"++", "+-", "-+", "--"}.issubset(pivot.columns):
            continue
        valid = pivot.dropna(subset=["++", "+-", "-+", "--"])
        interaction = (valid["++"] - valid["+-"]) - (
            valid["-+"] - valid["--"]
        )
        for key, value in interaction.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            rows.append(
                {
                    **dict(zip(base_keys, key_tuple)),
                    "metric": str(metric),
                    "interaction": float(value),
                }
            )
    return pd.DataFrame(rows)


def summarize_contrasts(
    contrasts: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Question-bootstrap each progress or whole-trajectory contrast."""
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    rng = np.random.default_rng(seed)
    group_cols = [
        column
        for column in ("metric", "contrast", "layer", "progress_bin")
        if column in contrasts.columns
    ]
    rows: list[dict[str, object]] = []
    for keys, group in contrasts.groupby(group_cols, sort=True, observed=True):
        values = group.set_index("question_id")["value"].dropna()
        if values.empty:
            continue
        samples = np.asarray(
            [
                rng.choice(values.to_numpy(), size=len(values), replace=True).mean()
                for _ in range(bootstrap)
            ],
            dtype=np.float64,
        )
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(group_cols, key_tuple)),
                "mean_contrast": float(values.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "negative_sign_fraction": float((values < 0).mean()),
                "n_questions": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PermutationResult:
    nulls: pd.DataFrame
    label_count_checks: np.ndarray


def _permuted_label_map(
    labels: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[dict[tuple[str, int], bool], bool]:
    mapping: dict[tuple[str, int], bool] = {}
    checks: list[bool] = []
    for question_id, group in labels.groupby("question_id", sort=True):
        ordered = group.sort_values("rollout_id", kind="stable")
        before = ordered["is_correct"].to_numpy(dtype=bool)
        after = rng.permutation(before)
        checks.append(int(before.sum()) == int(after.sum()))
        mapping.update(
            {
                (str(question_id), int(rollout_id)): bool(label)
                for rollout_id, label in zip(ordered["rollout_id"], after)
            }
        )
    return mapping, bool(all(checks))


def _assign_pair_types(
    pairs: pd.DataFrame,
    mapping: dict[tuple[str, int], bool],
) -> pd.DataFrame:
    relabeled = pairs.copy()
    left = np.asarray(
        [
            mapping[(str(row.question_id), int(row.pair_lo))]
            for row in relabeled.itertuples()
        ],
        dtype=bool,
    )
    right = np.asarray(
        [
            mapping[(str(row.question_id), int(row.pair_hi))]
            for row in relabeled.itertuples()
        ],
        dtype=bool,
    )
    relabeled["pair_type"] = np.where(
        left & right,
        "++",
        np.where(~left & ~right, "--", "+-"),
    )
    return relabeled


def _null_estimates(
    pairs: pd.DataFrame,
    permutation_id: int,
) -> pd.DataFrame:
    frames = []
    for scope, whole_trajectory in (("progress", False), ("whole", True)):
        means = pair_type_means(pairs, whole_trajectory=whole_trajectory)
        contrasts = question_contrasts(means)
        group_cols = ["metric", "contrast", "layer"]
        if not whole_trajectory:
            group_cols.append("progress_bin")
        estimates = (
            contrasts.groupby(group_cols, as_index=False, observed=True)["value"]
            .mean()
            .rename(columns={"value": "mean_contrast"})
        )
        estimates.insert(0, "scope", scope)
        estimates.insert(0, "permutation_id", int(permutation_id))
        frames.append(estimates)
    return pd.concat(frames, ignore_index=True)


def permutation_inference(
    pairs: pd.DataFrame,
    labels: pd.DataFrame,
    permutations: int,
    seed: int,
) -> PermutationResult:
    """Build within-question label-permutation nulls from fixed pair geometry."""
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    label_view = labels[["question_id", "rollout_id", "is_correct"]].drop_duplicates(
        ["question_id", "rollout_id"]
    )
    rng = np.random.default_rng(seed)
    frames = []
    checks = np.zeros(permutations, dtype=bool)
    for permutation_id in range(permutations):
        mapping, checks[permutation_id] = _permuted_label_map(label_view, rng)
        relabeled = _assign_pair_types(pairs, mapping)
        frames.append(_null_estimates(relabeled, permutation_id))
    return PermutationResult(
        nulls=pd.concat(frames, ignore_index=True),
        label_count_checks=checks,
    )


def adjusted_max_stat_pvalues(
    observed: pd.DataFrame,
    nulls: pd.DataFrame,
) -> pd.DataFrame:
    """Attach family-wise max-absolute-statistic p-values to progress cells."""
    result = observed.copy()
    result["max_stat_p"] = np.nan
    progress_nulls = nulls[nulls["scope"] == "progress"]
    for keys, group in result.groupby(["metric", "contrast"], sort=True):
        null_family = progress_nulls[
            (progress_nulls["metric"] == keys[0])
            & (progress_nulls["contrast"] == keys[1])
        ]
        maxima = null_family.groupby("permutation_id")["mean_contrast"].apply(
            lambda values: values.abs().max()
        )
        for index in group.index:
            observed_abs = abs(float(result.at[index, "mean_contrast"]))
            result.at[index, "max_stat_p"] = (
                1 + int((maxima >= observed_abs).sum())
            ) / (1 + len(maxima))
    return result
