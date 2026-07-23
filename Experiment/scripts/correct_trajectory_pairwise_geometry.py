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

