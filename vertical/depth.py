from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FAMILY_KEYS = ["model_family", "representation", "stage"]
STATE_KEYS = [
    "model_name",
    "condition",
    "checkpoint",
    "global_step",
    "training_progress",
]


def _metric_policy(policy: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = policy.copy()
    if "metric" in data:
        data = data.loc[data["metric"] == metric]
    required = set(FAMILY_KEYS + STATE_KEYS + ["layer_index", "relative_depth", "profile_mean"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"policy frame is missing columns: {missing}")
    return data


def _base_profile(family: pd.DataFrame, base_condition: str) -> pd.DataFrame:
    base = family.loc[family["condition"] == base_condition]
    if base.empty:
        raise ValueError(
            f"missing Base condition={base_condition} for model_family={family['model_family'].iloc[0]}"
        )
    return (
        base.groupby(["layer_index", "relative_depth"], as_index=False, dropna=False)
        .agg(base_profile_mean=("profile_mean", "mean"))
        .sort_values("layer_index")
    )


def _depth_summary(relative_depth: np.ndarray, delta: np.ndarray) -> dict[str, float]:
    mass = np.abs(np.asarray(delta, dtype=float))
    mass = np.nan_to_num(mass, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.asarray(relative_depth, dtype=float)
    total = float(mass.sum())
    if total <= 0:
        return {
            "depth_mass": 0.0,
            "peak_depth": np.nan,
            "depth_center": np.nan,
            "depth_spread": np.nan,
            "early_mass": 0.0,
            "middle_mass": 0.0,
            "late_mass": 0.0,
            "transition_depth": np.nan,
        }
    center = float((depth * mass).sum() / total)
    spread = float(np.sqrt((((depth - center) ** 2) * mass).sum() / total))
    transition = np.abs(np.diff(delta))
    transition_depth = (
        float(depth[int(np.nanargmax(transition)) + 1])
        if transition.size and np.isfinite(transition).any()
        else np.nan
    )
    return {
        "depth_mass": total,
        "peak_depth": float(depth[int(np.argmax(mass))]),
        "depth_center": center,
        "depth_spread": spread,
        "early_mass": float(mass[depth <= 1 / 3].sum() / total),
        "middle_mass": float(
            mass[(depth > 1 / 3) & (depth <= 2 / 3)].sum() / total
        ),
        "late_mass": float(mass[depth > 2 / 3].sum() / total),
        "transition_depth": transition_depth,
    }


def summarize_depth_change(
    policy: pd.DataFrame,
    *,
    metric: str,
    base_condition: str = "base",
) -> pd.DataFrame:
    data = _metric_policy(policy, metric)
    rows: list[dict[str, Any]] = []
    for family_keys, family in data.groupby(FAMILY_KEYS, dropna=False, sort=False):
        base = _base_profile(family, base_condition)
        for state_keys, current in family.groupby(STATE_KEYS, dropna=False, sort=False):
            current_profile = (
                current.groupby(["layer_index", "relative_depth"], as_index=False, dropna=False)
                .agg(profile_mean=("profile_mean", "mean"))
                .sort_values("layer_index")
            )
            merged = current_profile.merge(
                base,
                on=["layer_index", "relative_depth"],
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(base) or len(merged) != len(current_profile):
                raise ValueError("Base and condition profiles do not share identical layer grids")
            delta = merged["profile_mean"].to_numpy(float) - merged[
                "base_profile_mean"
            ].to_numpy(float)
            rows.append(
                {
                    **dict(zip(FAMILY_KEYS, family_keys, strict=True)),
                    **dict(zip(STATE_KEYS, state_keys, strict=True)),
                    "metric": metric,
                    **_depth_summary(merged["relative_depth"].to_numpy(float), delta),
                }
            )
    return pd.DataFrame(rows)


def analyze_condition_delta(
    policy: pd.DataFrame,
    *,
    base_condition: str,
    target_condition: str,
) -> pd.DataFrame:
    required = set(FAMILY_KEYS + ["condition", "layer_index", "relative_depth", "profile_mean"])
    missing = sorted(required - set(policy.columns))
    if missing:
        raise ValueError(f"policy frame is missing columns: {missing}")
    base = policy.loc[policy["condition"] == base_condition]
    target = policy.loc[policy["condition"] == target_condition]
    if base.empty or target.empty:
        raise ValueError("condition delta requires both Base and target conditions")
    layer_keys = FAMILY_KEYS + ["layer_index", "relative_depth"]
    base_mean = (
        base.groupby(layer_keys, dropna=False, as_index=False)
        .agg(base_profile_mean=("profile_mean", "mean"))
    )
    target_mean = (
        target.groupby(layer_keys, dropna=False, as_index=False)
        .agg(target_profile_mean=("profile_mean", "mean"))
    )
    merged = target_mean.merge(base_mean, on=layer_keys, how="inner", validate="one_to_one")
    if len(merged) != len(base_mean) or len(merged) != len(target_mean):
        raise ValueError("Base and target conditions do not share identical layer grids")
    merged["base_condition"] = base_condition
    merged["target_condition"] = target_condition
    merged["condition_delta"] = merged["target_profile_mean"] - merged["base_profile_mean"]
    merged["paired_layers"] = merged.groupby(FAMILY_KEYS, dropna=False)[
        "layer_index"
    ].transform("size")
    return merged.sort_values(FAMILY_KEYS + ["layer_index"]).reset_index(drop=True)

