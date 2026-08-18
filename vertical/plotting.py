from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .audit import atomic_parquet, sha256_file, write_json_atomic
from .depth import summarize_depth_change
from .profiles import METRIC_REGISTRY
from .statistics import (
    cluster_signflip,
    fit_nuisance_controlled_effects,
    paired_condition_delta,
    summarize_layer_auc,
    summarize_outcome_profiles,
    summarize_policy_profiles,
    standardize_within_family_base,
)


@dataclass(frozen=True)
class AnalysisAudit:
    passed: bool
    gates: dict[str, bool]
    metrics: tuple[str, ...]
    figure_count: int
    nonblank_figure_count: int
    input_sha256: str
    table_sha256: dict[str, str]
    metric_status: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": self.gates,
            "metrics": list(self.metrics),
            "figure_count": self.figure_count,
            "nonblank_figure_count": self.nonblank_figure_count,
            "input_sha256": self.input_sha256,
            "table_sha256": self.table_sha256,
            "metric_status": self.metric_status,
        }


def _state_label(frame: pd.DataFrame) -> pd.Series:
    condition = frame["condition"].fillna("unknown").astype(str)
    checkpoint = frame["checkpoint"].fillna("").astype(str)
    return np.where(checkpoint.str.len() > 0, condition + ":" + checkpoint, condition)


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _heatmap(
    frame: pd.DataFrame,
    *,
    value: str,
    title: str,
    color_label: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    if frame.empty or value not in frame or not frame[value].notna().any():
        axis.text(0.5, 0.5, "No supported values", ha="center", va="center")
        axis.set_axis_off()
        axis.set_title(title)
        _save(figure, path)
        return
    data = frame.copy()
    data["state_label"] = _state_label(data)
    reduced = (
        data.groupby(["layer_index", "state_label"], as_index=False, dropna=False)[value]
        .mean()
    )
    pivot = reduced.pivot(index="layer_index", columns="state_label", values=value)
    pivot = pivot.sort_index()
    image = axis.imshow(
        np.ma.masked_invalid(pivot.to_numpy(float)),
        origin="lower",
        aspect="auto",
    )
    axis.set_xticks(range(len(pivot.columns)), labels=[str(item) for item in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), labels=[str(item) for item in pivot.index])
    axis.set_xlabel("condition / checkpoint")
    axis.set_ylabel("hidden-state layer")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=color_label)
    _save(figure, path)


def plot_policy_heatmap(frame: pd.DataFrame, metric: str, path: Path) -> None:
    _heatmap(
        frame,
        value="profile_mean",
        title=f"Policy profile | {metric}",
        color_label="question-equal mean",
        path=Path(path),
    )


def plot_selected_profiles(
    frame: pd.DataFrame,
    metric: str,
    selected_conditions: Sequence[str],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    plotted = False
    for condition in selected_conditions:
        group = frame.loc[frame["condition"] == condition]
        if group.empty:
            continue
        curve = (
            group.groupby("relative_depth", as_index=False, dropna=False)["profile_mean"]
            .mean()
            .sort_values("relative_depth")
        )
        axis.plot(
            curve["relative_depth"],
            curve["profile_mean"],
            marker="o",
            markersize=3,
            label=str(condition),
        )
        plotted = True
    if plotted:
        axis.legend(title="condition")
        axis.grid(alpha=0.2)
    else:
        axis.text(0.5, 0.5, "No supported values", ha="center", va="center")
    axis.set_xlabel("relative depth")
    axis.set_ylabel("question-equal mean")
    axis.set_title(f"Selected conditions | {metric}")
    _save(figure, Path(path))


def plot_correct_minus_wrong(frame: pd.DataFrame, metric: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    plotted = False
    if not frame.empty:
        for condition, group in frame.groupby("condition", dropna=False):
            curve = (
                group.groupby("relative_depth", as_index=False, dropna=False)[
                    "correct_minus_wrong"
                ]
                .mean()
                .sort_values("relative_depth")
            )
            axis.plot(
                curve["relative_depth"],
                curve["correct_minus_wrong"],
                marker="o",
                markersize=3,
                label=str(condition),
            )
            plotted = True
    if plotted:
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.legend(title="condition")
        axis.grid(alpha=0.2)
    else:
        axis.text(0.5, 0.5, "No mixed-outcome questions", ha="center", va="center")
    axis.set_xlabel("relative depth")
    axis.set_ylabel("correct - wrong")
    axis.set_title(f"Outcome separation | {metric}")
    _save(figure, Path(path))


def plot_auc_heatmap(frame: pd.DataFrame, metric: str, path: Path) -> None:
    _heatmap(
        frame,
        value="question_equal_auc",
        title=f"Within-question AUROC | {metric}",
        color_label="AUROC",
        path=Path(path),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def render_v01_report(
    profiles_path: Path,
    output_dir: Path,
    metrics: Sequence[str],
) -> AnalysisAudit:
    profiles_path = Path(profiles_path)
    output_dir = Path(output_dir)
    if not profiles_path.is_file():
        raise FileNotFoundError(profiles_path)
    if not metrics:
        raise ValueError("at least one metric is required")
    profiles = pd.read_parquet(profiles_path)
    if profiles.empty:
        raise ValueError("profile input is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    policy_frames = []
    outcome_frames = []
    auc_frames = []
    depth_frames = []
    for metric in metrics:
        policy = summarize_policy_profiles(profiles, metric).assign(metric=metric)
        outcome = summarize_outcome_profiles(profiles, metric).assign(metric=metric)
        auc = summarize_layer_auc(profiles, metric).assign(metric=metric)
        policy_frames.append(policy)
        outcome_frames.append(outcome)
        auc_frames.append(auc)
        depth_frames.append(summarize_depth_change(policy, metric=metric))
    policy_all = pd.concat(policy_frames, ignore_index=True)
    outcome_all = pd.concat(outcome_frames, ignore_index=True)
    auc_all = pd.concat(auc_frames, ignore_index=True)
    depth_all = pd.concat(depth_frames, ignore_index=True)

    table_paths = {
        "policy_profiles": output_dir / "policy_profiles.parquet",
        "outcome_profiles": output_dir / "outcome_profiles.parquet",
        "layer_auc": output_dir / "layer_auc.parquet",
        "depth_summaries": output_dir / "depth_summaries.parquet",
    }
    for frame, path in (
        (policy_all, table_paths["policy_profiles"]),
        (outcome_all, table_paths["outcome_profiles"]),
        (auc_all, table_paths["layer_auc"]),
        (depth_all, table_paths["depth_summaries"]),
    ):
        atomic_parquet(frame, path)

    combinations = list(
        profiles[["model_family", "representation", "stage"]]
        .drop_duplicates()
        .sort_values(["model_family", "representation", "stage"])
        .itertuples(index=False, name=None)
    )
    figures: list[Path] = []
    for metric in metrics:
        for family, representation, stage in combinations:
            selection = (
                (policy_all["metric"] == metric)
                & (policy_all["model_family"] == family)
                & (policy_all["representation"] == representation)
                & (policy_all["stage"] == stage)
            )
            policy_group = policy_all.loc[selection]
            outcome_group = outcome_all.loc[
                (outcome_all["metric"] == metric)
                & (outcome_all["model_family"] == family)
                & (outcome_all["representation"] == representation)
                & (outcome_all["stage"] == stage)
            ]
            auc_group = auc_all.loc[
                (auc_all["metric"] == metric)
                & (auc_all["model_family"] == family)
                & (auc_all["representation"] == representation)
                & (auc_all["stage"] == stage)
            ]
            suffix = _slug(f"{family}__{metric}__{representation}__B{int(stage) + 1}")
            paths = [
                figure_dir / f"policy_heatmap__{suffix}.png",
                figure_dir / f"selected_profiles__{suffix}.png",
                figure_dir / f"correct_minus_wrong__{suffix}.png",
                figure_dir / f"auc_heatmap__{suffix}.png",
            ]
            plot_policy_heatmap(policy_group, metric, paths[0])
            plot_selected_profiles(
                policy_group,
                metric,
                sorted(policy_group["condition"].dropna().astype(str).unique()),
                paths[1],
            )
            plot_correct_minus_wrong(outcome_group, metric, paths[2])
            plot_auc_heatmap(auc_group, metric, paths[3])
            figures.extend(paths)

    expected_figure_count = len(metrics) * len(
        profiles[["model_family", "representation", "stage"]].drop_duplicates()
    ) * 4
    nonblank = [path for path in figures if path.is_file() and path.stat().st_size > 10_000]
    gates = {
        "tables_nonempty": len(policy_all) > 0 and len(auc_all) > 0 and len(depth_all) > 0,
        "figure_count": len(figures) == expected_figure_count,
        "figures_nonblank": len(nonblank) == len(figures),
    }
    audit = AnalysisAudit(
        passed=bool(all(gates.values())),
        gates=gates,
        metrics=tuple(metrics),
        figure_count=len(figures),
        nonblank_figure_count=len(nonblank),
        input_sha256=sha256_file(profiles_path),
        table_sha256={name: sha256_file(path) for name, path in table_paths.items()},
        metric_status={
            metric: str(METRIC_REGISTRY.get(metric, {}).get("status", "unregistered"))
            for metric in metrics
        },
    )
    write_json_atomic(audit.to_dict(), output_dir / "analysis_audit.json")
    if not audit.passed:
        raise RuntimeError(f"analysis audit failed: {audit.to_dict()}")
    return audit


INFERENCE_GROUP_KEYS = [
    "model_family",
    "model_name",
    "condition",
    "checkpoint",
    "global_step",
    "training_progress",
    "stage",
    "representation",
]
INFERENCE_CONTROLS = (
    "response_token_count",
    "trajectory_point_count",
    "policy_entropy",
)


def _question_bootstrap_table(
    profiles: pd.DataFrame,
    metrics: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(metrics):
        if metric not in profiles:
            continue
        data = profiles.copy()
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data = data.loc[np.isfinite(data[metric])]
        coverage = f"profile_coverage_count_{metric}"
        if coverage in data:
            data = data.loc[data[coverage] > 0]
        if data.empty:
            continue
        question = (
            data.groupby(INFERENCE_GROUP_KEYS + ["question_id"], dropna=False, as_index=False)
            .agg(question_mean=(metric, "mean"))
        )
        group_index = 0
        for keys, group in question.groupby(INFERENCE_GROUP_KEYS, dropna=False, sort=True):
            values = group["question_mean"].to_numpy(float)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            rng = np.random.default_rng(seed + metric_index * 100_000 + group_index)
            draw_indices = rng.integers(0, len(values), size=(n_boot, len(values)))
            estimates = values[draw_indices].mean(axis=1)
            rows.append(
                {
                    **dict(zip(INFERENCE_GROUP_KEYS, keys, strict=True)),
                    "metric": metric,
                    "estimate": float(values.mean()),
                    "ci_low": float(np.quantile(estimates, 0.025)),
                    "ci_high": float(np.quantile(estimates, 0.975)),
                    "n_questions": int(len(values)),
                    "n_boot": int(n_boot),
                    "seed": int(seed + metric_index * 100_000 + group_index),
                    "resampling_unit": "question_id",
                }
            )
            group_index += 1
    return pd.DataFrame(rows)


def _cluster_table(
    profiles: pd.DataFrame,
    metrics: Sequence[str],
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(metrics):
        if metric not in profiles or "is_correct" not in profiles:
            continue
        data = profiles.copy()
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data = data.loc[data["is_correct"].notna() & np.isfinite(data[metric])]
        coverage = f"profile_coverage_count_{metric}"
        if coverage in data:
            data = data.loc[data[coverage] > 0]
        if data.empty:
            continue
        for group_index, (keys, group) in enumerate(
            data.groupby(INFERENCE_GROUP_KEYS, dropna=False, sort=True)
        ):
            rollout = (
                group.groupby(["question_id", "is_correct", "layer_index"], dropna=False)
                .agg(value=(metric, "mean"))
                .reset_index()
            )
            correct = rollout.loc[rollout["is_correct"].astype(bool)]
            wrong = rollout.loc[~rollout["is_correct"].astype(bool)]
            correct = correct.groupby(["question_id", "layer_index"], as_index=False)[
                "value"
            ].mean()
            wrong = wrong.groupby(["question_id", "layer_index"], as_index=False)[
                "value"
            ].mean()
            paired = correct.merge(
                wrong,
                on=["question_id", "layer_index"],
                how="inner",
                suffixes=("_correct", "_wrong"),
            )
            if paired.empty:
                rows.append(
                    {
                        **dict(zip(INFERENCE_GROUP_KEYS, keys, strict=True)),
                        "metric": metric,
                        "cluster_id": -1,
                        "coverage_ok": False,
                        "unsupported_reason": "no mixed-outcome questions",
                        "n_questions": 0,
                        "n_permutations": int(n_permutations),
                        "seed": int(seed),
                        "correction_method": "question_sign_flip_max_cluster_mass",
                    }
                )
                continue
            layers = sorted(paired["layer_index"].unique())
            matrix = (
                paired.pivot(index="question_id", columns="layer_index", values="value_correct")
                - paired.pivot(index="question_id", columns="layer_index", values="value_wrong")
            ).reindex(columns=layers)
            matrix = matrix.dropna(how="all")
            if len(matrix) < 2:
                rows.append(
                    {
                        **dict(zip(INFERENCE_GROUP_KEYS, keys, strict=True)),
                        "metric": metric,
                        "cluster_id": -1,
                        "coverage_ok": False,
                        "unsupported_reason": "fewer than two mixed-outcome questions",
                        "n_questions": int(len(matrix)),
                        "n_permutations": int(n_permutations),
                        "seed": int(seed),
                        "correction_method": "question_sign_flip_max_cluster_mass",
                    }
                )
                continue
            clusters = cluster_signflip(
                matrix.to_numpy(float),
                n_permutations=n_permutations,
                seed=seed + metric_index * 100_000 + group_index,
            )
            if clusters.empty:
                rows.append(
                    {
                        **dict(zip(INFERENCE_GROUP_KEYS, keys, strict=True)),
                        "metric": metric,
                        "cluster_id": -1,
                        "coverage_ok": True,
                        "unsupported_reason": "no supra-threshold clusters",
                        "n_questions": int(len(matrix)),
                        "n_permutations": int(n_permutations),
                        "seed": int(seed),
                        "correction_method": "question_sign_flip_max_cluster_mass",
                    }
                )
                continue
            for cluster in clusters.to_dict(orient="records"):
                cluster.update(
                    {
                        **dict(zip(INFERENCE_GROUP_KEYS, keys, strict=True)),
                        "metric": metric,
                        "coverage_ok": True,
                        "unsupported_reason": "",
                        "relative_start_depth": float(
                            layers[int(cluster["start_index"])] / max(layers)
                        )
                        if max(layers)
                        else 0.0,
                        "relative_end_depth": float(
                            layers[int(cluster["end_index"])] / max(layers)
                        )
                        if max(layers)
                        else 0.0,
                    }
                )
                rows.append(cluster)
    return pd.DataFrame(rows)


def _nuisance_table(
    profiles: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = INFERENCE_GROUP_KEYS + ["layer_index", "relative_depth"]
    for metric in metrics:
        if metric not in profiles:
            continue
        data = profiles.copy()
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        for keys, group in data.groupby(group_keys, dropna=False, sort=True):
            try:
                result = fit_nuisance_controlled_effects(
                    group,
                    metric,
                    INFERENCE_CONTROLS,
                )
            except ValueError as exc:
                result = pd.DataFrame(
                    [
                        {
                            "term": "is_correct",
                            "coverage_ok": False,
                            "unsupported_reason": str(exc),
                            "n_questions": 0,
                        }
                    ]
                )
            for row in result.to_dict(orient="records"):
                rows.append(
                    {
                        **dict(zip(group_keys, keys, strict=True)),
                        "metric": metric,
                        **row,
                    }
                )
    return pd.DataFrame(rows)


def _paired_table(profiles: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    pair_keys = [
        "model_family",
        "question_id",
        "rollout_id",
        "representation",
        "stage",
        "layer_index",
    ]
    for metric in metrics:
        if metric not in profiles:
            continue
        for family, family_frame in profiles.groupby("model_family", dropna=False, sort=True):
            conditions = sorted(family_frame["condition"].dropna().unique())
            if "base" not in conditions:
                continue
            for target in [condition for condition in conditions if condition != "base"]:
                selected = family_frame[pair_keys + ["condition", metric]].copy()
                try:
                    paired = paired_condition_delta(
                        selected,
                        "base",
                        target,
                        pair_keys,
                    )
                    paired["model_family"] = family
                    paired["metric"] = metric
                    paired["base_value"] = paired[f"{metric}_base"]
                    paired["target_value"] = paired[f"{metric}_target"]
                    paired["value_delta"] = paired[f"{metric}_delta"]
                    rows.append(paired)
                except ValueError as exc:
                    rows.append(
                        pd.DataFrame(
                            [
                                {
                                    "model_family": family,
                                    "metric": metric,
                                    "target_condition": target,
                                    "pairing_status": "unavailable",
                                    "unsupported_reason": str(exc),
                                    "n_pairs": 0,
                                }
                            ]
                        )
                    )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _standardized_table(profiles: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for metric in metrics:
        if metric not in profiles:
            continue
        try:
            standardized = standardize_within_family_base(profiles, metric)
        except ValueError:
            continue
        standardized = standardized.copy()
        standardized["metric"] = metric
        standardized["base_standardized_value"] = standardized[
            f"base_standardized_{metric}"
        ]
        keep = [
            column
            for column in (
                "record_id",
                "model_family",
                "model_name",
                "condition",
                "question_id",
                "rollout_id",
                "is_correct",
                "representation",
                "stage",
                "layer_index",
                "relative_depth",
                "metric",
                metric,
                f"base_mean_{metric}",
                f"base_std_{metric}",
                "base_standardized_value",
                "standardization_ok",
                "standardization_scope",
            )
            if column in standardized
        ]
        rows.append(standardized[keep].rename(columns={metric: "value"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def render_inference_report(
    profiles_path: Path,
    output_dir: Path,
    metrics: Sequence[str],
    *,
    n_boot: int = 200,
    n_permutations: int = 199,
    seed: int = 20260818,
) -> dict[str, Any]:
    if n_boot < 1 or n_permutations < 1:
        raise ValueError("inference resampling counts must be positive")
    profiles = pd.read_parquet(profiles_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "question_bootstrap": _question_bootstrap_table(
            profiles, metrics, n_boot=n_boot, seed=seed
        ),
        "cluster_permutation": _cluster_table(
            profiles, metrics, n_permutations=n_permutations, seed=seed
        ),
        "nuisance_effects": _nuisance_table(profiles, metrics),
        "paired_condition_deltas": _paired_table(profiles, metrics),
        "base_standardized_profiles": _standardized_table(profiles, metrics),
    }
    paths: dict[str, str] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.parquet"
        atomic_parquet(table, path)
        paths[name] = str(path)
    audit = {
        "passed": True,
        "metrics": list(metrics),
        "n_boot": int(n_boot),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "question_bootstrap_rows": int(len(tables["question_bootstrap"])),
        "cluster_rows": int(len(tables["cluster_permutation"])),
        "nuisance_rows": int(len(tables["nuisance_effects"])),
        "paired_rows": int(len(tables["paired_condition_deltas"])),
        "standardized_rows": int(len(tables["base_standardized_profiles"])),
        "tables": paths,
    }
    write_json_atomic(audit, output_dir / "inference_audit.json")
    return audit
