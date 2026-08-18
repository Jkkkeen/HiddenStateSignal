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
    summarize_layer_auc,
    summarize_outcome_profiles,
    summarize_policy_profiles,
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
