from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import within_question_auc_table
from .common import sha256_file, write_json_atomic
from .vertical_profiles import PROFILE_COLUMNS


PROFILE_KEYS = [
    "model",
    "checkpoint",
    "global_step",
    "training_progress",
    "stage",
    "representation",
    "layer_index",
    "relative_depth",
]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _valid_metric_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in frame:
        raise ValueError(f"missing profile metric: {metric}")
    coverage = f"profile_coverage_count_{metric}"
    result = frame.loc[frame[metric].notna()].copy()
    if coverage in result:
        result = result.loc[result[coverage] > 0]
    return result


def summarize_policy_profiles(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = _valid_metric_rows(frame, metric)
    question = (
        data.groupby(PROFILE_KEYS + ["question_id"], dropna=False, as_index=False)
        .agg(profile_mean=(metric, "mean"), n_rollouts=("rollout_id", "nunique"))
    )
    return (
        question.groupby(PROFILE_KEYS, dropna=False, as_index=False)
        .agg(
            profile_mean=("profile_mean", "mean"),
            profile_std=("profile_mean", lambda values: float(np.std(values, ddof=0))),
            n_questions=("question_id", "nunique"),
            n_rollouts=("n_rollouts", "sum"),
        )
        .sort_values(["global_step", "stage", "representation", "layer_index"])
        .reset_index(drop=True)
    )


def summarize_outcome_profiles(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = _valid_metric_rows(frame, metric)
    question_label = (
        data.groupby(PROFILE_KEYS + ["question_id", "is_correct"], dropna=False, as_index=False)
        .agg(profile_mean=(metric, "mean"), n_rollouts=("rollout_id", "nunique"))
    )
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
        return pd.DataFrame(
            columns=PROFILE_KEYS
            + ["correct_mean", "wrong_mean", "correct_minus_wrong", "n_mixed_questions", "n_rollouts"]
        )
    mixed = values.loc[:, ["wrong", "correct"]].dropna()
    mixed_counts = counts.reindex(mixed.index).fillna(0).sum(axis=1)
    mixed = mixed.assign(
        correct_minus_wrong=mixed["correct"] - mixed["wrong"],
        n_rollouts=mixed_counts,
    ).reset_index()
    if mixed.empty:
        return pd.DataFrame(
            columns=PROFILE_KEYS
            + ["correct_mean", "wrong_mean", "correct_minus_wrong", "n_mixed_questions", "n_rollouts"]
        )
    return (
        mixed.groupby(PROFILE_KEYS, dropna=False, as_index=False)
        .agg(
            correct_mean=("correct", "mean"),
            wrong_mean=("wrong", "mean"),
            correct_minus_wrong=("correct_minus_wrong", "mean"),
            n_mixed_questions=("question_id", "nunique"),
            n_rollouts=("n_rollouts", "sum"),
        )
        .sort_values(["global_step", "stage", "representation", "layer_index"])
        .reset_index(drop=True)
    )


def summarize_layer_auc(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = _valid_metric_rows(frame, metric)
    rows: list[dict[str, Any]] = []
    for keys, group in data.groupby(PROFILE_KEYS, dropna=False, sort=False):
        table = within_question_auc_table(group, score_column=metric)
        base = dict(zip(PROFILE_KEYS, keys, strict=True))
        rows.append(
            {
                **base,
                "question_equal_auc": float(table["auc"].mean()) if len(table) else np.nan,
                "pair_weighted_auc": float(
                    np.average(table["auc"], weights=table["pair_count"])
                )
                if len(table)
                else np.nan,
                "n_mixed_questions": int(len(table)),
                "n_pairs": int(table["pair_count"].sum()) if len(table) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["global_step", "stage", "representation", "layer_index"]
    ).reset_index(drop=True)


def summarize_depth_change(
    policy: pd.DataFrame,
    *,
    metric: str,
    base_step: int = 0,
) -> pd.DataFrame:
    group_keys = ["model", "representation", "stage"]
    rows: list[dict[str, Any]] = []
    for keys, family in policy.groupby(group_keys, dropna=False, sort=False):
        base = family.loc[family["global_step"] == base_step, ["layer_index", "profile_mean"]]
        if base.empty:
            raise ValueError(f"missing base profile for {dict(zip(group_keys, keys, strict=True))}")
        base = base.rename(columns={"profile_mean": "base_profile_mean"})
        for (checkpoint, global_step, training_progress), current in family.groupby(
            ["checkpoint", "global_step", "training_progress"],
            dropna=False,
            sort=False,
        ):
            merged = current.merge(base, on="layer_index", how="inner", validate="one_to_one")
            relative_depth = merged["relative_depth"].to_numpy(float)
            mass = np.abs(
                merged["profile_mean"].to_numpy(float)
                - merged["base_profile_mean"].to_numpy(float)
            )
            mass = np.nan_to_num(mass, nan=0.0)
            total = float(mass.sum())
            if total > 0:
                peak_depth = float(relative_depth[int(np.argmax(mass))])
                depth_center = float((relative_depth * mass).sum() / total)
                depth_spread = float(
                    np.sqrt((((relative_depth - depth_center) ** 2) * mass).sum() / total)
                )
                early = float(mass[relative_depth <= 1 / 3].sum() / total)
                middle = float(
                    mass[(relative_depth > 1 / 3) & (relative_depth <= 2 / 3)].sum()
                    / total
                )
                late = float(mass[relative_depth > 2 / 3].sum() / total)
            else:
                peak_depth = np.nan
                depth_center = np.nan
                depth_spread = np.nan
                early = middle = late = 0.0
            rows.append(
                {
                    **dict(zip(group_keys, keys, strict=True)),
                    "metric": metric,
                    "checkpoint": checkpoint,
                    "global_step": int(global_step),
                    "training_progress": float(training_progress),
                    "depth_mass": total,
                    "peak_depth": peak_depth,
                    "depth_center": depth_center,
                    "depth_spread": depth_spread,
                    "early_mass": early,
                    "middle_mass": middle,
                    "late_mass": late,
                }
            )
    return pd.DataFrame(rows)


def _accuracy_by_step(generation_dir: Path, steps: list[int]) -> dict[int, float]:
    result = {}
    for step in steps:
        with (generation_dir / f"{step}.jsonl").open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            raise ValueError(f"empty generation file for step={step}")
        result[step] = float(np.mean([bool(row.get("acc", False)) for row in rows]))
    return result


def _heatmap(
    frame: pd.DataFrame,
    *,
    value: str,
    steps: list[int],
    title: str,
    path: Path,
    color_label: str,
) -> None:
    pivot = frame.pivot(index="layer_index", columns="global_step", values=value)
    pivot = pivot.reindex(columns=steps).sort_index()
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    image = axis.imshow(np.ma.masked_invalid(pivot.to_numpy(float)), origin="lower", aspect="auto")
    axis.set_xticks(range(len(steps)), labels=[str(step) for step in steps])
    axis.set_yticks(range(len(pivot.index)), labels=[str(value) for value in pivot.index])
    axis.set_xlabel("global step")
    axis.set_ylabel("hidden-state layer")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=color_label)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _profile_curves(
    frame: pd.DataFrame,
    *,
    value: str,
    selected_steps: list[int],
    title: str,
    path: Path,
    ylabel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for step in selected_steps:
        group = frame.loc[frame["global_step"] == step].sort_values("relative_depth")
        if len(group):
            axis.plot(group["relative_depth"], group[value], marker="o", markersize=2, label=str(step))
    axis.set_xlabel("relative depth")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(title="step")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_family(
    *,
    metric: str,
    representation: str,
    stage: int,
    policy: pd.DataFrame,
    outcome: pd.DataFrame,
    auc: pd.DataFrame,
    steps: list[int],
    selected_steps: list[int],
    figure_dir: Path,
) -> list[Path]:
    suffix = f"{metric}__{representation}__B{stage + 1}"
    policy_group = policy.loc[
        (policy["metric"] == metric)
        & (policy["representation"] == representation)
        & (policy["stage"] == stage)
    ]
    outcome_group = outcome.loc[
        (outcome["metric"] == metric)
        & (outcome["representation"] == representation)
        & (outcome["stage"] == stage)
    ]
    auc_group = auc.loc[
        (auc["metric"] == metric)
        & (auc["representation"] == representation)
        & (auc["stage"] == stage)
    ]
    paths = [
        figure_dir / f"policy_heatmap__{suffix}.png",
        figure_dir / f"selected_profiles__{suffix}.png",
        figure_dir / f"correct_minus_wrong__{suffix}.png",
        figure_dir / f"auc_heatmap__{suffix}.png",
    ]
    _heatmap(
        policy_group,
        value="profile_mean",
        steps=steps,
        title=f"Policy profile | {metric} | {representation} | B{stage + 1}",
        path=paths[0],
        color_label="question-equal mean",
    )
    _profile_curves(
        policy_group,
        value="profile_mean",
        selected_steps=selected_steps,
        title=f"Selected checkpoints | {metric} | {representation} | B{stage + 1}",
        path=paths[1],
        ylabel="question-equal mean",
    )
    _profile_curves(
        outcome_group,
        value="correct_minus_wrong",
        selected_steps=steps,
        title=f"Correct - wrong | {metric} | {representation} | B{stage + 1}",
        path=paths[2],
        ylabel="correct - wrong",
    )
    _heatmap(
        auc_group,
        value="question_equal_auc",
        steps=steps,
        title=f"Within-question AUROC | {metric} | {representation} | B{stage + 1}",
        path=paths[3],
        color_label="AUROC",
    )
    return paths


def analyze_profiles(
    *,
    profiles_dir: Path,
    generation_dir: Path,
    output_dir: Path,
    expected_steps: list[int],
    expected_questions: int,
    metrics: tuple[str, ...] = PROFILE_COLUMNS,
    approval_output: Path | None = None,
) -> dict[str, Any]:
    expected_steps = [int(step) for step in expected_steps]
    profile_paths = [profiles_dir / f"layer_profiles_step{step:03d}.parquet" for step in expected_steps]
    missing = [str(path) for path in profile_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing profile partitions: {missing}")
    profiles = pd.concat([pd.read_parquet(path) for path in profile_paths], ignore_index=True)
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
        depth_frames.append(summarize_depth_change(policy, metric=metric, base_step=0))
    policy_all = pd.concat(policy_frames, ignore_index=True)
    outcome_all = pd.concat(outcome_frames, ignore_index=True)
    auc_all = pd.concat(auc_frames, ignore_index=True)
    depth_all = pd.concat(depth_frames, ignore_index=True)
    _atomic_parquet(policy_all, output_dir / "policy_profiles.parquet")
    _atomic_parquet(outcome_all, output_dir / "outcome_profiles.parquet")
    _atomic_parquet(auc_all, output_dir / "layer_auc.parquet")
    _atomic_parquet(depth_all, output_dir / "depth_summaries.parquet")

    accuracy = _accuracy_by_step(generation_dir, expected_steps)
    nonzero = [step for step in expected_steps if step > 0]
    early = min(nonzero) if nonzero else 0
    best = max(expected_steps, key=lambda step: (accuracy[step], step))
    selected_steps = sorted({0, early, best, max(expected_steps)})
    figures: list[Path] = []
    combinations = (
        profiles[["representation", "stage"]]
        .drop_duplicates()
        .sort_values(["representation", "stage"])
        .itertuples(index=False, name=None)
    )
    combinations = list(combinations)
    for metric in metrics:
        for representation, stage in combinations:
            figures.extend(
                _plot_family(
                    metric=metric,
                    representation=str(representation),
                    stage=int(stage),
                    policy=policy_all,
                    outcome=outcome_all,
                    auc=auc_all,
                    steps=expected_steps,
                    selected_steps=selected_steps,
                    figure_dir=figure_dir,
                )
            )

    questions_by_step = {
        int(step): int(group["question_id"].nunique())
        for step, group in profiles.groupby("global_step")
    }
    extraction_audits = []
    for step in expected_steps:
        path = profiles_dir / f"extraction_audit_step{step:03d}.json"
        extraction_audits.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {})
    nonblank = [path for path in figures if path.is_file() and path.stat().st_size > 10_000]
    expected_figure_count = len(metrics) * len(combinations) * 4
    gates = {
        "checkpoint_steps": sorted(profiles["global_step"].unique().astype(int).tolist())
        == sorted(expected_steps),
        "question_counts": all(
            questions_by_step.get(step) == expected_questions for step in expected_steps
        ),
        "extraction_audits": all(audit.get("passed") for audit in extraction_audits),
        "figure_count": len(figures) == expected_figure_count,
        "figures_nonblank": len(nonblank) == len(figures),
    }
    audit = {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "expected_steps": expected_steps,
        "questions_by_step": questions_by_step,
        "metrics": list(metrics),
        "representations": sorted(profiles["representation"].unique().tolist()),
        "stages": sorted(profiles["stage"].unique().astype(int).tolist()),
        "accuracy_by_step": {str(key): value for key, value in accuracy.items()},
        "selected_steps": selected_steps,
        "figure_count": len(figures),
        "nonblank_figure_count": len(nonblank),
        "profile_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in profile_paths
        ],
    }
    write_json_atomic(output_dir / "analysis_audit.json", audit)
    if approval_output is not None:
        extraction_seconds = float(
            sum(float(audit_row.get("elapsed_seconds", 0.0)) for audit_row in extraction_audits)
        )
        approval = {
            "status": "passed"
            if audit["passed"] and expected_steps == [0, 50, 250] and expected_questions == 16
            else "failed",
            "analysis_passed": audit["passed"],
            "expected_steps": expected_steps,
            "expected_questions": expected_questions,
            "smoke_extraction_seconds": extraction_seconds,
            "projected_formal_hours": extraction_seconds * 22528 / 384 / 3600,
            "projection_method": "linear from 384 smoke rollouts to 22528 formal rollouts",
        }
        write_json_atomic(approval_output, approval)
    if not audit["passed"]:
        raise RuntimeError(f"layer-profile analysis audit failed: {audit}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Q3 layer-resolved hidden profiles.")
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-steps",
        type=lambda value: [int(item) for item in value.split(",")],
        required=True,
    )
    parser.add_argument("--expected-questions", type=int, required=True)
    parser.add_argument("--approval-output", type=Path)
    args = parser.parse_args()
    audit = analyze_profiles(
        profiles_dir=args.profiles_dir,
        generation_dir=args.generation_dir,
        output_dir=args.output_dir,
        expected_steps=args.expected_steps,
        expected_questions=args.expected_questions,
        approval_output=args.approval_output,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
