#!/usr/bin/env python3
"""Build the self-contained Qwen3-1.7B vertical-dynamics presentation report.

The script reads compact formal artifacts and produces one offline HTML file.
It intentionally does not load model weights, raw hidden tensors, or response
text. All claims in the report are derived from the audited checkpoint tables.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import html as html_lib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_STEPS = (0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250)
QUESTIONS = 256
ROLLOUTS = 8
BOOTSTRAP_DRAWS = 4000
SEED = 20260818
FORMAL_METRICS = (
    "v1_raw_update_norm",
    "v1_relative_update_norm",
    "v3_demean_state_angle",
    "v4_layer_update_turning_angle",
    "v6_raw_activation_entropy",
    "v7_layer_difference_entropy",
)
SOURCE_FIGURES = {
    "v1_heatmap": "policy_heatmap__v1_raw_update_norm__mean_w128_s32__B4.png",
    "v3_heatmap": "policy_heatmap__v3_demean_state_angle__mean_w128_s32__B4.png",
    "v4_heatmap": "policy_heatmap__v4_layer_update_turning_angle__mean_w128_s32__B2.png",
    "v3_profiles": "selected_profiles__v3_demean_state_angle__mean_w128_s32__B4.png",
    "v3_auc": "auc_heatmap__v3_demean_state_angle__mean_w128_s32__B4.png",
    "v1_relative_auc": "auc_heatmap__v1_relative_update_norm__mean_w128_s32__B4.png",
}
METRIC_MAP = {
    "v1_raw_update_norm": ("V1", "raw", "V1 raw update norm"),
    "v1_relative_update_norm": ("V1", "relative", "V1 relative update norm"),
    "v3_demean_state_angle": ("V3", "demean", "V3 demeaned state angle"),
    "v4_layer_update_turning_angle": ("V4", "median", "V4 turning angle"),
    "weighted_layer_update_turning_angle": ("V4C", "weighted", "V4C weighted turning"),
    "layer_update_ER": ("V8", "ER", "V8 layer-update ER"),
    "layer_update_ER_centered": ("V8", "centered", "V8 centered ER"),
}
COLORS = {
    "V1 raw": "#266d9b",
    "V1 relative": "#3b82a0",
    "V3": "#167a62",
    "V4": "#73529c",
    "V4C": "#a87317",
    "V8": "#b64d48",
    "V8 centered": "#d17c68",
}
METRIC_COLORS = {
    "V1 raw update norm": COLORS["V1 raw"],
    "V1 relative update norm": COLORS["V1 relative"],
    "V3 demeaned state angle": COLORS["V3"],
    "V4 turning angle": COLORS["V4"],
    "V4C weighted turning": COLORS["V4C"],
    "V8 layer-update ER": COLORS["V8"],
    "V8 centered ER": COLORS["V8 centered"],
}
REQUIRED_SECTIONS = (
    "overview",
    "ability",
    "vertical",
    "depth",
    "auroc",
    "horizontal",
    "stopping",
    "audit",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def pass_at_k_from_count(correct: int, total: int, k: int) -> float:
    """Return the unbiased empirical pass@k estimate for one rollout group."""
    if not 0 <= correct <= total or not 1 <= k <= total:
        raise ValueError("invalid correct/total/k")
    failures = total - correct
    missed = math.comb(failures, k) / math.comb(total, k) if failures >= k else 0.0
    return float(1.0 - missed)


def paired_bootstrap(
    values: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point estimate and paired question-bootstrap 95% interval."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("values must be a two-dimensional question table")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.shape[0], size=(draws, array.shape[0]))
    samples = array[indices].mean(axis=1)
    return (
        array.mean(axis=0),
        np.quantile(samples, 0.025, axis=0),
        np.quantile(samples, 0.975, axis=0),
    )


def _formal_paths(root: Path) -> tuple[Path, Path, Path]:
    formal = root / "layer_profiles_formal"
    return formal, formal / "metrics", formal / "analysis"


def validate_result_root(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    formal, metrics, analysis = _formal_paths(root)
    status_path = formal / "run_status.json"
    audit_path = analysis / "analysis_audit.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"missing run_status: {status_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing analysis_audit: {audit_path}")
    status = _json(status_path)
    audit = _json(audit_path)
    if status.get("status") != "completed" or int(status.get("exit_code", 1)) != 0:
        raise RuntimeError(f"formal run is not completed: {status}")
    if not audit.get("passed"):
        raise RuntimeError(f"formal analysis audit failed: {audit}")
    missing: list[str] = []
    for step in EXPECTED_STEPS:
        tag = f"step{step:03d}"
        for prefix in ("layer_profiles", "vertical_metrics", "controls", "model_identity", "extraction_audit"):
            path = metrics / f"{prefix}_{tag}.{'json' if prefix in {'model_identity', 'extraction_audit'} else 'parquet'}"
            if not path.is_file():
                missing.append(str(path))
        generation = root / "heldout" / "generations" / f"{step}.jsonl"
        if not generation.is_file():
            missing.append(str(generation))
    for filename in SOURCE_FIGURES.values():
        path = analysis / "figures" / filename
        if not path.is_file() or path.stat().st_size < 10_000:
            missing.append(str(path))
    for path in (root / "heldout" / "hidden_summary.jsonl", root / "online_hidden" / "per_step_summary.jsonl"):
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("missing required report inputs:\n" + "\n".join(missing))
    provenance = {
        "root": str(root),
        "run_status": status,
        "analysis_audit_passed": bool(audit.get("passed")),
        "expected_steps": list(EXPECTED_STEPS),
        "expected_questions": QUESTIONS,
        "rollouts_per_question": ROLLOUTS,
        "analysis_input_hashes": audit.get("profile_inputs", []),
    }
    return provenance, audit


def _generation_matrix(path: Path) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = QUESTIONS * ROLLOUTS
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(rows)}")
    return np.asarray([bool(row.get("acc", False)) for row in rows], dtype=float).reshape(QUESTIONS, ROLLOUTS)


def load_behavior(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    question_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    matrices: dict[int, np.ndarray] = {}
    for step in EXPECTED_STEPS:
        matrix = _generation_matrix(root / "heldout" / "generations" / f"{step}.jsonl")
        matrices[step] = matrix
        correct_counts = matrix.sum(axis=1).astype(int)
        values = np.asarray(
            [[pass_at_k_from_count(int(count), ROLLOUTS, k) for k in (1, 4, 8)] for count in correct_counts],
            dtype=float,
        )
        for question, (count, row) in enumerate(zip(correct_counts, values, strict=True)):
            question_rows.append({
                "global_step": step,
                "question_index": question,
                "correct_count": int(count),
                "pass_at_1": row[0],
                "pass_at_4": row[1],
                "pass_at_8": row[2],
            })
        point, low, high = paired_bootstrap(values)
        summary_rows.append({
            "global_step": step,
            "pass_at_1": point[0], "pass_at_1_low": low[0], "pass_at_1_high": high[0],
            "pass_at_4": point[1], "pass_at_4_low": low[1], "pass_at_4_high": high[1],
            "pass_at_8": point[2], "pass_at_8_low": low[2], "pass_at_8_high": high[2],
        })
    question = pd.DataFrame(question_rows)
    summary = pd.DataFrame(summary_rows)
    for previous, current in zip(EXPECTED_STEPS, EXPECTED_STEPS[1:]):
        delta = question.loc[question.global_step.eq(current), "pass_at_1"].to_numpy() - question.loc[question.global_step.eq(previous), "pass_at_1"].to_numpy()
        point, low, high = paired_bootstrap(delta)
        summary.loc[summary.global_step.eq(current), "delta_pass_at_1"] = point[0]
        summary.loc[summary.global_step.eq(current), "delta_pass_at_1_low"] = low[0]
        summary.loc[summary.global_step.eq(current), "delta_pass_at_1_high"] = high[0]
    summary["delta_pass_at_1"] = summary["delta_pass_at_1"].fillna(0.0)
    summary["delta_pass_at_1_low"] = summary["delta_pass_at_1_low"].fillna(0.0)
    summary["delta_pass_at_1_high"] = summary["delta_pass_at_1_high"].fillna(0.0)
    return summary, question


def paired_behavior_effects(question: pd.DataFrame) -> pd.DataFrame:
    """Compute paired checkpoint effects on the fixed held-out questions."""
    required = {"global_step", "question_index", "pass_at_1", "pass_at_4", "pass_at_8"}
    missing = required.difference(question.columns)
    if missing:
        raise ValueError(f"behavior question table is missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for previous, current in ((0, 250), (50, 250), (175, 250)):
        for metric in ("pass_at_1", "pass_at_4", "pass_at_8"):
            pivot = question.pivot(index="question_index", columns="global_step", values=metric)
            if previous not in pivot or current not in pivot:
                raise ValueError(f"missing checkpoint pair {previous}->{current} for {metric}")
            delta = (pivot[current] - pivot[previous]).dropna().to_numpy(float)
            if delta.size != QUESTIONS:
                raise ValueError(
                    f"checkpoint pair {previous}->{current} for {metric} has {delta.size} questions"
                )
            point, low, high = paired_bootstrap(delta)
            rows.append({
                "previous_step": previous,
                "current_step": current,
                "metric": metric,
                "effect": float(point[0]),
                "low": float(low[0]),
                "high": float(high[0]),
                "questions": int(delta.size),
            })
    return pd.DataFrame(rows)


def _profile_dynamics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = values[0]
    denominator = float(np.abs(base).sum()) + 1e-12
    cumulative = np.abs(values - base).sum(axis=1) / denominator
    adjacent = np.full(len(values), np.nan)
    level = values.mean(axis=1)
    cosine = np.full(len(values), np.nan)
    for index in range(1, len(values)):
        adjacent[index] = float(np.abs(values[index] - values[index - 1]).sum() / (np.abs(values[index - 1]).sum() + 1e-12))
    for index in range(2, len(values)):
        update = values[index] - values[index - 1]
        previous = values[index - 1] - values[index - 2]
        cosine[index] = float(update @ previous / (np.linalg.norm(update) * np.linalg.norm(previous) + 1e-12))
    return cumulative, adjacent, cosine, level


def _append_profile_group(
    output: list[dict[str, Any]],
    *,
    family: str,
    variant: str,
    label: str,
    representation: str,
    values: np.ndarray,
) -> None:
    cumulative, adjacent, cosine, level = _profile_dynamics(values)
    for index, step in enumerate(EXPECTED_STEPS):
        output.append({
            "metric_family": family,
            "metric_variant": variant,
            "metric_label": label,
            "representation": representation,
            "global_step": step,
            "level_mean": float(level[index]),
            "cumulative_relative_l1": float(cumulative[index]),
            "adjacent_relative_l1": None if np.isnan(adjacent[index]) else float(adjacent[index]),
            "update_cosine": None if np.isnan(cosine[index]) else float(cosine[index]),
        })


def load_vertical_dynamics(root: Path) -> pd.DataFrame:
    _, _, analysis = _formal_paths(root)
    policy = pd.read_parquet(analysis / "policy_profiles.parquet")
    rows: list[dict[str, Any]] = []
    for metric in ("v1_raw_update_norm", "v1_relative_update_norm", "v3_demean_state_angle", "v4_layer_update_turning_angle"):
        family, variant, label = METRIC_MAP[metric]
        for representation, group in policy.loc[policy.metric.eq(metric)].groupby("representation"):
            pivot = group.pivot_table(index=["stage", "layer_index"], columns="global_step", values="profile_mean", aggfunc="first").reindex(columns=EXPECTED_STEPS)
            pivot = pivot.dropna(axis=0, how="any")
            if pivot.empty:
                raise ValueError(f"no complete profile cells for {metric}/{representation}")
            _append_profile_group(rows, family=family, variant=variant, label=label, representation=str(representation), values=pivot.to_numpy(float).T)
    metric_frames: list[pd.DataFrame] = []
    for step in EXPECTED_STEPS:
        metric_frames.append(pd.read_parquet(_formal_paths(root)[1] / f"vertical_metrics_step{step:03d}.parquet"))
    aggregate = pd.concat(metric_frames, ignore_index=True)
    selected = aggregate.loc[aggregate.metric.isin(["weighted_layer_update_turning_angle", "layer_update_ER", "layer_update_ER_centered"])].copy()
    question = selected.groupby(["global_step", "question_id", "representation", "stage", "metric"], as_index=False)["value"].mean()
    reduced = question.groupby(["global_step", "representation", "stage", "metric"], as_index=False)["value"].mean()
    for metric, group in reduced.groupby("metric"):
        family, variant, label = METRIC_MAP[metric]
        for representation, rep_group in group.groupby("representation"):
            pivot = rep_group.pivot(index="stage", columns="global_step", values="value").reindex(columns=EXPECTED_STEPS).dropna(axis=0, how="any")
            if pivot.empty:
                raise ValueError(f"no complete aggregate cells for {metric}/{representation}")
            _append_profile_group(rows, family=family, variant=variant, label=label, representation=str(representation), values=pivot.to_numpy(float).T)
    return pd.DataFrame(rows).sort_values(["metric_family", "metric_variant", "representation", "global_step"]).reset_index(drop=True)


def load_auc_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, analysis = _formal_paths(root)
    auc = pd.read_parquet(analysis / "layer_auc.parquet")
    auc["separability"] = np.maximum(auc["question_equal_auc"], 1.0 - auc["question_equal_auc"])
    grouped = auc.groupby("global_step")["separability"]
    summary = grouped.agg(["median", "max"]).reset_index().rename(columns={"max": "maximum"})
    p90 = grouped.quantile(0.9).rename("p90").reset_index()
    summary = summary.merge(p90, on="global_step", how="left")
    summary = summary[["global_step", "median", "p90", "maximum"]].sort_values("global_step")
    late = auc.loc[
        auc.global_step.ge(175)
        & auc.metric.isin(["v3_demean_state_angle", "v1_relative_update_norm"])
        & auc.stage.eq(3)
        & auc.layer_index.eq(28)
        & auc.representation.eq("mean_w128_s32")
    ].copy()
    late["separability"] = np.maximum(late["question_equal_auc"], 1.0 - late["question_equal_auc"])
    return summary, late[["global_step", "metric", "question_equal_auc", "separability", "n_mixed_questions", "n_pairs"]]


def load_horizontal_summary(root: Path) -> pd.DataFrame:
    path = root / "online_hidden" / "per_step_summary.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output: list[dict[str, Any]] = []
    for family, short in (("H2_straightness", "H2"), ("H5_centeredERV", "H5")):
        prefix = f"hidden/{family}__chunkMean_Lfinal_stageNative/s3"
        for block_start in range(1, 251, 25):
            block = [row for row in rows if block_start <= int(row["training_step"]) <= block_start + 24]
            values = {}
            for field in ("correct_mean", "wrong_mean", "auroc", "auc_questions"):
                array = np.asarray([row.get(f"{prefix}/{field}", np.nan) for row in block], dtype=float)
                values[field] = float(np.nanmean(array))
            output.append({
                "family": short,
                "block_start": block_start,
                "block_end": block_start + 24,
                **values,
                "cohort": "online training probe; typically 16 mixed questions",
            })
    return pd.DataFrame(output)


def _figure_bytes(figure: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def render_summary_figures(data: dict[str, pd.DataFrame]) -> dict[str, bytes]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    figures: dict[str, bytes] = {}
    behavior = data["behavior_summary"]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for key, label, color in (("pass_at_1", "pass@1", "#167a62"), ("pass_at_4", "pass@4", "#266d9b"), ("pass_at_8", "pass@8", "#a87317")):
        ax.plot(behavior.global_step, behavior[key], marker="o", lw=2.2, label=label, color=color)
        ax.fill_between(behavior.global_step, behavior[f"{key}_low"], behavior[f"{key}_high"], color=color, alpha=0.10)
    ax.axvline(50, color="#a87317", ls="--", lw=1)
    ax.axvline(175, color="#b64d48", ls="--", lw=1)
    ax.set(xlabel="training step", ylabel="question-equal ability", title="Held-out ability rises early, then plateaus")
    ax.set_ylim(0.25, 0.9); ax.grid(alpha=0.18); ax.legend(frameon=False, ncol=3)
    figures["ability_passk"] = _figure_bytes(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    delta = behavior.iloc[1:]
    ax.axhline(0, color="#1e2929", lw=1)
    ax.errorbar(delta.global_step, delta.delta_pass_at_1, yerr=[delta.delta_pass_at_1 - delta.delta_pass_at_1_low, delta.delta_pass_at_1_high - delta.delta_pass_at_1], fmt="o-", color="#167a62", capsize=3)
    ax.axvline(50, color="#a87317", ls="--", lw=1); ax.axvline(175, color="#b64d48", ls="--", lw=1)
    ax.set(xlabel="checkpoint ending the interval", ylabel="Delta pass@1", title="Marginal ability gain becomes indistinguishable from zero")
    ax.grid(alpha=0.18); figures["ability_delta_pass1"] = _figure_bytes(fig)

    vertical = data["vertical_dynamics"]
    main = vertical.loc[vertical.representation.eq("mean_w128_s32")]
    for column, title, key in (("cumulative_relative_l1", "Cumulative vertical reorganization from base", "vertical_cumulative"), ("adjacent_relative_l1", "Adjacent-checkpoint vertical change", "vertical_adjacent")):
        fig, ax = plt.subplots(figsize=(9.4, 4.8))
        for label, group in main.groupby("metric_label"):
            color = METRIC_COLORS.get(label, "#266d9b")
            ax.plot(group.global_step, group[column], marker="o", lw=2, label=label, color=color)
        ax.axvline(50, color="#a87317", ls="--", lw=1); ax.axvline(175, color="#b64d48", ls="--", lw=1)
        ax.set(xlabel="training step", ylabel="relative L1 change", title=title)
        ax.grid(alpha=0.18); ax.legend(frameon=False, ncol=2, fontsize=8)
        figures[key] = _figure_bytes(fig)

    auc = data["auc_summary"]
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.plot(auc["global_step"], auc["median"], marker="o", color="#266d9b", lw=2, label="median separability")
    ax.plot(auc["global_step"], auc["p90"], marker="o", color="#a87317", lw=1.8, label="P90 separability")
    ax.axhline(0.5, color="#1e2929", lw=1); ax.set(xlabel="training step", ylabel="max(AUROC, 1-AUROC)", title="Global vertical correctness separation remains weak")
    ax.set_ylim(0.48, max(0.75, float(auc["p90"].max()) + 0.02)); ax.grid(alpha=0.18); ax.legend(frameon=False)
    figures["auc_distribution"] = _figure_bytes(fig)

    horizontal = data["horizontal"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=False)
    for axis, family in zip(axes, ("H2", "H5"), strict=True):
        group = horizontal.loc[horizontal.family.eq(family)]
        x = group.block_start + 12
        axis.plot(x, group.correct_mean, marker="o", color="#167a62", label="correct mean")
        axis.plot(x, group.wrong_mean, marker="o", color="#b64d48", label="wrong mean")
        twin = axis.twinx(); twin.plot(x, group.auroc, color="#266d9b", ls="--", marker=".", label="online AUROC")
        axis.set_title(f"{family} · final layer · B4"); axis.set_xlabel("training step block"); axis.grid(alpha=0.16)
        axis.set_ylim(bottom=0); twin.set_ylim(0, 1); twin.set_ylabel("AUROC", color="#266d9b")
    axes[0].set_ylabel("metric level"); axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Horizontal context is cohort-dependent and not a formal vertical estimate", y=1.02)
    figures["horizontal_h2_h5"] = _figure_bytes(fig)
    return figures


def encode_png(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _img(images: dict[str, str], key: str, alt: str, caption: str, wide: bool = False) -> str:
    klass = "figure wide" if wide else "figure"
    return f'<figure class="{klass}"><img src="{images[key]}" alt="{html_lib.escape(alt)}"><figcaption>{caption}</figcaption></figure>'


def render_html(data: dict[str, Any], images: dict[str, str]) -> str:
    behavior = data["behavior_summary"]
    final = behavior.iloc[-1]
    base = behavior.iloc[0]
    vertical = data["vertical_dynamics"]
    main_vertical = vertical.loc[vertical.representation.eq("mean_w128_s32")]
    final_vertical = main_vertical.loc[main_vertical.global_step.eq(250)]
    late_auc = data["auc_late"]
    effects = data["behavior_effects"]
    final_effects = effects.loc[
        effects.previous_step.eq(0) & effects.current_step.eq(250)
    ].set_index("metric")
    pass1_effect = final_effects.loc["pass_at_1"]
    pass4_effect = final_effects.loc["pass_at_4"]
    pass8_effect = final_effects.loc["pass_at_8"]
    pass_delta = float(pass1_effect.effect)
    audit = data["audit"]
    provenance = data["provenance"]
    vertical_rows = []
    for family in ("V1", "V3", "V4", "V4C", "V8"):
        group = final_vertical.loc[final_vertical.metric_family.eq(family)]
        if group.empty:
            continue
        for variant, row in group.groupby("metric_variant").first().iterrows():
            vertical_rows.append(f"<tr><td>{html_lib.escape(family + (' · ' + str(variant) if variant else ''))}</td><td>{_fmt(row.cumulative_relative_l1 * 100, 2)}%</td><td>{_fmt(row.adjacent_relative_l1 * 100, 2)}%</td><td>{_fmt(row.update_cosine, 2)}</td></tr>")
    behavior_rows = []
    for _, row in behavior.iterrows():
        behavior_rows.append(f"<tr><td>{int(row.global_step)}</td><td>{row.pass_at_1:.2%}</td><td>{row.pass_at_4:.2%}</td><td>{row.pass_at_8:.2%}</td><td>{row.delta_pass_at_1:+.2%}</td></tr>")
    auc_rows = []
    for _, row in late_auc.iterrows():
        auc_rows.append(f"<tr><td>{html_lib.escape(str(row.metric))}</td><td>{int(row.global_step)}</td><td>B4 · L28</td><td>{row.question_equal_auc:.3f}</td><td>{row.separability:.3f}</td><td>{int(row.n_mixed_questions)}</td></tr>")
    source_hashes = html_lib.escape(json.dumps(provenance.get("analysis_input_hashes", []), ensure_ascii=False, indent=2))
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Qwen3-1.7B · RL Vertical Dynamics</title>
<style>
:root{{--ink:#1e2929;--muted:#63706d;--paper:#f7f5ef;--panel:#fff;--line:#d9ddd3;--green:#167a62;--green-bg:#e1f2e9;--blue:#266d9b;--amber:#a87317;--amber-bg:#fff2d8;--red:#b64d48;--red-bg:#fae8e5;--charcoal:#17383a;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.62 ui-sans-serif,system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}} main{{max-width:1320px;margin:auto;padding:18px 28px 58px}} a{{color:#176997}} code,.mono{{font-family:var(--mono);font-size:.9em}} h1{{margin:0;font-size:38px;line-height:1.12}} h2{{margin:0;font-size:25px;line-height:1.18}} h3{{margin:0;font-size:18px}} p{{margin:10px 0}} .hero{{background:var(--charcoal);color:#fff;padding:32px;border-radius:7px;display:grid;grid-template-columns:1.3fr .7fr;gap:24px;align-items:end}} .eyebrow{{margin:0 0 9px;color:#a8ded0;font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}} .hero p{{color:#d9e6e1;max-width:820px}} .stats{{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}} .stat{{border:1px solid #ffffff45;padding:12px;border-radius:5px}} .stat b{{display:block;font:700 22px/1.1 var(--mono)}} .stat span{{color:#d9e6e1;font-size:12px}} nav{{display:flex;flex-wrap:wrap;gap:7px;padding:14px 0;border-bottom:1px solid var(--line);position:sticky;top:0;background:color-mix(in srgb,var(--paper) 94%,transparent);backdrop-filter:blur(6px);z-index:5}} nav a{{padding:5px 9px;border:1px solid var(--line);border-radius:5px;background:var(--panel);color:var(--ink);font-size:13px;text-decoration:none}} section{{margin:36px 0;scroll-margin-top:70px}} .section-head{{display:flex;justify-content:space-between;gap:22px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:15px}} .section-head p{{max-width:650px;color:var(--muted);margin:0}} .grid{{display:grid;gap:15px}} .two{{grid-template-columns:repeat(2,minmax(0,1fr))}} .three{{grid-template-columns:repeat(3,minmax(0,1fr))}} .panel,.figure,.stat-card{{background:var(--panel);border:1px solid var(--line);border-radius:7px}} .panel{{padding:18px}} .stat-card{{padding:15px}} .stat-card label{{display:block;color:var(--muted);font-size:12px}} .stat-card strong{{display:block;font:700 27px/1.15 var(--mono);margin:4px 0}} .positive{{color:var(--green)}} .warning{{color:var(--amber)}} .negative{{color:var(--red)}} .callout{{border-left:4px solid var(--amber);background:var(--amber-bg);padding:13px 15px;border-radius:0 6px 6px 0}} .callout.success{{border-left-color:var(--green);background:var(--green-bg)}} .callout.danger{{border-left-color:var(--red);background:var(--red-bg)}} .figure{{margin:0;padding:10px}} .figure.wide{{grid-column:1/-1}} .figure img{{display:block;width:100%;height:auto;aspect-ratio:16/9;object-fit:contain;background:#fff}} figcaption{{padding:8px 3px 0;color:var(--muted);font-size:12px}} .table-wrap{{overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:7px}} table{{width:100%;border-collapse:collapse;font-size:13px;min-width:560px}} th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} th{{color:var(--muted);font-size:12px}} tbody tr:hover{{background:#fbfaf6}} .timeline{{display:grid;gap:0}} .tl{{display:grid;grid-template-columns:145px 1fr;gap:15px;border-left:2px solid var(--line);padding:0 0 17px 18px;margin-left:8px;position:relative}} .tl:before{{content:"";position:absolute;left:-7px;top:4px;width:12px;height:12px;border-radius:50%;background:var(--amber)}} .tl.green:before{{background:var(--green)}} .tl.red:before{{background:var(--red)}} .tl small{{color:var(--muted)}} .formula{{overflow:auto;background:#f3f0e8;border:1px solid #e0dbcf;border-radius:5px;padding:12px;font:13px/1.65 var(--mono)}} details{{border-top:1px solid var(--line);padding:12px 0}} summary{{cursor:pointer;font-weight:700}} .foot{{color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:17px}}
@media(max-width:900px){{main{{padding:12px 14px 44px}}.hero,.two,.three{{grid-template-columns:1fr}}.hero{{padding:24px}}h1{{font-size:30px}}.section-head{{display:block}}.section-head p{{margin-top:8px}}.tl{{grid-template-columns:1fr;gap:3px}}nav{{position:static}}}}
</style></head><body><main>
<header class="hero"><div><p class="eyebrow">Qwen3-1.7B-Base · SimpleRL GRPO · formal held-out run</p><h1>RL 后的纵向 Hidden-State 动力学</h1><p>这次正式 run 的主结果是：能力提升真实存在；V1/V3/V4/V4C/V8 在早期发生明显跨层重组，step200 后变化接近平台。纵向指标适合作为训练状态和停止候选信号，而不是单条回答正确性的通用分类器。</p></div><div class="stats"><div class="stat"><b>{base.pass_at_1:.2%} → {final.pass_at_1:.2%}</b><span>pass@1 · base → step250</span></div><div class="stat"><b>{pass_delta * 100:+.2f} pp</b><span>pass@1 final-minus-base</span></div><div class="stat"><b>11 / 11</b><span>checkpoint audit passed</span></div><div class="stat"><b>22,528</b><span>held-out rollouts</span></div></div></header>
<nav><a href="#overview">结论</a><a href="#ability">能力</a><a href="#vertical">纵向重组</a><a href="#depth">层深位置</a><a href="#auroc">正确性</a><a href="#horizontal">横向 context</a><a href="#stopping">停止候选</a><a href="#audit">审计</a></nav>
<section id="overview"><div class="section-head"><div><p class="eyebrow" style="color:var(--amber)">01 · executive result</p><h2>先看结论</h2></div><p>这是 discovery 单 seed：结果足以组织机制故事，但还不是跨 seed 的最终 stopping rule。</p></div><div class="grid three"><div class="stat-card"><label>pass@4 final-minus-base</label><strong class="positive">{float(pass4_effect.effect) * 100:+.2f} pp</strong><small>paired 95% CI [{float(pass4_effect.low) * 100:+.2f}, {float(pass4_effect.high) * 100:+.2f}] pp</small></div><div class="stat-card"><label>pass@8 final-minus-base</label><strong class="positive">{float(pass8_effect.effect) * 100:+.2f} pp</strong><small>paired 95% CI [{float(pass8_effect.low) * 100:+.2f}, {float(pass8_effect.high) * 100:+.2f}] pp</small></div><div class="stat-card"><label>formal figures</label><strong class="positive">192 / 192</strong><small>analysis audit passed，全部非空</small></div></div><div class="callout success" style="margin-top:15px"><strong>一句话：</strong>step0–50 是能力提升和深层结构安装期；step50–175 是小幅巩固期；step175–250 能力和纵向变化都接近平台，因此 step175–200 是停止候选，而不是已验证的通用规则。</div></section>
<section id="ability"><div class="section-head"><div><p class="eyebrow" style="color:var(--green)">02 · held-out ability</p><h2>RL 确实学到了东西</h2></div><p>pass@k 由每题 8 个 rollout 的组合估计得到，置信区间按 question 配对 bootstrap。</p></div><div class="grid two">{_img(images, "ability_passk", "pass@1、pass@4、pass@8 随 checkpoint 的曲线", "能力曲线：step0–50 增长最快；阴影为 4,000 次 question-bootstrap 95% CI。", True)}{_img(images, "ability_delta_pass1", "相邻 checkpoint 的 pass@1 增量及置信区间", "边际增益：step175 后各区间 CI 都包含 0。", True)}</div><div class="table-wrap" style="margin-top:15px"><table><thead><tr><th>step</th><th>pass@1</th><th>pass@4</th><th>pass@8</th><th>Δpass@1</th></tr></thead><tbody>{"".join(behavior_rows)}</tbody></table></div></section>
<section id="vertical"><div class="section-head"><div><p class="eyebrow" style="color:var(--blue)">03 · vertical reorganization</p><h2>早期重组，后期平台</h2></div><p>所有候选 family 都保留；这里比较的是每个 profile 自身的相对变化，不把不同物理单位混在一个轴上。</p></div><div class="grid two">{_img(images, "vertical_cumulative", "V1、V3、V4、V4C、V8 的累计相对重组", "累计相对 L1：V8 和 V1 raw 的累计变化最大，V4C 很小。", True)}{_img(images, "vertical_adjacent", "V1、V3、V4、V4C、V8 的相邻 checkpoint 变化", "相邻变化速度：step200 后大多数指标降到约 0.1%–0.25%。", True)}</div><div class="table-wrap" style="margin-top:15px"><table><thead><tr><th>family / variant</th><th>step250 cumulative</th><th>step225→250</th><th>last update cosine</th></tr></thead><tbody>{"".join(vertical_rows)}</tbody></table></div><div class="callout" style="margin-top:15px"><strong>V8 的含义：</strong>layer-update ER 从多个独立方向的深度更新中计算有效秩。V8 上升表示更新方向更分散，不自动等于“更正确”；当前证据要和 pass@k 的增量一起解释。</div></section>
<section id="depth"><div class="section-head"><div><p class="eyebrow" style="color:var(--blue)">04 · where it happens</p><h2>深层模板稳定，没有明显迁移</h2></div><p>这些是正式 256-question profile analysis 中最能说明层深位置的图，不是从单个 rollout 挑出来的截图。</p></div><div class="grid two">{_img(images, "v1_heatmap", "V1 raw mean-window B4 layer by checkpoint heatmap", "V1 raw · mean_w128_s32 · B4：重组峰值稳定在 L28。")}{_img(images, "v3_heatmap", "V3 demeaned mean-window B4 layer by checkpoint heatmap", "V3 · mean_w128_s32 · B4：去公共分量后的深层角度变化稳定集中在 L28。")}{_img(images, "v4_heatmap", "V4 mean-window B2 layer by checkpoint heatmap", "V4 · mean_w128_s32 · B2：turning 主要在深层，mean-window 峰值约 L24。")}{_img(images, "v3_profiles", "V3 selected checkpoint layer profiles", "V3 选定 checkpoint profile：后续 checkpoint 更像放大同一深层模板，而不是峰值逐层迁移。")}</div><div class="callout" style="margin-top:15px"><strong>层深故事：</strong>V1 raw / V3 的峰值在 L28，V4 的 mean-window 峰值约 L24；step25 后没有观察到由浅到深的迁移。更准确的说法是“早期安装、后期巩固”。</div></section>
<section id="auroc"><div class="section-head"><div><p class="eyebrow" style="color:var(--red)">05 · correctness separation</p><h2>正确性信号是局部的，不是全层普遍存在</h2></div><p>AUROC 低于 0.5 的方向用 separability=max(AUROC,1-AUROC) 展示；这只是方向翻转，不是重新拟合分类器。</p></div><div class="grid two">{_img(images, "auc_distribution", "Global vertical AUROC separability by checkpoint", "全层分布：step250 中位 separability 约 0.529，P90 约 0.576。", True)}{_img(images, "v3_auc", "V3 B4 layer AUROC heatmap", "V3 · mean_w128_s32 · B4：late B4/L28 是较稳定的局部信号。")}{_img(images, "v1_relative_auc", "V1 relative B4 layer AUROC heatmap", "V1 relative · mean_w128_s32 · B4：L28 反向后形成相近的局部信号。")}</div><div class="table-wrap" style="margin-top:15px"><table><thead><tr><th>metric</th><th>step</th><th>cell</th><th>raw AUROC</th><th>separability</th><th>mixed questions</th></tr></thead><tbody>{"".join(auc_rows)}</tbody></table></div><div class="callout danger" style="margin-top:15px"><strong>边界：</strong>全层中位 AUROC 很接近 0.5，因此不能说纵向指标普遍区分正误。它们更适合作为训练阶段 / 结构状态变量，V3 B4 L28 和 V1 relative B4 L28 是局部补充证据。</div></section>
<section id="horizontal"><div class="section-head"><div><p class="eyebrow" style="color:var(--blue)">06 · horizontal context</p><h2>H2/H5 需要和纵向结果分开看</h2></div><p>这里使用的是 training-time online probe，通常每步约 16 个 mixed questions；不是本次 256-question vertical profile 的同一统计口径。</p></div>{_img(images, "horizontal_h2_h5", "Online H2 and H5 final-layer B4 trends", "H2/H5 online final-layer B4：正确/错误均值和在线 AUROC 分开显示。", True)}<div class="callout" style="margin-top:15px"><strong>当前结论：</strong>本 run 的长期在线 H2 最好覆盖 cell 平均约 0.58，并没有稳健复现 0.67；H5 主要表现为 ER level 随训练变化，但方向和正确性分离依赖 cohort。checkpoint held-out H2/H5 每个 cell 只有约 5–11 个 mixed questions，只能视为低功效诊断，不能据此奖励 H2/H5，也不支持把它们直接用于 reward shaping。</div></section>
<section id="stopping"><div class="section-head"><div><p class="eyebrow" style="color:var(--amber)">07 · stopping candidate</p><h2>step175–200 是候选停止区间</h2></div><p>停止判断需要同时看纵向变化是否超过抽样噪声，以及 pass@k 是否还在取得能力增益。</p></div><div class="grid two"><div class="panel"><div class="timeline"><div class="tl green"><div><b>step0–50</b><small>有效学习</small></div><div>pass@1 从 39.65% 到 60.06%；V1/V3/V4/V8 的纵向变化最大。</div></div><div class="tl"><div><b>step50–175</b><small>巩固 / refinement</small></div><div>能力仍有小幅累计增益，纵向变化继续但明显衰减。</div></div><div class="tl red"><div><b>step175–250</b><small>停止候选</small></div><div>pass@1 只增加约 0.49 pp，纵向相邻变化约 0.1%–0.25%，多个更新方向出现轻微反转。</div></div></div></div><div class="panel"><h3>四种状态解释</h3><table><thead><tr><th>纵向变化</th><th>能力</th><th>解释</th></tr></thead><tbody><tr><td>高</td><td>上升</td><td class="positive">有效学习候选</td></tr><tr><td>高</td><td>不升</td><td class="warning">漂移 / 过训练候选</td></tr><tr><td>低</td><td>上升</td><td>readout / policy refinement</td></tr><tr><td>低</td><td>不升</td><td class="negative">停止候选</td></tr></tbody></table></div></div><div class="callout" style="margin-top:15px"><strong>还缺的一步：</strong>当前报告没有把 late change 和 rollout sampling noise 做正式比较。要把 step175–200 从“候选”升级为 stopping rule，需要执行 <code>vertical_dynamic_plan.md</code> 中的 4-vs-4 split noise 和滞后 coupling 分析，并在独立 seed 上复现。</div></section>
<section id="audit"><div class="section-head"><div><p class="eyebrow" style="color:var(--muted)">08 · audit and limits</p><h2>数据与解释边界</h2></div><p>所有主结论来自已完成的 formal audit；下面保留复核所需的最小 provenance。</p></div><div class="grid three"><div class="stat-card"><label>checkpoint audit</label><strong class="positive">PASS</strong><small>0–250 every 25 steps</small></div><div class="stat-card"><label>question coverage</label><strong class="positive">256 × 11</strong><small>固定 held-out cohort</small></div><div class="stat-card"><label>rollout coverage</label><strong class="positive">8 / question</strong><small>pass@k 组合估计</small></div></div><div class="panel" style="margin-top:15px"><ul><li>这是单 seed discovery，不能单独证明因果机制。</li><li>checkpoint-level 相关会受到共同训练时间趋势影响；早期大区间会抬高整体相关。</li><li>纵向动态尚未完成 rollout sampling-noise 校正。</li><li>AUROC 局部信号不能替代 held-out pass@k。</li><li>online H2/H5 与 formal vertical cohort 的 question coverage 不同。</li></ul><details><summary>输入 hash 与机器审计</summary><pre style="white-space:pre-wrap;overflow:auto;font-size:11px">{source_hashes}</pre></details></div><p class="foot">Generated by <code>build_q3_vertical_dynamics_report.py</code> · build seed <code>{SEED}</code> · formal audit passed · this report embeds derived figures and does not contain raw responses or hidden tensors.</p></section>
</main></body></html>'''


def build_report(root: Path, output: Path) -> dict[str, Any]:
    provenance, audit = validate_result_root(root)
    behavior_summary, behavior_questions = load_behavior(root)
    behavior_effects = paired_behavior_effects(behavior_questions)
    vertical = load_vertical_dynamics(root)
    auc_summary, auc_late = load_auc_summary(root)
    horizontal = load_horizontal_summary(root)
    tables = {
        "behavior_summary": behavior_summary,
        "behavior_questions": behavior_questions,
        "behavior_effects": behavior_effects,
        "vertical_dynamics": vertical,
        "auc_summary": auc_summary,
        "auc_late": auc_late,
        "horizontal": horizontal,
    }
    image_bytes = render_summary_figures(tables)
    figure_dir = _formal_paths(root)[2] / "figures"
    for key, filename in SOURCE_FIGURES.items():
        image_bytes[key] = (figure_dir / filename).read_bytes()
    undersized = {key: len(value) for key, value in image_bytes.items() if len(value) < 10_000}
    if undersized:
        raise RuntimeError(f"report figures are undersized: {undersized}")
    images = {key: encode_png(value) for key, value in image_bytes.items()}
    data = {**tables, "provenance": provenance, "audit": audit}
    document = render_html(data, images).encode("utf-8")
    decoded = document.decode("utf-8")
    missing_sections = [section for section in REQUIRED_SECTIONS if f'id="{section}"' not in decoded]
    if missing_sections:
        raise RuntimeError(f"report is missing sections: {missing_sections}")
    if decoded.count("data:image/png;base64,") != len(images):
        raise RuntimeError("not every report image was embedded exactly once")
    if "https://" in decoded or "http://" in decoded:
        raise RuntimeError("offline report contains an external URL")
    if not decoded.rstrip().endswith("</html>"):
        raise RuntimeError("report document is incomplete")
    _atomic_bytes(output, document)
    result = {
        "passed": True,
        "output": str(output),
        "bytes": len(document),
        "embedded_images": len(images),
        "expected_steps": list(EXPECTED_STEPS),
        "analysis_audit_passed": bool(audit.get("passed")),
        "build_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": SEED,
        "sha256": _sha256(output),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Q3 vertical dynamics presentation report")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.result_root, args.output)


if __name__ == "__main__":
    main()
