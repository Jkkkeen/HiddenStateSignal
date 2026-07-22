#!/usr/bin/env python3
"""Analyze Experiment 0 all-layer long-response hidden dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from long_success_trajectory_common import length_bin, pairwise_auc


TOKEN_FEATURES = [
    "horizontal_norm_mean",
    "horizontal_norm_median",
    "horizontal_norm_p90",
    "horizontal_norm_delta_mean",
    "horizontal_norm_delta_median",
    "vertical_norm_mean",
    "vertical_norm_median",
    "vertical_norm_p90",
    "vertical_norm_delta_mean",
    "vertical_norm_delta_median",
    "coordinate_entropy_mean",
    "effective_dimensions_mean",
    "token_turn_cos_mean",
]
SPAN_FEATURES = [
    "span_movement_norm_mean",
    "span_movement_norm_median",
    "span_movement_norm_p90",
    "span_turn_cos_mean",
    "span_turn_cos_median",
    "span_layer_update_norm_mean",
    "span_layer_turn_cos_mean",
    "span_layer_turn_cos_median",
    "cross_set_direction",
    "cross_prototype_direction",
    "cross_length_support",
]
ALL_FEATURES = TOKEN_FEATURES + SPAN_FEATURES
PRIMARY_REPRESENTATION = "mean_w128_s64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 0 hidden dynamics.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-representation", default=PRIMARY_REPRESENTATION)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--run-label", default="Smoke")
    return parser.parse_args()


def hedges_g(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size < 2 or negative.size < 2:
        return float("nan")
    degrees = positive.size + negative.size - 2
    pooled_variance = (
        (positive.size - 1) * positive.var(ddof=1)
        + (negative.size - 1) * negative.var(ddof=1)
    ) / degrees
    if pooled_variance <= 0:
        return float("nan")
    correction = 1.0 - 3.0 / (4.0 * (positive.size + negative.size) - 9.0)
    return float(correction * (positive.mean() - negative.mean()) / np.sqrt(pooled_variance))


def _cluster_bootstrap_g(
    frame: pd.DataFrame,
    feature: str,
    bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    question_groups = {
        str(question_id): group for question_id, group in frame.groupby("question_id", sort=True)
    }
    question_ids = np.asarray(sorted(question_groups), dtype=object)
    if question_ids.size < 2 or bootstrap <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        selected = rng.choice(question_ids, size=question_ids.size, replace=True)
        positive = np.concatenate(
            [
                question_groups[str(question)].loc[
                    question_groups[str(question)]["is_correct"], feature
                ].to_numpy(dtype=np.float64)
                for question in selected
            ]
        )
        negative = np.concatenate(
            [
                question_groups[str(question)].loc[
                    ~question_groups[str(question)]["is_correct"], feature
                ].to_numpy(dtype=np.float64)
                for question in selected
            ]
        )
        samples.append(hedges_g(positive, negative))
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _question_effects(frame: pd.DataFrame, feature: str) -> np.ndarray:
    values = []
    for _, group in frame.groupby("question_id", sort=True):
        values.append(
            hedges_g(
                group.loc[group["is_correct"], feature].to_numpy(),
                group.loc[~group["is_correct"], feature].to_numpy(),
            )
        )
    return np.asarray(values, dtype=np.float64)


def _within_question_auc(frame: pd.DataFrame, score: str) -> pd.Series:
    records = {}
    for question_id, group in frame.groupby("question_id", sort=True):
        records[str(question_id)] = pairwise_auc(
            group.loc[group["is_correct"], score].to_numpy(dtype=np.float64),
            group.loc[~group["is_correct"], score].to_numpy(dtype=np.float64),
        )
    return pd.Series(records, dtype=np.float64)


def _paired_delta_interval(
    left: pd.Series,
    right: pd.Series,
    bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    delta = aligned["left"] - aligned["right"]
    estimate = float(delta.mean()) if not delta.empty else float("nan")
    if delta.size < 2 or bootstrap <= 0:
        return estimate, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = delta.to_numpy(dtype=np.float64)
    samples = np.asarray(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(bootstrap)]
    )
    return estimate, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def compare_feature_to_length(
    frame: pd.DataFrame,
    feature: str,
    seed: int,
    bootstrap: int,
) -> dict[str, Any]:
    required = {"question_id", "rollout_id", "is_correct", "think_length", feature}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"predictor frame missing columns: {sorted(missing)}")
    view = frame[
        ["question_id", "rollout_id", "is_correct", "think_length", feature]
    ].dropna()
    view = view.drop_duplicates(["question_id", "rollout_id"], keep="first").reset_index(drop=True)
    n_questions = int(view["question_id"].nunique())
    if n_questions < 2 or view["is_correct"].nunique() < 2:
        return {
            "n_rows_feature": len(view),
            "n_rows_length": len(view),
            "n_rows_combined": len(view),
            "n_questions": n_questions,
            "fold_question_overlap": 0,
            "feature_only_auc": float("nan"),
            "length_only_auc": float("nan"),
            "combined_auc": float("nan"),
            "delta_feature_vs_length": float("nan"),
            "delta_feature_vs_length_low": float("nan"),
            "delta_feature_vs_length_high": float("nan"),
            "delta_combined_vs_length": float("nan"),
            "delta_combined_vs_length_low": float("nan"),
            "delta_combined_vs_length_high": float("nan"),
            "passes_length_gate": False,
        }

    feature_score = np.full(len(view), np.nan, dtype=np.float64)
    length_score = np.full(len(view), np.nan, dtype=np.float64)
    combined_score = np.full(len(view), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=min(5, n_questions))
    fold_overlap = 0
    for fold_id, (train_index, test_index) in enumerate(
        splitter.split(view, view["is_correct"], groups=view["question_id"])
    ):
        train_questions = set(view.iloc[train_index]["question_id"])
        test_questions = set(view.iloc[test_index]["question_id"])
        fold_overlap = max(fold_overlap, len(train_questions.intersection(test_questions)))
        for columns, destination in (
            ([feature], feature_score),
            (["think_length"], length_score),
            ([feature, "think_length"], combined_score),
        ):
            pipeline = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed + fold_id,
                ),
            )
            pipeline.fit(view.iloc[train_index][columns], view.iloc[train_index]["is_correct"])
            classes = list(pipeline.named_steps["logisticregression"].classes_)
            positive_index = classes.index(True) if True in classes else classes.index(1)
            destination[test_index] = pipeline.predict_proba(view.iloc[test_index][columns])[
                :, positive_index
            ]

    scored = view.copy()
    scored["feature_score"] = feature_score
    scored["length_score"] = length_score
    scored["combined_score"] = combined_score
    feature_auc = _within_question_auc(scored, "feature_score")
    length_auc = _within_question_auc(scored, "length_score")
    combined_auc = _within_question_auc(scored, "combined_score")
    feature_delta = _paired_delta_interval(
        feature_auc, length_auc, bootstrap=bootstrap, seed=seed + 101
    )
    combined_delta = _paired_delta_interval(
        combined_auc, length_auc, bootstrap=bootstrap, seed=seed + 202
    )
    return {
        "n_rows_feature": int(len(view)),
        "n_rows_length": int(len(view)),
        "n_rows_combined": int(len(view)),
        "n_questions": n_questions,
        "fold_question_overlap": int(fold_overlap),
        "feature_only_auc": float(feature_auc.mean()),
        "length_only_auc": float(length_auc.mean()),
        "combined_auc": float(combined_auc.mean()),
        "delta_feature_vs_length": feature_delta[0],
        "delta_feature_vs_length_low": feature_delta[1],
        "delta_feature_vs_length_high": feature_delta[2],
        "delta_combined_vs_length": combined_delta[0],
        "delta_combined_vs_length_low": combined_delta[1],
        "delta_combined_vs_length_high": combined_delta[2],
        "passes_length_gate": passes_length_gate(feature_delta[1], combined_delta[1]),
    }


def passes_length_gate(delta_feature_low: float, delta_combined_low: float) -> bool:
    return bool(
        np.isfinite(delta_feature_low)
        and np.isfinite(delta_combined_low)
        and delta_feature_low > 0
        and delta_combined_low > 0
    )


def _load_parquet_directory(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("question_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no question parquet shards: {path}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def compute_effect_table(
    features: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    offset = 0
    for feature in ALL_FEATURES:
        if feature not in features.columns:
            continue
        available = features.dropna(subset=[feature])
        if available.empty:
            continue
        for keys, group in available.groupby(
            ["representation", "layer", "progress_bin"], sort=True, observed=True
        ):
            positive = group.loc[group["is_correct"], feature].to_numpy(dtype=np.float64)
            negative = group.loc[~group["is_correct"], feature].to_numpy(dtype=np.float64)
            effect = hedges_g(positive, negative)
            low, high = _cluster_bootstrap_g(group, feature, bootstrap, seed + offset)
            question_effects = _question_effects(group, feature)
            finite_question = question_effects[np.isfinite(question_effects)]
            if np.isfinite(effect) and finite_question.size:
                sign_consistency = float(np.mean(np.sign(finite_question) == np.sign(effect)))
            else:
                sign_consistency = float("nan")
            rows.append(
                {
                    "feature": feature,
                    "representation": keys[0],
                    "layer": int(keys[1]),
                    "progress_bin": int(keys[2]),
                    "hedges_g": effect,
                    "ci_low": low,
                    "ci_high": high,
                    "abs_hedges_g": abs(effect) if np.isfinite(effect) else float("nan"),
                    "question_sign_consistency": sign_consistency,
                    "n_questions": int(group["question_id"].nunique()),
                    "n_correct": int(group["is_correct"].sum()),
                    "n_wrong": int((~group["is_correct"]).sum()),
                }
            )
            offset += 1
    return pd.DataFrame(rows)


def compute_predictor_table(
    features: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    offset = 0
    for feature in ALL_FEATURES:
        if feature not in features.columns:
            continue
        available = features.dropna(subset=[feature])
        for keys, group in available.groupby(
            ["representation", "layer", "progress_bin"], sort=True, observed=True
        ):
            result = compare_feature_to_length(
                group, feature=feature, seed=seed + offset, bootstrap=bootstrap
            )
            rows.append(
                {
                    "feature": feature,
                    "representation": keys[0],
                    "layer": int(keys[1]),
                    "progress_bin": int(keys[2]),
                    **result,
                }
            )
            offset += 1
    return pd.DataFrame(rows)


def compute_length_effect(
    features: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    rollouts = features[
        ["question_id", "rollout_id", "is_correct", "think_length"]
    ].drop_duplicates(["question_id", "rollout_id"])
    effect = hedges_g(
        rollouts.loc[rollouts["is_correct"], "think_length"].to_numpy(),
        rollouts.loc[~rollouts["is_correct"], "think_length"].to_numpy(),
    )
    low, high = _cluster_bootstrap_g(rollouts, "think_length", bootstrap, seed)
    return {
        "hedges_g": effect,
        "abs_hedges_g": abs(effect) if np.isfinite(effect) else float("nan"),
        "ci_low": low,
        "ci_high": high,
        "n_questions": int(rollouts["question_id"].nunique()),
        "n_rollouts": int(len(rollouts)),
    }


def _sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(maximum, random_state=seed)


def _finish_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _scatter_by_correct(
    axis: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
) -> None:
    colors = np.where(frame["is_correct"].to_numpy(dtype=bool), "#167d5a", "#c04a3f")
    axis.scatter(frame[x], frame[y], c=colors, s=10, alpha=0.35, edgecolors="none")
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.set_title(title)
    axis.grid(alpha=0.2)


def write_figures(
    token: pd.DataFrame,
    span: pd.DataFrame,
    effects: pd.DataFrame,
    predictors: pd.DataFrame,
    length_effect: dict[str, Any],
    output_dir: Path,
    primary_representation: str,
    seed: int,
) -> list[Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    keys = ["question_id", "rollout_id", "is_correct", "progress_bin", "layer", "think_length"]
    primary = span[span["representation"] == primary_representation]

    horizontal = token.merge(
        primary[keys + ["span_turn_cos_mean"]], on=keys, how="inner", validate="one_to_one"
    ).dropna(subset=["horizontal_norm_mean", "span_turn_cos_mean"])
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    _scatter_by_correct(
        axis,
        _sample(horizontal, 20000, seed),
        "horizontal_norm_mean",
        "span_turn_cos_mean",
        "Horizontal amplitude and pooled turning",
    )
    paths = [_finish_figure(fig, figure_dir / "E0_F1_horizontal_length_angle.png")]

    cross = primary.dropna(subset=["cross_length_support", "cross_set_direction"])
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    _scatter_by_correct(
        axis,
        _sample(cross, 20000, seed + 1),
        "cross_length_support",
        "cross_set_direction",
        "Progress-matched cross-rollout support",
    )
    paths.append(_finish_figure(fig, figure_dir / "E0_F2_cross_rollout_support.png"))

    vertical = token.merge(
        primary[keys + ["span_layer_turn_cos_mean"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    ).dropna(subset=["coordinate_entropy_mean", "vertical_norm_mean", "span_layer_turn_cos_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    shown = _sample(vertical, 20000, seed + 2)
    _scatter_by_correct(
        axes[0], shown, "coordinate_entropy_mean", "vertical_norm_mean", "Vertical activity"
    )
    _scatter_by_correct(
        axes[1],
        shown,
        "coordinate_entropy_mean",
        "span_layer_turn_cos_mean",
        "Vertical pooled turning",
    )
    paths.append(_finish_figure(fig, figure_dir / "E0_F3_vertical_entropy_activity.png"))

    selected_effects = effects.sort_values("abs_hedges_g", ascending=False).dropna(
        subset=["hedges_g"]
    )
    selected_names = selected_effects["feature"].drop_duplicates().head(4).tolist()
    if not selected_names:
        selected_names = ["no_finite_effect"]
    fig, axes = plt.subplots(
        len(selected_names), 1, figsize=(9, max(3.2, 3.0 * len(selected_names))), squeeze=False
    )
    for axis, feature in zip(axes[:, 0], selected_names):
        view = effects[effects["feature"] == feature]
        if view.empty:
            axis.text(0.5, 0.5, "No finite effect", ha="center", va="center")
            axis.axis("off")
            continue
        pivot = view.pivot_table(index="layer", columns="progress_bin", values="hedges_g")
        bound = max(float(np.nanmax(np.abs(pivot.to_numpy()))), 0.1)
        image = axis.imshow(pivot, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
        axis.set_title(feature)
        axis.set_xlabel("progress bin")
        axis.set_ylabel("hidden-state index")
        axis.set_xticks(range(len(pivot.columns)), pivot.columns)
        fig.colorbar(image, ax=axis, fraction=0.025)
    paths.append(_finish_figure(fig, figure_dir / "E0_F4_layer_progress_effects.png"))

    reliability_rows = []
    for representation, group in span.groupby("representation", sort=True):
        for feature, split_a, split_b in (
            ("span_turn_cos_mean", "span_turn_cos_split_a", "span_turn_cos_split_b"),
            (
                "span_layer_turn_cos_mean",
                "span_layer_turn_cos_split_a",
                "span_layer_turn_cos_split_b",
            ),
        ):
            valid = group[[split_a, split_b]].dropna()
            reliability_rows.append(
                {
                    "label": f"{representation}\n{feature}",
                    "reliability": float(valid[split_a].corr(valid[split_b]))
                    if len(valid) >= 3
                    else float("nan"),
                }
            )
    reliability = pd.DataFrame(reliability_rows)
    fig, axis = plt.subplots(figsize=(max(8, len(reliability) * 0.65), 5.5))
    axis.bar(np.arange(len(reliability)), reliability["reliability"].fillna(0.0), color="#3267a8")
    axis.set_xticks(np.arange(len(reliability)), reliability["label"], rotation=45, ha="right")
    axis.set_ylim(-1, 1)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("split-half correlation")
    axis.set_title("Angle reliability across pooling specifications")
    paths.append(_finish_figure(fig, figure_dir / "E0_F5_angle_snr_reliability.png"))

    length_view = token.dropna(subset=["horizontal_norm_mean"]).copy()
    length_view["length_bin"] = length_view["think_length"].map(length_bin)
    strata_rows = []
    for bin_name, group in length_view.groupby("length_bin", sort=False):
        strata_rows.append(
            {
                "length_bin": bin_name,
                "hedges_g": hedges_g(
                    group.loc[group["is_correct"], "horizontal_norm_mean"].to_numpy(),
                    group.loc[~group["is_correct"], "horizontal_norm_mean"].to_numpy(),
                ),
                "n": len(group),
            }
        )
    strata = pd.DataFrame(strata_rows)
    fig, axis = plt.subplots(figsize=(7.5, 5))
    axis.bar(strata["length_bin"], strata["hedges_g"], color="#7b6d3a")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Hedges' g")
    axis.set_title("Horizontal norm effect by think-length stratum")
    paths.append(_finish_figure(fig, figure_dir / "E0_F6_length_strata.png"))

    top_predictors = predictors.sort_values("feature_only_auc", ascending=False).head(20)
    fig, axis = plt.subplots(figsize=(10, max(5, len(top_predictors) * 0.35)))
    positions = np.arange(len(top_predictors))
    labels = [
        f"{row.feature} L{row.layer} B{row.progress_bin}"
        for row in top_predictors.itertuples()
    ]
    axis.scatter(top_predictors["length_only_auc"], positions, label="length-only", marker="|", s=120)
    axis.scatter(top_predictors["feature_only_auc"], positions, label="feature-only", s=24)
    axis.scatter(top_predictors["combined_auc"], positions, label="feature+length", s=24)
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, 1)
    axis.set_xlabel("held-out within-question pairwise AUC")
    axis.set_title(
        f"Raw think-length floor | |g|={float(length_effect['abs_hedges_g']):.3f}"
    )
    axis.legend()
    paths.append(_finish_figure(fig, figure_dir / "E0_F7_length_baseline.png"))
    return paths


def write_report(
    output_dir: Path,
    run_label: str,
    token: pd.DataFrame,
    span: pd.DataFrame,
    prototype: pd.DataFrame,
    effects: pd.DataFrame,
    predictors: pd.DataFrame,
    length_effect: dict[str, Any],
    figures: Iterable[Path],
) -> Path:
    path = output_dir / "LONG_EXPERIMENT_0_RESULTS.md"
    top_effects = effects.sort_values("abs_hedges_g", ascending=False).dropna(
        subset=["hedges_g"]
    ).head(10)
    top_predictors = predictors.sort_values("feature_only_auc", ascending=False).head(10)
    lines = [
        f"# Long Experiment 0 {run_label} Results",
        "",
        "Discovery-only all-layer hidden-dynamics analysis on fixed long responses; no new generation and no RL update.",
        "",
        "## Cohort",
        "",
        f"- Questions: {token['question_id'].nunique()}",
        f"- Rollouts: {token[['question_id', 'rollout_id']].drop_duplicates().shape[0]}",
        f"- Hidden-state indexes: {token['layer'].nunique()} ({int(token['layer'].min())}-{int(token['layer'].max())})",
        f"- Think-token range: {int(token['think_length'].min())}-{int(token['think_length'].max())}",
        f"- Span representations: {', '.join(sorted(span['representation'].unique()))}",
        "",
        "## Raw Length Floor",
        "",
        f"- signed Hedges' g: {float(length_effect['hedges_g']):.4f}",
        f"- absolute Hedges' g: {float(length_effect['abs_hedges_g']):.4f}",
        f"- question-bootstrap 95% CI: [{float(length_effect['ci_low']):.4f}, {float(length_effect['ci_high']):.4f}]",
        "",
        "## Prototype Shrinkage",
        "",
    ]
    for threshold in (10, 20, 30):
        column = f"prototype_valid_0{threshold}"
        if column in prototype.columns:
            lines.append(f"- kappa >= 0.{threshold:02d} valid fraction: {prototype[column].mean():.4f}")
    lines.extend(["", "## Strongest Correct/Wrong Effects", ""])
    if top_effects.empty:
        lines.append("No finite Hedges' g estimates.")
    else:
        lines.extend(
            [
                "| Feature | Representation | Layer | Bin | g | 95% CI | Sign consistency |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in top_effects.itertuples():
            lines.append(
                f"| {row.feature} | {row.representation} | {row.layer} | {row.progress_bin} | "
                f"{row.hedges_g:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] | "
                f"{row.question_sign_consistency:.3f} |"
            )
    lines.extend(["", "## Length-Controlled Predictors", ""])
    if top_predictors.empty:
        lines.append("No eligible grouped predictor comparison.")
    else:
        lines.extend(
            [
                "| Feature | Rep | Layer | Bin | Feature AUC | Length AUC | Combined AUC | Pass |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in top_predictors.itertuples():
            lines.append(
                f"| {row.feature} | {row.representation} | {row.layer} | {row.progress_bin} | "
                f"{row.feature_only_auc:.4f} | {row.length_only_auc:.4f} | "
                f"{row.combined_auc:.4f} | {bool(row.passes_length_gate)} |"
            )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Candidates passing both paired length-baseline comparisons: {int(predictors['passes_length_gate'].sum())}",
            "- Passing this smoke gate does not establish a confirmatory result; representation and layer choices must be frozen before locked evaluation.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend([f"- `{figure.relative_to(output_dir).as_posix()}`" for figure in figures])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = _load_parquet_directory(input_dir / "bin_features")
    span = _load_parquet_directory(input_dir / "span_features")
    prototype = _load_parquet_directory(input_dir / "prototype_diagnostics")
    combined = pd.concat([token, span], ignore_index=True, sort=False)

    effects = compute_effect_table(combined, bootstrap=args.bootstrap, seed=args.seed)
    predictors = compute_predictor_table(combined, bootstrap=args.bootstrap, seed=args.seed)
    length_effect = compute_length_effect(combined, bootstrap=args.bootstrap, seed=args.seed + 9000)

    combined.to_parquet(output_dir / "long_experiment_0_bin_features.parquet", index=False)
    effects.to_csv(output_dir / "long_experiment_0_question_effects.csv", index=False)
    predictors.to_csv(output_dir / "long_experiment_0_predictor_comparisons.csv", index=False)
    prototype.to_parquet(
        output_dir / "long_experiment_0_prototype_diagnostics.parquet", index=False
    )
    figures = write_figures(
        token,
        span,
        effects,
        predictors,
        length_effect,
        output_dir,
        primary_representation=args.primary_representation,
        seed=args.seed,
    )
    report = write_report(
        output_dir,
        run_label=args.run_label,
        token=token,
        span=span,
        prototype=prototype,
        effects=effects,
        predictors=predictors,
        length_effect=length_effect,
        figures=figures,
    )
    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "run_label": args.run_label,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "primary_representation": args.primary_representation,
        "questions": int(token["question_id"].nunique()),
        "rollouts": int(token[["question_id", "rollout_id"]].drop_duplicates().shape[0]),
        "layers": sorted(map(int, token["layer"].unique())),
        "length_effect": length_effect,
        "passing_candidates": int(predictors["passes_length_gate"].sum()),
        "report": report.name,
        "new_generation": False,
    }
    (output_dir / "analysis_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
