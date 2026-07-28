#!/usr/bin/env python3
"""Analyze the frozen Experiment 02 replication and exploratory families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_experiment0_hidden_dynamics import compare_feature_to_length, hedges_g
from long_success_trajectory_common import pairwise_auc


MAIN_CELLS = (
    {
        "metric": "movement",
        "feature": "movement_norm_median",
        "progress_bin": 5,
        "layer": 24,
        "representation": "mean_w128_s64",
    },
    {
        "metric": "activity",
        "feature": "activity_norm_p90",
        "progress_bin": 9,
        "layer": 15,
        "representation": "token",
    },
)
ENTROPY_FEATURES = (
    "activation_entropy_raw_mean",
    "activation_entropy_z_mean",
    "update_entropy_mean",
)
PATH_FEATURES = (
    "path_length",
    "net_displacement",
    "straightness",
    "log_detour",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 02.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--run-label", default="Replication96")
    return parser.parse_args()


def load_shards(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("question_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards in {directory}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def question_aucs(frame: pd.DataFrame, feature: str) -> pd.Series:
    records = {}
    for question_id, group in frame.groupby("question_id", sort=True):
        positive = group.loc[group["is_correct"], feature].to_numpy(dtype=np.float64)
        negative = group.loc[~group["is_correct"], feature].to_numpy(dtype=np.float64)
        positive = positive[np.isfinite(positive)]
        negative = negative[np.isfinite(negative)]
        records[str(question_id)] = pairwise_auc(positive, negative)
    return pd.Series(records, dtype=np.float64).dropna()


def bootstrap_mean(
    values: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or bootstrap <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(bootstrap, finite.size))
    means = finite[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def strict_questions(progress: pd.DataFrame) -> set[str]:
    labels = (
        progress[["question_id", "rollout_id", "is_correct"]]
        .drop_duplicates()
        .groupby("question_id")["is_correct"]
        .agg(n_rollouts="size", n_correct="sum")
    )
    labels["n_wrong"] = labels["n_rollouts"] - labels["n_correct"]
    return set(
        labels.index[(labels["n_correct"] >= 3) & (labels["n_wrong"] >= 3)].astype(str)
    )


def confirmatory_results(
    progress: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict_ids = strict_questions(progress)
    result_rows = []
    curve_rows = []
    predictor_rows = []
    for cell_index, cell in enumerate(MAIN_CELLS):
        for cohort, cohort_frame in (
            ("main_2plus2", progress),
            (
                "strict_3plus3",
                progress[progress["question_id"].astype(str).isin(strict_ids)],
            ),
        ):
            for progress_bin in range(10):
                view = cohort_frame[
                    cohort_frame["progress_bin"] == progress_bin
                ].copy()
                aucs = question_aucs(view, cell["feature"])
                low, high = bootstrap_mean(
                    aucs.to_numpy(),
                    bootstrap=bootstrap,
                    seed=seed + 1000 * cell_index + progress_bin,
                )
                row = {
                    "metric": cell["metric"],
                    "feature": cell["feature"],
                    "cohort": cohort,
                    "progress_bin": progress_bin,
                    "is_confirmatory_cell": progress_bin == cell["progress_bin"],
                    "n_questions": int(aucs.size),
                    "within_q_auc": float(aucs.mean()) if not aucs.empty else float("nan"),
                    "auc_ci_low": low,
                    "auc_ci_high": high,
                    "sign_fraction": float((aucs > 0.5).mean())
                    if not aucs.empty
                    else float("nan"),
                }
                curve_rows.append(row)
                if row["is_confirmatory_cell"]:
                    result_rows.append(row)
                    comparison = compare_feature_to_length(
                        view,
                        cell["feature"],
                        seed=seed + cell_index,
                        bootstrap=bootstrap,
                    )
                    predictor_rows.append(
                        {
                            "metric": cell["metric"],
                            "feature": cell["feature"],
                            "cohort": cohort,
                            **comparison,
                        }
                    )
    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(curve_rows),
        pd.DataFrame(predictor_rows),
    )


def question_weighted_curve(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    per_question = (
        frame.groupby(
            ["question_id", "is_correct", "progress_bin"],
            as_index=False,
            observed=True,
        )[feature]
        .mean()
    )
    return (
        per_question.groupby(
            ["is_correct", "progress_bin"],
            as_index=False,
            observed=True,
        )[feature]
        .agg(["mean", "sem"])
        .reset_index()
    )


def plot_progress_spaghetti(progress: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    colors = {False: "#c44e52", True: "#2878b5"}
    for axis, cell in zip(axes, MAIN_CELLS):
        feature = cell["feature"]
        for _, group in progress.groupby(
            ["question_id", "rollout_id"],
            sort=False,
        ):
            group = group.sort_values("progress_bin")
            label = bool(group["is_correct"].iloc[0])
            axis.plot(
                group["progress_bin"],
                group[feature],
                color=colors[label],
                alpha=0.035,
                linewidth=0.65,
            )
        curve = question_weighted_curve(progress, feature)
        for label in (False, True):
            group = curve[curve["is_correct"] == label].sort_values("progress_bin")
            axis.plot(
                group["progress_bin"],
                group["mean"],
                color=colors[label],
                linewidth=2.5,
                label="correct" if label else "wrong",
            )
            axis.fill_between(
                group["progress_bin"],
                group["mean"] - 1.96 * group["sem"].fillna(0.0),
                group["mean"] + 1.96 * group["sem"].fillna(0.0),
                color=colors[label],
                alpha=0.15,
            )
        axis.axvline(cell["progress_bin"], color="#333333", linestyle="--", alpha=0.6)
        axis.set_title(
            f"{cell['metric'].title()} full rollout trend\n"
            f"frozen cell: bin {cell['progress_bin']}"
        )
        axis.set_xlabel("Relative-progress bin")
        axis.set_ylabel(feature)
        axis.set_xticks(range(10))
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "E02_F1_progress_spaghetti.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_auc_curves(curves: pd.DataFrame, figure_dir: Path) -> Path:
    main = curves[curves["cohort"] == "main_2plus2"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for axis, cell in zip(axes, MAIN_CELLS):
        group = main[main["metric"] == cell["metric"]].sort_values("progress_bin")
        axis.plot(group["progress_bin"], group["within_q_auc"], marker="o")
        axis.fill_between(
            group["progress_bin"],
            group["auc_ci_low"],
            group["auc_ci_high"],
            alpha=0.18,
        )
        axis.axhline(0.5, color="#555555", linestyle="--")
        point = group.loc[
            group["progress_bin"] == cell["progress_bin"],
            "within_q_auc",
        ]
        axis.scatter(
            [cell["progress_bin"]],
            point,
            s=90,
            facecolor="#d62728",
            edgecolor="black",
            zorder=4,
        )
        axis.set_title(cell["metric"].title())
        axis.set_xlabel("Relative-progress bin")
        axis.set_xticks(range(10))
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Question-equal within-Q AUC")
    fig.tight_layout()
    path = figure_dir / "E02_F2_confirmatory_auc_curves.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def exploratory_effect_table(
    frame: pd.DataFrame,
    *,
    group_keys: list[str],
    features: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_keys, sort=True, observed=True):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_keys, key_values))
        for feature in features:
            if feature not in group:
                continue
            aucs = question_aucs(group, feature)
            positive = group.loc[group["is_correct"], feature].to_numpy(dtype=np.float64)
            negative = group.loc[~group["is_correct"], feature].to_numpy(dtype=np.float64)
            rows.append(
                {
                    **base,
                    "feature": feature,
                    "n_questions": int(aucs.size),
                    "within_q_auc": float(aucs.mean())
                    if not aucs.empty
                    else float("nan"),
                    "sign_fraction": float((aucs > 0.5).mean())
                    if not aucs.empty
                    else float("nan"),
                    "hedges_g": hedges_g(positive, negative),
                }
            )
    return pd.DataFrame(rows)


def plot_entropy_heatmaps(effects: pd.DataFrame, figure_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(ENTROPY_FEATURES), figsize=(16, 6), sharey=True)
    for axis, feature in zip(axes, ENTROPY_FEATURES):
        view = effects[effects["feature"] == feature]
        pivot = view.pivot(index="layer", columns="progress_bin", values="hedges_g")
        values = pivot.to_numpy(dtype=np.float64)
        limit = np.nanquantile(np.abs(values), 0.98) if np.isfinite(values).any() else 1.0
        limit = max(float(limit), 0.1)
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(feature.replace("_mean", ""))
        axis.set_xlabel("Progress bin")
        axis.set_xticks(range(10))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Hidden-state layer")
    fig.suptitle("Correct minus wrong standardized gap")
    fig.tight_layout()
    path = figure_dir / "E02_F3_entropy_layer_progress.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_path_auc(path_effects: pd.DataFrame, figure_dir: Path) -> Path:
    view = path_effects[
        (path_effects["representation"] == "mean_w128_s64")
        & (path_effects["scope"] == "progress_block")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, feature in zip(axes.flat, PATH_FEATURES):
        part = view[view["feature"] == feature]
        for layer, group in part.groupby("layer"):
            group = group.sort_values("progress_bin")
            axis.plot(
                group["progress_bin"],
                group["within_q_auc"],
                marker="o",
                label=f"L{layer}",
            )
        axis.axhline(0.5, color="#555555", linestyle="--")
        axis.set_title(feature)
        axis.set_xticks(range(10))
        axis.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False)
    fig.supxlabel("Relative-progress block")
    fig.supylabel("Question-equal within-Q AUC")
    fig.tight_layout()
    path = figure_dir / "E02_F4_path_geometry_auc.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    output_dir: Path,
    confirmatory: pd.DataFrame,
    predictors: pd.DataFrame,
    progress: pd.DataFrame,
    entropy: pd.DataFrame,
    path: pd.DataFrame,
    figure_paths: list[Path],
    run_label: str,
) -> Path:
    lines = [
        f"# Experiment 02 Results: {run_label}",
        "",
        f"- Questions: {progress['question_id'].nunique()}",
        f"- Rollouts: {progress[['question_id', 'rollout_id']].drop_duplicates().shape[0]}",
        f"- Strict 3+3 questions: {len(strict_questions(progress))}",
        f"- Entropy layers: {entropy['layer'].nunique()}",
        f"- Path rows: {len(path)}",
        "",
        "## Frozen Confirmatory Cells",
        "",
        "| metric | cohort | bin | questions | within-Q AUC | 95% CI | sign fraction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in confirmatory.iterrows():
        lines.append(
            f"| {row['metric']} | {row['cohort']} | {int(row['progress_bin'])} | "
            f"{int(row['n_questions'])} | {row['within_q_auc']:.4f} | "
            f"[{row['auc_ci_low']:.4f}, {row['auc_ci_high']:.4f}] | "
            f"{row['sign_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Main replication is supported only when the main 2+2 cohort CI lower bound exceeds 0.5.",
            "All entropy and path results remain exploratory.",
            "",
            "## Length Controls",
            "",
        ]
    )
    for _, row in predictors.iterrows():
        lines.append(
            f"- {row['metric']} / {row['cohort']}: feature={row['feature_only_auc']:.4f}, "
            f"length={row['length_only_auc']:.4f}, combined={row['combined_auc']:.4f}."
        )
    lines.extend(["", "## Figures", ""])
    lines.extend(f"- {path.name}" for path in figure_paths)
    report = output_dir / "EXPERIMENT02_RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = load_shards(input_dir / "progress_features")
    entropy = load_shards(input_dir / "entropy_features")
    path = load_shards(input_dir / "path_features")

    confirmatory, curves, predictors = confirmatory_results(
        progress,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    entropy_effects = exploratory_effect_table(
        entropy,
        group_keys=["layer", "progress_bin"],
        features=ENTROPY_FEATURES,
    )
    path_effects = exploratory_effect_table(
        path,
        group_keys=["representation", "layer", "scope", "progress_bin"],
        features=PATH_FEATURES,
    )

    progress.to_parquet(output_dir / "experiment02_progress_features.parquet", index=False)
    entropy.to_parquet(output_dir / "experiment02_entropy_features.parquet", index=False)
    path.to_parquet(output_dir / "experiment02_path_features.parquet", index=False)
    confirmatory.to_csv(output_dir / "experiment02_confirmatory.csv", index=False)
    curves.to_csv(output_dir / "experiment02_auc_curves.csv", index=False)
    predictors.to_csv(output_dir / "experiment02_length_controls.csv", index=False)
    entropy_effects.to_csv(output_dir / "experiment02_entropy_effects.csv", index=False)
    path_effects.to_csv(output_dir / "experiment02_path_effects.csv", index=False)

    figures = [
        plot_progress_spaghetti(progress, figure_dir),
        plot_auc_curves(curves, figure_dir),
        plot_entropy_heatmaps(entropy_effects, figure_dir),
        plot_path_auc(path_effects, figure_dir),
    ]
    report = write_report(
        output_dir,
        confirmatory,
        predictors,
        progress,
        entropy,
        path,
        figures,
        args.run_label,
    )
    meta = {
        "run_label": args.run_label,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "questions": int(progress["question_id"].nunique()),
        "rollouts": int(
            progress[["question_id", "rollout_id"]].drop_duplicates().shape[0]
        ),
        "strict_questions": len(strict_questions(progress)),
        "confirmatory_cells": list(MAIN_CELLS),
        "figures": [str(path) for path in figures],
        "report": str(report),
    }
    (output_dir / "analysis_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
