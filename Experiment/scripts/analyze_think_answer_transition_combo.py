#!/usr/bin/env python3
"""No-forward think-to-answer transition combo analysis.

This script joins existing semantic-step basin outputs for the think and answer
segments. It tests whether correctness is better predicted by transition and
stability features than by either segment alone.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_COLUMNS = {
    "question_id",
    "rollout_id",
    "layer",
    "is_correct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze think-to-answer transition features.")
    parser.add_argument("--think-dir", default="server_results/angle_results_smoke500")
    parser.add_argument("--answer-dir", default="server_results/angle_results_answer_smoke500")
    parser.add_argument("--output-dir", default="server_results/think_answer_transition_combo")
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def safe_mean(values: Any) -> float:
    arr = finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_std(values: Any) -> float:
    arr = finite(values)
    return float(np.std(arr)) if arr.size else float("nan")


def safe_percentile(values: Any, q: float) -> float:
    arr = finite(values)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(score)
    y_true = y_true[valid]
    score = score[valid]
    pos = score[y_true]
    neg = score[~y_true]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    cmp = pos[:, None] - neg[None, :]
    return float((np.sum(cmp > 0) + 0.5 * np.sum(cmp == 0)) / cmp.size)


def aucs_by_question(df: pd.DataFrame, score_col: str) -> np.ndarray:
    aucs: list[float] = []
    for _, group in df.groupby("question_id"):
        y = group["is_correct"].to_numpy(dtype=bool)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        auc = binary_auc(y, group[score_col].to_numpy(dtype=np.float64))
        if np.isfinite(auc):
            aucs.append(auc)
    return np.asarray(aucs, dtype=np.float64)


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = finite(values)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or n_boot <= 0:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        means[idx] = float(np.mean(rng.choice(values, size=values.size, replace=True)))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def zscore_by_layer(df: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=np.float64)
    for _, idx in df.groupby("layer").groups.items():
        values = df.loc[idx, col].to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        if not valid.any():
            continue
        mean = float(np.mean(values[valid]))
        std = float(np.std(values[valid]))
        if std <= 1e-12:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (values - mean) / std
    return out


def summarize_step_segment(steps: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    needed = {"question_id", "rollout_id", "layer", "is_correct", "relative_step_pos"}
    missing = needed - set(steps.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    steps = steps.copy()
    steps["question_id"] = steps["question_id"].astype(str)
    for (qid, rid, layer, is_correct), group in steps.groupby(
        ["question_id", "rollout_id", "layer", "is_correct"], sort=False
    ):
        group = group.sort_values("relative_step_pos")
        early = group[group["relative_step_pos"].astype(float) <= 0.30]
        late = group[group["relative_step_pos"].astype(float) >= 0.75]
        if early.empty:
            early = group.head(max(1, math.ceil(0.25 * len(group))))
        if late.empty:
            late = group.tail(max(1, math.ceil(0.25 * len(group))))

        margin = group.get("state_margin", pd.Series(dtype=np.float64))
        gain = group.get("step_correct_gain", pd.Series(dtype=np.float64))
        turn = group.get("turn_to_correct", pd.Series(dtype=np.float64))
        productive = group.get("productive_turn", pd.Series(dtype=np.float64))

        rows.append(
            {
                "question_id": str(qid),
                "rollout_id": int(rid),
                "layer": int(layer),
                "is_correct": bool(is_correct),
                f"{prefix}_step_count": int(len(group)),
                f"{prefix}_early_margin_mean": safe_mean(early.get("state_margin", [])),
                f"{prefix}_late_margin_mean": safe_mean(late.get("state_margin", [])),
                f"{prefix}_final_margin": safe_mean(group.tail(1).get("state_margin", [])),
                f"{prefix}_margin_mean": safe_mean(margin),
                f"{prefix}_margin_std": safe_std(margin),
                f"{prefix}_mean_gain": safe_mean(gain),
                f"{prefix}_early_gain_mean": safe_mean(early.get("step_correct_gain", [])),
                f"{prefix}_late_gain_mean": safe_mean(late.get("step_correct_gain", [])),
                f"{prefix}_gain_abs_mean": safe_mean(np.abs(finite(gain))),
                f"{prefix}_gain_std": safe_std(gain),
                f"{prefix}_mean_turn": safe_mean(turn),
                f"{prefix}_early_turn_mean": safe_mean(early.get("turn_to_correct", [])),
                f"{prefix}_late_turn_mean": safe_mean(late.get("turn_to_correct", [])),
                f"{prefix}_productive_turn_p90": safe_percentile(productive, 90),
            }
        )
    return pd.DataFrame(rows)


def build_transition_features(think_steps: pd.DataFrame, answer_steps: pd.DataFrame) -> pd.DataFrame:
    think = summarize_step_segment(think_steps, "think")
    answer = summarize_step_segment(answer_steps, "answer")
    merged = think.merge(
        answer,
        on=["question_id", "rollout_id", "layer", "is_correct"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return merged

    merged["transition_margin_drop"] = (
        merged["answer_early_margin_mean"] - merged["think_late_margin_mean"]
    )
    merged["transition_final_drop"] = merged["answer_final_margin"] - merged["think_final_margin"]
    merged["transition_margin_abs_drop"] = np.abs(merged["transition_margin_drop"])
    merged["transition_stability_score"] = -merged["transition_margin_abs_drop"]
    merged["answer_stability_score"] = -np.abs(merged["answer_mean_gain"])
    merged["answer_lock_score"] = -merged["answer_mean_gain"]
    merged["answer_low_early_margin_score"] = -merged["answer_early_margin_mean"]
    merged["answer_low_late_margin_score"] = -merged["answer_late_margin_mean"]
    merged["think_low_late_margin_score"] = -merged["think_late_margin_mean"]
    merged["answer_low_gain_abs_score"] = -merged["answer_gain_abs_mean"]

    combo_cols = [
        "answer_lock_score",
        "answer_low_early_margin_score",
        "answer_stability_score",
        "transition_stability_score",
    ]
    for col in combo_cols:
        merged[f"z_{col}"] = zscore_by_layer(merged, col)
    merged["combo_answer_lock_stability"] = merged[[f"z_{col}" for col in combo_cols]].mean(axis=1)

    combo_cols2 = [
        "think_low_late_margin_score",
        "answer_low_early_margin_score",
        "answer_lock_score",
    ]
    for col in combo_cols2:
        zcol = f"z_{col}"
        if zcol not in merged.columns:
            merged[zcol] = zscore_by_layer(merged, col)
    merged["combo_think_answer_low_margin"] = merged[[f"z_{col}" for col in combo_cols2]].mean(axis=1)

    return merged


def feature_columns(features: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in features.columns:
        if col in ID_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(features[col]):
            cols.append(col)
    return cols


def evaluate_transition_features(features: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if features.empty:
        return pd.DataFrame(rows)
    for layer in sorted(features["layer"].unique()):
        sub = features[features["layer"] == layer]
        for feature in feature_columns(sub):
            aucs = aucs_by_question(sub, feature)
            if aucs.size == 0:
                continue
            neg = 1.0 - aucs
            low, high = bootstrap_ci(aucs, n_boot, seed)
            nlow, nhigh = bootstrap_ci(neg, n_boot, seed + 17)
            rows.append(
                {
                    "layer": int(layer),
                    "feature": feature,
                    "n_questions": int(aucs.size),
                    "mean_auc_pos": float(np.mean(aucs)),
                    "ci_low_pos": low,
                    "ci_high_pos": high,
                    "mean_auc_neg": float(np.mean(neg)),
                    "ci_low_neg": nlow,
                    "ci_high_neg": nhigh,
                    "best_auc": float(max(np.mean(aucs), np.mean(neg))),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("best_auc", ascending=False).reset_index(drop=True)
    return result


def plot_top_auc(eval_df: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    import matplotlib.pyplot as plt

    view = eval_df.sort_values("best_auc", ascending=False).head(top_n).copy()
    if view.empty:
        return
    view["label"] = view.apply(lambda r: f"L{int(r['layer'])} {r['feature']}", axis=1)
    fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.32 * len(view))))
    y = np.arange(len(view))
    ax.barh(y, view["best_auc"].to_numpy(dtype=np.float64), color="#3A78C2")
    ax.axvline(0.5, color="#222222", linewidth=1.0, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(view["label"])
    ax.invert_yaxis()
    ax.set_xlabel("within-question AUROC (best direction)")
    ax.set_title("Think-to-answer transition combo features")
    ax.set_xlim(0.45, max(0.75, float(view["best_auc"].max()) + 0.03))
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_margin_scatter(features: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    layers = sorted(features["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(5.8 * len(layers), 4.6), sharex=False, sharey=False)
    if len(layers) == 1:
        axes = [axes]
    colors = {True: "#2F6BFF", False: "#D94B4B"}
    labels = {True: "correct rollout", False: "wrong rollout"}
    for ax, layer in zip(axes, layers):
        sub = features[features["layer"] == layer]
        for is_correct in [True, False]:
            part = sub[sub["is_correct"] == is_correct]
            ax.scatter(
                part["think_late_margin_mean"],
                part["answer_early_margin_mean"],
                s=13,
                alpha=0.42,
                color=colors[is_correct],
                label=labels[is_correct],
                edgecolors="none",
            )
        ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
        ax.axvline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("think late state margin")
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("answer early state margin")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Think late margin vs answer early margin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_think_final_margin(features: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    layers = sorted(features["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(5.2 * len(layers), 4.6), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    colors = {True: "#2F6BFF", False: "#D94B4B"}
    labels = {True: "correct rollout", False: "wrong rollout"}
    rng = np.random.default_rng(2026)
    for ax, layer in zip(axes, layers):
        sub = features[features["layer"] == layer]
        data = [
            finite(sub[sub["is_correct"] == True]["think_final_margin"]),
            finite(sub[sub["is_correct"] == False]["think_final_margin"]),
        ]
        parts = ax.violinplot(data, positions=[0, 1], widths=0.72, showmeans=True, showextrema=False)
        for body, is_correct in zip(parts["bodies"], [True, False]):
            body.set_facecolor(colors[is_correct])
            body.set_edgecolor(colors[is_correct])
            body.set_alpha(0.22)
        if "cmeans" in parts:
            parts["cmeans"].set_color("#222222")
            parts["cmeans"].set_linewidth(1.2)
        for xpos, values, is_correct in zip([0, 1], data, [True, False]):
            if values.size == 0:
                continue
            sample = values if values.size <= 900 else rng.choice(values, size=900, replace=False)
            jitter = rng.normal(0.0, 0.045, size=sample.size)
            ax.scatter(
                np.full(sample.size, xpos) + jitter,
                sample,
                s=7,
                alpha=0.18,
                color=colors[is_correct],
                edgecolors="none",
            )
        ax.axhline(0.0, color="#333333", linewidth=0.9, alpha=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([labels[True], labels[False]], rotation=12, ha="right")
        ax.set_title(f"Layer {layer}")
        ax.grid(True, axis="y", alpha=0.22)
    axes[0].set_ylabel("think final state margin")
    fig.suptitle("Thinking final correct-answer margin by rollout correctness")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, features: pd.DataFrame, eval_df: pd.DataFrame, figures: list[Path]) -> Path:
    report = output_dir / "H_TRANSITION_COMBO_RESULTS.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Think-to-Answer Transition Combo Results\n\n")
        f.write("No model forward is used. This analysis joins existing think-stage and answer-stage basin parquet files.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Joined rollout-layer rows: {len(features)}\n")
        f.write(f"- Rollouts: {features[['question_id', 'rollout_id']].drop_duplicates().shape[0] if not features.empty else 0}\n")
        f.write(f"- Questions: {features['question_id'].nunique() if not features.empty else 0}\n")
        f.write(f"- Layers: {', '.join(str(x) for x in sorted(features['layer'].unique())) if not features.empty else ''}\n")
        f.write("\n## Top Features\n\n")
        f.write("| layer | feature | questions | AUROC(+feature) | 95% CI | AUROC(-feature) | 95% CI | best |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for row in eval_df.head(30).itertuples(index=False):
            f.write(
                f"| {int(row.layer)} | {row.feature} | {int(row.n_questions)} | "
                f"{row.mean_auc_pos:.4f} | [{row.ci_low_pos:.4f}, {row.ci_high_pos:.4f}] | "
                f"{row.mean_auc_neg:.4f} | [{row.ci_low_neg:.4f}, {row.ci_high_neg:.4f}] | "
                f"{row.best_auc:.4f} |\n"
            )
        f.write("\n## Figures\n\n")
        for fig in figures:
            f.write(f"- `figures/{fig.name}`\n")
        f.write("\n## Notes\n\n")
        f.write("- `transition_margin_drop = answer_early_margin_mean - think_late_margin_mean`.\n")
        f.write("- `answer_lock_score = -answer_mean_gain`, so larger means less answer-stage gain.\n")
        f.write("- `answer_stability_score = -abs(answer_mean_gain)`, so larger means less answer-stage drift.\n")
    return report


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    think_dir = Path(args.think_dir)
    answer_dir = Path(args.answer_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    think_steps = pd.read_parquet(think_dir / "semantic_step_basin_steps.parquet")
    answer_steps = pd.read_parquet(answer_dir / "semantic_step_basin_steps.parquet")
    think_steps = think_steps[think_steps["layer"].isin(layers)].copy()
    answer_steps = answer_steps[answer_steps["layer"].isin(layers)].copy()

    features = build_transition_features(think_steps, answer_steps)
    eval_df = evaluate_transition_features(features, args.bootstrap, args.seed)

    feature_path = output_dir / "think_answer_transition_features.parquet"
    eval_path = output_dir / "think_answer_transition_eval.csv"
    features.to_parquet(feature_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    figures: list[Path] = []
    fig = figures_dir / "H1_transition_combo_top_auc.png"
    plot_top_auc(eval_df, fig)
    figures.append(fig)
    fig = figures_dir / "H2_think_late_vs_answer_early_margin.png"
    plot_margin_scatter(features, fig)
    figures.append(fig)
    fig = figures_dir / "H3_think_final_margin_by_correctness.png"
    plot_think_final_margin(features, fig)
    figures.append(fig)
    report = write_report(output_dir, features, eval_df, figures)

    print(f"think: {think_dir}")
    print(f"answer: {answer_dir}")
    print(f"saved features: {feature_path} ({len(features)} rows)")
    print(f"saved eval: {eval_path} ({len(eval_df)} rows)")
    print(f"saved report: {report}")
    for fig in figures:
        print(f"saved figure: {fig}")
    if not eval_df.empty:
        print(eval_df.head(20))


if __name__ == "__main__":
    main()
