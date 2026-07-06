#!/usr/bin/env python3
"""Analyze Experiment F training-step hidden trajectory dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_SCORE_SPECS = {
    "path_score": "pos",
    "d_late_mean_score": "pos",
    "erv36_adj_late_min": "pos",
}
ID_COLS = {"step", "question_id", "rollout_id", "is_correct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment F checkpoint sweep.")
    parser.add_argument(
        "--steps",
        required=True,
        help="Comma-separated step specs like 0:experiment_f/step0,20:experiment_f/step20",
    )
    parser.add_argument("--output-dir", default="experiment_f/results")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_steps(raw: str) -> list[tuple[int, Path]]:
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Step spec must be step:path, got {part}")
        step_raw, path_raw = part.split(":", 1)
        items.append((int(step_raw), Path(path_raw)))
    return sorted(items, key=lambda x: x[0])


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


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        means.append(float(np.mean(rng.choice(values, size=values.size, replace=True))))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def load_npz_score(path: Path, step: int, value_name: str, score_name: str) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    return pd.DataFrame(
        {
            "step": step,
            "question_id": z["id"].astype(str),
            "rollout_id": z["roll_index"].astype(int),
            "is_correct": z["is_correct"].astype(bool),
            value_name: z["value"].astype(float),
            score_name: z["score"].astype(float),
        }
    )


def load_step_rollouts(step_dir: Path, step: int) -> pd.DataFrame:
    candidates = [
        step_dir / "rollouts_labeled.jsonl",
        step_dir / "rollouts_filtered_labeled.jsonl",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "step": step,
                    "question_id": str(row.get("question_id", "")),
                    "rollout_id": int(row.get("rollout_id", -1)),
                    "is_correct": bool(row.get("is_correct", False)),
                    "response_length": int(row.get("response_length", len(str(row.get("response", "")).split()))),
                }
            )
    return pd.DataFrame(rows)


def load_step_features(step: int, step_dir: Path) -> pd.DataFrame:
    frames = []
    scores_dir = step_dir / "scores"
    path_score = scores_dir / "M3_L36_path_length.npz"
    if path_score.exists():
        frames.append(load_npz_score(path_score, step, "path_length", "path_score"))

    path_dyn = step_dir / "path_dynamics_results" / "path_dynamics_features.parquet"
    if path_dyn.exists():
        df = pd.read_parquet(path_dyn)
        df = df[df["layer"] == 36].copy()
        df["step"] = step
        df["d_late_mean_score"] = -df["d_late_mean"].astype(float)
        keep = ["step", "question_id", "rollout_id", "is_correct", "d_late_mean", "d_late_mean_score", "num_chunks"]
        frames.append(df[keep])

    er_dyn = step_dir / "er_local_dynamics_results" / "local_er_dynamics_rollout_summary.parquet"
    if er_dyn.exists():
        df = pd.read_parquet(er_dyn)
        df = df[df["layer"] == 36].copy()
        df["step"] = step
        df = df.rename(
            columns={
                "erv_adj_late_min": "erv36_adj_late_min",
                "erv_hist_late_min": "erv36_hist_late_min",
            }
        )
        keep_cols = [
            "step",
            "question_id",
            "rollout_id",
            "is_correct",
            "erv36_adj_late_min",
            "erv36_hist_late_min",
            "num_chunks",
            "response_length",
        ]
        frames.append(df[[c for c in keep_cols if c in df.columns]])

    rollouts = load_step_rollouts(step_dir, step)
    if not rollouts.empty:
        frames.append(rollouts)

    if not frames:
        raise FileNotFoundError(f"No Experiment F feature files found in {step_dir}")

    merged = frames[0]
    keys = ["step", "question_id", "rollout_id"]
    for frame in frames[1:]:
        value_cols = [c for c in frame.columns if c not in keys]
        if "is_correct" in value_cols and "is_correct" in merged.columns:
            value_cols.remove("is_correct")
        value_cols = [c for c in value_cols if c not in merged.columns]
        if not value_cols:
            continue
        merged = merged.merge(frame[keys + value_cols], on=keys, how="outer")

    if "is_correct" not in merged.columns:
        raise ValueError(f"Missing is_correct in {step_dir}")
    return merged


def load_all_steps(step_specs: list[tuple[int, Path]]) -> pd.DataFrame:
    frames = []
    for step, step_dir in step_specs:
        print(f"loading step {step}: {step_dir}")
        frame = load_step_features(step, step_dir)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def per_question_aucs(df: pd.DataFrame, metric: str, direction: str) -> np.ndarray:
    aucs = []
    for _, group in df.groupby("question_id"):
        y = group["is_correct"].to_numpy(dtype=bool)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        score = group[metric].to_numpy(dtype=np.float64)
        if direction == "neg":
            score = -score
        auc = binary_auc(y, score)
        if np.isfinite(auc):
            aucs.append(auc)
    return np.asarray(aucs, dtype=np.float64)


def summarize_step_metrics(
    df: pd.DataFrame,
    score_specs: dict[str, str] | None = None,
    behavior_cols: list[str] | None = None,
    bootstrap: int = 1000,
    seed: int = 2026,
) -> pd.DataFrame:
    score_specs = score_specs or DEFAULT_SCORE_SPECS
    behavior_cols = behavior_cols or ["response_length", "num_chunks"]
    rows: list[dict[str, Any]] = []

    for step, step_df in df.groupby("step"):
        for metric, direction in score_specs.items():
            if metric not in step_df.columns:
                continue
            aucs = per_question_aucs(step_df, metric, direction)
            lo, hi = bootstrap_ci(aucs, bootstrap, seed) if aucs.size else (float("nan"), float("nan"))
            rows.append(
                {
                    "step": int(step),
                    "metric": metric,
                    "group": "auroc",
                    "direction": direction,
                    "mean_auc": float(np.mean(aucs)) if aucs.size else float("nan"),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_questions": int(aucs.size),
                    "mean_value": float("nan"),
                    "n_rollouts": int(len(step_df)),
                }
            )
            for group_name, group_df in [
                ("overall", step_df),
                ("correct", step_df[step_df["is_correct"] == True]),
                ("incorrect", step_df[step_df["is_correct"] == False]),
            ]:
                vals = group_df[metric].to_numpy(dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                rows.append(
                    {
                        "step": int(step),
                        "metric": metric,
                        "group": group_name,
                        "direction": direction,
                        "mean_auc": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n_questions": int(group_df["question_id"].nunique()),
                        "mean_value": float(np.mean(vals)) if vals.size else float("nan"),
                        "n_rollouts": int(vals.size),
                    }
                )

        for metric in behavior_cols:
            if metric not in step_df.columns:
                continue
            for group_name, group_df in [
                ("overall", step_df),
                ("correct", step_df[step_df["is_correct"] == True]),
                ("incorrect", step_df[step_df["is_correct"] == False]),
            ]:
                vals = group_df[metric].to_numpy(dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                rows.append(
                    {
                        "step": int(step),
                        "metric": metric,
                        "group": group_name,
                        "direction": "",
                        "mean_auc": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n_questions": int(group_df["question_id"].nunique()),
                        "mean_value": float(np.mean(vals)) if vals.size else float("nan"),
                        "n_rollouts": int(vals.size),
                    }
                )

        acc = float(step_df["is_correct"].mean()) if len(step_df) else float("nan")
        rows.append(
            {
                "step": int(step),
                "metric": "accuracy",
                "group": "overall",
                "direction": "",
                "mean_auc": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_questions": int(step_df["question_id"].nunique()),
                "mean_value": acc,
                "n_rollouts": int(len(step_df)),
            }
        )
    return pd.DataFrame(rows)


def plot_auroc(summary: pd.DataFrame, fig_dir: Path) -> Path:
    work = summary[summary["group"] == "auroc"].copy()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=work, x="step", y="mean_auc", hue="metric", marker="o")
    plt.axhline(0.5, color="black", linewidth=1, alpha=0.5)
    plt.title("Experiment F: within-question AUROC over training")
    plt.xlabel("Training step")
    plt.ylabel("Mean per-question AUROC")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = fig_dir / "F3_auroc_over_training.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_raw_trends(summary: pd.DataFrame, fig_dir: Path) -> list[Path]:
    paths = []
    for metric in ["path_score", "d_late_mean_score", "erv36_adj_late_min", "response_length", "num_chunks"]:
        work = summary[(summary["metric"] == metric) & (summary["group"].isin(["overall", "correct", "incorrect"]))].copy()
        if work.empty:
            continue
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=work, x="step", y="mean_value", hue="group", marker="o")
        plt.title(f"Experiment F raw trend: {metric}")
        plt.xlabel("Training step")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        path = fig_dir / f"F2_raw_{metric}.png"
        plt.savefig(path, dpi=220)
        plt.close()
        paths.append(path)
    return paths


def plot_behavior(summary: pd.DataFrame, fig_dir: Path) -> Path:
    work = summary[(summary["metric"].isin(["accuracy", "response_length"])) & (summary["group"] == "overall")].copy()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    acc = work[work["metric"] == "accuracy"]
    length = work[work["metric"] == "response_length"]
    if not acc.empty:
        ax1.plot(acc["step"], acc["mean_value"], marker="o", color="purple", label="accuracy")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Accuracy", color="purple")
    ax1.tick_params(axis="y", labelcolor="purple")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    if not length.empty:
        ax2.plot(length["step"], length["mean_value"], marker="o", color="darkorange", label="response length")
    ax2.set_ylabel("Response length", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    plt.title("Experiment F: behavior anchors")
    fig.tight_layout()
    path = fig_dir / "F1_behavior_anchors.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def write_report(output_dir: Path, summary: pd.DataFrame, figures: list[Path]) -> Path:
    report = output_dir / "TRAINING_STEP_DYNAMICS_RESULTS.md"
    auroc = summary[summary["group"] == "auroc"].copy()
    with report.open("w", encoding="utf-8") as f:
        f.write("# Training-Step Hidden Trajectory Dynamics\n\n")
        f.write("Experiment F tracks raw hidden-trajectory trends and within-question AUROC across checkpoints.\n\n")
        f.write("## AUROC Over Training\n\n")
        f.write("| step | metric | direction | questions | mean AUROC | 95% CI |\n")
        f.write("|---:|---|---|---:|---:|---:|\n")
        for _, row in auroc.sort_values(["metric", "step"]).iterrows():
            f.write(
                f"| {int(row['step'])} | {row['metric']} | {row['direction']} | "
                f"{int(row['n_questions'])} | {row['mean_auc']:.4f} | "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |\n"
            )

        f.write("\n## Behavior Anchors\n\n")
        behavior = summary[(summary["metric"].isin(["accuracy", "response_length", "num_chunks"])) & (summary["group"] == "overall")]
        f.write("| step | metric | mean value | n rollouts |\n")
        f.write("|---:|---|---:|---:|\n")
        for _, row in behavior.sort_values(["metric", "step"]).iterrows():
            f.write(f"| {int(row['step'])} | {row['metric']} | {row['mean_value']:.4f} | {int(row['n_rollouts'])} |\n")

        f.write("\n## Figures\n\n")
        for path in figures:
            f.write(f"- `{path}`\n")

        f.write("\n## Interpretation Notes\n\n")
        f.write("- Raw trends describe geometry changes; AUROC trends evaluate diagnostic validity.\n")
        f.write("- Overall raw means can move mechanically as accuracy changes, so correct/incorrect groups and AUROC are primary.\n")
        f.write("- Cross-check response length and num_chunks before interpreting raw L2 trends.\n")
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    step_specs = parse_steps(args.steps)
    features = load_all_steps(step_specs)
    summary = summarize_step_metrics(features, DEFAULT_SCORE_SPECS, ["response_length", "num_chunks"], args.bootstrap, args.seed)

    features_path = output_dir / "training_step_features.parquet"
    summary_path = output_dir / "training_step_summary.parquet"
    features.to_parquet(features_path, index=False)
    summary.to_parquet(summary_path, index=False)

    figures = [plot_behavior(summary, fig_dir), plot_auroc(summary, fig_dir)]
    figures.extend(plot_raw_trends(summary, fig_dir))
    report = write_report(output_dir, summary, figures)

    print(f"saved {features_path}: {len(features)} rows")
    print(f"saved {summary_path}: {len(summary)} rows")
    print(f"saved report: {report}")
    for path in figures:
        print(f"saved figure: {path}")


if __name__ == "__main__":
    main()
