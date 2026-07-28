#!/usr/bin/env python3
"""Apply the two preregistered gates to entropy-band confirm120 features."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE = "raw_entropy_L14_19_bin6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-questions", type=int, default=120)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    differences = positive[:, None] - negative[None, :]
    return float((np.mean(differences > 0) + 0.5 * np.mean(differences == 0)))


def question_aucs(frame: pd.DataFrame, score: str = FEATURE) -> pd.Series:
    records = {}
    for question_id, group in frame.groupby("question_id", sort=True):
        records[str(question_id)] = pairwise_auc(
            group.loc[group["is_correct"], score].to_numpy(dtype=np.float64),
            group.loc[~group["is_correct"], score].to_numpy(dtype=np.float64),
        )
    return pd.Series(records, dtype=np.float64).dropna()


def bootstrap_mean_interval(
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


def _paired_delta_interval(
    left: pd.Series,
    right: pd.Series,
    *,
    bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    delta = (aligned["left"] - aligned["right"]).to_numpy(dtype=np.float64)
    estimate = float(delta.mean()) if delta.size else float("nan")
    low, high = bootstrap_mean_interval(delta, bootstrap=bootstrap, seed=seed)
    return estimate, low, high


def grouped_oof_scores(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"question_id", "rollout_id", "is_correct", "think_length", FEATURE}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"analysis frame missing columns: {sorted(missing)}")
    view = frame[list(required)].dropna().copy()
    view = view.drop_duplicates(["question_id", "rollout_id"], keep="first").reset_index(drop=True)
    eligible = (
        view.groupby("question_id")["is_correct"]
        .nunique()
        .loc[lambda values: values == 2]
        .index
    )
    view = view[view["question_id"].isin(eligible)].reset_index(drop=True)
    n_questions = int(view["question_id"].nunique())
    if n_questions < 2:
        raise ValueError("at least two mixed-label questions are required for grouped OOF")

    destinations = {
        "feature_score": np.full(len(view), np.nan, dtype=np.float64),
        "length_score": np.full(len(view), np.nan, dtype=np.float64),
        "combined_score": np.full(len(view), np.nan, dtype=np.float64),
    }
    fold_ids = np.full(len(view), -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=min(5, n_questions))
    max_overlap = 0
    for fold_id, (train_index, test_index) in enumerate(
        splitter.split(view, view["is_correct"], groups=view["question_id"])
    ):
        train_questions = set(view.iloc[train_index]["question_id"].astype(str))
        test_questions = set(view.iloc[test_index]["question_id"].astype(str))
        max_overlap = max(max_overlap, len(train_questions.intersection(test_questions)))
        fold_ids[test_index] = fold_id
        for columns, destination_name in (
            ([FEATURE], "feature_score"),
            (["think_length"], "length_score"),
            ([FEATURE, "think_length"], "combined_score"),
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
            model = pipeline.named_steps["logisticregression"]
            classes = list(model.classes_)
            positive_index = classes.index(True) if True in classes else classes.index(1)
            destinations[destination_name][test_index] = pipeline.predict_proba(
                view.iloc[test_index][columns]
            )[:, positive_index]

    scored = view.copy()
    scored["fold_id"] = fold_ids
    for name, values in destinations.items():
        scored[name] = values
    if scored[[*destinations]].isna().any().any() or (scored["fold_id"] < 0).any():
        raise RuntimeError("OOF scoring left unassigned rows")
    return scored, {
        "n_questions": n_questions,
        "n_rows": len(scored),
        "n_folds": int(scored["fold_id"].nunique()),
        "fold_question_overlap": max_overlap,
    }


def classify_gate_outcome(gate1_pass: bool, gate2_pass: bool) -> str:
    if not gate1_pass:
        return "replication_failed"
    if not gate2_pass:
        return "internal_correlate_not_rl_candidate"
    return "advance_to_rl_credit_assignment"


def strict_questions(frame: pd.DataFrame) -> set[str]:
    labels = (
        frame[["question_id", "rollout_id", "is_correct"]]
        .drop_duplicates()
        .groupby("question_id")["is_correct"]
        .agg(n_rollouts="size", n_correct="sum")
    )
    labels["n_wrong"] = labels["n_rollouts"] - labels["n_correct"]
    return set(labels.index[(labels["n_correct"] >= 3) & (labels["n_wrong"] >= 3)].astype(str))


def evaluate_cohort(
    frame: pd.DataFrame,
    *,
    cohort: str,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    raw_aucs = question_aucs(frame, FEATURE)
    gate1_low, gate1_high = bootstrap_mean_interval(
        raw_aucs.to_numpy(), bootstrap=bootstrap, seed=seed
    )
    oof, oof_meta = grouped_oof_scores(frame, seed=seed + 100)
    feature_aucs = question_aucs(oof, "feature_score")
    length_aucs = question_aucs(oof, "length_score")
    combined_aucs = question_aucs(oof, "combined_score")
    feature_delta = _paired_delta_interval(
        feature_aucs, length_aucs, bootstrap=bootstrap, seed=seed + 200
    )
    combined_delta = _paired_delta_interval(
        combined_aucs, length_aucs, bootstrap=bootstrap, seed=seed + 300
    )
    gate1_pass = bool(np.isfinite(gate1_low) and gate1_low > 0.5)
    gate2_pass = bool(np.isfinite(combined_delta[1]) and combined_delta[1] > 0.0)
    result = {
        "cohort": cohort,
        "n_questions": int(raw_aucs.size),
        "n_rollouts": int(len(frame)),
        "within_q_auc": float(raw_aucs.mean()),
        "within_q_auc_ci_low": gate1_low,
        "within_q_auc_ci_high": gate1_high,
        "sign_fraction": float((raw_aucs > 0.5).mean()),
        "feature_only_oof_auc": float(feature_aucs.mean()),
        "length_only_oof_auc": float(length_aucs.mean()),
        "combined_oof_auc": float(combined_aucs.mean()),
        "delta_feature_vs_length": feature_delta[0],
        "delta_feature_vs_length_ci_low": feature_delta[1],
        "delta_feature_vs_length_ci_high": feature_delta[2],
        "delta_combined_vs_length": combined_delta[0],
        "delta_combined_vs_length_ci_low": combined_delta[1],
        "delta_combined_vs_length_ci_high": combined_delta[2],
        "fold_question_overlap": oof_meta["fold_question_overlap"],
        "gate1_signal_replication": gate1_pass,
        "gate2_length_increment": gate2_pass,
        "outcome": classify_gate_outcome(gate1_pass, gate2_pass),
    }
    per_question = pd.DataFrame(
        {
            "question_id": raw_aucs.index,
            "raw_feature_auc": raw_aucs.values,
            "feature_only_oof_auc": feature_aucs.reindex(raw_aucs.index).values,
            "length_only_oof_auc": length_aucs.reindex(raw_aucs.index).values,
            "combined_oof_auc": combined_aucs.reindex(raw_aucs.index).values,
            "cohort": cohort,
        }
    )
    oof["cohort"] = cohort
    return result, per_question, oof


def load_shards(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("question_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no feature shards in {directory}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def plot_question_means(frame: pd.DataFrame, output_dir: Path) -> Path:
    summary = (
        frame.groupby(["question_id", "is_correct"], as_index=False)[FEATURE]
        .mean()
        .pivot(index="question_id", columns="is_correct", values=FEATURE)
        .dropna()
    )
    fig, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.scatter(summary[False], summary[True], alpha=0.55, s=28, color="#2878b5")
    limits = [float(summary.min().min()), float(summary.max().max())]
    axis.plot(limits, limits, linestyle="--", color="#555555", linewidth=1)
    axis.set_xlabel("Mean raw entropy, wrong rollouts")
    axis.set_ylabel("Mean raw entropy, correct rollouts")
    axis.set_title("Frozen L14-L19 / bin6 entropy by question")
    axis.grid(alpha=0.16)
    fig.tight_layout()
    path = output_dir / "figures" / "E03_F1_question_correct_wrong.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_question_aucs(per_question: pd.DataFrame, result: dict[str, Any], output_dir: Path) -> Path:
    view = per_question[per_question["cohort"] == "main_2plus2"].sort_values("raw_feature_auc")
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.scatter(np.arange(len(view)), view["raw_feature_auc"], s=18, alpha=0.65)
    axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axis.axhline(result["within_q_auc"], color="#c44e52", linewidth=2, label="question-equal mean")
    axis.fill_between(
        [-1, len(view)],
        result["within_q_auc_ci_low"],
        result["within_q_auc_ci_high"],
        color="#c44e52",
        alpha=0.15,
        label="question-bootstrap 95% CI",
    )
    axis.set_xlim(-1, len(view))
    axis.set_xlabel("Questions sorted by within-question AUC")
    axis.set_ylabel("Within-question AUC")
    axis.set_title("Preregistered entropy-band replication")
    axis.legend(frameon=False)
    axis.grid(alpha=0.14)
    fig.tight_layout()
    path = output_dir / "figures" / "E03_F2_question_auc.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(output_dir: Path, results: pd.DataFrame, figures: list[Path]) -> Path:
    main = results.loc[results["cohort"] == "main_2plus2"].iloc[0]
    lines = [
        "# Entropy Band Confirm120 Results",
        "",
        "Frozen feature: per-token centered-coordinate raw activation entropy, averaged over L14-L19 and progress bin6. Higher values were preregistered to predict correct rollouts.",
        "",
        "| cohort | questions | within-Q AUC (95% CI) | combined-length delta (95% CI) | Gate 1 | Gate 2 | outcome |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['cohort']} | {int(row['n_questions'])} | "
            f"{row['within_q_auc']:.4f} [{row['within_q_auc_ci_low']:.4f}, {row['within_q_auc_ci_high']:.4f}] | "
            f"{row['delta_combined_vs_length']:.4f} [{row['delta_combined_vs_length_ci_low']:.4f}, {row['delta_combined_vs_length_ci_high']:.4f}] | "
            f"{bool(row['gate1_signal_replication'])} | {bool(row['gate2_length_increment'])} | {row['outcome']} |"
        )
    lines.extend(
        [
            "",
            "## Primary Decision",
            "",
            f"- Gate 1 passes only when the main-cohort AUC CI lower bound is above 0.5: **{bool(main['gate1_signal_replication'])}**.",
            f"- Gate 2 passes only when the grouped-OOF combined-minus-length AUC CI lower bound is above 0: **{bool(main['gate2_length_increment'])}**.",
            f"- Frozen outcome: **{main['outcome']}**.",
            "",
            "## Figures",
            "",
            *[f"- {path.name}" for path in figures],
        ]
    )
    path = output_dir / "ENTROPY_BAND_CONFIRM120_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_shards(Path(args.input_dir) / "features")
    n_questions = int(frame["question_id"].nunique())
    if n_questions != args.expected_questions:
        raise ValueError(f"expected exactly {args.expected_questions} questions, found {n_questions}")
    if frame[FEATURE].isna().any() or not np.isfinite(frame[FEATURE]).all():
        raise ValueError("frozen entropy feature contains missing or non-finite values")

    strict_ids = strict_questions(frame)
    cohort_specs = [
        ("main_2plus2", frame),
        ("strict_3plus3", frame[frame["question_id"].astype(str).isin(strict_ids)]),
    ]
    results = []
    question_frames = []
    oof_frames = []
    for cohort_index, (cohort, subset) in enumerate(cohort_specs):
        if subset["question_id"].nunique() < 2:
            continue
        result, per_question, oof = evaluate_cohort(
            subset,
            cohort=cohort,
            bootstrap=args.bootstrap,
            seed=args.seed + 1000 * cohort_index,
        )
        results.append(result)
        question_frames.append(per_question)
        oof_frames.append(oof)
    result_frame = pd.DataFrame(results)
    question_frame = pd.concat(question_frames, ignore_index=True)
    oof_frame = pd.concat(oof_frames, ignore_index=True)
    result_frame.to_csv(output_dir / "confirmatory_gate_results.csv", index=False)
    question_frame.to_csv(output_dir / "question_aucs.csv", index=False)
    oof_frame.to_parquet(output_dir / "oof_scores.parquet", index=False)
    frame.to_parquet(output_dir / "entropy_band_rollout_features.parquet", index=False)
    main_result = results[0]
    figures = [
        plot_question_means(frame, output_dir),
        plot_question_aucs(question_frame, main_result, output_dir),
    ]
    report = write_report(output_dir, result_frame, figures)
    metadata = {
        "feature": FEATURE,
        "direction": "higher_is_correct",
        "layers": list(range(14, 20)),
        "progress_bin": 6,
        "questions": n_questions,
        "rollouts": len(frame),
        "strict_questions": len(strict_ids),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "primary_outcome": main_result["outcome"],
        "report": str(report),
        "figures": [str(path) for path in figures],
        "alternate_cell_scan": False,
    }
    _atomic_json(output_dir / "analysis_meta.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
