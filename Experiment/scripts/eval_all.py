#!/usr/bin/env python3
"""Evaluate trajectory score NPZ files with per-question AUROC."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all score NPZ files.")
    parser.add_argument("--scores-glob", default="scores/*.npz")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
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
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_file(path: Path, n_boot: int, seed: int) -> dict[str, object]:
    z = np.load(path, allow_pickle=True)
    ids = z["id"].astype(str)
    y = z["is_correct"].astype(bool)
    score = z["score"].astype(np.float64)

    by_q: dict[str, list[int]] = defaultdict(list)
    for idx, qid in enumerate(ids):
        if np.isfinite(score[idx]):
            by_q[str(qid)].append(idx)

    per_q_auc = []
    per_q_auc_opposite = []
    used_questions = 0
    skipped_questions = 0

    for qid, indices in by_q.items():
        idx = np.asarray(indices, dtype=np.int64)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == yy.size:
            skipped_questions += 1
            continue
        used_questions += 1
        auc = binary_auc(yy, score[idx])
        per_q_auc.append(auc)
        per_q_auc_opposite.append(1.0 - auc)

    aucs = np.asarray(per_q_auc, dtype=np.float64)
    aucs_opp = np.asarray(per_q_auc_opposite, dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(aucs, n_boot, seed)
    opp_ci_low, opp_ci_high = bootstrap_ci(aucs_opp, n_boot, seed + 17)

    return {
        "metric": path.stem,
        "path": str(path),
        "n_rows": int(len(ids)),
        "n_questions_total": int(len(by_q)),
        "n_questions_used": int(used_questions),
        "n_questions_skipped": int(skipped_questions),
        "mean_auc": float(np.nanmean(aucs)) if aucs.size else float("nan"),
        "median_auc": float(np.nanmedian(aucs)) if aucs.size else float("nan"),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "mean_auc_opposite": float(np.nanmean(aucs_opp)) if aucs_opp.size else float("nan"),
        "median_auc_opposite": float(np.nanmedian(aucs_opp)) if aucs_opp.size else float("nan"),
        "opposite_ci95_low": opp_ci_low,
        "opposite_ci95_high": opp_ci_high,
    }


def verdict(row: dict[str, object]) -> str:
    mean_auc = float(row["mean_auc"])
    low = float(row["ci95_low"])
    if mean_auc > 0.55 and low > 0.50:
        return "PASS"
    if mean_auc > 0.50:
        return "MARGINAL"
    return "FAIL"


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(p) for p in glob.glob(args.scores_glob))
    if not paths:
        raise FileNotFoundError(f"No score files matched {args.scores_glob}")

    rows = [evaluate_file(path, args.bootstrap, args.seed) for path in paths]
    rows.sort(key=lambda item: float(item["mean_auc"]), reverse=True)

    json_path = results_dir / "eval_all.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = results_dir / "RESULTS.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Chunk Trajectory Evaluation Results\n\n")
        f.write("| metric | verdict | mean AUROC | median | 95% CI | opposite mean | questions |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['metric']} | {verdict(row)} | "
                f"{float(row['mean_auc']):.4f} | {float(row['median_auc']):.4f} | "
                f"[{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}] | "
                f"{float(row['mean_auc_opposite']):.4f} | {row['n_questions_used']} |\n"
            )

    print(f"saved {json_path}")
    print(f"saved {md_path}")
    print("top 10:")
    for row in rows[:10]:
        print(
            row["metric"],
            verdict(row),
            f"mean={float(row['mean_auc']):.4f}",
            f"ci=[{float(row['ci95_low']):.4f},{float(row['ci95_high']):.4f}]",
            f"opp={float(row['mean_auc_opposite']):.4f}",
            f"q={row['n_questions_used']}",
        )


if __name__ == "__main__":
    main()
