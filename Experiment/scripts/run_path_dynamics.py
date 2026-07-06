#!/usr/bin/env python3
"""Experiment E: path velocity and acceleration over chunk hidden states.

This script reads the chunk-mean hidden-state shards produced by
extract_chunk_hidden_qwen3vl.py and computes adjacent cross-chunk displacement
dynamics.  No model forward is used.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only for very small local envs.
    def tqdm(iterable, **_: Any):
        return iterable


EPS = 1e-12
PRIMARY_DIRECTIONS = {
    "d_late_mean": "neg",
    "d_ratio_late_early": "neg",
    "pv_late_max": "neg",
    "d_hist_late_min": "pos",
}
ID_COLUMNS = {
    "question_id",
    "rollout_id",
    "layer",
    "is_correct",
    "num_chunks",
    "n_steps",
    "response_length",
    "source_shard",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment E path dynamics.")
    parser.add_argument("--input-glob", default="chunklayer/h_gpu*.npz")
    parser.add_argument("--output-dir", default="path_dynamics_results")
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-shards", type=int, default=-1)
    parser.add_argument("--limit-rollouts-per-shard", type=int, default=-1)
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def path_dynamics_from_points(points: np.ndarray) -> dict[str, np.ndarray]:
    """Return adjacent L2 distance d, first difference pv, second difference pa."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"points must be [n_points, dim], got shape {points.shape}")
    if points.shape[0] < 2:
        empty = np.asarray([], dtype=np.float64)
        return {"d": empty, "pv": empty, "pa": empty}

    steps = points[1:] - points[:-1]
    d = np.linalg.norm(steps, axis=1).astype(np.float64)
    pv = np.diff(d).astype(np.float64) if d.size >= 2 else np.asarray([], dtype=np.float64)
    pa = np.diff(pv).astype(np.float64) if pv.size >= 2 else np.asarray([], dtype=np.float64)
    return {"d": d, "pv": pv, "pa": pa}


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def late_values(values: np.ndarray) -> np.ndarray:
    values = finite(values)
    if values.size == 0:
        return values
    start = int(math.floor(0.75 * values.size))
    start = min(start, values.size - 1)
    return values[start:]


def early_values(values: np.ndarray) -> np.ndarray:
    values = finite(values)
    if values.size == 0:
        return values
    end = int(math.ceil(0.25 * values.size))
    end = max(1, min(end, values.size))
    return values[:end]


def safe_mean(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.mean(values)) if values.size else float("nan")


def safe_std(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.std(values)) if values.size else float("nan")


def safe_min(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.min(values)) if values.size else float("nan")


def safe_max(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.max(values)) if values.size else float("nan")


def summarize_path_dynamics(d: np.ndarray) -> dict[str, float]:
    """Summarize one rollout-layer distance sequence into scalar features."""
    d = finite(d)
    if d.size == 0:
        return {
            "path_length": float("nan"),
            "d_mean": float("nan"),
            "d_std": float("nan"),
            "d_cv": float("nan"),
            "d_max": float("nan"),
            "d_early_mean": float("nan"),
            "d_late_mean": float("nan"),
            "d_late_max": float("nan"),
            "d_late_std": float("nan"),
            "d_ratio_late_early": float("nan"),
            "pv_late_min": float("nan"),
            "pv_late_max": float("nan"),
            "pv_late_mean": float("nan"),
            "pv_late_std": float("nan"),
            "pv_abs_late_mean": float("nan"),
            "d_hist_late_min": float("nan"),
            "d_hist_late_max": float("nan"),
            "d_hist_late_mean": float("nan"),
        }

    pv = np.diff(d) if d.size >= 2 else np.asarray([], dtype=np.float64)
    hist_delta = np.full(d.shape, np.nan, dtype=np.float64)
    for idx in range(1, d.size):
        hist_delta[idx] = d[idx] - float(np.mean(d[:idx]))

    late_d = late_values(d)
    early_d = early_values(d)
    late_pv = late_values(pv)
    late_hist = late_values(hist_delta)
    d_mean = safe_mean(d)
    d_std = safe_std(d)
    early_mean = safe_mean(early_d)
    late_mean = safe_mean(late_d)

    return {
        "path_length": float(np.sum(d)),
        "d_mean": d_mean,
        "d_std": d_std,
        "d_cv": float(d_std / d_mean) if np.isfinite(d_mean) and abs(d_mean) > EPS else float("nan"),
        "d_max": safe_max(d),
        "d_early_mean": early_mean,
        "d_late_mean": late_mean,
        "d_late_max": safe_max(late_d),
        "d_late_std": safe_std(late_d),
        "d_ratio_late_early": (
            float(late_mean / early_mean)
            if np.isfinite(late_mean) and np.isfinite(early_mean) and abs(early_mean) > EPS
            else float("nan")
        ),
        "pv_late_min": safe_min(late_pv),
        "pv_late_max": safe_max(late_pv),
        "pv_late_mean": safe_mean(late_pv),
        "pv_late_std": safe_std(late_pv),
        "pv_abs_late_mean": safe_mean(np.abs(late_pv)),
        "d_hist_late_min": safe_min(late_hist),
        "d_hist_late_max": safe_max(late_hist),
        "d_hist_late_mean": safe_mean(late_hist),
    }


def sorted_chunk_indices(z: np.lib.npyio.NpzFile, rollout_id: int) -> np.ndarray:
    idx = np.where(z["rollout_idx"] == rollout_id)[0]
    if idx.size == 0:
        return idx
    starts = z["chunk_start"][idx]
    return idx[np.argsort(starts)]


def rollout_order(z: np.lib.npyio.NpzFile, limit: int) -> list[int]:
    order = [idx for idx, skipped in enumerate(z["meta_skipped"]) if not skipped]
    if limit is not None and limit >= 0:
        order = order[:limit]
    return order


def rows_for_shard(
    z: np.lib.npyio.NpzFile,
    layers: list[int],
    source_shard: str,
    limit_rollouts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    h_chunk = z["h_chunk"]
    n_layers = h_chunk.shape[1]
    dynamics_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for local_idx in tqdm(rollout_order(z, limit_rollouts), desc=f"path {Path(source_shard).name}", leave=False):
        chunk_idx = sorted_chunk_indices(z, local_idx)
        if chunk_idx.size < 2:
            continue

        qid = str(z["meta_id"][local_idx])
        rollout_id = int(z["meta_roll_index"][local_idx])
        is_correct = bool(z["meta_is_correct"][local_idx])
        response_length = int(z["meta_response_len"][local_idx])
        chunk_starts = z["chunk_start"][chunk_idx]
        chunk_ends = z["chunk_end"][chunk_idx]

        for layer in layers:
            if layer < 0 or layer >= n_layers:
                continue
            points = h_chunk[chunk_idx, layer, :].astype(np.float32, copy=False)
            dyn = path_dynamics_from_points(points)
            d = dyn["d"]
            pv = dyn["pv"]
            pa = dyn["pa"]
            if d.size == 0:
                continue

            base = {
                "question_id": qid,
                "rollout_id": rollout_id,
                "layer": int(layer),
                "is_correct": is_correct,
                "response_length": response_length,
                "num_chunks": int(chunk_idx.size),
                "n_steps": int(d.size),
                "source_shard": source_shard,
            }
            summary_rows.append({**base, **summarize_path_dynamics(d)})

            for step_id, d_value in enumerate(d):
                row = {
                    **base,
                    "step_id": int(step_id),
                    "from_chunk_id": int(step_id),
                    "to_chunk_id": int(step_id + 1),
                    "from_chunk_start": int(chunk_starts[step_id]),
                    "from_chunk_end": int(chunk_ends[step_id]),
                    "to_chunk_start": int(chunk_starts[step_id + 1]),
                    "to_chunk_end": int(chunk_ends[step_id + 1]),
                    "relative_pos": float(step_id / max(d.size - 1, 1)),
                    "d": float(d_value),
                    "pv": float(pv[step_id - 1]) if step_id >= 1 and step_id - 1 < pv.size else np.nan,
                    "pa": float(pa[step_id - 2]) if step_id >= 2 and step_id - 2 < pa.size else np.nan,
                }
                dynamics_rows.append(row)

    return dynamics_rows, summary_rows


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
    values = finite(values)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        means.append(float(np.mean(rng.choice(values, size=values.size, replace=True))))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def aucs_by_question(df: pd.DataFrame, score_col: str) -> np.ndarray:
    aucs = []
    for _, group in df.groupby("question_id"):
        y = group["is_correct"].to_numpy(dtype=bool)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        auc = binary_auc(y, group[score_col].to_numpy(dtype=np.float64))
        if np.isfinite(auc):
            aucs.append(auc)
    return np.asarray(aucs, dtype=np.float64)


def evaluate_summary_features(summary: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    feature_cols = [col for col in summary.columns if col not in ID_COLUMNS]
    rows = []
    for layer in sorted(summary["layer"].unique()):
        sub_layer = summary[summary["layer"] == layer]
        for feature in feature_cols:
            aucs = aucs_by_question(sub_layer, feature)
            if aucs.size == 0:
                continue
            neg_aucs = 1.0 - aucs
            pos_low, pos_high = bootstrap_ci(aucs, n_boot, seed)
            neg_low, neg_high = bootstrap_ci(neg_aucs, n_boot, seed)
            direction = PRIMARY_DIRECTIONS.get(feature, "")
            committed = float(np.mean(neg_aucs)) if direction == "neg" else float(np.mean(aucs))
            rows.append(
                {
                    "layer": int(layer),
                    "feature": feature,
                    "is_primary": feature in PRIMARY_DIRECTIONS,
                    "direction": direction,
                    "n_questions": int(aucs.size),
                    "mean_auc_pos": float(np.mean(aucs)),
                    "median_auc_pos": float(np.median(aucs)),
                    "ci_low_pos": pos_low,
                    "ci_high_pos": pos_high,
                    "mean_auc_neg": float(np.mean(neg_aucs)),
                    "median_auc_neg": float(np.median(neg_aucs)),
                    "ci_low_neg": neg_low,
                    "ci_high_neg": neg_high,
                    "committed_auc": committed,
                    "best_auc": float(max(np.mean(aucs), np.mean(neg_aucs))),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["is_primary", "best_auc"], ascending=[False, False])
    return result


def evaluate_chunk_auc(dynamics: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for feature in ["d", "pv"]:
        for (layer, step_id), group in dynamics.groupby(["layer", "step_id"]):
            aucs = aucs_by_question(group, feature)
            if aucs.size == 0:
                continue
            neg_aucs = 1.0 - aucs
            pos_low, pos_high = bootstrap_ci(aucs, n_boot, seed)
            neg_low, neg_high = bootstrap_ci(neg_aucs, n_boot, seed)
            rows.append(
                {
                    "layer": int(layer),
                    "step_id": int(step_id),
                    "feature": feature,
                    "n_questions": int(aucs.size),
                    "mean_auc_pos": float(np.mean(aucs)),
                    "ci_low_pos": pos_low,
                    "ci_high_pos": pos_high,
                    "mean_auc_neg": float(np.mean(neg_aucs)),
                    "ci_low_neg": neg_low,
                    "ci_high_neg": neg_high,
                    "best_auc": float(max(np.mean(aucs), np.mean(neg_aucs))),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["feature", "layer", "step_id"])
    return result


def format_ci(row: pd.Series, suffix: str) -> str:
    return f"[{row[f'ci_low_{suffix}']:.4f}, {row[f'ci_high_{suffix}']:.4f}]"


def write_report(
    output_dir: Path,
    dynamics: pd.DataFrame,
    summary: pd.DataFrame,
    eval_df: pd.DataFrame,
    chunk_auc: pd.DataFrame,
    figure_paths: list[Path] | None = None,
) -> Path:
    report = output_dir / "PATH_DYNAMICS_RESULTS.md"
    figure_paths = figure_paths or []
    primary = eval_df[eval_df["is_primary"]].copy() if not eval_df.empty else pd.DataFrame()
    top = eval_df.sort_values("best_auc", ascending=False).head(30) if not eval_df.empty else pd.DataFrame()

    with report.open("w", encoding="utf-8") as f:
        f.write("# Path Dynamics Results\n\n")
        f.write("Experiment E decomposes Method 3 path length into adjacent cross-chunk displacement dynamics.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Chunk-step rows: {len(dynamics)}\n")
        f.write(f"- Rollout-layer rows: {len(summary)}\n")
        f.write(f"- Questions: {summary['question_id'].nunique() if not summary.empty else 0}\n")
        f.write(f"- Rollouts: {summary[['question_id', 'rollout_id']].drop_duplicates().shape[0] if not summary.empty else 0}\n")
        f.write(f"- Layers: {', '.join(map(str, sorted(summary['layer'].unique()))) if not summary.empty else 'none'}\n")
        f.write("- Source representation: chunk-mean hidden states from `chunklayer/h_gpu*.npz`\n\n")

        f.write("## Primary Metrics\n\n")
        f.write("| layer | feature | direction | questions | committed AUROC | AUROC(+feature) | AUROC(-feature) |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|\n")
        for _, row in primary.iterrows():
            pos = f"{row['mean_auc_pos']:.4f} {format_ci(row, 'pos')}"
            neg = f"{row['mean_auc_neg']:.4f} {format_ci(row, 'neg')}"
            f.write(
                f"| {int(row['layer'])} | {row['feature']} | {row['direction'] or 'both'} | "
                f"{int(row['n_questions'])} | {row['committed_auc']:.4f} | {pos} | {neg} |\n"
            )

        f.write("\n## Top Rollout-Level Features\n\n")
        f.write("| layer | feature | primary | questions | best AUROC | AUROC(+feature) | AUROC(-feature) |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for _, row in top.iterrows():
            f.write(
                f"| {int(row['layer'])} | {row['feature']} | {bool(row['is_primary'])} | "
                f"{int(row['n_questions'])} | {row['best_auc']:.4f} | "
                f"{row['mean_auc_pos']:.4f} | {row['mean_auc_neg']:.4f} |\n"
            )

        if not chunk_auc.empty:
            d_auc = chunk_auc[chunk_auc["feature"] == "d"].copy()
            f.write("\n## Chunk-Level AUROC Using -d\n\n")
            f.write("| layer | step | questions | AUROC(-d) | 95% CI |\n")
            f.write("|---:|---:|---:|---:|---:|\n")
            for _, row in d_auc.iterrows():
                f.write(
                    f"| {int(row['layer'])} | {int(row['step_id'])} | {int(row['n_questions'])} | "
                    f"{row['mean_auc_neg']:.4f} | {format_ci(row, 'neg')} |\n"
                )

        f.write("\n## Figures\n\n")
        if figure_paths:
            for path in figure_paths:
                f.write(f"- `{path}`\n")
        else:
            f.write("- Figures not generated yet. Run `scripts/plot_path_dynamics.py`.\n")

        f.write("\n## Interpretation Notes\n\n")
        f.write("- Smaller late displacement means the rollout is moving less in hidden space near the end.\n")
        f.write("- `pv` is the change in displacement, so positive late `pv` indicates late acceleration.\n")
        f.write("- This experiment tests whether temporal patterns add information beyond total path length.\n")
        f.write("- No new model forward is required.\n")

    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(args.input_glob))
    if args.max_shards is not None and args.max_shards >= 0:
        paths = paths[: args.max_shards]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input_glob}")

    layers = parse_layers(args.layers)
    print("Experiment E: Path Velocity & Acceleration")
    print("input files:", len(paths))
    print("layers:", layers)
    print("output_dir:", output_dir)

    all_dynamics: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    for path in paths:
        print(f"loading {path}")
        with np.load(path, allow_pickle=True) as z:
            dyn_rows, summary_rows = rows_for_shard(
                z=z,
                layers=layers,
                source_shard=path,
                limit_rollouts=args.limit_rollouts_per_shard,
            )
            all_dynamics.extend(dyn_rows)
            all_summary.extend(summary_rows)

    dynamics = pd.DataFrame(all_dynamics)
    summary = pd.DataFrame(all_summary)
    if dynamics.empty or summary.empty:
        raise RuntimeError("No path dynamics rows were produced.")

    eval_df = evaluate_summary_features(summary, args.bootstrap, args.seed)
    chunk_auc = evaluate_chunk_auc(dynamics, args.bootstrap, args.seed)

    dynamics_path = output_dir / "path_dynamics.parquet"
    summary_path = output_dir / "path_dynamics_features.parquet"
    eval_path = output_dir / "path_dynamics_eval.parquet"
    chunk_auc_path = output_dir / "path_dynamics_chunk_auc.parquet"
    dynamics.to_parquet(dynamics_path, index=False)
    summary.to_parquet(summary_path, index=False)
    eval_df.to_parquet(eval_path, index=False)
    chunk_auc.to_parquet(chunk_auc_path, index=False)
    report = write_report(output_dir, dynamics, summary, eval_df, chunk_auc)

    print(f"saved {dynamics_path}: {len(dynamics)} rows")
    print(f"saved {summary_path}: {len(summary)} rows")
    print(f"saved {eval_path}: {len(eval_df)} rows")
    print(f"saved {chunk_auc_path}: {len(chunk_auc)} rows")
    print(f"saved {report}")
    print("top 10:")
    for _, row in eval_df.sort_values("best_auc", ascending=False).head(10).iterrows():
        print(
            f"L{int(row['layer'])} {row['feature']} "
            f"best={row['best_auc']:.4f} pos={row['mean_auc_pos']:.4f} "
            f"neg={row['mean_auc_neg']:.4f} q={int(row['n_questions'])}"
        )


if __name__ == "__main__":
    main()

