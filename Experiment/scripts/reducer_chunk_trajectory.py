#!/usr/bin/env python3
"""Reduce chunk hidden-state shards into trajectory scores.

Inputs are the NPZ files produced by extract_chunk_hidden_qwen3vl.py:
  chunklayer/h_gpu0.npz ... chunklayer/h_gpu7.npz

Outputs are small score NPZ files under scores/.  Each score file contains
one scalar per rollout with fields:
  id, roll_index, is_correct, value, score

`value` is the raw metric. `score` follows the committed direction where larger
means more likely correct.
"""

from __future__ import annotations

import argparse
import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute chunk trajectory scores.")
    parser.add_argument("--input-glob", default="chunklayer/h_gpu*.npz")
    parser.add_argument("--scores-dir", default="scores")
    parser.add_argument(
        "--methods",
        default="M3,M1,M0",
        help="Comma-separated subset of M3,M1,M0.",
    )
    parser.add_argument(
        "--layers",
        default="24,36",
        help="Comma-separated layer indices for Method 3.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=-1,
        help="Use only the first N shards for smoke tests. -1 means all.",
    )
    parser.add_argument(
        "--limit-rollouts-per-shard",
        type=int,
        default=-1,
        help="Use only first N rollouts in each shard for smoke tests.",
    )
    return parser.parse_args()


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cos = dot / np.maximum(denom, EPS)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def trajectory_metrics(points: np.ndarray) -> dict[str, float]:
    """Compute CoE-style and trajectory metrics for [n_points, dim]."""
    points = points.astype(np.float32, copy=False)
    n_points = points.shape[0]
    if n_points < 2:
        return {
            "mag": math.nan,
            "ang": math.nan,
            "coe_c": math.nan,
            "coe_r": math.nan,
            "path_length": 0.0,
            "curvature_ratio": math.nan,
            "m_std": math.nan,
            "a_std": math.nan,
            "m_max": math.nan,
            "late_m_mean": math.nan,
            "end_drift": math.nan,
            "direction_consistency": math.nan,
            "late_curvature": math.nan,
        }

    steps = points[1:] - points[:-1]
    step_norm = np.linalg.norm(steps, axis=1)
    path_length = float(step_norm.sum())
    total_disp = float(np.linalg.norm(points[-1] - points[0]))
    total_angle = float(angle_between(points[:1], points[-1:])[0])

    m_t = step_norm / max(total_disp, EPS)
    a_raw = angle_between(points[:-1], points[1:])
    a_t = a_raw / max(total_angle, EPS)

    mag = float(np.mean(m_t))
    ang = float(np.mean(a_t))
    coe_r = float(np.mean(m_t - a_t))
    coe_c = float(
        np.sqrt(np.mean(m_t * np.cos(a_t)) ** 2 + np.mean(m_t * np.sin(a_t)) ** 2)
    )

    if len(steps) >= 2:
        s0 = steps[:-1]
        s1 = steps[1:]
        denom = np.linalg.norm(s0, axis=1) * np.linalg.norm(s1, axis=1)
        direction_cos = np.sum(s0 * s1, axis=1) / np.maximum(denom, EPS)
        direction_consistency = float(np.mean(np.clip(direction_cos, -1.0, 1.0)))
    else:
        direction_consistency = math.nan

    late_start = max(0, int(math.floor(0.75 * len(a_t))))
    late_curvature = float(np.std(a_t[late_start:])) if len(a_t[late_start:]) else math.nan
    late_m_mean = float(np.mean(m_t[late_start:])) if len(m_t[late_start:]) else math.nan

    drift_start = max(0, n_points - 4)
    end_drift = float(np.linalg.norm(points[-1] - points[drift_start]) / max(total_disp, EPS))

    return {
        "mag": mag,
        "ang": ang,
        "coe_c": coe_c,
        "coe_r": coe_r,
        "path_length": path_length,
        "curvature_ratio": float(path_length / max(total_disp, EPS)),
        "m_std": float(np.std(m_t)),
        "a_std": float(np.std(a_t)),
        "m_max": float(np.max(m_t)),
        "late_m_mean": late_m_mean,
        "end_drift": end_drift,
        "direction_consistency": direction_consistency,
        "late_curvature": late_curvature,
    }


def layer_prefix_coe_c(layer_points: np.ndarray) -> float:
    """CoE-C over a layer trajectory [n_layers, dim]."""
    return trajectory_metrics(layer_points)["coe_c"]


def init_score_store() -> dict[str, dict[str, list[Any]]]:
    return defaultdict(lambda: {"id": [], "roll_index": [], "is_correct": [], "value": [], "score": []})


def add_score(
    store: dict[str, dict[str, list[Any]]],
    name: str,
    qid: str,
    roll_index: int,
    is_correct: bool,
    value: float,
    direction: str,
) -> None:
    if value is None or not np.isfinite(value):
        return
    if direction == "neg":
        score = -float(value)
    elif direction == "pos":
        score = float(value)
    elif direction == "neg_abs":
        score = -abs(float(value))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    item = store[name]
    item["id"].append(str(qid))
    item["roll_index"].append(int(roll_index))
    item["is_correct"].append(bool(is_correct))
    item["value"].append(float(value))
    item["score"].append(float(score))


def write_score_files(scores_dir: Path, store: dict[str, dict[str, list[Any]]]) -> None:
    scores_dir.mkdir(parents=True, exist_ok=True)
    for name, item in sorted(store.items()):
        path = scores_dir / f"{name}.npz"
        np.savez_compressed(
            path,
            id=np.asarray(item["id"], dtype=str),
            roll_index=np.asarray(item["roll_index"], dtype=np.int64),
            is_correct=np.asarray(item["is_correct"], dtype=bool),
            value=np.asarray(item["value"], dtype=np.float64),
            score=np.asarray(item["score"], dtype=np.float64),
            metric=np.asarray(name),
        )
        print(f"saved {path}: {len(item['id'])} rows")


def rollout_order(z: np.lib.npyio.NpzFile, limit: int) -> list[int]:
    meta_skipped = z["meta_skipped"]
    order = [idx for idx, skipped in enumerate(meta_skipped) if not skipped]
    if limit is not None and limit >= 0:
        order = order[:limit]
    return order


def sorted_chunk_indices(z: np.lib.npyio.NpzFile, rollout_id: int) -> np.ndarray:
    idx = np.where(z["rollout_idx"] == rollout_id)[0]
    if idx.size == 0:
        return idx
    starts = z["chunk_start"][idx]
    return idx[np.argsort(starts)]


def compute_m3_for_shard(
    z: np.lib.npyio.NpzFile,
    layers: list[int],
    store: dict[str, dict[str, list[Any]]],
    limit_rollouts: int,
) -> None:
    h_chunk = z["h_chunk"]
    n_layers = h_chunk.shape[1]
    order = rollout_order(z, limit_rollouts)
    for local_idx in tqdm(order, desc="M3", leave=False):
        chunk_idx = sorted_chunk_indices(z, local_idx)
        if chunk_idx.size < 2:
            continue
        qid = str(z["meta_id"][local_idx])
        roll = int(z["meta_roll_index"][local_idx])
        label = bool(z["meta_is_correct"][local_idx])
        for layer in layers:
            if layer < 0 or layer >= n_layers:
                continue
            metrics = trajectory_metrics(h_chunk[chunk_idx, layer, :])
            prefix = f"M3_L{layer}"
            add_score(store, f"{prefix}_mag", qid, roll, label, metrics["mag"], "neg")
            add_score(store, f"{prefix}_ang", qid, roll, label, metrics["ang"], "neg")
            add_score(store, f"{prefix}_coe_c", qid, roll, label, metrics["coe_c"], "neg")
            add_score(store, f"{prefix}_curvature_ratio", qid, roll, label, metrics["curvature_ratio"], "neg")
            add_score(store, f"{prefix}_path_length", qid, roll, label, metrics["path_length"], "neg")
            add_score(store, f"{prefix}_end_drift", qid, roll, label, metrics["end_drift"], "neg")
            add_score(
                store,
                f"{prefix}_dir_consistency",
                qid,
                roll,
                label,
                metrics["direction_consistency"],
                "pos",
            )
            add_score(store, f"{prefix}_late_curvature", qid, roll, label, metrics["late_curvature"], "neg")


def compute_m1_for_shard(
    z: np.lib.npyio.NpzFile,
    store: dict[str, dict[str, list[Any]]],
    limit_rollouts: int,
) -> None:
    h_chunk = z["h_chunk"]
    order = rollout_order(z, limit_rollouts)
    for local_idx in tqdm(order, desc="M1", leave=False):
        chunk_idx = sorted_chunk_indices(z, local_idx)
        if chunk_idx.size < 1:
            continue
        qid = str(z["meta_id"][local_idx])
        roll = int(z["meta_roll_index"][local_idx])
        label = bool(z["meta_is_correct"][local_idx])
        trajectory = h_chunk[chunk_idx].reshape(-1, h_chunk.shape[-1])
        metrics = trajectory_metrics(trajectory)
        add_score(store, "M1_tat_mag", qid, roll, label, metrics["mag"], "neg")
        add_score(store, "M1_tat_ang", qid, roll, label, metrics["ang"], "neg")
        add_score(store, "M1_tat_coe_c", qid, roll, label, metrics["coe_c"], "neg")
        add_score(store, "M1_tat_curvature_ratio", qid, roll, label, metrics["curvature_ratio"], "neg")
        add_score(store, "M1_tat_m_std", qid, roll, label, metrics["m_std"], "neg")
        add_score(store, "M1_tat_a_std", qid, roll, label, metrics["a_std"], "neg")
        add_score(store, "M1_tat_m_max", qid, roll, label, metrics["m_max"], "neg")
        add_score(store, "M1_tat_late_m_mean", qid, roll, label, metrics["late_m_mean"], "neg")


def collect_m0_prefixes_for_shard(
    z: np.lib.npyio.NpzFile,
    limit_rollouts: int,
) -> list[dict[str, Any]]:
    h_chunk = z["h_chunk"]
    order = rollout_order(z, limit_rollouts)
    records: list[dict[str, Any]] = []
    for local_idx in tqdm(order, desc="M0 collect", leave=False):
        chunk_idx = sorted_chunk_indices(z, local_idx)
        if chunk_idx.size < 1:
            continue
        qid = str(z["meta_id"][local_idx])
        roll = int(z["meta_roll_index"][local_idx])
        label = bool(z["meta_is_correct"][local_idx])
        lengths = (z["chunk_end"][chunk_idx] - z["chunk_start"][chunk_idx]).astype(np.float32)
        chunks = h_chunk[chunk_idx].astype(np.float32)
        weighted = chunks * lengths[:, None, None]
        cum_hidden = np.cumsum(weighted, axis=0)
        cum_len = np.cumsum(lengths)
        prefix_values = []
        for pos in range(len(chunk_idx)):
            layer_points = cum_hidden[pos] / max(float(cum_len[pos]), EPS)
            prefix_values.append(layer_prefix_coe_c(layer_points))
        records.append(
            {
                "id": qid,
                "roll_index": roll,
                "is_correct": label,
                "prefix_values": np.asarray(prefix_values, dtype=np.float64),
            }
        )
    return records


def compute_m0_scores(
    prefix_records: list[dict[str, Any]],
    store: dict[str, dict[str, list[Any]]],
) -> None:
    by_question_step: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in prefix_records:
        for step, value in enumerate(record["prefix_values"]):
            if np.isfinite(value):
                by_question_step[(record["id"], step)].append(float(value))

    stats: dict[tuple[str, int], tuple[float, float]] = {}
    for key, values in by_question_step.items():
        arr = np.asarray(values, dtype=np.float64)
        stats[key] = (float(np.mean(arr)), float(np.std(arr)))

    for record in tqdm(prefix_records, desc="M0 score"):
        z_values = []
        for step, value in enumerate(record["prefix_values"]):
            mu, sigma = stats[(record["id"], step)]
            if sigma <= EPS:
                z = 0.0
            else:
                z = (float(value) - mu) / sigma
            z_values.append(z)
        zs = np.asarray(z_values, dtype=np.float64)
        if zs.size == 0:
            continue

        half = zs.size // 2
        late = zs[half:] if zs[half:].size else zs
        last3 = zs[-3:] if zs.size >= 3 else zs
        z_delta = float(zs[-1] - zs[-2]) if zs.size >= 2 else 0.0
        denom = max(zs.size - 1, 1)
        z_argmax_rel = float(np.argmax(zs) / denom)

        qid = record["id"]
        roll = int(record["roll_index"])
        label = bool(record["is_correct"])
        add_score(store, "M0_Z_max", qid, roll, label, float(np.max(zs)), "neg")
        add_score(store, "M0_Z_late_std", qid, roll, label, float(np.std(late)), "neg")
        add_score(store, "M0_Z_late_mean", qid, roll, label, float(np.mean(last3)), "neg")
        add_score(store, "M0_Z_argmax_rel", qid, roll, label, z_argmax_rel, "neg")
        add_score(store, "M0_Z_delta_last", qid, roll, label, z_delta, "neg_abs")
        add_score(store, "M0_Z_mean_abs", qid, roll, label, float(np.mean(np.abs(zs))), "neg")


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if args.max_shards is not None and args.max_shards >= 0:
        paths = paths[: args.max_shards]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input_glob}")

    methods = {item.strip().upper() for item in args.methods.split(",") if item.strip()}
    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    scores_dir = Path(args.scores_dir)
    store = init_score_store()
    prefix_records: list[dict[str, Any]] = []

    print("input files:")
    for path in paths:
        print(" ", path)
    print("methods:", sorted(methods))
    print("layers:", layers)

    for path in paths:
        print(f"loading {path}")
        with np.load(path, allow_pickle=True) as z:
            print("  h_chunk:", z["h_chunk"].shape, z["h_chunk"].dtype)
            print("  rollouts:", len(z["meta_id"]), "skipped:", int(z["meta_skipped"].sum()))
            if "M3" in methods:
                compute_m3_for_shard(z, layers, store, args.limit_rollouts_per_shard)
            if "M1" in methods:
                compute_m1_for_shard(z, store, args.limit_rollouts_per_shard)
            if "M0" in methods:
                prefix_records.extend(collect_m0_prefixes_for_shard(z, args.limit_rollouts_per_shard))

    if "M0" in methods:
        compute_m0_scores(prefix_records, store)

    write_score_files(scores_dir, store)
    print("done")


if __name__ == "__main__":
    main()
