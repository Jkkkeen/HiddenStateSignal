#!/usr/bin/env python3
"""Experiment B: Cross-rollout ER from chunk-mean hidden states.

This script uses existing chunklayer/h_gpu*.npz files.  For each question,
chunk position, and layer, it stacks chunk-mean hidden vectors from all
available rollouts into G(q, k, l) in R^{N x D}, then computes effective rank.

The current version uses chunk-mean hidden states.  A last-token version needs
new forward output with h_last and should be run later.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cross-rollout ER experiment.")
    parser.add_argument("--input-glob", default="chunklayer/h_gpu*.npz")
    parser.add_argument("--output-dir", default="er_results")
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--min-rollouts", type=int, default=4)
    parser.add_argument("--max-shards", type=int, default=-1)
    parser.add_argument("--limit-questions", type=int, default=-1)
    return parser.parse_args()


def effective_rank(matrix: np.ndarray, center: bool = False) -> float:
    matrix = matrix.astype(np.float32, copy=False)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return float("nan")
    if center:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    total = float(singular_values.sum())
    if total <= EPS:
        return 0.0
    probs = singular_values / total
    entropy = -float(np.sum(probs * np.log(np.clip(probs, EPS, None))))
    return float(np.exp(entropy))


def entropy_from_values(values: list[str]) -> tuple[float, float, int, float]:
    clean = [str(v) for v in values if str(v) and str(v).lower() != "none"]
    if not clean:
        return 0.0, 0.0, 0, 0.0
    counts = Counter(clean)
    total = sum(counts.values())
    probs = np.asarray([count / total for count in counts.values()], dtype=np.float64)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, EPS, None))))
    norm = entropy / math.log(total) if total > 1 else 0.0
    majority_conf = max(counts.values()) / total
    return entropy, norm, len(counts), majority_conf


def accuracy_group(acc: float) -> str:
    if acc <= 0.25:
        return "low"
    if acc >= 0.75:
        return "high"
    return "mid"


def sorted_chunk_indices(z: np.lib.npyio.NpzFile, rollout_id: int) -> np.ndarray:
    idx = np.where(z["rollout_idx"] == rollout_id)[0]
    if idx.size == 0:
        return idx
    starts = z["chunk_start"][idx]
    return idx[np.argsort(starts)]


def load_rollout_records(
    paths: list[str],
    layers: list[int],
) -> dict[str, list[dict[str, Any]]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in paths:
        print(f"loading {path}")
        with np.load(path, allow_pickle=True) as z:
            h_chunk = z["h_chunk"]
            n_layers = h_chunk.shape[1]
            for layer in layers:
                if layer < 0 or layer >= n_layers:
                    raise ValueError(f"Layer {layer} out of range for {path}; n_layers={n_layers}")

            skipped = z["meta_skipped"].astype(bool)
            for local_idx in tqdm(range(len(z["meta_id"])), desc=Path(path).name, leave=False):
                if skipped[local_idx]:
                    continue
                chunk_idx = sorted_chunk_indices(z, local_idx)
                if chunk_idx.size == 0:
                    continue
                layer_hidden = {
                    layer: h_chunk[chunk_idx, layer, :].astype(np.float16, copy=True)
                    for layer in layers
                }
                qid = str(z["meta_id"][local_idx])
                by_question[qid].append(
                    {
                        "question_id": qid,
                        "rollout_id": int(z["meta_roll_index"][local_idx]),
                        "is_correct": bool(z["meta_is_correct"][local_idx]),
                        "pred_answer": str(z["meta_pred_answer"][local_idx])
                        if "meta_pred_answer" in z.files
                        else "",
                        "answer": str(z["meta_answer"][local_idx]) if "meta_answer" in z.files else "",
                        "response_len": int(z["meta_response_len"][local_idx])
                        if "meta_response_len" in z.files
                        else int(z["chunk_end"][chunk_idx[-1]]),
                        "num_chunks": int(chunk_idx.size),
                        "chunk_starts": z["chunk_start"][chunk_idx].astype(np.int64, copy=True),
                        "chunk_ends": z["chunk_end"][chunk_idx].astype(np.int64, copy=True),
                        "layer_hidden": layer_hidden,
                    }
                )

    return by_question


def limit_questions_if_needed(
    by_question: dict[str, list[dict[str, Any]]],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    if limit is None or limit < 0:
        return by_question
    kept_keys = sorted(by_question.keys())[:limit]
    return {key: by_question[key] for key in kept_keys}


def build_question_summary(by_question: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for qid, rollouts in sorted(by_question.items()):
        labels = [bool(row["is_correct"]) for row in rollouts]
        preds = [str(row.get("pred_answer", "")) for row in rollouts]
        entropy, entropy_norm, unique_answers, majority_conf = entropy_from_values(preds)
        accuracy = float(np.mean(labels)) if labels else float("nan")
        rows.append(
            {
                "question_id": qid,
                "total_rollouts": len(rollouts),
                "question_accuracy": accuracy,
                "accuracy_group": accuracy_group(accuracy),
                "answer_entropy": entropy,
                "answer_entropy_norm": entropy_norm,
                "unique_pred_answers": unique_answers,
                "majority_confidence": majority_conf,
                "mean_response_len": float(np.mean([r["response_len"] for r in rollouts])),
                "mean_num_chunks": float(np.mean([r["num_chunks"] for r in rollouts])),
            }
        )
    return pd.DataFrame(rows)


def compute_cross_rollout_er(
    by_question: dict[str, list[dict[str, Any]]],
    question_summary: pd.DataFrame,
    layers: list[int],
    min_rollouts: int,
) -> pd.DataFrame:
    summary_by_q = question_summary.set_index("question_id").to_dict(orient="index")
    rows: list[dict[str, Any]] = []

    for qid, rollouts in tqdm(sorted(by_question.items()), desc="cross-rollout ER"):
        max_chunks = max(row["num_chunks"] for row in rollouts)
        qmeta = summary_by_q[qid]
        for chunk_pos in range(max_chunks):
            valid = [row for row in rollouts if row["num_chunks"] > chunk_pos]
            if len(valid) < min_rollouts:
                continue
            chunk_start_mean = float(np.mean([row["chunk_starts"][chunk_pos] for row in valid]))
            chunk_end_mean = float(np.mean([row["chunk_ends"][chunk_pos] for row in valid]))
            rel_pos_mean = float(
                np.mean(
                    [
                        row["chunk_starts"][chunk_pos] / max(row["response_len"], 1)
                        for row in valid
                    ]
                )
            )
            for layer in layers:
                matrix = np.stack([row["layer_hidden"][layer][chunk_pos] for row in valid], axis=0)
                er_raw = effective_rank(matrix, center=False)
                er_centered = effective_rank(matrix, center=True)
                rows.append(
                    {
                        "question_id": qid,
                        "chunk_pos": chunk_pos,
                        "layer": layer,
                        "valid_rollouts": len(valid),
                        "chunk_start_mean": chunk_start_mean,
                        "chunk_end_mean": chunk_end_mean,
                        "relative_pos_mean": rel_pos_mean,
                        "er_group_raw": er_raw,
                        "er_group_centered": er_centered,
                        "question_accuracy": qmeta["question_accuracy"],
                        "accuracy_group": qmeta["accuracy_group"],
                        "answer_entropy": qmeta["answer_entropy"],
                        "answer_entropy_norm": qmeta["answer_entropy_norm"],
                        "unique_pred_answers": qmeta["unique_pred_answers"],
                        "majority_confidence": qmeta["majority_confidence"],
                        "total_rollouts": qmeta["total_rollouts"],
                        "mean_response_len": qmeta["mean_response_len"],
                        "mean_num_chunks": qmeta["mean_num_chunks"],
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    paths = sorted(glob.glob(args.input_glob))
    if args.max_shards is not None and args.max_shards >= 0:
        paths = paths[: args.max_shards]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment B: Cross-rollout ER")
    print("input files:", len(paths))
    print("layers:", layers)
    print("min_rollouts:", args.min_rollouts)
    print("output_dir:", output_dir)

    by_question = load_rollout_records(paths, layers)
    by_question = limit_questions_if_needed(by_question, args.limit_questions)
    print("questions:", len(by_question))

    question_summary = build_question_summary(by_question)
    cross_er = compute_cross_rollout_er(by_question, question_summary, layers, args.min_rollouts)

    cross_path = output_dir / "cross_rollout_er.parquet"
    summary_path = output_dir / "question_summary.parquet"
    json_path = output_dir / "cross_rollout_er_meta.json"

    cross_er.to_parquet(cross_path, index=False)
    question_summary.to_parquet(summary_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "input_glob": args.input_glob,
                "input_files": paths,
                "layers": layers,
                "min_rollouts": args.min_rollouts,
                "questions": int(len(question_summary)),
                "cross_er_rows": int(len(cross_er)),
                "note": "Current version uses chunk-mean hidden states from h_chunk.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"saved {cross_path}: {len(cross_er)} rows")
    print(f"saved {summary_path}: {len(question_summary)} rows")
    print(f"saved {json_path}")
    print(cross_er.head())


if __name__ == "__main__":
    main()
