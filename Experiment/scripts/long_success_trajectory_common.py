#!/usr/bin/env python3
"""Pure helpers shared by long success-trajectory extraction and analysis."""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd


VALID_CHOICES = {"A", "B", "C", "D"}
LENGTH_BIN_ORDER = ["<2k", "2-4k", "4-8k", "8k+"]


def think_length(row: dict[str, Any]) -> int:
    for key in ("think_token_count", "think_length"):
        value = row.get(key)
        if value is not None:
            return int(value)
    return 0


def length_bin(value: int | float) -> str:
    value = float(value)
    if value < 2000:
        return "<2k"
    if value < 4000:
        return "2-4k"
    if value < 8000:
        return "4-8k"
    return "8k+"


def is_clean_complete_row(row: dict[str, Any]) -> bool:
    pred = str(row.get("pred_answer") or "").strip().upper()
    return (
        not bool(row.get("truncated"))
        and bool(row.get("has_think_close"))
        and pred in VALID_CHOICES
    )


def eligible_question_summaries(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not is_clean_complete_row(row):
            continue
        question_id = str(row.get("question_id", ""))
        if not question_id:
            continue
        grouped.setdefault(question_id, []).append(row)

    records = []
    for question_id, group in grouped.items():
        labels = [bool(row.get("is_correct")) for row in group]
        lengths = [think_length(row) for row in group]
        n_correct = int(sum(labels))
        n_wrong = int(len(labels) - n_correct)
        median_length = float(np.median(lengths)) if lengths else 0.0
        is_primary = n_correct >= 3 and n_wrong >= 3
        is_secondary = n_correct >= 2 and n_wrong >= 2 and not is_primary
        records.append(
            {
                "question_id": question_id,
                "n_rollouts": int(len(group)),
                "n_correct": n_correct,
                "n_wrong": n_wrong,
                "median_think_length": median_length,
                "length_bin": length_bin(median_length),
                "is_primary": is_primary,
                "is_secondary": is_secondary,
            }
        )
    columns = [
        "question_id",
        "n_rollouts",
        "n_correct",
        "n_wrong",
        "median_think_length",
        "length_bin",
        "is_primary",
        "is_secondary",
    ]
    return pd.DataFrame(records, columns=columns).sort_values("question_id").reset_index(drop=True)


def _stable_key(question_id: str, seed: int, salt: str) -> str:
    payload = f"{seed}:{salt}:{question_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_primary_questions(
    summaries: pd.DataFrame,
    discovery_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < discovery_fraction < 1.0:
        raise ValueError("discovery_fraction must be in (0, 1)")
    primary = summaries[summaries["is_primary"]].copy()
    discovery: list[str] = []
    confirmatory: list[str] = []
    for bin_name in LENGTH_BIN_ORDER:
        ids = primary.loc[primary["length_bin"] == bin_name, "question_id"].astype(str).tolist()
        ids.sort(key=lambda qid: _stable_key(qid, seed, f"split:{bin_name}"))
        if len(ids) <= 1:
            n_discovery = len(ids)
        else:
            n_discovery = int(round(len(ids) * discovery_fraction))
            n_discovery = min(max(n_discovery, 1), len(ids) - 1)
        discovery.extend(ids[:n_discovery])
        confirmatory.extend(ids[n_discovery:])
    return sorted(discovery), sorted(confirmatory)


def select_stratified_questions(
    summaries: pd.DataFrame,
    candidate_ids: Iterable[str],
    limit: int,
    seed: int,
) -> list[str]:
    candidate_set = {str(item) for item in candidate_ids}
    if limit <= 0 or not candidate_set:
        return []
    view = summaries[summaries["question_id"].isin(candidate_set)].copy()
    limit = min(limit, len(view))
    counts = view.groupby("length_bin")["question_id"].count().to_dict()
    total = int(sum(counts.values()))
    quotas = {name: int(np.floor(limit * counts.get(name, 0) / total)) for name in LENGTH_BIN_ORDER}
    if limit >= sum(1 for count in counts.values() if count > 0):
        for name, count in counts.items():
            if count > 0:
                quotas[name] = max(quotas.get(name, 0), 1)
    while sum(quotas.values()) > limit:
        choices = [name for name in LENGTH_BIN_ORDER if quotas.get(name, 0) > 1]
        if not choices:
            choices = [name for name in LENGTH_BIN_ORDER if quotas.get(name, 0) > 0]
        quotas[choices[-1]] -= 1
    while sum(quotas.values()) < limit:
        choices = [name for name in LENGTH_BIN_ORDER if quotas.get(name, 0) < counts.get(name, 0)]
        if not choices:
            break
        choices.sort(
            key=lambda name: limit * counts.get(name, 0) / total - quotas.get(name, 0),
            reverse=True,
        )
        quotas[choices[0]] += 1

    selected = []
    for bin_name in LENGTH_BIN_ORDER:
        ids = view.loc[view["length_bin"] == bin_name, "question_id"].astype(str).tolist()
        ids.sort(key=lambda qid: _stable_key(qid, seed, f"smoke:{bin_name}"))
        selected.extend(ids[: quotas.get(bin_name, 0)])
    return sorted(selected)


def full_span_bounds(length: int, window: int, stride: int) -> list[tuple[int, int]]:
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if length < window:
        return []
    return [(start, start + window) for start in range(0, length - window + 1, stride)]


def balanced_reference_subsets(
    labels: dict[int, bool],
    query_id: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if query_id not in labels:
        raise KeyError(f"query_id not present: {query_id}")
    total_correct = int(sum(bool(value) for value in labels.values()))
    total_wrong = int(len(labels) - total_correct)
    m = min(total_correct - 1, total_wrong - 1)
    if m < 1:
        return []
    positives = sorted(rid for rid, label in labels.items() if rid != query_id and label)
    negatives = sorted(rid for rid, label in labels.items() if rid != query_id and not label)
    return [
        (tuple(pos_subset), tuple(neg_subset))
        for pos_subset in combinations(positives, m)
        for neg_subset in combinations(negatives, m)
    ]


def cosine_similarity(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= eps:
        return 0.0
    return float(np.dot(left, right) / denom)


def pairwise_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    if positive_scores.size == 0 or negative_scores.size == 0:
        return float("nan")
    differences = positive_scores[:, None] - negative_scores[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))

