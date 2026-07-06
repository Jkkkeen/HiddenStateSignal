#!/usr/bin/env python3
"""Semantic-step correct-answer basin helpers.

This module contains only pure utilities: semantic-step splitting and the
geometry reductions used by the Qwen3-VL forward script.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


EPS = 1e-12
RETHINK_TRIGGERS = (
    "wait",
    "actually",
    "reconsider",
    "however",
    "but",
    "alternatively",
    "on second thought",
    "made a mistake",
    "re-check",
    "recheck",
)


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _safe_mean(values: list[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_std(values: list[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.std(arr)) if arr.size else float("nan")


def _safe_percentile(values: list[float] | np.ndarray, q: float) -> float:
    arr = _finite(values)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _late_slice(values: list[Any]) -> list[Any]:
    if not values:
        return []
    start = int(math.floor(0.75 * len(values)))
    start = min(start, len(values) - 1)
    return values[start:]


def _early_slice(values: list[Any]) -> list[Any]:
    if not values:
        return []
    end = int(math.ceil(0.25 * len(values)))
    end = max(1, min(end, len(values)))
    return values[:end]


def has_rethink_trigger(text: str) -> bool:
    lower = text.lower()
    return any(trigger in lower for trigger in RETHINK_TRIGGERS)


def split_semantic_steps(text: str, min_chars: int = 12) -> list[dict[str, Any]]:
    """Split thinking text into weak semantic steps.

    This first version deliberately uses deterministic discourse/sentence
    boundaries rather than fixed token chunks. Very short fragments are merged
    into the previous step so downstream hidden states correspond to usable
    reasoning units.
    """
    text = str(text or "")
    raw_steps: list[dict[str, Any]] = []
    pattern = re.compile(r".+?(?:[.!?;。！？；]+|\n+|$)", re.DOTALL)
    for match in pattern.finditer(text):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        start = match.start() + leading
        end = match.start() + trailing
        raw_steps.append({"text": stripped, "start_char": start, "end_char": end})

    merged: list[dict[str, Any]] = []
    for step in raw_steps:
        if merged and len(step["text"]) < min_chars:
            prev = merged[-1]
            prev["text"] = f"{prev['text']} {step['text']}".strip()
            prev["end_char"] = step["end_char"]
            prev["has_rethink_trigger"] = has_rethink_trigger(prev["text"])
            continue
        step["has_rethink_trigger"] = has_rethink_trigger(step["text"])
        merged.append(step)

    return merged


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= EPS:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _max_wrong_cos(vector: np.ndarray, option_vectors: dict[str, np.ndarray], correct_label: str) -> float:
    values = [
        cosine(vector, option)
        for label, option in option_vectors.items()
        if label != correct_label
    ]
    return float(max(values)) if values else 0.0


def compute_step_basin_metrics(
    states: np.ndarray,
    option_vectors: dict[str, np.ndarray],
    correct_label: str,
    chosen_label: str | None = None,
    rethink_flags: list[bool] | None = None,
) -> list[dict[str, Any]]:
    """Compute step-level correct-answer basin metrics.

    `states[0]` is the starting prompt/think baseline. Each subsequent row is
    the hidden state after one semantic step.
    """
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"states must be [n_states, dim], got {states.shape}")
    if states.shape[0] < 2:
        return []
    if correct_label not in option_vectors:
        raise KeyError(f"correct_label {correct_label!r} missing from option vectors")

    option_vectors = {label: normalize(vec) for label, vec in option_vectors.items()}
    correct_vec = option_vectors[correct_label]
    n_steps = states.shape[0] - 1
    rethink_flags = rethink_flags or [False] * n_steps

    rows: list[dict[str, Any]] = []
    prev_margin = 0.0
    prev_disp: np.ndarray | None = None
    for step_idx in range(n_steps):
        state_idx = step_idx + 1
        state_delta = states[state_idx] - states[0]
        disp = states[state_idx] - states[state_idx - 1]

        align_correct_state = cosine(state_delta, correct_vec)
        align_wrong_state = _max_wrong_cos(state_delta, option_vectors, correct_label)
        state_margin = align_correct_state - align_wrong_state
        step_correct_gain = state_margin - prev_margin

        align_correct_step = cosine(disp, correct_vec)
        align_wrong_step = _max_wrong_cos(disp, option_vectors, correct_label)
        turn_to_correct = align_correct_step - align_wrong_step

        if prev_disp is None:
            turn_angle = float("nan")
            productive_turn = 0.0
        else:
            turn_cos = cosine(disp, prev_disp)
            turn_angle = float(math.acos(np.clip(turn_cos, -1.0, 1.0)))
            productive_turn = float(turn_angle * max(0.0, step_correct_gain))

        if chosen_label and chosen_label != correct_label and chosen_label in option_vectors:
            chosen_wrong_margin = cosine(state_delta, option_vectors[chosen_label]) - align_correct_state
        else:
            chosen_wrong_margin = float("nan")

        rows.append(
            {
                "step_index": int(step_idx),
                "relative_step_pos": float(step_idx / max(n_steps - 1, 1)),
                "align_correct_state": align_correct_state,
                "align_wrong_state": align_wrong_state,
                "state_margin": float(state_margin),
                "step_correct_gain": float(step_correct_gain),
                "align_correct_step": align_correct_step,
                "align_wrong_step": align_wrong_step,
                "turn_to_correct": float(turn_to_correct),
                "turn_angle": turn_angle,
                "productive_turn": productive_turn,
                "chosen_wrong_margin": float(chosen_wrong_margin),
                "has_rethink_trigger": bool(
                    rethink_flags[step_idx] if step_idx < len(rethink_flags) else False
                ),
            }
        )
        prev_margin = state_margin
        prev_disp = disp

    return rows


def summarize_rollout_basin_features(rows: list[dict[str, Any]]) -> dict[str, float | int | bool]:
    """Aggregate step-level basin rows into rollout-level features."""
    if not rows:
        return {
            "n_semantic_steps": 0,
            "mean_step_correct_gain": float("nan"),
            "positive_gain_rate": float("nan"),
            "late_margin_mean": float("nan"),
            "early_to_late_margin_gain": float("nan"),
            "mean_turn_to_correct": float("nan"),
            "late_turn_to_correct": float("nan"),
            "productive_turn_p90": float("nan"),
            "productive_turn_mean": float("nan"),
            "first_cross_pos": float("nan"),
            "final_margin": float("nan"),
            "late_margin_std": float("nan"),
            "stable_correct_basin": False,
            "wrong_lock_score": float("nan"),
            "rethink_step_count": 0,
            "rethink_gain_mean": float("nan"),
        }

    margins = [float(row.get("state_margin", float("nan"))) for row in rows]
    gains = [float(row.get("step_correct_gain", float("nan"))) for row in rows]
    turns = [float(row.get("turn_to_correct", float("nan"))) for row in rows]
    productive = [float(row.get("productive_turn", float("nan"))) for row in rows]
    chosen_wrong = [float(row.get("chosen_wrong_margin", float("nan"))) for row in rows]

    late_rows = _late_slice(rows)
    early_rows = _early_slice(rows)
    late_margins = [float(row.get("state_margin", float("nan"))) for row in late_rows]
    early_margins = [float(row.get("state_margin", float("nan"))) for row in early_rows]
    late_turns = [float(row.get("turn_to_correct", float("nan"))) for row in late_rows]
    late_chosen_wrong = [
        float(row.get("chosen_wrong_margin", float("nan"))) for row in late_rows
    ]

    first_cross_pos = float("nan")
    for idx, margin in enumerate(margins):
        if np.isfinite(margin) and margin >= 0.0:
            first_cross_pos = float((idx + 1) / len(rows))
            break

    rethink_gains = [
        float(row.get("step_correct_gain", float("nan")))
        for row in rows
        if row.get("has_rethink_trigger")
    ]
    late_margin_mean = _safe_mean(late_margins)
    late_margin_std = _safe_std(late_margins)

    finite_gains = _finite(gains)
    return {
        "n_semantic_steps": int(len(rows)),
        "mean_step_correct_gain": _safe_mean(gains),
        "positive_gain_rate": (
            float(np.mean(finite_gains > 0.0)) if finite_gains.size else float("nan")
        ),
        "late_margin_mean": late_margin_mean,
        "early_to_late_margin_gain": float(late_margin_mean - _safe_mean(early_margins)),
        "mean_turn_to_correct": _safe_mean(turns),
        "late_turn_to_correct": _safe_mean(late_turns),
        "productive_turn_p90": _safe_percentile(productive, 90),
        "productive_turn_mean": _safe_mean(productive),
        "first_cross_pos": first_cross_pos,
        "final_margin": float(margins[-1]) if margins else float("nan"),
        "late_margin_std": late_margin_std,
        "stable_correct_basin": bool(
            np.isfinite(late_margin_mean)
            and late_margin_mean > 0.0
            and np.isfinite(late_margin_std)
            and late_margin_std < 0.25
        ),
        "wrong_lock_score": _safe_mean(late_chosen_wrong),
        "rethink_step_count": int(sum(bool(row.get("has_rethink_trigger")) for row in rows)),
        "rethink_gain_mean": _safe_mean(rethink_gains),
    }
