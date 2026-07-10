#!/usr/bin/env python3
"""Utilities for C2 actor-forward option-logit probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


VALID_OPTIONS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class ActorOptionGain:
    gain: float
    margins: list[float]
    probe_count: int
    probe_failed: float


def parse_probe_fracs(raw: str | Sequence[float]) -> list[float]:
    if isinstance(raw, str):
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = [float(item) for item in raw]
    if not values:
        raise ValueError("probe fracs are empty")
    return [max(0.0, min(1.0, value)) for value in values]


def probe_positions_for_lengths(
    total_lengths: Sequence[int],
    response_lengths: Sequence[int],
    fracs: Sequence[float],
) -> list[list[int]]:
    fracs = parse_probe_fracs(fracs)
    positions: list[list[int]] = []
    for total_len, response_len in zip(total_lengths, response_lengths, strict=True):
        total_len = int(total_len)
        response_len = int(response_len)
        prompt_len = max(total_len - response_len, 0)
        row: list[int] = []
        for frac in fracs:
            prefix_len = int(round(frac * response_len)) if frac > 0 else 0
            prefix_len = max(0, min(response_len, prefix_len))
            pos = prompt_len + prefix_len - 1
            row.append(max(0, min(total_len - 1, pos)))
        positions.append(row)
    return positions


def correct_margin_from_option_logits(option_logits: Sequence[float], correct: str | None) -> float:
    correct = str(correct or "").strip().upper()
    if correct not in VALID_OPTIONS:
        return float("nan")
    values = [float(value) for value in option_logits]
    correct_idx = VALID_OPTIONS.index(correct)
    wrong = [value for idx, value in enumerate(values) if idx != correct_idx]
    return float(values[correct_idx] - max(wrong))


def clipped_late_gain(margins: Sequence[float], clip_value: float, reward_start_index: int = 1) -> float:
    clean = [float(value) for value in margins if np.isfinite(value)]
    if len(clean) <= reward_start_index + 1:
        return 0.0
    reward_margins = np.asarray(clean[int(reward_start_index) :], dtype=np.float64)
    diffs = np.diff(reward_margins)
    clipped = np.clip(diffs, -abs(float(clip_value)), abs(float(clip_value)))
    return float(clipped.mean()) if clipped.size else 0.0


def option_gain_from_actor_logits(
    option_logits: Sequence[Sequence[float]],
    correct: str | None,
    clip_value: float = 0.5,
    reward_start_index: int = 1,
) -> ActorOptionGain:
    margins = [correct_margin_from_option_logits(row, correct) for row in option_logits]
    failed = 0.0 if margins and all(np.isfinite(value) for value in margins) else 1.0
    gain = 0.0 if failed else clipped_late_gain(margins, clip_value, reward_start_index)
    return ActorOptionGain(gain=float(gain), margins=margins, probe_count=len(margins), probe_failed=failed)
