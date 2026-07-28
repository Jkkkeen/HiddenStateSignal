#!/usr/bin/env python3
"""Pure helpers for option-content recoverability candidate scoring."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


VALID_OPTIONS = ("A", "B", "C", "D")
CHOICE_LINE = re.compile(r"^\s*([A-Z])\s*(?::|\)|\.)\s*(.*?)\s*$")


def _contiguous_labels(labels: Sequence[str]) -> tuple[str, ...]:
    ordered = tuple(sorted(str(label).strip().upper() for label in labels))
    expected = tuple(chr(ord("A") + index) for index in range(len(ordered)))
    if len(ordered) < 2 or ordered != expected:
        raise ValueError("choices must use at least two contiguous labels starting at A")
    return ordered


def parse_mathverse_choices(prompt: str) -> dict[str, str]:
    """Parse one-line contiguous option contents from a MathVerse prompt."""

    choices: dict[str, str] = {}
    for line in str(prompt or "").splitlines():
        match = CHOICE_LINE.match(line)
        if match and match.group(2):
            choices[match.group(1)] = match.group(2).strip()
    labels = _contiguous_labels(choices)
    return {label: choices[label] for label in labels}


def actual_prompt_token_logprob(row: Any, expected_token_id: int) -> float:
    """Extract the actual token's log probability from a vLLM prompt row."""

    if row is None or not isinstance(row, Mapping):
        raise ValueError("prompt logprob row must be a mapping")
    expected = int(expected_token_id)
    value = row.get(expected)
    if value is None:
        value = row.get(str(expected))
    if value is None:
        raise ValueError(f"prompt logprob row is missing actual token {expected}")
    logprob = getattr(value, "logprob", value)
    return float(logprob)


def candidate_token_indices(
    offsets: Sequence[tuple[int, int] | list[int]],
    candidate_start: int,
    candidate_end: int,
) -> list[int]:
    """Return token indices whose character offsets overlap candidate content."""

    if candidate_end <= candidate_start:
        raise ValueError("candidate span must be non-empty")
    indices = [
        index
        for index, offset in enumerate(offsets)
        if int(offset[1]) > int(offset[0])
        and int(offset[1]) > int(candidate_start)
        and int(offset[0]) < int(candidate_end)
    ]
    if not indices:
        raise ValueError("candidate span does not overlap any prompt tokens")
    return indices


def sequence_logprob(
    prompt_token_ids: Sequence[int],
    prompt_logprobs: Sequence[Any],
    token_indices: Sequence[int],
) -> dict[str, float | int]:
    """Aggregate actual-token prompt log probabilities over a candidate span."""

    if len(prompt_token_ids) != len(prompt_logprobs):
        raise ValueError("prompt token ids and logprobs must have equal length")
    indices = [int(index) for index in token_indices]
    if not indices:
        raise ValueError("candidate token indices must be non-empty")
    values = [
        actual_prompt_token_logprob(prompt_logprobs[index], int(prompt_token_ids[index]))
        for index in indices
    ]
    total = float(sum(values))
    return {"sum": total, "mean": total / len(values), "token_count": len(values)}


def choice_features(
    scores: Mapping[str, float],
    gold: str | None,
    selected: str | None,
) -> dict[str, float | str]:
    """Derive gold-support, wrong-answer commitment, and uncertainty features."""

    labels = _contiguous_labels(scores)
    values = {label: float(scores[label]) for label in labels}
    if gold not in labels:
        raise ValueError(f"invalid gold option {gold!r}")
    if selected not in labels:
        raise ValueError(f"invalid selected option {selected!r}")

    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    top_option, top_score = ordered[0]
    top2_gap = top_score - ordered[1][1]
    gold_score = values[str(gold)]
    wrong_values = [value for label, value in values.items() if label != gold]
    gold_margin = gold_score - max(wrong_values)
    gold_margin_mean = gold_score - sum(wrong_values) / len(wrong_values)
    commitment_gap = values[str(selected)] - gold_score

    normalizer = max(values.values())
    weights = [math.exp(value - normalizer) for value in values.values()]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return {
        "gold_margin": float(gold_margin),
        "gold_margin_mean": float(gold_margin_mean),
        "commitment_gap": float(commitment_gap),
        "top2_gap": float(top2_gap),
        "entropy": float(entropy),
        "top_option": top_option,
    }


def calibrate_scores(
    reasoning_scores: Mapping[str, float],
    prompt_only_scores: Mapping[str, float],
) -> dict[str, float]:
    """Subtract each option's prompt-only score from its reasoning score."""

    labels = _contiguous_labels(reasoning_scores)
    prompt_labels = _contiguous_labels(prompt_only_scores)
    if labels != prompt_labels:
        raise ValueError("reasoning and prompt-only scores must use the same labels")
    return {
        label: float(reasoning_scores[label]) - float(prompt_only_scores[label])
        for label in labels
    }
