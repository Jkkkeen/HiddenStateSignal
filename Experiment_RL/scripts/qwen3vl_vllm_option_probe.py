#!/usr/bin/env python3
"""Pure helpers for Qwen3-VL C2.1 vLLM option-logprob probes."""

from __future__ import annotations

from typing import Any

import numpy as np


VALID_OPTIONS = ("A", "B", "C", "D")
DEFAULT_PROBE_SUFFIX = "\n\nGiven the reasoning so far, the answer is ("


def _one_value(row: Any) -> Any:
    if isinstance(row, (list, tuple)):
        if not row:
            raise ValueError("empty prompt-logprob row")
        return row[0]
    return row


def prompt_actual_token_logprob(extra_fields: dict[str, Any], expected_token_id: int) -> float:
    """Return the logprob of an appended prompt token from vLLM prompt_logprobs.

    VERL's vLLM adapter appends a dummy prompt-logprob row for the final prompt
    position. When we append an option token to the probe prompt, the option
    token's actual logprob is therefore in the second-to-last row.
    """

    prompt_ids = list(extra_fields.get("prompt_ids") or [])
    prompt_logprobs = list(extra_fields.get("prompt_logprobs") or [])
    if len(prompt_ids) < 2 or len(prompt_logprobs) < 2:
        raise ValueError("prompt_ids and prompt_logprobs must contain at least two rows")
    if len(prompt_ids) != len(prompt_logprobs):
        raise ValueError(f"prompt_ids/logprobs length mismatch: {len(prompt_ids)} != {len(prompt_logprobs)}")

    token_id = int(_one_value(prompt_ids[-2]))
    if token_id != int(expected_token_id):
        raise ValueError(f"expected appended token {expected_token_id}, got {token_id}")
    return float(_one_value(prompt_logprobs[-2]))


def option_logprobs_from_extra_fields(
    extra_fields_by_label: dict[str, dict[str, Any]],
    label_token_ids: dict[str, int],
) -> dict[str, float]:
    """Parse A/B/C/D option logprobs from four vLLM probe outputs."""

    out: dict[str, float] = {}
    for label in VALID_OPTIONS:
        if label not in extra_fields_by_label:
            raise ValueError(f"missing vLLM probe output for option {label}")
        if label not in label_token_ids:
            raise ValueError(f"missing token id for option {label}")
        out[label] = prompt_actual_token_logprob(extra_fields_by_label[label], label_token_ids[label])
    return out


def margin_from_option_logprobs(option_logprobs: dict[str, float], correct: str | None) -> float:
    """Compute correct-option logprob minus the best wrong-option logprob."""

    if correct not in VALID_OPTIONS:
        return float("nan")
    correct_value = float(option_logprobs[str(correct)])
    wrong_values = [float(option_logprobs[label]) for label in VALID_OPTIONS if label != correct]
    return float(correct_value - max(wrong_values))


def clipped_late_gain(margins: list[float], clip_value: float, reward_start_index: int = 1) -> float:
    """Mean clipped margin gain, excluding the prompt-only diagnostic anchor."""

    clean = [float(value) for value in margins if np.isfinite(value)]
    if len(clean) <= reward_start_index + 1:
        return 0.0
    reward_margins = np.asarray(clean[reward_start_index:], dtype=np.float64)
    diffs = np.diff(reward_margins)
    clipped = np.clip(diffs, -abs(float(clip_value)), abs(float(clip_value)))
    return float(np.mean(clipped)) if clipped.size else 0.0


def build_probe_text(
    prompt_text: str,
    response_prefix: str,
    suffix: str = DEFAULT_PROBE_SUFFIX,
    option_label: str = "A",
) -> str:
    """Build the text portion of a vLLM prompt-logprobs option probe."""

    label = str(option_label).strip().upper()
    if label not in VALID_OPTIONS:
        raise ValueError(f"option_label must be one of {VALID_OPTIONS}, got {option_label!r}")
    return f"{str(prompt_text).strip()}\n\n{response_prefix}{suffix}{label}"
