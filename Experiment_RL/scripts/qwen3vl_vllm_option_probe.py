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


def prompt_last_matching_token_logprob(extra_fields: dict[str, Any], expected_token_id: int) -> float:
    """Return the logprob for the last prompt row matching a token id."""

    prompt_ids = list(extra_fields.get("prompt_actual_ids") or extra_fields.get("prompt_ids") or [])
    prompt_logprobs = list(extra_fields.get("prompt_actual_logprobs") or extra_fields.get("prompt_logprobs") or [])
    if not prompt_ids or not prompt_logprobs:
        raise ValueError("prompt_ids and prompt_logprobs must be present")
    if len(prompt_ids) != len(prompt_logprobs):
        raise ValueError(f"prompt_ids/logprobs length mismatch: {len(prompt_ids)} != {len(prompt_logprobs)}")

    expected = int(expected_token_id)
    for token_row, logprob_row in reversed(list(zip(prompt_ids, prompt_logprobs, strict=True))):
        token_values = list(token_row if isinstance(token_row, (list, tuple)) else [token_row])
        logprob_values = list(logprob_row if isinstance(logprob_row, (list, tuple)) else [logprob_row])
        for token_id, logprob in zip(token_values, logprob_values, strict=False):
            if token_id is not None and int(token_id) == expected:
                return float(logprob)
    raise ValueError(f"missing prompt logprob for token {expected}")


def option_logprobs_from_prompt_tail_extra_fields(
    extra_fields_by_label: dict[str, dict[str, Any]],
    label_token_ids: dict[str, int],
) -> dict[str, float]:
    """Parse A/B/C/D option logprobs when label tokens have a trailing prompt token."""

    out: dict[str, float] = {}
    for label in VALID_OPTIONS:
        if label not in extra_fields_by_label:
            raise ValueError(f"missing vLLM probe output for option {label}")
        if label not in label_token_ids:
            raise ValueError(f"missing token id for option {label}")
        out[label] = prompt_last_matching_token_logprob(extra_fields_by_label[label], label_token_ids[label])
    return out


def generated_token_option_logprobs_from_extra_fields(
    extra_fields: dict[str, Any],
    label_token_ids: dict[str, int],
    step_index: int = 0,
) -> dict[str, float]:
    """Parse A/B/C/D logprobs from generated-token top-k logprob rows."""

    generation_ids = list(extra_fields.get("generation_ids") or [])
    generation_logprobs = list(extra_fields.get("generation_logprobs") or [])
    if len(generation_ids) <= step_index or len(generation_logprobs) <= step_index:
        raise ValueError("generation_ids and generation_logprobs must contain the requested step")
    if len(generation_ids) != len(generation_logprobs):
        raise ValueError(
            f"generation_ids/logprobs length mismatch: {len(generation_ids)} != {len(generation_logprobs)}"
        )

    token_ids = list(generation_ids[step_index] or [])
    token_logprobs = list(generation_logprobs[step_index] or [])
    if len(token_ids) != len(token_logprobs):
        raise ValueError(f"generated token row length mismatch: {len(token_ids)} != {len(token_logprobs)}")

    by_token_id: dict[int, float] = {}
    for token_id, logprob in zip(token_ids, token_logprobs, strict=True):
        if token_id is None or logprob is None:
            continue
        by_token_id[int(token_id)] = float(logprob)

    out: dict[str, float] = {}
    for label in VALID_OPTIONS:
        if label not in label_token_ids:
            raise ValueError(f"missing token id for option {label}")
        token_id = int(label_token_ids[label])
        if token_id not in by_token_id:
            raise ValueError(f"missing generated logprob for option {label} token {token_id}")
        out[label] = by_token_id[token_id]
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
