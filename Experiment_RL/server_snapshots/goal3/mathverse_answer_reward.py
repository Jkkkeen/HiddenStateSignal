#!/usr/bin/env python3
"""Rule reward for MathVerse A/B/C/D final-answer correctness in VERL."""

from __future__ import annotations

import re
from typing import Any


VALID_OPTIONS = {"A", "B", "C", "D"}


def normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in VALID_OPTIONS else None


def extract_final_choice(text: str | None) -> str | None:
    if not text:
        return None
    value = str(text)
    # Prefer content after the thinking block if present.
    close_idx = value.rfind("</think>")
    candidates = [value[close_idx + len("</think>") :], value] if close_idx >= 0 else [value]
    patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\**\s*([ABCD])\b",
        r"\b([ABCD])\s*(?:is\s+the\s+answer|is\s+correct)\b",
        r"(?:option|choice)\s*\**\s*([ABCD])\b",
        r"\*\*([ABCD])\*\*",
        r"\(([ABCD])\)",
        r"\b([ABCD])\s*[:\.\)]",
    ]
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        for pattern in patterns:
            matches = re.findall(pattern, candidate, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        tail = candidate[-500:]
        matches = re.findall(r"\b([ABCD])\b", tail)
        if matches:
            return matches[-1].upper()
    return None


def _ground_truth_choice(ground_truth: Any, extra_info: dict[str, Any] | None = None) -> str | None:
    direct = normalize_choice(ground_truth)
    if direct is not None:
        return direct
    if isinstance(ground_truth, dict):
        for key in ("ground_truth", "answer", "label"):
            direct = normalize_choice(ground_truth.get(key))
            if direct is not None:
                return direct
    if extra_info:
        for key in ("answer", "ground_truth", "label"):
            direct = normalize_choice(extra_info.get(key))
            if direct is not None:
                return direct
    return None


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    """VERL-compatible rule reward.

    Returns 1.0 for a correct final choice and 0.0 otherwise. Additional
    keyword arguments are accepted because VERL reward managers may pass extra
    metadata depending on the version.
    """

    answer = _ground_truth_choice(ground_truth, extra_info)
    pred = extract_final_choice(solution_str)
    return 1.0 if answer is not None and pred == answer else 0.0
