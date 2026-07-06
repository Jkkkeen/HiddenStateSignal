#!/usr/bin/env python3
"""Response-stage segment extraction and weak semantic step splitting."""

from __future__ import annotations

import re
from typing import Any


HEADING_RE = re.compile(
    r"(?im)^(?:\s*(?:#{1,6}\s*)?(?:\*{1,3}\s*)?(?:step\s*\d+|final\s+answer|answer)\b[^\n]*|\s*\d+\.\s+\S[^\n]*)"
)
SENTENCE_RE = re.compile(r".+?(?:[.!?;。！？；]+|\n+|$)", re.DOTALL)


def answer_after_think(response: str) -> tuple[str, int, int, str]:
    text = str(response or "")
    close_tag = "</think>"
    close_idx = text.find(close_tag)
    if close_idx < 0:
        return "", len(text), len(text), "no_close"
    start = close_idx + len(close_tag)
    raw_segment = text[start:]
    leading = len(raw_segment) - len(raw_segment.lstrip())
    trailing = len(raw_segment.rstrip())
    segment_start = start + leading
    segment_end = start + trailing
    return text[segment_start:segment_end], segment_start, segment_end, "after_close"


def _has_block_headings(text: str) -> bool:
    return len(list(HEADING_RE.finditer(text))) >= 2


def _split_block_steps(text: str) -> list[dict[str, Any]]:
    matches = list(HEADING_RE.finditer(text))
    steps: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            continue
        heading = match.group(0).strip()
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        steps.append(
            {
                "text": stripped,
                "heading": heading,
                "start_char": start + leading,
                "end_char": start + trailing,
            }
        )
    return steps


def _split_sentence_steps(text: str, min_chars: int) -> list[dict[str, Any]]:
    raw_steps: list[dict[str, Any]] = []
    for match in SENTENCE_RE.finditer(text):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        raw_steps.append(
            {
                "text": stripped,
                "heading": "",
                "start_char": match.start() + leading,
                "end_char": match.start() + trailing,
            }
        )

    merged: list[dict[str, Any]] = []
    for step in raw_steps:
        if merged and len(step["text"]) < min_chars:
            prev = merged[-1]
            prev["text"] = f"{prev['text']} {step['text']}".strip()
            prev["end_char"] = step["end_char"]
            continue
        merged.append(step)
    return merged


def split_response_steps(
    text: str,
    min_chars: int = 12,
    short_answer_chars: int = 40,
) -> tuple[list[dict[str, Any]], str]:
    text = str(text or "").strip()
    if not text:
        return [], "empty"
    if _has_block_headings(text):
        steps = _split_block_steps(text)
        if steps:
            return steps, "block"
    if len(text) <= short_answer_chars:
        return [{"text": text, "heading": "", "start_char": 0, "end_char": len(text)}], "terminal"
    steps = _split_sentence_steps(text, min_chars=min_chars)
    return steps, "sentence" if len(steps) > 1 else "terminal"
