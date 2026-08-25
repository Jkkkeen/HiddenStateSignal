"""Independent answer/format scoring for the frozen ER-style four-level reward."""

from __future__ import annotations

import re
from typing import Any, Callable

from experiment_2e.math_reward import extract_last_boxed, math_verify_equivalent


Verifier = Callable[[str, str], bool]


def _clean_candidate(value: str) -> str | None:
    candidate = (value or "").strip()
    if candidate.startswith("$") and candidate.endswith("$") and len(candidate) >= 2:
        candidate = candidate[1:-1].strip()
    if candidate.startswith(r"\(") and candidate.endswith(r"\)"):
        candidate = candidate[2:-2].strip()
    candidate = re.sub(r"^[\s:$=]+", "", candidate)
    candidate = re.sub(r"[\s.$。]+$", "", candidate)
    return candidate or None


def extract_unboxed_final_answer(text: str) -> str | None:
    """Extract an explicitly stated final answer without requiring ``\\boxed``.

    The fallback is deliberately conservative. It accepts an explicit answer
    phrase, a ``####`` answer, or a short math-like final line. The calibration
    audit must report extraction disagreements before formal training.
    """

    source = text or ""
    patterns = (
        r"(?im)^\s*####\s*(.+?)\s*$",
        r"(?im)^\s*(?:therefore[, ]+)?(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*(.+?)\s*$",
    )
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        matches.extend((match.start(), match.group(1)) for match in re.finditer(pattern, source))
    if matches:
        return _clean_candidate(max(matches, key=lambda item: item[0])[1])

    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if not lines:
        return None
    final = _clean_candidate(lines[-1])
    if final is None or len(final) > 128:
        return None
    alphabetic_words = re.findall(r"[A-Za-z]{2,}", final)
    latex_words = {"frac", "sqrt", "pi", "infty", "pm", "cdot", "times"}
    if sum(word.lower() not in latex_words for word in alphabetic_words) > 1:
        return None
    return final


def score_response(
    solution_str: str,
    ground_truth: str,
    verifier: Verifier | None = None,
) -> dict[str, Any]:
    boxed = extract_last_boxed(solution_str or "")
    candidate = boxed if boxed is not None else extract_unboxed_final_answer(solution_str or "")
    format_correct = boxed is not None
    answer_correct = bool(candidate is not None and (verifier or math_verify_equivalent)(ground_truth, candidate))

    if answer_correct and format_correct:
        reward = 1.0
    elif answer_correct:
        reward = 0.5
    elif format_correct:
        reward = -0.5
    else:
        reward = -1.0

    if candidate is None:
        failure_reason = "missing_answer"
    elif not answer_correct:
        failure_reason = "answer_mismatch"
    elif not format_correct:
        failure_reason = "missing_boxed_answer"
    else:
        failure_reason = None
    return {
        "reward": reward,
        "answer_reward": float(answer_correct),
        "format_reward": float(format_correct),
        "parsed_answer": candidate,
        "parser_source": "boxed" if boxed is not None else ("unboxed" if candidate is not None else None),
        "failure_reason": failure_reason,
    }


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    del data_source, extra_info
    if solution_str is None or ground_truth is None:
        return -1.0
    return float(score_response(solution_str, ground_truth)["reward"])
