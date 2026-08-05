from __future__ import annotations

import re
from typing import Any, Callable


Verifier = Callable[[str, str], bool]


def extract_last_boxed(text: str) -> str | None:
    """Return the contents of the last balanced ``\\boxed{...}`` expression."""
    marker = r"\boxed"
    starts = [match.start() for match in re.finditer(re.escape(marker), text or "")]
    for start in reversed(starts):
        cursor = start + len(marker)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            continue
        depth = 1
        content_start = cursor + 1
        cursor += 1
        while cursor < len(text):
            if text[cursor] == "{" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth += 1
            elif text[cursor] == "}" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    return text[content_start:cursor].strip()
            cursor += 1
    return None


def normalize_answer(value: str) -> str:
    normalized = (value or "").strip()
    if normalized.startswith("$") and normalized.endswith("$") and len(normalized) >= 2:
        normalized = normalized[1:-1]
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.rstrip(".")


def math_verify_equivalent(gold: str, prediction: str) -> bool:
    """Check symbolic equivalence with math-verify, with exact fallback for simple values."""
    if normalize_answer(gold) == normalize_answer(prediction):
        return True
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

        # Boxing supplies the extraction anchor expected by math-verify for
        # expressions such as bare ``\sqrt{4}``.
        parsed_gold = parse(r"\boxed{" + gold + "}", (LatexExtractionConfig(),))
        parsed_prediction = parse(
            r"\boxed{" + prediction + "}",
            (ExprExtractionConfig(), LatexExtractionConfig()),
        )
        if not parsed_gold or not parsed_prediction:
            return False
        return bool(verify(parsed_gold, parsed_prediction))
    except (ImportError, RuntimeError, TypeError, ValueError):
        return False


def score_response(solution_str: str, ground_truth: str, verifier: Verifier | None = None) -> dict[str, Any]:
    """Score a response while keeping formatting and mathematical correctness separate.

    The scalar GRPO reward is correctness. A boxed final answer is mandatory, so an
    unboxed answer cannot receive correctness reward. ``format_reward`` is retained
    as an audit field and is not added as an extra shaping bonus.
    """
    candidate = extract_last_boxed(solution_str or "")
    if candidate is None:
        return {
            "reward": 0.0,
            "answer_reward": 0.0,
            "format_reward": 0.0,
            "parsed_answer": None,
            "failure_reason": "missing_boxed_answer",
        }

    equivalent = (verifier or math_verify_equivalent)(ground_truth, candidate)
    answer_reward = float(bool(equivalent))
    return {
        "reward": answer_reward,
        "answer_reward": answer_reward,
        "format_reward": 1.0,
        "parsed_answer": candidate,
        "failure_reason": None if equivalent else "answer_mismatch",
    }


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    """veRL-compatible custom reward entry point."""
    del data_source, extra_info
    if solution_str is None or ground_truth is None:
        return 0.0
    return float(score_response(solution_str, ground_truth)["reward"])
