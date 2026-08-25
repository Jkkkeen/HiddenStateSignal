"""Binary correctness reward with audit fields for Qwen3 Base GRPO."""

from __future__ import annotations

from typing import Any

from experiment_2e.four_level_reward import extract_unboxed_final_answer
from experiment_2e.math_reward import score_response


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    del data_source, extra_info
    response = solution_str or ""
    result = score_response(response, ground_truth or "")
    correct = bool(result["answer_reward"] >= 0.5)
    parseable_answer = result["parsed_answer"] or extract_unboxed_final_answer(response)
    return {
        "score": float(correct),
        "acc": correct,
        "format_correct": bool(result["format_reward"] >= 0.5),
        "parse_correct": parseable_answer is not None,
    }
