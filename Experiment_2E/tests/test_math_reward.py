from experiment_2e.math_reward import extract_last_boxed, score_response


def test_extract_last_boxed_handles_nested_latex_and_uses_last_answer():
    text = r"First \boxed{1}, finally \boxed{\frac{1}{2}}."
    assert extract_last_boxed(text) == r"\frac{1}{2}"


def test_score_requires_boxed_format():
    result = score_response("The answer is 4.", "4", verifier=lambda gold, pred: gold == pred)
    assert result["format_reward"] == 0.0
    assert result["answer_reward"] == 0.0
    assert result["failure_reason"] == "missing_boxed_answer"


def test_score_separates_format_and_answer_correctness():
    result = score_response(r"Thus \boxed{5}.", "4", verifier=lambda gold, pred: gold == pred)
    assert result["format_reward"] == 1.0
    assert result["answer_reward"] == 0.0
    assert result["failure_reason"] == "answer_mismatch"


def test_score_accepts_equivalent_answer_via_injected_verifier():
    result = score_response(
        r"Thus \boxed{0.5}.",
        r"\frac{1}{2}",
        verifier=lambda gold, pred: (gold, pred) == (r"\frac{1}{2}", "0.5"),
    )
    assert result["format_reward"] == 1.0
    assert result["answer_reward"] == 1.0
    assert result["reward"] == 1.0
