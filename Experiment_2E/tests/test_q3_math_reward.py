from experiment_2e.q3_math_reward import compute_score


def test_q3_reward_keeps_binary_boxed_score_but_audits_unboxed_parseability(monkeypatch):
    monkeypatch.setattr(
        "experiment_2e.math_reward.math_verify_equivalent",
        lambda gold, prediction: gold == prediction,
    )

    boxed = compute_score(solution_str=r"Therefore \\boxed{4}.", ground_truth="4")
    assert boxed == {
        "score": 1.0,
        "acc": True,
        "format_correct": True,
        "parse_correct": True,
    }

    unboxed = compute_score(solution_str="The final answer is 4.", ground_truth="4")
    assert unboxed == {
        "score": 0.0,
        "acc": False,
        "format_correct": False,
        "parse_correct": True,
    }

    missing = compute_score(solution_str="Reasoning stopped mid-sentence", ground_truth="4")
    assert missing["score"] == 0.0
    assert missing["format_correct"] is False
    assert missing["parse_correct"] is False
