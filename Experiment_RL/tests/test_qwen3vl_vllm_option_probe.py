import math

import pytest

from scripts.qwen3vl_vllm_option_probe import (
    build_probe_text,
    clipped_late_gain,
    margin_from_option_logprobs,
    option_logprobs_from_extra_fields,
    prompt_actual_token_logprob,
)


def test_prompt_actual_token_logprob_reads_appended_option_before_dummy_tail():
    extra_fields = {
        "prompt_ids": [[101], [102], [35], [0]],
        "prompt_logprobs": [[-0.1], [-0.2], [-3.5], [0.0]],
    }

    assert prompt_actual_token_logprob(extra_fields, expected_token_id=35) == -3.5


def test_prompt_actual_token_logprob_rejects_unexpected_appended_token():
    extra_fields = {
        "prompt_ids": [[101], [102], [33], [0]],
        "prompt_logprobs": [[-0.1], [-0.2], [-1.25], [0.0]],
    }

    with pytest.raises(ValueError, match="expected appended token"):
        prompt_actual_token_logprob(extra_fields, expected_token_id=35)


def test_option_logprobs_from_extra_fields_maps_abcd_outputs():
    by_label = {
        "A": {"prompt_ids": [[10], [32], [0]], "prompt_logprobs": [[-0.1], [-0.7], [0.0]]},
        "B": {"prompt_ids": [[10], [33], [0]], "prompt_logprobs": [[-0.1], [-1.5], [0.0]]},
        "C": {"prompt_ids": [[10], [34], [0]], "prompt_logprobs": [[-0.1], [-0.4], [0.0]]},
        "D": {"prompt_ids": [[10], [35], [0]], "prompt_logprobs": [[-0.1], [-2.0], [0.0]]},
    }

    out = option_logprobs_from_extra_fields(by_label, {"A": 32, "B": 33, "C": 34, "D": 35})

    assert out == {"A": -0.7, "B": -1.5, "C": -0.4, "D": -2.0}


def test_margin_from_option_logprobs_uses_correct_minus_best_wrong():
    margin = margin_from_option_logprobs({"A": -0.7, "B": -1.5, "C": -0.4, "D": -2.0}, correct="A")

    assert math.isclose(margin, -0.3)


def test_clipped_late_gain_excludes_prompt_only_anchor():
    margins = [10.0, 0.0, 0.4, 1.0]

    assert math.isclose(clipped_late_gain(margins, clip_value=0.5, reward_start_index=1), 0.45)


def test_build_probe_text_appends_suffix_and_option_label():
    text = build_probe_text(
        prompt_text="Question text",
        response_prefix="<think>work",
        suffix="\nAnswer: ",
        option_label="C",
    )

    assert text == "Question text\n\n<think>work\nAnswer: C"
