import math

from scripts.mathverse_qwen3vl_option_gain_reward import (
    clipped_late_gain,
    compute_score,
    extract_final_choice,
    normalize_choice,
    prompt_and_images_from_extra_info,
    response_prefixes_by_frac,
)


def test_extract_final_choice_prefers_after_think_close():
    text = "<think>Maybe B. Final answer: B</think>\nAfter checking, Answer: C"

    assert extract_final_choice(text) == "C"


def test_prompt_and_images_from_extra_info_preserves_multimodal_question():
    prompt, images = prompt_and_images_from_extra_info(
        {
            "question": "<image>\nQuestion: choose one\nA: x\nB: y",
            "image_path": "/tmp/image.png",
            "question_id": "q1",
        }
    )

    assert "Question: choose one" in prompt
    assert "<image>" not in prompt
    assert images == ["/tmp/image.png"]


def test_response_prefixes_keep_zero_diagnostic_and_reward_fracs():
    prefixes = response_prefixes_by_frac("abcdefghij", [0.0, 0.25, 0.5, 0.9])

    assert prefixes == ["", "ab", "abcde", "abcdefghi"]


def test_clipped_late_gain_excludes_prompt_only_margin():
    margins = [10.0, 0.0, 0.4, 1.0]

    assert math.isclose(clipped_late_gain(margins, clip_value=0.5, reward_start_index=1), 0.45)


def test_normalize_choice_accepts_only_abcd():
    assert normalize_choice(" c ") == "C"
    assert normalize_choice("E") is None


def test_compute_score_returns_fixed_margin_keys_when_probe_fails(monkeypatch):
    monkeypatch.setenv("OPTION_GAIN_RESPONSE_FRACS", "0.0,0.25,0.50,0.90")

    result = compute_score(solution_str="Answer: A", ground_truth="A", extra_info={})

    assert result["option_probe_failed"] == 1.0
    assert result["option_probe_count"] == 0.0
    for index in range(4):
        assert f"option_margin_{index}" in result
        assert math.isnan(result[f"option_margin_{index}"])
    assert "option_probe_elapsed_s" in result
    assert "option_probe_response_chars" in result


def test_compute_score_deferred_vllm_backend_does_not_probe(monkeypatch):
    monkeypatch.setenv("OPTION_GAIN_BACKEND", "vllm")
    monkeypatch.setenv("OPTION_GAIN_RESPONSE_FRACS", "0.0,0.25,0.50,0.90")

    result = compute_score(solution_str="Answer: B", ground_truth="B", extra_info={})

    assert result["score"] == 1.0
    assert result["answer_reward"] == 1.0
    assert result["option_logit_gain_raw"] == 0.0
    assert result["option_probe_deferred"] == 1.0
    assert result["option_probe_failed"] == 0.0
    assert result["option_probe_count"] == 0.0
    for index in range(4):
        assert f"option_margin_{index}" in result
        assert math.isnan(result[f"option_margin_{index}"])


def test_compute_score_carries_question_identity_for_vllm_grouping(monkeypatch):
    monkeypatch.setenv("OPTION_GAIN_BACKEND", "vllm")

    result = compute_score(
        solution_str="Answer: D",
        ground_truth="D",
        extra_info={"question_id": "mathverse-q99", "source_index": 99},
    )

    assert result["question_id"] == "mathverse-q99"
    assert result["source_index"] == 99.0
