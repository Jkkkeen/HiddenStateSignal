import math

import pytest

from scripts.recoverability_candidate_scoring import (
    actual_prompt_token_logprob,
    calibrate_scores,
    candidate_token_indices,
    choice_features,
    parse_mathverse_choices,
    sequence_logprob,
)
from scripts.score_recoverability_candidates_vllm import (
    aggregate_scored_requests,
    build_candidate_messages,
    build_scoring_requests,
    candidate_ids_from_rendered,
    last_subsequence_indices,
    resolve_image_path,
)


def test_parse_mathverse_choices_accepts_colon_and_parenthesized_labels():
    prompt = """Question: Pick one.
Choices:
A: 4
B) 4 sqrt(2)
C. 4 sqrt(3)
D:8
"""

    assert parse_mathverse_choices(prompt) == {
        "A": "4",
        "B": "4 sqrt(2)",
        "C": "4 sqrt(3)",
        "D": "8",
    }


def test_parse_mathverse_choices_accepts_three_and_five_contiguous_labels():
    assert parse_mathverse_choices("Choices:\nA: one\nB: two\nC: three") == {
        "A": "one",
        "B": "two",
        "C": "three",
    }
    assert tuple(parse_mathverse_choices("A.1\nB.2\nC.3\nD.4\nE.5")) == tuple("ABCDE")


def test_parse_mathverse_choices_rejects_missing_middle_label():
    with pytest.raises(ValueError, match="contiguous"):
        parse_mathverse_choices("Choices:\nA: one\nC: three")


def test_actual_prompt_token_logprob_reads_actual_token_from_vllm_row():
    row = {11: -2.5, 12: -0.25}

    assert actual_prompt_token_logprob(row, expected_token_id=11) == -2.5


def test_actual_prompt_token_logprob_accepts_vllm_logprob_object():
    class FakeLogprob:
        logprob = -1.75

    assert actual_prompt_token_logprob({7: FakeLogprob()}, expected_token_id=7) == -1.75


def test_candidate_token_indices_selects_tokens_overlapping_content_span():
    offsets = [(0, 3), (3, 6), (6, 9), (9, 12)]

    assert candidate_token_indices(offsets, candidate_start=4, candidate_end=10) == [1, 2, 3]


def test_sequence_logprob_returns_sum_mean_and_token_count():
    prompt_ids = [10, 11, 12, 13]
    prompt_logprobs = [None, {11: -1.0}, {12: -2.0}, {13: -3.0}]

    result = sequence_logprob(prompt_ids, prompt_logprobs, token_indices=[1, 2])

    assert result == {"sum": -3.0, "mean": -1.5, "token_count": 2}


def test_choice_features_reports_gold_commitment_gap_top2_gap_and_entropy():
    scores = {"A": -0.2, "B": -1.2, "C": -2.2, "D": -3.2}

    result = choice_features(scores, gold="B", selected="A")

    assert math.isclose(result["gold_margin"], -1.0)
    assert math.isclose(result["gold_margin_mean"], 2.0 / 3.0)
    assert math.isclose(result["commitment_gap"], 1.0)
    assert math.isclose(result["top2_gap"], 1.0)
    assert result["top_option"] == "A"
    expected_entropy = -sum(
        probability * math.log(probability)
        for probability in (0.6439142599, 0.2368828181, 0.0871443187, 0.0320586033)
    )
    assert math.isclose(result["entropy"], expected_entropy, rel_tol=1e-8)


def test_choice_features_supports_variable_candidate_count():
    result = choice_features({"A": -0.2, "B": -1.2, "C": -2.2}, gold="B", selected="A")

    assert math.isclose(result["gold_margin"], -1.0)
    assert math.isclose(result["gold_margin_mean"], 0.0, abs_tol=1e-12)
    assert math.isclose(result["commitment_gap"], 1.0)
    assert result["top_option"] == "A"


def test_calibrate_scores_subtracts_prompt_only_option_prior():
    reasoning = {"A": -1.0, "B": -2.0, "C": -3.0, "D": -4.0}
    prompt_only = {"A": -1.5, "B": -1.0, "C": -3.5, "D": -3.0}

    assert calibrate_scores(reasoning, prompt_only) == {
        "A": 0.5,
        "B": -1.0,
        "C": 0.5,
        "D": -1.0,
    }


def test_calibrate_scores_supports_matching_variable_candidate_sets():
    assert calibrate_scores(
        {"A": -1.0, "B": -2.0, "C": -3.0},
        {"A": -1.5, "B": -1.0, "C": -3.5},
    ) == {"A": 0.5, "B": -1.0, "C": 0.5}


def test_last_subsequence_indices_uses_appended_candidate_not_question_copy():
    prompt_ids = [1, 20, 21, 2, 20, 21, 3]

    assert last_subsequence_indices(prompt_ids, [20, 21]) == [4, 5]


def test_resolve_image_path_keeps_absolute_and_resolves_project_relative(tmp_path):
    relative = resolve_image_path("data/images/q.png", tmp_path)
    absolute_input = tmp_path / "already.png"

    assert relative == str((tmp_path / "data/images/q.png").resolve())
    assert resolve_image_path(str(absolute_input.resolve()), tmp_path) == str(
        absolute_input.resolve()
    )


def test_build_candidate_messages_separates_prompt_only_and_reasoning_context(tmp_path):
    record = {
        "prompt": "Question: Pick.\nChoices:\nA: one\nB: two\nC: three\nD: four",
        "image_path": "image.png",
        "revision_prefix": "first calculate the value",
    }

    prompt_only = build_candidate_messages(record, tmp_path, "two", "prompt_only")
    reasoning = build_candidate_messages(record, tmp_path, "two", "reasoning")

    assert prompt_only[0]["content"][0]["image"] == str((tmp_path / "image.png").resolve())
    assert prompt_only[-1]["content"][0]["text"] == "Candidate answer: two"
    assert reasoning[-1]["content"][0]["text"] == (
        "<think>\nfirst calculate the value\n</think>\n\nCandidate answer: two"
    )


def test_build_scoring_requests_deduplicates_prompt_only_by_question(tmp_path):
    base = {
        "question_id": "q1",
        "prompt": "Question: Pick.\nChoices:\nA: one\nB: two\nC: three\nD: four",
        "image_path": "image.png",
        "answer": "B",
        "pred_answer": "A",
    }
    records = [
        {**base, "rollout_id": 1, "revision_prefix": "work one"},
        {**base, "rollout_id": 2, "revision_prefix": "work two"},
    ]

    requests = build_scoring_requests(records, tmp_path)

    prompt_only = [row for row in requests if row["context"] == "prompt_only"]
    reasoning = [row for row in requests if row["context"] == "reasoning"]
    assert len(prompt_only) == 4
    assert len(reasoning) == 8
    assert {(row["rollout_id"], row["option_label"]) for row in reasoning} == {
        (1, "A"), (1, "B"), (1, "C"), (1, "D"),
        (2, "A"), (2, "B"), (2, "C"), (2, "D"),
    }


def test_build_scoring_requests_uses_metadata_choices_for_vision_only_prompt(tmp_path):
    records = [{
        "question_id": "130",
        "rollout_id": 7,
        "prompt": "According to the question shown in the image, answer with a letter.",
        "image_path": "image.png",
        "revision_prefix": "read the diagram",
        "answer": "C",
        "pred_answer": "A",
    }]
    metadata = {
        "130": {
            "question_for_eval": "Height?\nChoices:\nA:8.8\nB:10\nC:12\nD:14"
        }
    }

    requests = build_scoring_requests(records, tmp_path, metadata)

    assert len(requests) == 8
    assert {row["option_text"] for row in requests} == {"8.8", "10", "12", "14"}


def test_candidate_ids_from_rendered_uses_last_text_occurrence_and_offsets():
    rendered = "Choices: A: blue\nAssistant: Candidate answer: blue<end>"
    input_ids = [10, 11, 12, 13, 14, 15]
    offsets = [(0, 12), (12, 16), (16, 28), (28, 46), (46, 50), (50, 55)]

    assert candidate_ids_from_rendered(rendered, "blue", input_ids, offsets) == [14]


def test_aggregate_scored_requests_builds_raw_and_calibrated_content_features():
    records = [{
        "question_id": "q1",
        "rollout_id": 7,
        "answer": "B",
        "pred_answer": "A",
    }]
    prompt_scores = {"A": -2.0, "B": -1.0, "C": -3.0, "D": -4.0}
    reasoning_scores = {"A": -0.2, "B": -1.2, "C": -2.2, "D": -3.2}
    scored = []
    for label in "ABCD":
        scored.append({
            "question_id": "q1",
            "rollout_id": None,
            "context": "prompt_only",
            "option_label": label,
            "score_mean": prompt_scores[label],
            "score_sum": prompt_scores[label] * 2,
            "token_count": 2,
        })
        scored.append({
            "question_id": "q1",
            "rollout_id": 7,
            "context": "reasoning",
            "option_label": label,
            "score_mean": reasoning_scores[label],
            "score_sum": reasoning_scores[label] * 2,
            "token_count": 2,
        })

    row = aggregate_scored_requests(records, scored)[0]

    assert math.isclose(row["content_gold_margin"], -1.0)
    assert math.isclose(row["content_commitment_gap"], 1.0)
    assert math.isclose(row["content_top2_gap"], 1.0)
    assert math.isclose(row["content_calibrated_gold_margin"], -2.0)
    assert row["content_score_A"] == -0.2
    assert row["content_prompt_score_A"] == -2.0
    assert row["content_calibrated_score_A"] == 1.8
