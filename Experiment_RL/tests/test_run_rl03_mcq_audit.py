from scripts.build_rl03_mcq_audit_manifest import build_probe_prefixes
from scripts.run_rl03_mcq_audit import build_scoring_requests, score_rendered_request


def _record():
    think = "x" * 100
    probes = build_probe_prefixes(think, "x" * 73)
    return {
        "question_id": "q1",
        "rollout_id": 7,
        "gold_letter": "B",
        "pred_letter": "A",
        "prompt": "Question?\nA: one\nB: two\nC: three\nD: four",
        "image_path": "data/q1.png",
        "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "gold_content_surfaces": ["two"],
        "think_text": think,
        "trimmed_text": "x" * 73,
        "probes": [
            {key: value for key, value in probe.items() if key != "reasoning_prefix"}
            for probe in probes
        ],
    }


def test_build_scoring_requests_materializes_frozen_sparse_and_dense_matrix():
    requests = build_scoring_requests([_record()])

    assert len(requests) == 240
    i0 = [request for request in requests if request["interface_id"] == "I0"]
    assert len(i0) == 60
    assert {request["representation"] for request in i0} == {"letter_first"}

    dense_content = [
        request
        for request in requests
        if request["interface_id"] == "I1"
        and request["representation"] == "content_sequence"
        and request["probe_id"] == "think_0.20"
    ]
    assert [(request["target_id"], request["target_text"]) for request in dense_content] == [
        ("gold_surface_0", "two")
    ]

    sparse_content = [
        request
        for request in requests
        if request["interface_id"] == "I1"
        and request["representation"] == "content_sequence"
        and request["probe_id"] == "think_0.25"
    ]
    assert {request["target_id"] for request in sparse_content} == {
        "option_A",
        "option_B",
        "option_C",
        "option_D",
    }


def test_build_scoring_requests_reconstructs_prompt_think_and_trimmed_prefixes():
    requests = build_scoring_requests([_record()])
    lookup = {
        (request["probe_id"], request["interface_id"], request["representation"], request["target_id"]): request
        for request in requests
    }

    prompt = lookup[("prompt_0.00", "I0", "letter_first", "letter_A")]
    think = lookup[("think_0.25", "I0", "letter_first", "letter_A")]
    trimmed = lookup[("trimmed_final", "I0", "letter_first", "letter_A")]
    assert prompt["reasoning_prefix"] == ""
    assert think["reasoning_prefix"] == "x" * 25
    assert trimmed["reasoning_prefix"] == "x" * 73
    assert prompt["assistant_text"][prompt["target_start"] : prompt["target_end"]] == "A"


def test_build_scoring_requests_reproduces_legacy_prompt_and_neutralizes_content_cells():
    record = _record()
    instruction = (
        "Please first conduct reasoning, and then answer the question and provide "
        "the correct option letter, e.g., A, B, C, D, at the end.\n"
    )
    record["prompt"] = instruction + "\n  " + record["prompt"] + "  \n"

    requests = build_scoring_requests([record])
    i0 = next(request for request in requests if request["interface_id"] == "I0")
    i1_legacy = next(
        request
        for request in requests
        if request["interface_id"] == "I1"
        and request["representation"] == "letter_first"
        and request["prompt_mode"] == "legacy_letter_instruction"
    )
    i1_neutral = next(
        request
        for request in requests
        if request["interface_id"] == "I1"
        and request["representation"] == "letter_first"
        and request["prompt_mode"] == "neutralized_letter_instruction"
    )
    i1_content = next(
        request
        for request in requests
        if request["interface_id"] == "I1"
        and request["representation"] == "content_sequence"
    )

    wrapper = (
        "Solve the following MathVerse problem. Give concise reasoning and the final answer.\n\n"
    )
    assert i0["prompt"] == wrapper + instruction + "\n  " + _record()["prompt"] + "  \n"
    assert i0["prompt_mode"] == "legacy_letter_instruction"
    assert i1_legacy["prompt"] == i0["prompt"]
    assert i1_neutral["prompt"] == wrapper + "\n  " + _record()["prompt"] + "  \n"
    assert i1_content["prompt"] == i1_neutral["prompt"]


def test_score_rendered_request_scores_final_target_and_terminator_only():
    request = next(
        request
        for request in build_scoring_requests([_record()])
        if request["probe_id"] == "think_0.25"
        and request["interface_id"] == "I1"
        and request["representation"] == "content_sequence"
        and request["target_id"] == "option_B"
    )
    rendered = f"<user>two appears in the question</user><assistant>{request['assistant_text']}<end>"
    input_ids = list(range(1000, 1000 + len(rendered)))
    offsets = [(index, index + 1) for index in range(len(rendered))]
    prompt_logprobs = [
        None if index == 0 else {input_ids[index]: -1.0}
        for index in range(len(input_ids))
    ]

    result = score_rendered_request(
        request,
        rendered,
        input_ids,
        offsets,
        prompt_logprobs,
    )

    assert result["token_count"] == len("two\n")
    assert result["score_sum"] == -len("two\n")
    assert result["score_mean"] == -1.0
    target_start = rendered.rfind("two\n")
    assert result["target_token_indices"] == list(
        range(target_start, target_start + len("two\n"))
    )


def test_request_identity_changes_when_target_content_changes():
    first = _record()
    second = _record()
    second["choices"] = {**second["choices"], "B": "a different answer"}
    second["gold_content_surfaces"] = ["a different answer"]

    first_request = next(
        request
        for request in build_scoring_requests([first])
        if request["probe_id"] == "think_0.25"
        and request["interface_id"] == "I1"
        and request["target_id"] == "option_B"
    )
    second_request = next(
        request
        for request in build_scoring_requests([second])
        if request["probe_id"] == "think_0.25"
        and request["interface_id"] == "I1"
        and request["target_id"] == "option_B"
    )

    assert first_request["request_id"] != second_request["request_id"]
