from types import SimpleNamespace

import pytest

from scripts.build_rl03_mcq_audit_manifest import build_probe_prefixes
from scripts.run_rl03_mcq_audit_vllm import (
    build_run_contract,
    build_stage_a0_requests,
    ensure_run_contract,
    pending_requests,
    render_request,
    score_requests,
    scored_row,
    target_token_ids_from_rendered,
)


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        ids = [ord(char) for char in text]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class _Processor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is False
        user_text = next(
            item["text"] for item in messages[0]["content"] if item["type"] == "text"
        )
        assistant_text = messages[1]["content"][0]["text"]
        return f"<user>{user_text}</user><assistant>{assistant_text}<end>"


class _FakeLlm:
    def generate(self, prompts, sampling_params, use_tqdm=False):
        del sampling_params
        assert use_tqdm is False
        outputs = []
        for payload in prompts:
            ids = [ord(char) for char in payload["prompt"]]
            logprobs = [None] + [{token_id: -1.0} for token_id in ids[1:]]
            outputs.append(
                SimpleNamespace(prompt_token_ids=ids, prompt_logprobs=logprobs)
            )
        return outputs


def _record():
    probes = build_probe_prefixes("x" * 100, "x" * 73)
    return {
        "question_id": "q1",
        "rollout_id": 7,
        "gold_letter": "B",
        "pred_letter": "A",
        "is_correct": False,
        "prompt": "Question?\nA: one\nB: two\nC: three\nD: four",
        "image_path": "data/q1.png",
        "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "gold_content_surfaces": ["two"],
        "think_text": "x" * 100,
        "trimmed_text": "x" * 73,
        "probes": [
            {key: value for key, value in probe.items() if key != "reasoning_prefix"}
            for probe in probes
        ],
    }


def test_target_token_ids_cover_final_answer_and_fixed_terminator_only():
    tokenizer = _Tokenizer()
    rendered = "reasoning mentions 1/2\n\nFinal answer: 1/2\n"

    target_ids = target_token_ids_from_rendered(
        rendered,
        interface="\n\nFinal answer: ",
        target_text="1/2",
        terminator="\n",
        tokenizer=tokenizer,
    )

    assert target_ids == [ord(char) for char in "1/2\n"]


def test_scored_row_uses_last_matching_target_sequence():
    request = {"request_id": "q1:7:target", "question_id": "q1", "rollout_id": 7}
    output = SimpleNamespace(
        prompt_token_ids=[20, 30, 10, 20, 30, 40],
        prompt_logprobs=[
            None,
            {30: -99.0},
            {10: -99.0},
            {20: -2.0},
            {30: -4.0},
            {40: -6.0},
        ],
    )

    row = scored_row(request, output, target_ids=[20, 30, 40])

    assert row["score_sum"] == -12.0
    assert row["score_mean"] == -4.0
    assert row["token_count"] == 3
    assert row["target_token_indices"] == [3, 4, 5]


def test_scored_row_rejects_multi_token_letter_first_target():
    request = {
        "request_id": "q1:7:letter",
        "question_id": "q1",
        "rollout_id": 7,
        "representation": "letter_first",
    }
    output = SimpleNamespace(
        prompt_token_ids=[10, 20, 30],
        prompt_logprobs=[None, {20: -2.0}, {30: -3.0}],
    )

    with pytest.raises(ValueError, match="one context token"):
        scored_row(request, output, target_ids=[20, 30])


def test_stage_a0_materializes_the_frozen_240_request_matrix():
    rows = build_stage_a0_requests([_record()])

    assert len(rows) == 240
    assert len({row["request_id"] for row in rows}) == 240


def test_pending_requests_skips_only_completed_request_ids():
    requests = [
        {"request_id": "r1"},
        {"request_id": "r2"},
        {"request_id": "r3"},
    ]

    pending = pending_requests(requests, [{"request_id": "r2"}])

    assert [request["request_id"] for request in pending] == ["r1", "r3"]


def test_run_contract_refuses_model_or_manifest_changes_on_resume(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"question_id":"q1"}\n', encoding="utf-8")
    requests = [{"request_id": "r1"}, {"request_id": "r2"}]
    contract = build_run_contract(
        manifest,
        model="/models/qwen",
        requests=requests,
        dtype="bfloat16",
        max_model_len=32768,
        seed=17,
    )
    path = tmp_path / "run_contract.json"

    ensure_run_contract(path, contract, overwrite=False)
    ensure_run_contract(path, contract, overwrite=False)
    changed = {**contract, "model": "/models/other"}
    with pytest.raises(ValueError, match="run contract"):
        ensure_run_contract(path, changed, overwrite=False)

    manifest.write_text('{"question_id":"q2"}\n', encoding="utf-8")
    changed_manifest = build_run_contract(
        manifest,
        model="/models/qwen",
        requests=requests,
        dtype="bfloat16",
        max_model_len=32768,
        seed=17,
    )
    with pytest.raises(ValueError, match="run contract"):
        ensure_run_contract(path, changed_manifest, overwrite=False)


def test_render_request_keeps_multimodal_payload_and_final_target_span(tmp_path):
    from PIL import Image

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    request = next(
        request
        for request in build_stage_a0_requests([{**_record(), "image_path": str(image_path)}])
        if request["probe_id"] == "think_0.25"
        and request["interface_id"] == "I1"
        and request["representation"] == "content_sequence"
        and request["target_id"] == "option_B"
    )

    payload, target_ids, opened = render_request(request, _Processor(), tmp_path)
    try:
        assert payload["multi_modal_data"]["image"].size == (2, 2)
        assert target_ids == [ord(char) for char in "two\n"]
    finally:
        for image in opened:
            image.close()


def test_scoring_is_identical_one_by_one_and_batched(tmp_path):
    requests = [
        request
        for request in build_stage_a0_requests([{**_record(), "image_path": ""}])
        if request["probe_id"] == "think_0.25"
        and request["interface_id"] == "I1"
        and request["representation"] == "content_sequence"
    ][:2]

    one_by_one, _ = score_requests(
        _FakeLlm(),
        _Processor(),
        object(),
        requests,
        tmp_path,
        1,
        tmp_path / "one.jsonl",
        tmp_path / "one_progress.json",
    )
    batched, _ = score_requests(
        _FakeLlm(),
        _Processor(),
        object(),
        requests,
        tmp_path,
        2,
        tmp_path / "batch.jsonl",
        tmp_path / "batch_progress.json",
    )

    fields = ("request_id", "score_sum", "score_mean", "token_count")
    assert [[row[field] for field in fields] for row in one_by_one] == [
        [row[field] for field in fields] for row in batched
    ]
