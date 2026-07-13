#!/usr/bin/env python3
"""Score the frozen RL03 MathVerse interface-by-representation audit."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

try:
    from answer_likelihood_scoring import (
        ANSWER_END,
        INTERFACES,
        PROTOCOL_VERSION,
        build_assistant_scoring_text,
        sequence_logprob,
        target_token_indices,
        target_span_from_rendered,
    )
except ModuleNotFoundError:
    from scripts.answer_likelihood_scoring import (
        ANSWER_END,
        INTERFACES,
        PROTOCOL_VERSION,
        build_assistant_scoring_text,
        sequence_logprob,
        target_token_indices,
        target_span_from_rendered,
    )


MATHVERSE_WRAPPER = (
    "Solve the following MathVerse problem. Give concise reasoning and the final answer.\n\n"
)
LETTER_INSTRUCTION_RE = re.compile(
    r"(?im)^Please first conduct reasoning, and then answer the question and provide "
    r"the correct option letter, e\.g\., A, B, C, D, at the end\.(?:\r?\n)?"
)


def _scoring_prompt(record: Mapping[str, Any], prompt_mode: str) -> tuple[str, str]:
    prompt = str(record["prompt"])
    if prompt_mode == "legacy_letter_instruction":
        return MATHVERSE_WRAPPER + prompt, "legacy_letter_instruction"
    if prompt_mode != "neutralized_letter_instruction":
        raise ValueError(f"unknown prompt mode {prompt_mode!r}")
    neutral = LETTER_INSTRUCTION_RE.sub("", prompt, count=1)
    return MATHVERSE_WRAPPER + neutral, "neutralized_letter_instruction"


def _reasoning_prefix(record: Mapping[str, Any], probe: Mapping[str, Any]) -> str:
    kind = str(probe["probe_kind"])
    if kind == "prompt":
        return ""
    if kind == "trimmed":
        return str(record["trimmed_text"])
    if kind == "think":
        return str(record["think_text"])[: int(probe["char_end"])]
    raise ValueError(f"unknown probe kind {kind!r}")


def _request(
    record: Mapping[str, Any],
    probe: Mapping[str, Any],
    interface_id: str,
    representation: str,
    target_id: str,
    target_text: str,
    include_terminator: bool,
    *,
    is_gold_target: bool,
    prompt_mode: str,
    surface_id: str | None = None,
) -> dict[str, Any]:
    reasoning_prefix = _reasoning_prefix(record, probe)
    prompt, prompt_mode = _scoring_prompt(record, prompt_mode)
    terminator = ANSWER_END if include_terminator else ""
    assistant_text, target_start, target_end = build_assistant_scoring_text(
        reasoning_prefix,
        INTERFACES[interface_id],
        target_text,
        terminator,
    )
    request_stem = ":".join(
        [
            str(record["question_id"]),
            str(record["rollout_id"]),
            str(probe["probe_id"]),
            interface_id,
            representation,
            target_id,
            prompt_mode,
        ]
    )
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "request_stem": request_stem,
        "prompt": prompt,
        "reasoning_prefix": reasoning_prefix,
        "interface": INTERFACES[interface_id],
        "target_text": str(target_text),
        "terminator": terminator,
        "surface_id": surface_id,
    }
    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    request_id = f"{request_stem}:{identity_hash}"
    return {
        "request_id": request_id,
        "question_id": str(record["question_id"]),
        "rollout_id": int(record["rollout_id"]),
        "is_correct": bool(record.get("is_correct", False)),
        "gold_letter": str(record["gold_letter"]),
        "pred_letter": str(record.get("pred_letter", "")),
        "prompt": prompt,
        "prompt_mode": prompt_mode,
        "image_path": str(record.get("image_path", "")),
        "probe_id": str(probe["probe_id"]),
        "probe_kind": str(probe["probe_kind"]),
        "frac": probe.get("frac"),
        "is_sparse": bool(probe["is_sparse"]),
        "is_dense": bool(probe["is_dense"]),
        "interface_id": interface_id,
        "interface": INTERFACES[interface_id],
        "representation": representation,
        "target_id": target_id,
        "target_text": str(target_text),
        "surface_id": surface_id,
        "is_gold_target": bool(is_gold_target),
        "include_terminator": bool(include_terminator),
        "reasoning_prefix": reasoning_prefix,
        "assistant_text": assistant_text,
        "target_start": target_start,
        "target_end": target_end,
    }


def build_scoring_requests(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand manifest records into the frozen sparse and dense atomic score matrix."""

    requests: list[dict[str, Any]] = []
    for record in records:
        choices = {str(label): str(text) for label, text in record["choices"].items()}
        gold_letter = str(record["gold_letter"])
        gold_surfaces = [str(value) for value in record["gold_content_surfaces"]]
        for probe in record["probes"]:
            if bool(probe["is_dense"]):
                for label in choices:
                    requests.append(
                        _request(
                            record,
                            probe,
                            "I0",
                            "letter_first",
                            f"letter_{label}",
                            label,
                            False,
                            is_gold_target=label == gold_letter,
                            prompt_mode="legacy_letter_instruction",
                        )
                    )
            if bool(probe["is_sparse"]):
                for interface_id in ("I1", "I2"):
                    for label in choices:
                        for prompt_mode in (
                            "legacy_letter_instruction",
                            "neutralized_letter_instruction",
                        ):
                            requests.append(
                                _request(
                                    record,
                                    probe,
                                    interface_id,
                                    "letter_first",
                                    f"letter_{label}",
                                    label,
                                    False,
                                    is_gold_target=label == gold_letter,
                                    prompt_mode=prompt_mode,
                                )
                            )
                        requests.append(
                            _request(
                                record,
                                probe,
                                interface_id,
                                "letter_sequence",
                                f"letter_{label}",
                                label,
                                True,
                                is_gold_target=label == gold_letter,
                                prompt_mode="neutralized_letter_instruction",
                            )
                        )
                    for label, content in choices.items():
                        surface_id = "gold_surface_0" if label == gold_letter else None
                        requests.append(
                            _request(
                                record,
                                probe,
                                interface_id,
                                "content_sequence",
                                f"option_{label}",
                                content,
                                True,
                                is_gold_target=label == gold_letter,
                                prompt_mode="neutralized_letter_instruction",
                                surface_id=surface_id,
                            )
                        )
                    for surface_index, surface in enumerate(gold_surfaces[1:], start=1):
                        requests.append(
                            _request(
                                record,
                                probe,
                                interface_id,
                                "content_sequence",
                                f"gold_surface_{surface_index}",
                                surface,
                                True,
                                is_gold_target=True,
                                prompt_mode="neutralized_letter_instruction",
                                surface_id=f"gold_surface_{surface_index}",
                            )
                        )
            elif bool(probe["is_dense"]):
                for interface_id in ("I1", "I2"):
                    for surface_index, surface in enumerate(gold_surfaces):
                        requests.append(
                            _request(
                                record,
                                probe,
                                interface_id,
                                "content_sequence",
                                f"gold_surface_{surface_index}",
                                surface,
                                True,
                                is_gold_target=True,
                                prompt_mode="neutralized_letter_instruction",
                                surface_id=f"gold_surface_{surface_index}",
                            )
                        )
    request_ids = [request["request_id"] for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("scoring request ids are not unique")
    return requests


def score_rendered_request(
    request: Mapping[str, Any],
    rendered_text: str,
    input_ids: list[int],
    offsets: list[tuple[int, int] | list[int]],
    prompt_logprobs: list[Any],
) -> dict[str, Any]:
    """Score only the final teacher-forced target and optional terminator."""

    terminator = ANSWER_END if bool(request["include_terminator"]) else ""
    target_start, target_end = target_span_from_rendered(
        rendered_text,
        str(request["interface"]),
        str(request["target_text"]),
        terminator,
    )
    token_indices = target_token_indices(offsets, target_start, target_end)
    score = sequence_logprob(input_ids, prompt_logprobs, token_indices)
    return {
        **dict(request),
        "score_sum": float(score["sum"]),
        "score_mean": float(score["mean"]),
        "token_count": int(score["token_count"]),
        "target_token_indices": token_indices,
    }
