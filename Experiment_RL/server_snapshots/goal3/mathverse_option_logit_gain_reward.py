#!/usr/bin/env python3
"""MathVerse reward with a frozen option-logit margin-gain bonus for VERL."""

from __future__ import annotations

import os
import re
from threading import Lock
from typing import Any

import numpy as np


VALID_OPTIONS = ("A", "B", "C", "D")
INSTRUCTION = 'Think briefly. End with exactly "Answer: X" where X is A, B, C, or D.'
DEFAULT_PROBE_SUFFIX = "\n\nGiven the reasoning so far, the answer is ("

_MODEL: Any | None = None
_TOKENIZER: Any | None = None
_LABEL_TOKEN_IDS: dict[str, int] | None = None
_LOAD_ERROR: str | None = None
_MODEL_LOCK = Lock()


def normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in VALID_OPTIONS else None


def extract_final_choice(text: str | None) -> str | None:
    if not text:
        return None
    value = str(text)
    close_idx = value.rfind("</think>")
    candidates = [value[close_idx + len("</think>") :], value] if close_idx >= 0 else [value]
    patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\**\s*([ABCD])\b",
        r"\b([ABCD])\s*(?:is\s+the\s+answer|is\s+correct)\b",
        r"(?:option|choice)\s*\**\s*([ABCD])\b",
        r"\*\*([ABCD])\*\*",
        r"\(([ABCD])\)",
        r"\b([ABCD])\s*[:\.\)]",
    ]
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        for pattern in patterns:
            matches = re.findall(pattern, candidate, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        matches = re.findall(r"\b([ABCD])\b", candidate[-500:])
        if matches:
            return matches[-1].upper()
    return None


def ground_truth_choice(ground_truth: Any, extra_info: dict[str, Any] | None = None) -> str | None:
    direct = normalize_choice(ground_truth)
    if direct is not None:
        return direct
    if isinstance(ground_truth, dict):
        for key in ("ground_truth", "answer", "label"):
            direct = normalize_choice(ground_truth.get(key))
            if direct is not None:
                return direct
    if extra_info:
        for key in ("answer", "ground_truth", "label"):
            direct = normalize_choice(extra_info.get(key))
            if direct is not None:
                return direct
    return None


def answer_reward(solution_str: str | None, ground_truth: Any, extra_info: dict[str, Any] | None) -> tuple[float, str | None, str | None]:
    answer = ground_truth_choice(ground_truth, extra_info)
    pred = extract_final_choice(solution_str)
    reward = 1.0 if answer is not None and pred == answer else 0.0
    return reward, answer, pred


def prompt_text_from_extra_info(extra_info: dict[str, Any] | None) -> str:
    extra_info = extra_info or {}
    prompt_text = str(extra_info.get("prompt_text") or "").strip()
    if prompt_text:
        return prompt_text
    question = str(extra_info.get("question") or "").replace("<image>", "").strip()
    if not question:
        raise ValueError("extra_info must contain prompt_text or question for option-logit probing")
    return f"{INSTRUCTION}\nQuestion (image omitted): {question}"


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("empty float list")
    return values


def response_prefixes(response: str | None, fracs: list[float]) -> list[str]:
    text = str(response or "")
    prefixes: list[str] = []
    seen: set[int] = set()
    for frac in fracs:
        frac = max(0.0, min(float(frac), 1.0))
        end = int(round(frac * len(text))) if frac > 0 else 0
        end = max(0, min(len(text), end))
        if end not in seen:
            prefixes.append(text[:end])
            seen.add(end)
    return prefixes


def correct_margin_from_logits(logits_by_label: dict[str, float], labels: list[str], correct: str | None) -> float:
    if correct not in labels:
        return float("nan")
    wrong = [label for label in labels if label != correct]
    correct_logit = float(logits_by_label[correct])
    wrong_logits = [float(logits_by_label[label]) for label in wrong]
    return float(correct_logit - max(wrong_logits))


def clipped_mean_gain(margins: list[float], clip_value: float) -> float:
    clean = [float(value) for value in margins if np.isfinite(value)]
    if len(clean) < 2:
        return 0.0
    diffs = np.diff(np.asarray(clean, dtype=np.float64))
    clipped = np.clip(diffs, -abs(float(clip_value)), abs(float(clip_value)))
    return float(np.mean(clipped))


def score_with_option_gain(answer_reward: float, option_gain: float, bonus_lambda: float) -> float:
    return float(answer_reward + bonus_lambda * option_gain)


def _label_token_ids(tokenizer: Any, labels: list[str]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"label {label!r} is not one token: {token_ids}")
        ids[label] = int(token_ids[0])
    return ids


def _load_probe_model() -> tuple[Any, Any, dict[str, int]]:
    global _MODEL, _TOKENIZER, _LABEL_TOKEN_IDS, _LOAD_ERROR
    if _MODEL is not None and _TOKENIZER is not None and _LABEL_TOKEN_IDS is not None:
        return _MODEL, _TOKENIZER, _LABEL_TOKEN_IDS
    with _MODEL_LOCK:
        if _MODEL is not None and _TOKENIZER is not None and _LABEL_TOKEN_IDS is not None:
            return _MODEL, _TOKENIZER, _LABEL_TOKEN_IDS
        if _LOAD_ERROR is not None:
            raise RuntimeError(_LOAD_ERROR)

        model_path = os.environ.get("OPTION_GAIN_MODEL_PATH") or os.environ.get("MODEL_PATH")
        if not model_path:
            raise RuntimeError("OPTION_GAIN_MODEL_PATH or MODEL_PATH must be set")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            device = os.environ.get("OPTION_GAIN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype_name = os.environ.get("OPTION_GAIN_DTYPE", "bfloat16").lower()
            dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} and device.startswith("cuda") else torch.float32
            attn_impl = os.environ.get("OPTION_GAIN_ATTN_IMPLEMENTATION", "sdpa")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                local_files_only=True,
                trust_remote_code=True,
            )
            model.to(device)
            model.eval()
            labels = list(VALID_OPTIONS)
            label_ids = _label_token_ids(tokenizer, labels)
            _MODEL, _TOKENIZER, _LABEL_TOKEN_IDS = model, tokenizer, label_ids
            print(
                "[option_gain_reward] loaded probe model",
                f"path={model_path}",
                f"device={device}",
                f"dtype={dtype}",
                f"label_token_ids={label_ids}",
                flush=True,
            )
            return model, tokenizer, label_ids
        except Exception as exc:  # pragma: no cover - exercised in smoke logs.
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(_LOAD_ERROR) from exc


def _probe_texts(prompt_text: str, prefixes: list[str], suffix: str) -> list[str]:
    _, tokenizer, _ = _load_probe_model()
    texts = []
    for prefix in prefixes:
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": f"{prefix}{suffix}"},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        texts.append(text)
    return texts


def option_margin_trajectory(prompt_text: str, response: str | None, correct: str | None) -> list[float]:
    import torch

    model, tokenizer, label_ids = _load_probe_model()
    fracs = parse_float_list(os.environ.get("OPTION_GAIN_RESPONSE_FRACS", "0.0,0.33,0.67,0.90"))
    suffix = os.environ.get("OPTION_GAIN_PROBE_SUFFIX", DEFAULT_PROBE_SUFFIX)
    max_response_chars = int(os.environ.get("OPTION_GAIN_MAX_RESPONSE_CHARS", "4096"))
    clipped_response = str(response or "")[:max_response_chars]
    prefixes = response_prefixes(clipped_response, fracs)
    if len(prefixes) < 2:
        prefixes = ["", clipped_response]
    texts = _probe_texts(prompt_text, prefixes, suffix)
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        outputs = model(**batch, use_cache=False)
    last_positions = batch["attention_mask"].sum(dim=1) - 1
    logits = outputs.logits[torch.arange(len(texts), device=device), last_positions].float().detach().cpu().numpy()
    labels = list(VALID_OPTIONS)
    margins: list[float] = []
    for row in logits:
        logits_by_label = {label: float(row[label_ids[label]]) for label in labels}
        margins.append(correct_margin_from_logits(logits_by_label, labels, correct))
    return margins


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    base_reward, correct, pred = answer_reward(solution_str, ground_truth, extra_info)
    bonus_lambda = float(os.environ.get("OPTION_GAIN_LAMBDA", "0.2"))
    clip_value = float(os.environ.get("OPTION_GAIN_CLIP", "0.5"))

    option_gain = 0.0
    margins: list[float] = []
    probe_failed = 0.0
    try:
        prompt_text = prompt_text_from_extra_info(extra_info)
        margins = option_margin_trajectory(prompt_text, solution_str, correct)
        option_gain = clipped_mean_gain(margins, clip_value=clip_value)
    except Exception as exc:  # pragma: no cover - visible via reward_extra_info.
        probe_failed = 1.0
        print(f"[option_gain_reward] probe failed: {type(exc).__name__}: {exc}", flush=True)

    score = score_with_option_gain(base_reward, option_gain, bonus_lambda)
    margin_start = float(margins[0]) if margins else 0.0
    margin_end = float(margins[-1]) if margins else 0.0
    return {
        "score": float(score),
        "acc": float(base_reward),
        "answer_reward": float(base_reward),
        "parsed_answer": 1.0 if pred is not None else 0.0,
        "option_logit_gain": float(option_gain),
        "option_bonus": float(bonus_lambda * option_gain),
        "option_margin_start": margin_start,
        "option_margin_end": margin_end,
        "option_margin_delta": float(margin_end - margin_start),
        "option_probe_count": float(len(margins)),
        "option_probe_failed": float(probe_failed),
    }
