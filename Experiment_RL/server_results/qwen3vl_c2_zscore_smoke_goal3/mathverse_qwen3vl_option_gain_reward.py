#!/usr/bin/env python3
"""MathVerse Qwen3-VL option-logit gain reward components for VERL C2."""

from __future__ import annotations

import os
import re
import time
from threading import Lock
from typing import Any

import numpy as np


VALID_OPTIONS = ("A", "B", "C", "D")
DEFAULT_PROBE_SUFFIX = "\n\nGiven the reasoning so far, the answer is ("

_MODEL: Any | None = None
_PROCESSOR: Any | None = None
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


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("empty float list")
    return values


def prompt_and_images_from_extra_info(extra_info: dict[str, Any] | None) -> tuple[str, list[str]]:
    extra_info = extra_info or {}
    prompt_text = str(extra_info.get("prompt_text") or "").replace("<image>", "").strip()
    if not prompt_text:
        prompt_text = str(extra_info.get("question") or "").replace("<image>", "").strip()
    if not prompt_text:
        raise ValueError("extra_info must contain prompt_text or question")

    images: list[str] = []
    for key in ("image_path", "image", "figure_path"):
        value = extra_info.get(key)
        if value:
            images.append(str(value))
    raw_images = extra_info.get("images")
    if raw_images is not None:
        if hasattr(raw_images, "tolist"):
            raw_images = raw_images.tolist()
        if isinstance(raw_images, (list, tuple)):
            images.extend(str(item) for item in raw_images if item)
        elif raw_images:
            images.append(str(raw_images))

    # Preserve order, remove duplicates.
    deduped = list(dict.fromkeys(images))
    return prompt_text, deduped


def response_prefixes_by_frac(response: str | None, fracs: list[float]) -> list[str]:
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


def correct_margin_from_logits(logits_by_label: dict[str, float], correct: str | None) -> float:
    if correct not in VALID_OPTIONS:
        return float("nan")
    correct_logit = float(logits_by_label[str(correct)])
    wrong_logits = [float(logits_by_label[label]) for label in VALID_OPTIONS if label != correct]
    return float(correct_logit - max(wrong_logits))


def clipped_late_gain(margins: list[float], clip_value: float, reward_start_index: int = 1) -> float:
    clean = [float(value) for value in margins if np.isfinite(value)]
    if len(clean) <= reward_start_index + 1:
        return 0.0
    reward_margins = np.asarray(clean[reward_start_index:], dtype=np.float64)
    diffs = np.diff(reward_margins)
    clipped = np.clip(diffs, -abs(float(clip_value)), abs(float(clip_value)))
    return float(np.mean(clipped)) if clipped.size else 0.0


def _label_token_ids(tokenizer: Any) -> dict[str, int]:
    ids: dict[str, int] = {}
    for label in VALID_OPTIONS:
        candidates = [
            tokenizer.encode(label, add_special_tokens=False),
            tokenizer.encode(" " + label, add_special_tokens=False),
        ]
        token_ids = next((item for item in candidates if len(item) == 1), candidates[0])
        if len(token_ids) != 1:
            raise ValueError(f"label {label!r} is not one token: {token_ids}")
        ids[label] = int(token_ids[0])
    return ids


def _load_probe_model() -> tuple[Any, Any, Any, dict[str, int]]:
    global _MODEL, _PROCESSOR, _TOKENIZER, _LABEL_TOKEN_IDS, _LOAD_ERROR
    if _MODEL is not None and _PROCESSOR is not None and _TOKENIZER is not None and _LABEL_TOKEN_IDS is not None:
        return _MODEL, _PROCESSOR, _TOKENIZER, _LABEL_TOKEN_IDS
    with _MODEL_LOCK:
        if _MODEL is not None and _PROCESSOR is not None and _TOKENIZER is not None and _LABEL_TOKEN_IDS is not None:
            return _MODEL, _PROCESSOR, _TOKENIZER, _LABEL_TOKEN_IDS
        if _LOAD_ERROR is not None:
            raise RuntimeError(_LOAD_ERROR)
        model_path = os.environ.get("OPTION_GAIN_MODEL_PATH") or os.environ.get("MODEL_PATH")
        if not model_path:
            raise RuntimeError("OPTION_GAIN_MODEL_PATH or MODEL_PATH must be set")
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            tokenizer = processor.tokenizer
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            device = os.environ.get("OPTION_GAIN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype_name = os.environ.get("OPTION_GAIN_DTYPE", "bfloat16").lower()
            dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} and device.startswith("cuda") else torch.float32
            attn_impl = os.environ.get("OPTION_GAIN_ATTN_IMPLEMENTATION", "sdpa")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                local_files_only=True,
                trust_remote_code=True,
            )
            model.to(device)
            model.eval()
            label_ids = _label_token_ids(tokenizer)
            _MODEL, _PROCESSOR, _TOKENIZER, _LABEL_TOKEN_IDS = model, processor, tokenizer, label_ids
            print(
                "[qwen3vl_option_gain_reward] loaded probe model",
                f"path={model_path}",
                f"device={device}",
                f"dtype={dtype}",
                f"label_token_ids={label_ids}",
                flush=True,
            )
            return model, processor, tokenizer, label_ids
        except Exception as exc:  # pragma: no cover - exercised in H200 smoke.
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(_LOAD_ERROR) from exc


def _probe_messages(prompt_text: str, image_paths: list[str], prefix: str, suffix: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in image_paths[:1]:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": f"{prefix}{suffix}"},
    ]


def option_margin_trajectory(
    prompt_text: str,
    image_paths: list[str],
    response: str | None,
    correct: str | None,
) -> list[float]:
    import torch
    from PIL import Image

    model, processor, tokenizer, label_ids = _load_probe_model()
    fracs = parse_float_list(os.environ.get("OPTION_GAIN_RESPONSE_FRACS", "0.0,0.25,0.50,0.75,0.90"))
    suffix = os.environ.get("OPTION_GAIN_PROBE_SUFFIX", DEFAULT_PROBE_SUFFIX)
    max_response_chars = int(os.environ.get("OPTION_GAIN_MAX_RESPONSE_CHARS", "20000"))
    clipped_response = str(response or "")[:max_response_chars]
    prefixes = response_prefixes_by_frac(clipped_response, fracs)
    if len(prefixes) < 2:
        prefixes = ["", clipped_response]

    texts: list[str] = []
    images_per_text: list[list[Any]] = []
    for prefix in prefixes:
        messages = _probe_messages(prompt_text, image_paths, prefix, suffix)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=True,
        )
        texts.append(text)
        pil_images = [Image.open(path).convert("RGB") for path in image_paths[:1]]
        images_per_text.append(pil_images)

    inputs = processor(
        text=texts,
        images=images_per_text if any(images_per_text) else None,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
    attention_mask = inputs["attention_mask"]
    last_positions = attention_mask.sum(dim=1) - 1
    logits = outputs.logits[torch.arange(len(texts), device=device), last_positions].float().detach().cpu().numpy()

    margins: list[float] = []
    for row in logits:
        logits_by_label = {label: float(row[label_ids[label]]) for label in VALID_OPTIONS}
        margins.append(correct_margin_from_logits(logits_by_label, correct))
    return margins


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    base_reward, correct, pred = answer_reward(solution_str, ground_truth, extra_info)
    clip_value = float(os.environ.get("OPTION_GAIN_CLIP", "0.5"))
    reward_start_index = int(os.environ.get("OPTION_GAIN_REWARD_START_INDEX", "1"))
    expected_margin_count = len(parse_float_list(os.environ.get("OPTION_GAIN_RESPONSE_FRACS", "0.0,0.25,0.50,0.75,0.90")))

    margins: list[float] = []
    probe_failed = 0.0
    option_gain_raw = 0.0
    probe_started = time.perf_counter()
    response_chars = len(str(solution_str or ""))
    try:
        prompt_text, image_paths = prompt_and_images_from_extra_info(extra_info)
        margins = option_margin_trajectory(prompt_text, image_paths, solution_str, correct)
        option_gain_raw = clipped_late_gain(margins, clip_value=clip_value, reward_start_index=reward_start_index)
    except Exception as exc:  # pragma: no cover - visible in H200 logs.
        probe_failed = 1.0
        print(f"[qwen3vl_option_gain_reward] probe failed: {type(exc).__name__}: {exc}", flush=True)
    probe_elapsed_s = time.perf_counter() - probe_started

    result = {
        "score": float(base_reward),
        "acc": float(base_reward),
        "answer_reward": float(base_reward),
        "parsed_answer": 1.0 if pred is not None else 0.0,
        "option_logit_gain_raw": float(option_gain_raw),
        "option_probe_count": float(len(margins)),
        "option_probe_failed": float(probe_failed),
        "option_probe_elapsed_s": float(probe_elapsed_s),
        "option_probe_response_chars": float(response_chars),
    }
    for i in range(expected_margin_count):
        result[f"option_margin_{i}"] = float("nan")
    for i, value in enumerate(margins):
        result[f"option_margin_{i}"] = float(value)
    return result
