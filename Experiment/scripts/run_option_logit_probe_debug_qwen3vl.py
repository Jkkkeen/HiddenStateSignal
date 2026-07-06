#!/usr/bin/env python3
"""Debug fixed-grid option-logit probes for Qwen3-VL Thinking rollouts.

This is intentionally a small smoke/debug script. It forwards a few rollout
prefixes with a fixed answer probe prompt and prints A/B/C/D logits so we can
check leakage and whether margins change across progress positions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from response_segment_steps import answer_after_think
from run_long_path_smoke_qwen3vl import (
    build_prompt_messages,
    encode_messages,
    load_jsonl,
    select_rows,
    to_device,
)
from run_semantic_step_basin_qwen3vl import normalize_label, think_text_and_char_span


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_LABELS = "A,B,C,D"
DEFAULT_PROBE_SUFFIX = "\n\nGiven the reasoning so far, the answer is ("


@dataclass(frozen=True)
class ProbePoint:
    segment: str
    frac: float
    char_end: int
    text_prefix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug option-logit probe trajectory.")
    parser.add_argument("--input", required=True, help="Labeled long rollout JSONL.")
    parser.add_argument("--output-dir", default="option_logit_probe_debug")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--option-labels", default=DEFAULT_LABELS)
    parser.add_argument(
        "--think-fracs",
        default="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,1.00",
    )
    parser.add_argument("--answer-fracs", default="0.00,0.25,0.50,0.75")
    parser.add_argument("--limit-rollouts", type=int, default=3)
    parser.add_argument("--clean-only", action="store_true", default=True)
    parser.add_argument("--probe-suffix", default=DEFAULT_PROBE_SUFFIX)
    parser.add_argument("--tail-chars", type=int, default=220)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def parse_csv_floats(raw: str) -> list[float]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def parse_labels(raw: str) -> list[str]:
    return [item.strip().upper() for item in str(raw).split(",") if item.strip()]


def char_end_for_frac(text: str, frac: float) -> int:
    text = str(text or "")
    if not text:
        return 0
    frac = max(0.0, min(float(frac), 1.0))
    if frac <= 0.0:
        return 1
    return max(1, min(len(text), int(round(frac * len(text)))))


def probe_char_positions(
    think_text: str,
    answer_text: str,
    think_fracs: list[float],
    answer_fracs: list[float],
) -> list[ProbePoint]:
    probes: list[ProbePoint] = []
    seen: set[tuple[str, int]] = set()
    for frac in think_fracs:
        end = char_end_for_frac(think_text, frac)
        key = ("think", end)
        if end > 0 and key not in seen:
            probes.append(ProbePoint("think", float(frac), end, think_text[:end]))
            seen.add(key)
    for frac in answer_fracs:
        end = char_end_for_frac(answer_text, frac)
        key = ("answer", end)
        if end > 0 and key not in seen:
            probes.append(ProbePoint("answer", float(frac), end, answer_text[:end]))
            seen.add(key)
    return probes


def entropy_from_logits(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values)
    probs = np.exp(values)
    probs = probs / np.sum(probs)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def logit_metrics(
    logits_by_label: dict[str, float],
    labels: list[str],
    correct: str | None,
    chosen: str | None,
) -> dict[str, Any]:
    values = np.asarray([logits_by_label[label] for label in labels], dtype=np.float64)
    top_idx = int(np.argmax(values))
    top_option = labels[top_idx]
    result: dict[str, Any] = {
        "top_option": top_option,
        "top_logit": float(values[top_idx]),
        "abcd_entropy": entropy_from_logits(values),
    }
    for label in labels:
        result[f"logit_{label}"] = float(logits_by_label[label])

    if correct in labels:
        wrong = [label for label in labels if label != correct]
        correct_logit = float(logits_by_label[correct])
        wrong_logits = np.asarray([logits_by_label[label] for label in wrong], dtype=np.float64)
        result["correct_margin_max"] = float(correct_logit - np.max(wrong_logits))
        result["correct_margin_mean"] = float(correct_logit - np.mean(wrong_logits))
        result["is_top_correct"] = bool(top_option == correct)
    else:
        result["correct_margin_max"] = float("nan")
        result["correct_margin_mean"] = float("nan")
        result["is_top_correct"] = False

    if chosen in labels:
        other = [label for label in labels if label != chosen]
        chosen_logit = float(logits_by_label[chosen])
        other_logits = np.asarray([logits_by_label[label] for label in other], dtype=np.float64)
        result["chosen_margin_max"] = float(chosen_logit - np.max(other_logits))
        result["chosen_margin_mean"] = float(chosen_logit - np.mean(other_logits))
        result["is_top_chosen"] = bool(top_option == chosen)
    else:
        result["chosen_margin_max"] = float("nan")
        result["chosen_margin_mean"] = float("nan")
        result["is_top_chosen"] = False
    return result


def label_token_ids(tokenizer: Any, labels: list[str]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for label in labels:
        candidates = [
            tokenizer.encode(label, add_special_tokens=False),
            tokenizer.encode(f" {label}", add_special_tokens=False),
        ]
        chosen: list[int] | None = None
        for cand in candidates:
            if len(cand) == 1:
                chosen = cand
                break
        if chosen is None:
            chosen = candidates[0]
        if len(chosen) != 1:
            raise ValueError(f"Label {label!r} is not a single token: {chosen}")
        ids[label] = int(chosen[0])
    return ids


def build_probe_messages(row: dict[str, Any], prefix_text: str, suffix: str) -> list[dict[str, Any]]:
    messages = build_prompt_messages(row)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"{prefix_text}{suffix}"}],
        }
    )
    return messages


def tail(text: str, n: int) -> str:
    clean = str(text or "").replace("\n", " ")
    return clean[-n:] if n > 0 else ""


def main() -> None:
    args = parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    labels = parse_labels(args.option_labels)
    think_fracs = parse_csv_floats(args.think_fracs)
    answer_fracs = parse_csv_floats(args.answer_fracs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)

    rows = select_rows(load_jsonl(Path(args.input)), args.clean_only, args.limit_rollouts)
    if not rows:
        raise ValueError("No rows selected.")

    print("Option logit probe debug")
    print(f"input: {args.input}")
    print(f"selected rollouts: {len(rows)}")
    print(f"output_dir: {output_dir}")
    print(f"model: {args.model}")
    print(f"labels: {labels}")
    print(f"think_fracs: {think_fracs}")
    print(f"answer_fracs: {answer_fracs}")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    tokenizer = getattr(processor, "tokenizer", processor)
    label_ids = label_token_ids(tokenizer, labels)
    print(f"label_token_ids: {label_ids}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    model.eval()
    device = next(model.parameters()).device

    out_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="option probes"):
        qid = str(row.get("question_id", ""))
        rid = int(row.get("rollout_id", -1))
        correct = normalize_label(row.get("answer"))
        chosen = normalize_label(row.get("pred_answer"))
        try:
            response = str(row.get("response", ""))
            think_text, _, _, think_status = think_text_and_char_span(response)
            answer_text, _, _, answer_status = answer_after_think(response)
            probes = probe_char_positions(think_text, answer_text, think_fracs, answer_fracs)
            if not probes:
                raise ValueError("no probe points")

            for probe_idx, probe in enumerate(probes):
                messages = build_probe_messages(row, probe.text_prefix, args.probe_suffix)
                batch = encode_messages(processor, messages, add_generation_prompt=False)
                batch = to_device(batch, device)
                with torch.inference_mode():
                    outputs = model(**batch, use_cache=False)
                last_logits = outputs.logits[0, -1].float().detach().cpu().numpy()
                logits = {label: float(last_logits[token_id]) for label, token_id in label_ids.items()}
                metrics = logit_metrics(logits, labels, correct, chosen)
                out_rows.append(
                    {
                        "question_id": qid,
                        "rollout_id": rid,
                        "source_line": int(row.get("_source_line", row.get("source_line", -1))),
                        "is_correct": bool(row.get("is_correct", False)),
                        "answer": correct,
                        "pred_answer": chosen,
                        "probe_idx": int(probe_idx),
                        "segment": probe.segment,
                        "frac": float(probe.frac),
                        "char_end": int(probe.char_end),
                        "think_chars": int(len(think_text)),
                        "answer_chars": int(len(answer_text)),
                        "think_status": think_status,
                        "answer_status": answer_status,
                        "prefix_tail": tail(probe.text_prefix, args.tail_chars),
                        **metrics,
                    }
                )
                del outputs, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as exc:
            skipped.append({"question_id": qid, "rollout_id": rid, "reason": str(exc)})
            print(f"[skip] question={qid} rollout={rid} reason={exc}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    jsonl_path = output_dir / "option_logit_probe_debug.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in out_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    skipped_path = output_dir / "option_logit_probe_skipped.jsonl"
    with skipped_path.open("w", encoding="utf-8") as f:
        for item in skipped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report_path = output_dir / "OPTION_LOGIT_PROBE_DEBUG.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Option Logit Probe Debug\n\n")
        f.write(f"- Rollouts requested: {len(rows)}\n")
        f.write(f"- Probe rows: {len(out_rows)}\n")
        f.write(f"- Skipped rollouts: {len(skipped)}\n")
        f.write(f"- Labels: {labels}\n")
        f.write(f"- Label token ids: {label_ids}\n")
        f.write("\n## Probe Rows\n\n")
        f.write("| qid | rid | seg | frac | top | correct | chosen | corr max | corr mean | chosen max | entropy | prefix tail |\n")
        f.write("|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---|\n")
        for item in out_rows[:220]:
            f.write(
                f"| {item['question_id']} | {item['rollout_id']} | {item['segment']} | {item['frac']:.2f} | "
                f"{item['top_option']} | {item['answer']} | {item['pred_answer']} | "
                f"{item['correct_margin_max']:.3f} | {item['correct_margin_mean']:.3f} | "
                f"{item['chosen_margin_max']:.3f} | {item['abcd_entropy']:.3f} | "
                f"{str(item['prefix_tail']).replace('|', '/')} |\n"
            )

    print(f"saved jsonl: {jsonl_path} ({len(out_rows)} rows)")
    print(f"saved skipped: {skipped_path} ({len(skipped)} rows)")
    print(f"saved report: {report_path}")


if __name__ == "__main__":
    main()
