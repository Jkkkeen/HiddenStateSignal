#!/usr/bin/env python3
"""Semantic-step correct-answer basin alignment for Qwen3-VL Thinking.

This script forwards existing labeled rollouts. It does not generate new
answers and does not save full token hidden states. It saves scalar step-level
and rollout-level basin metrics only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inspect_long_rollouts import segment_thinking_text
from run_long_path_smoke_qwen3vl import (
    common_prefix_len,
    encode_messages,
    image_part,
    load_jsonl,
    parse_layers,
    select_rows,
    to_device,
)
from run_path_dynamics import aucs_by_question, bootstrap_ci
from response_segment_steps import answer_after_think, split_response_steps
from semantic_step_basin import (
    compute_step_basin_metrics,
    has_rethink_trigger,
    split_semantic_steps,
    summarize_rollout_basin_features,
)


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_LABELS = "A,B,C,D"
PRIMARY_FEATURES = [
    "late_margin_mean",
    "early_to_late_margin_gain",
    "mean_step_correct_gain",
    "late_turn_to_correct",
    "productive_turn_p90",
]
SECONDARY_FEATURES = [
    "positive_gain_rate",
    "mean_turn_to_correct",
    "productive_turn_mean",
    "first_cross_pos",
    "final_margin",
    "wrong_lock_score",
    "rethink_gain_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic-step answer-basin alignment.")
    parser.add_argument("--input", required=True, help="Labeled long rollout JSONL.")
    parser.add_argument("--output-dir", default="angle_results")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--option-labels", default=DEFAULT_LABELS)
    parser.add_argument("--limit-rollouts", type=int, default=-1)
    parser.add_argument("--clean-only", action="store_true", default=True)
    parser.add_argument(
        "--segment",
        choices=("think", "answer"),
        default="think",
        help="Analyze thinking text or the answer segment after </think>.",
    )
    parser.add_argument(
        "--step-mode",
        choices=("auto", "block", "sentence"),
        default="auto",
        help="Answer segment step splitter. Think segment always uses semantic sentence steps.",
    )
    parser.add_argument("--min-step-chars", type=int, default=12)
    parser.add_argument("--min-semantic-steps", type=int, default=2)
    parser.add_argument(
        "--short-answer-chars",
        type=int,
        default=40,
        help="Answer segments at or below this length become one terminal step.",
    )
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_labels(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text.startswith("(") and ")" in text:
        text = text.strip("() ")
    return text[0] if text and text[0].isalpha() else None


def prompt_text(row: dict[str, Any]) -> str:
    return (
        "Solve the following MathVerse problem. Give concise reasoning and the final answer.\n\n"
        f"{row.get('prompt', '')}"
    )


def build_text_messages(row: dict[str, Any], text: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    part = image_part(row.get("image_path"))
    if part is not None:
        content.append(part)
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def build_prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return build_text_messages(row, prompt_text(row))


def build_option_messages(row: dict[str, Any], label: str) -> list[dict[str, Any]]:
    text = f"{prompt_text(row)}\n\nCandidate final answer: {label}."
    return build_text_messages(row, text)


def build_full_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = build_prompt_messages(row)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(row.get("response", ""))}],
        }
    )
    return messages


def row_cache_key(row: dict[str, Any]) -> str:
    raw = json.dumps(
        {
            "question_id": row.get("question_id"),
            "prompt": row.get("prompt", ""),
            "image_path": row.get("image_path", ""),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def think_text_and_char_span(response: str) -> tuple[str, int, int, str]:
    text = str(response or "")
    open_tag = "<think>"
    close_tag = "</think>"
    open_idx = text.find(open_tag)
    close_idx = text.find(close_tag)
    if open_idx >= 0 and close_idx > open_idx:
        start = open_idx + len(open_tag)
        return text[start:close_idx], start, close_idx, "ok"
    if close_idx >= 0:
        return text[:close_idx], 0, close_idx, "implicit_think_close_only"
    if open_idx >= 0:
        start = open_idx + len(open_tag)
        return text[start:], start, len(text), "missing_think_close"
    return text, 0, len(text), "no_think_tags"


def find_token_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(values):
        return -1
    last = len(values) - len(needle)
    for idx in range(last + 1):
        if values[idx : idx + len(needle)] == needle:
            return idx
    return -1


def think_token_span(
    response_ids: list[int],
    open_tag_ids: list[int],
    close_tag_ids: list[int],
) -> tuple[int, int, str]:
    start = 0
    status = "no_think_tags"
    open_idx = find_token_subsequence(response_ids, open_tag_ids)
    if open_idx >= 0:
        start = open_idx + len(open_tag_ids)
        status = "ok"
    close_idx = find_token_subsequence(response_ids, close_tag_ids)
    if close_idx >= 0 and close_idx >= start:
        return start, close_idx, status if open_idx >= 0 else "implicit_think_close_only"
    return start, len(response_ids), "missing_think_close" if open_idx >= 0 else status


def char_token_span(
    tokenizer: Any,
    text: str,
    start_char: int,
    end_char: int,
) -> tuple[int, int, str]:
    """Map a character span in response text to tokenizer-relative indices."""
    text = str(text or "")
    start_char = max(0, min(int(start_char), len(text)))
    end_char = max(start_char, min(int(end_char), len(text)))
    token_start = len(tokenizer.encode(text[:start_char], add_special_tokens=False))
    token_end = len(tokenizer.encode(text[:end_char], add_special_tokens=False))
    return token_start, token_end, "char_span"


def _with_rethink_flags(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for step in steps:
        item = dict(step)
        item["has_rethink_trigger"] = has_rethink_trigger(str(item.get("text", "")))
        result.append(item)
    return result


def select_response_segment_steps(
    response: str,
    segment: str = "think",
    step_mode: str = "auto",
    min_step_chars: int = 12,
    short_answer_chars: int = 40,
) -> dict[str, Any]:
    """Select text span and weak semantic steps for think or answer analysis."""
    response = str(response or "")
    if segment == "think":
        segment_text, start_char, end_char, text_status = think_text_and_char_span(response)
        if not segment_text.strip():
            seg = segment_thinking_text(response)
            segment_text = str(seg.get("think_text", ""))
            start_char = 0
            end_char = len(segment_text)
            text_status = str(seg.get("segment_status", text_status))
        steps = split_semantic_steps(segment_text, min_chars=min_step_chars)
        return {
            "segment": "think",
            "segment_text": segment_text,
            "segment_start_char": int(start_char),
            "segment_end_char": int(end_char),
            "text_segment_status": text_status,
            "steps": steps,
            "step_mode_used": "semantic",
        }

    if segment != "answer":
        raise ValueError(f"Unsupported segment {segment!r}")

    segment_text, start_char, end_char, text_status = answer_after_think(response)
    if step_mode == "sentence":
        steps, used_mode = split_response_steps(
            segment_text,
            min_chars=min_step_chars,
            short_answer_chars=max(0, short_answer_chars),
        )
        if used_mode == "block":
            from response_segment_steps import _split_sentence_steps

            steps = _split_sentence_steps(segment_text, min_chars=min_step_chars)
            used_mode = "sentence" if len(steps) > 1 else "terminal"
    elif step_mode == "block":
        from response_segment_steps import _split_block_steps

        steps = _split_block_steps(segment_text)
        used_mode = "block" if steps else "empty"
    else:
        steps, used_mode = split_response_steps(
            segment_text,
            min_chars=min_step_chars,
            short_answer_chars=max(0, short_answer_chars),
        )
    return {
        "segment": "answer",
        "segment_text": segment_text,
        "segment_start_char": int(start_char),
        "segment_end_char": int(end_char),
        "text_segment_status": text_status,
        "steps": _with_rethink_flags(steps),
        "step_mode_used": used_mode,
    }


def step_end_token_indices(
    tokenizer: Any,
    segment_text: str,
    steps: list[dict[str, Any]],
    segment_token_len: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    indices: list[int] = []
    kept_steps: list[dict[str, Any]] = []
    last_idx = -1
    for step in steps:
        cumulative = segment_text[: int(step["end_char"])]
        ids = tokenizer.encode(cumulative, add_special_tokens=False)
        if not ids:
            continue
        idx = min(len(ids) - 1, segment_token_len - 1)
        if idx <= last_idx:
            continue
        indices.append(idx)
        kept_steps.append(step)
        last_idx = idx
    return indices, kept_steps


def option_vectors_for_row(
    row: dict[str, Any],
    labels: list[str],
    layers: list[int],
    processor: Any,
    model: Any,
    device: Any,
    local_files_only: bool,
) -> dict[int, dict[str, np.ndarray]]:
    import torch

    prompt_inputs = encode_messages(processor, build_prompt_messages(row), add_generation_prompt=True)
    prompt_inputs = to_device(prompt_inputs, device)
    with torch.inference_mode():
        prompt_outputs = model(**prompt_inputs, output_hidden_states=True, use_cache=False)

    prompt_tail: dict[int, np.ndarray] = {}
    n_layers = len(prompt_outputs.hidden_states)
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} out of range; n_layers={n_layers}")
        prompt_tail[layer] = (
            prompt_outputs.hidden_states[layer][0, -1].float().detach().cpu().numpy()
        )

    result: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in layers}
    for label in labels:
        option_inputs = encode_messages(
            processor, build_option_messages(row, label), add_generation_prompt=True
        )
        option_inputs = to_device(option_inputs, device)
        with torch.inference_mode():
            option_outputs = model(**option_inputs, output_hidden_states=True, use_cache=False)
        for layer in layers:
            option_tail = (
                option_outputs.hidden_states[layer][0, -1].float().detach().cpu().numpy()
            )
            result[layer][label] = (option_tail - prompt_tail[layer]).astype(np.float32, copy=False)
        del option_outputs, option_inputs

    del prompt_outputs, prompt_inputs
    return result


def evaluate_basin_features(summary: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = PRIMARY_FEATURES + SECONDARY_FEATURES
    if summary.empty:
        return pd.DataFrame(rows)
    for layer in sorted(summary["layer"].unique()):
        sub = summary[summary["layer"] == layer]
        for feature in features:
            if feature not in sub.columns:
                continue
            aucs = aucs_by_question(sub, feature)
            if aucs.size == 0:
                continue
            neg = 1.0 - aucs
            low, high = bootstrap_ci(aucs, n_boot, seed)
            nlow, nhigh = bootstrap_ci(neg, n_boot, seed + 17)
            rows.append(
                {
                    "layer": int(layer),
                    "feature": feature,
                    "is_primary": feature in PRIMARY_FEATURES,
                    "n_questions": int(aucs.size),
                    "mean_auc_pos": float(np.mean(aucs)),
                    "ci_low_pos": low,
                    "ci_high_pos": high,
                    "mean_auc_neg": float(np.mean(neg)),
                    "ci_low_neg": nlow,
                    "ci_high_neg": nhigh,
                    "best_auc": float(max(np.mean(aucs), np.mean(neg))),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["is_primary", "best_auc"], ascending=[False, False])
    return result


def write_report(
    output_dir: Path,
    step_df: pd.DataFrame,
    summary: pd.DataFrame,
    eval_df: pd.DataFrame,
    skipped: list[dict[str, Any]],
    segment: str,
    step_mode: str,
) -> Path:
    report = output_dir / "SEMANTIC_STEP_BASIN_RESULTS.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Semantic-Step Correct-Answer Basin Results\n\n")
        f.write("This experiment forwards existing rollouts and computes answer-basin alignment on semantic steps only.\n\n")
        f.write("## Data\n\n")
        f.write(f"- Segment: {segment}\n")
        f.write(f"- Step mode: {step_mode}\n")
        f.write(f"- Step-layer rows: {len(step_df)}\n")
        f.write(f"- Rollout-layer rows: {len(summary)}\n")
        f.write(f"- Rollouts: {summary[['question_id', 'rollout_id']].drop_duplicates().shape[0] if not summary.empty else 0}\n")
        f.write(f"- Questions: {summary['question_id'].nunique() if not summary.empty else 0}\n")
        f.write(f"- Skipped rollouts: {len(skipped)}\n")
        if not summary.empty:
            f.write(f"- Mean semantic steps: {summary['n_semantic_steps'].mean():.2f}\n")
            f.write(f"- Mean rethink steps: {summary['rethink_step_count'].mean():.2f}\n")
        f.write("\n## Primary AUROC\n\n")
        f.write("| layer | feature | questions | AUROC(+feature) | 95% CI | AUROC(-feature) | 95% CI | best |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        view = eval_df[eval_df["is_primary"]].copy() if not eval_df.empty else pd.DataFrame()
        for _, row in view.iterrows():
            f.write(
                f"| {int(row['layer'])} | {row['feature']} | {int(row['n_questions'])} | "
                f"{row['mean_auc_pos']:.4f} | [{row['ci_low_pos']:.4f}, {row['ci_high_pos']:.4f}] | "
                f"{row['mean_auc_neg']:.4f} | [{row['ci_low_neg']:.4f}, {row['ci_high_neg']:.4f}] | "
                f"{row['best_auc']:.4f} |\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write("- `late_margin_mean`: late semantic states are closer to correct option than wrong options.\n")
        f.write("- `mean_step_correct_gain`: semantic steps increase correct-answer margin on average.\n")
        f.write("- `late_turn_to_correct`: late step displacements point toward the correct option direction.\n")
        f.write("- `productive_turn_p90`: large turns that also increase correct-answer margin.\n")
        f.write("\n## Files\n\n")
        f.write("- `semantic_step_basin_steps.parquet`\n")
        f.write("- `semantic_step_basin_rollout_features.parquet`\n")
        f.write("- `semantic_step_basin_eval.csv`\n")
        f.write("- `semantic_step_basin_skipped.jsonl`\n")
    return report


def main() -> None:
    args = parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    layers = parse_layers(args.layers)
    labels = parse_labels(args.option_labels)
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

    print("Semantic-step correct-answer basin alignment")
    print(f"input: {args.input}")
    print(f"selected rollouts: {len(rows)}")
    print(f"output_dir: {output_dir}")
    print(f"model: {args.model}")
    print(f"layers: {layers}")
    print(f"labels: {labels}")
    print(f"segment: {args.segment}")
    print(f"step_mode: {args.step_mode}")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    tokenizer = getattr(processor, "tokenizer", processor)
    open_tag_ids = tokenizer.encode("<think>", add_special_tokens=False)
    close_tag_ids = tokenizer.encode("</think>", add_special_tokens=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    model.eval()
    device = next(model.parameters()).device

    option_cache: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    step_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="semantic basin"):
        qid = str(row.get("question_id", ""))
        rollout_id = int(row.get("rollout_id", -1))
        answer = normalize_label(row.get("answer"))
        pred_answer = normalize_label(row.get("pred_answer"))
        try:
            if answer not in labels:
                raise ValueError(f"answer {answer!r} not in labels {labels}")
            response = str(row.get("response", ""))
            selected = select_response_segment_steps(
                response,
                segment=args.segment,
                step_mode=args.step_mode,
                min_step_chars=args.min_step_chars,
                short_answer_chars=args.short_answer_chars,
            )
            segment_text = str(selected["segment_text"])
            segment_start_char = int(selected["segment_start_char"])
            segment_end_char = int(selected["segment_end_char"])
            text_segment_status = str(selected["text_segment_status"])
            step_mode_used = str(selected["step_mode_used"])
            steps = list(selected["steps"])
            if len(steps) < args.min_semantic_steps:
                raise ValueError(f"too few semantic steps: {len(steps)}")

            prompt_inputs = encode_messages(
                processor, build_prompt_messages(row), add_generation_prompt=True
            )
            full_inputs = encode_messages(
                processor, build_full_messages(row), add_generation_prompt=False
            )
            response_start = common_prefix_len(prompt_inputs["input_ids"], full_inputs["input_ids"])
            response_end = int(full_inputs["input_ids"].shape[-1])
            response_ids = (
                full_inputs["input_ids"][0, response_start:response_end].detach().cpu().tolist()
            )
            if args.segment == "think":
                segment_start, segment_end, token_segment_status = think_token_span(
                    response_ids, open_tag_ids, close_tag_ids
                )
            else:
                segment_start, segment_end, token_segment_status = char_token_span(
                    tokenizer, response, segment_start_char, segment_end_char
                )
            segment_token_len = segment_end - segment_start
            if segment_token_len <= 0:
                raise ValueError(f"empty {args.segment} token span: {segment_token_len}")

            step_token_indices, kept_steps = step_end_token_indices(
                tokenizer, segment_text, steps, segment_token_len
            )
            if len(step_token_indices) < args.min_semantic_steps:
                raise ValueError(f"too few token-aligned semantic steps: {len(step_token_indices)}")

            cache_key = row_cache_key(row)
            if cache_key not in option_cache:
                option_cache[cache_key] = option_vectors_for_row(
                    row,
                    labels,
                    layers,
                    processor,
                    model,
                    device,
                    args.local_files_only,
                )

            full_inputs = to_device(full_inputs, device)
            with torch.inference_mode():
                outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)

            base = {
                "question_id": qid,
                "rollout_id": rollout_id,
                "source_line": int(row.get("_source_line", row.get("source_line", -1))),
                "is_correct": bool(row.get("is_correct", False)),
                "answer": answer,
                "pred_answer": pred_answer,
                "segment": args.segment,
                "step_mode": args.step_mode,
                "step_mode_used": step_mode_used,
                "finish_reason": str(row.get("finish_reason", "")),
                "truncated": bool(row.get("truncated", False)),
                "think_length": int(row.get("think_length", 0)),
                "segment_token_count": int(segment_token_len),
                "response_token_count": int(row.get("response_token_count", response_end - response_start)),
                "n_semantic_steps_text": int(len(steps)),
                "n_semantic_steps_aligned": int(len(kept_steps)),
                "text_segment_status": text_segment_status,
                "token_segment_status": token_segment_status,
            }

            for layer in layers:
                if layer < 0 or layer >= len(outputs.hidden_states):
                    raise ValueError(f"Layer {layer} out of range; n_layers={len(outputs.hidden_states)}")
                full_hidden = outputs.hidden_states[layer][0].float().detach().cpu().numpy()
                baseline_idx = response_start + segment_start - 1
                baseline_idx = max(0, baseline_idx)
                states = [full_hidden[baseline_idx].astype(np.float32, copy=False)]
                for rel_idx in step_token_indices:
                    full_idx = response_start + segment_start + rel_idx
                    full_idx = min(full_idx, full_hidden.shape[0] - 1)
                    states.append(full_hidden[full_idx].astype(np.float32, copy=False))
                states_arr = np.stack(states, axis=0).astype(np.float32, copy=False)
                metrics = compute_step_basin_metrics(
                    states_arr,
                    option_cache[cache_key][layer],
                    correct_label=answer,
                    chosen_label=pred_answer,
                    rethink_flags=[bool(step["has_rethink_trigger"]) for step in kept_steps],
                )
                if not metrics:
                    continue
                layer_base = {**base, "layer": int(layer)}
                for metric_row, step, rel_token_idx in zip(metrics, kept_steps, step_token_indices):
                    step_rows.append(
                        {
                            **layer_base,
                            **metric_row,
                            "step_start_char": int(step["start_char"]),
                            "step_end_char": int(step["end_char"]),
                            "step_text_len": int(len(step["text"])),
                            "step_end_token_rel": int(rel_token_idx),
                        }
                    )
                summary_rows.append(
                    {
                        **layer_base,
                        **summarize_rollout_basin_features(metrics),
                    }
                )

            del outputs, full_inputs, prompt_inputs
        except Exception as exc:
            skipped.append(
                {
                    "question_id": qid,
                    "rollout_id": rollout_id,
                    "reason": str(exc),
                }
            )
            print(f"[skip] question={qid} rollout={rollout_id} reason={exc}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    step_df = pd.DataFrame(step_rows)
    summary = pd.DataFrame(summary_rows)
    eval_df = evaluate_basin_features(summary, args.bootstrap, args.seed)

    step_path = output_dir / "semantic_step_basin_steps.parquet"
    summary_path = output_dir / "semantic_step_basin_rollout_features.parquet"
    eval_path = output_dir / "semantic_step_basin_eval.csv"
    skipped_path = output_dir / "semantic_step_basin_skipped.jsonl"
    step_df.to_parquet(step_path, index=False)
    summary.to_parquet(summary_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    with skipped_path.open("w", encoding="utf-8") as f:
        for item in skipped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = write_report(output_dir, step_df, summary, eval_df, skipped, args.segment, args.step_mode)

    print(f"saved steps: {step_path} ({len(step_df)} rows)")
    print(f"saved summary: {summary_path} ({len(summary)} rows)")
    print(f"saved eval: {eval_path} ({len(eval_df)} rows)")
    print(f"saved skipped: {skipped_path} ({len(skipped)} rows)")
    print(f"saved report: {report}")
    if not eval_df.empty:
        print(eval_df.head(20))


if __name__ == "__main__":
    main()
