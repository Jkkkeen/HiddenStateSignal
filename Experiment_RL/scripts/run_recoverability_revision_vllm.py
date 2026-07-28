#!/usr/bin/env python3
"""Sample revision continuations for trimmed incorrect Qwen3-VL rollouts."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from mathverse_qwen3vl_option_gain_reward import normalize_choice


DEFAULT_REVISION_INSTRUCTION = (
    "The previous reasoning stopped before its final conclusion. Independently review it "
    "against the original question and image. Correct any mistake you find. Continue the "
    "reasoning briefly, then give exactly one final option letter in the format "
    '"Final answer: X", where X is A, B, C, or D.'
)

SIGNAL_FIELDS = (
    "split_role",
    "signal_rank",
    "signal_quantile",
    "selection_feature",
    "answer_leakage_flag",
    "original_prediction_claim_flag",
    "trim_status_recomputed",
    "trimmed_margin_mean",
    "trimmed_margin_max",
    "trimmed_entropy",
    "prompt_only_margin_mean",
    "prompt_only_margin_max",
    "think_final_margin_mean",
    "think_final_margin_max",
    "think_early_to_final_gain_mean",
    "think_early_to_final_gain_max",
    "think_late_margin_drop_mean",
    "think_late_margin_drop_max",
    "think_entropy_delta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="recoverability_discovery_smoke_v1")
    parser.add_argument("--project-root", default="/data2/hjk/projects/AI-HiddenState-ER")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--hf-home", default="/data2/hjk/cache/huggingface")
    parser.add_argument("--revision-instruction", default=DEFAULT_REVISION_INSTRUCTION)
    parser.add_argument("--write-response-chars", type=int, default=-1)
    return parser.parse_args()


def load_jsonl(path: Path, limit: int = -1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit >= 0 and len(rows) >= limit:
                break
    return rows


def extract_revision_choice(text: str | None) -> str | None:
    if not text:
        return None
    value = str(text)
    close_idx = value.rfind("</think>")
    candidates = [value[close_idx + len("</think>") :], value] if close_idx >= 0 else [value]
    patterns = (
        r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\**\s*([ABCD])\b",
        r"(?:option|choice)\s*\**\s*([ABCD])\s*(?:is\s+correct)?\b",
    )
    for candidate in candidates:
        for pattern in patterns:
            matches = re.findall(pattern, candidate, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()
    return None


def resolve_image_path(image_path: Any, project_root: Path) -> str:
    path = Path(str(image_path or ""))
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def build_revision_messages(
    record: dict[str, Any],
    project_root: Path,
    instruction: str = DEFAULT_REVISION_INSTRUCTION,
) -> list[dict[str, Any]]:
    image_path = resolve_image_path(record.get("image_path"), project_root)
    prompt = str(record.get("prompt", "")).strip()
    prefix = str(record.get("revision_prefix", "")).strip()
    if not prompt:
        raise ValueError("record has no prompt")
    if not prefix:
        raise ValueError("record has no revision_prefix")
    user_content: list[dict[str, Any]] = []
    if image_path:
        user_content.append({"type": "image", "image": image_path})
    user_content.append({"type": "text", "text": prompt})
    return [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"<think>\n{prefix}\n</think>"}],
        },
        {"role": "user", "content": [{"type": "text", "text": instruction}]},
    ]


def to_vllm_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from PIL import Image

    converted: list[dict[str, Any]] = []
    for message in messages:
        content: list[dict[str, Any]] = []
        for item in message.get("content", []):
            if item.get("type") == "image":
                image = Image.open(item["image"]).convert("RGB")
                content.append({"type": "image_pil", "image_pil": image})
            else:
                content.append(item)
        converted.append({"role": message.get("role", "user"), "content": content})
    return converted


def batched(values: list[Any], batch_size: int) -> list[list[Any]]:
    size = max(1, int(batch_size))
    return [values[index : index + size] for index in range(0, len(values), size)]


def summarize_prefixes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record["question_id"]), int(record["rollout_id"]))
        grouped[key].append(record)

    summaries: list[dict[str, Any]] = []
    for (question_id, rollout_id), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["sample_index"]))
        first = group[0]
        recovered_count = sum(bool(row["correct"]) for row in group)
        invalid_count = sum(row.get("prediction") is None for row in group)
        truncated_count = sum(bool(row.get("truncated_by_length")) for row in group)
        summary: dict[str, Any] = {
            "question_id": question_id,
            "rollout_id": rollout_id,
            "answer": first.get("answer"),
            "original_prediction": first.get("original_prediction"),
            "revision_samples": len(group),
            "recovered_count": int(recovered_count),
            "recovery_rate": float(recovered_count / len(group)),
            "any_recovery": bool(recovered_count > 0),
            "all_recovered": bool(recovered_count == len(group)),
            "invalid_revision_count": int(invalid_count),
            "truncated_revision_count": int(truncated_count),
            "mean_output_tokens": float(mean(int(row.get("output_token_count") or 0) for row in group)),
            "revision_predictions": [row.get("prediction") for row in group],
        }
        for field in SIGNAL_FIELDS:
            if field in first:
                summary[field] = first[field]
        summaries.append(summary)
    return summaries


def summarize_run(
    generation_records: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    predictions = Counter(
        str(row.get("prediction") or "INVALID") for row in generation_records
    )
    recovery_rates = [float(row["recovery_rate"]) for row in prefix_rows]
    invalid = sum(int(row["invalid_revision_count"]) for row in prefix_rows)
    total = len(generation_records)
    nontruncated = [row for row in generation_records if not row.get("truncated_by_length")]
    nontruncated_invalid = sum(row.get("prediction") is None for row in nontruncated)
    exhausted_without_answer = sum(
        bool(row.get("truncated_by_length")) and row.get("prediction") is None
        for row in generation_records
    )
    return {
        "run_name": args.run_name,
        "model": args.model,
        "prefixes": len(prefix_rows),
        "questions": len({row["question_id"] for row in prefix_rows}),
        "revision_samples_per_prefix": args.k,
        "total_generations": total,
        "mean_recovery_rate": float(mean(recovery_rates)) if recovery_rates else 0.0,
        "any_recovery_prefixes": int(sum(bool(row["any_recovery"]) for row in prefix_rows)),
        "all_recovered_prefixes": int(sum(bool(row["all_recovered"]) for row in prefix_rows)),
        "zero_recovery_prefixes": int(sum(float(row["recovery_rate"]) == 0.0 for row in prefix_rows)),
        "invalid_generations": int(invalid),
        "invalid_rate": float(invalid / total) if total else 0.0,
        "nontruncated_invalid_generations": int(nontruncated_invalid),
        "nontruncated_invalid_rate": (
            float(nontruncated_invalid / len(nontruncated)) if nontruncated else 0.0
        ),
        "budget_exhausted_without_answer": int(exhausted_without_answer),
        "truncated_generations": int(sum(bool(row.get("truncated_by_length")) for row in generation_records)),
        "prediction_counts": dict(predictions),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "revision_instruction": args.revision_instruction,
    }


def write_markdown(
    path: Path,
    run_summary: dict[str, Any],
    prefix_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Recoverability Revision Smoke",
        "",
        "## Run Summary",
        "",
        f"- Prefixes: {run_summary['prefixes']}",
        f"- Questions: {run_summary['questions']}",
        f"- Revisions per prefix: {run_summary['revision_samples_per_prefix']}",
        f"- Mean recovery rate: {run_summary['mean_recovery_rate']:.4f}",
        f"- Prefixes with any recovery: {run_summary['any_recovery_prefixes']}",
        f"- Zero-recovery prefixes: {run_summary['zero_recovery_prefixes']}",
        f"- Invalid generation rate: {run_summary['invalid_rate']:.4f}",
        f"- Truncated generations: {run_summary['truncated_generations']}",
        "",
        "## Prefix Preview",
        "",
        "| question | rollout | signal rank | margin mean | recovered/K | predictions |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in prefix_rows[:24]:
        lines.append(
            "| {question_id} | {rollout_id} | {signal_rank} | {margin:.4f} | "
            "{recovered}/{samples} | {predictions} |".format(
                question_id=row["question_id"],
                rollout_id=row["rollout_id"],
                signal_rank=row.get("signal_rank", ""),
                margin=float(row.get("trimmed_margin_mean") or 0.0),
                recovered=row["recovered_count"],
                samples=row["revision_samples"],
                predictions=",".join(str(item or "INVALID") for item in row["revision_predictions"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.k < 2:
        raise ValueError("K must be at least 2 for a recoverability estimate")
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from vllm import LLM, SamplingParams

    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.input), args.limit)
    prepared: list[dict[str, Any]] = []
    for record in rows:
        messages = build_revision_messages(record, project_root, args.revision_instruction)
        prepared.append({"record": record, "messages": to_vllm_messages(messages)})

    print(f"Loading vLLM model from {args.model}", flush=True)
    print(f"prefixes={len(prepared)} K={args.k} max_tokens={args.max_tokens}", flush=True)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1, "video": 0},
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    generation_records: list[dict[str, Any]] = []
    for batch in batched(prepared, args.batch_size):
        outputs = llm.chat(
            [item["messages"] for item in batch],
            sampling_params=sampling,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": True},
        )
        for item, request_output in zip(batch, outputs, strict=True):
            source = item["record"]
            answer = normalize_choice(source.get("answer"))
            for sample_index, completion in enumerate(request_output.outputs):
                response = str(completion.text or "")
                prediction = extract_revision_choice(response)
                token_ids = getattr(completion, "token_ids", None)
                output_tokens = len(token_ids) if token_ids is not None else 0
                finish_reason = getattr(completion, "finish_reason", None)
                stored_response = (
                    response
                    if args.write_response_chars < 0 or len(response) <= args.write_response_chars
                    else response[: args.write_response_chars] + "\n...[truncated stored response]..."
                )
                record: dict[str, Any] = {
                    "run_name": args.run_name,
                    "question_id": str(source.get("question_id")),
                    "rollout_id": int(source.get("rollout_id", -1)),
                    "sample_index": int(sample_index),
                    "answer": answer,
                    "original_prediction": normalize_choice(source.get("pred_answer")),
                    "prediction": prediction,
                    "correct": bool(answer is not None and prediction == answer),
                    "output_token_count": int(output_tokens),
                    "finish_reason": finish_reason,
                    "truncated_by_length": bool(
                        str(finish_reason).lower() == "length" or output_tokens >= args.max_tokens
                    ),
                    "response": stored_response,
                }
                for field in SIGNAL_FIELDS:
                    if field in source:
                        record[field] = source[field]
                generation_records.append(record)
                print(
                    "RECOVERY_ROW",
                    record["question_id"],
                    record["rollout_id"],
                    sample_index,
                    "answer",
                    answer,
                    "pred",
                    prediction,
                    "correct",
                    int(record["correct"]),
                    "tokens",
                    output_tokens,
                    "finish",
                    finish_reason,
                    flush=True,
                )

    prefix_rows = summarize_prefixes(generation_records)
    run_summary = summarize_run(generation_records, prefix_rows, args)

    generations_path = output_dir / "revision_generations.jsonl"
    prefixes_jsonl_path = output_dir / "recoverability_scores.jsonl"
    prefixes_csv_path = output_dir / "recoverability_scores.csv"
    summary_path = output_dir / "run_summary.json"
    report_path = output_dir / "RECOVERABILITY_SMOKE_REPORT.md"
    with generations_path.open("w", encoding="utf-8") as handle:
        for row in generation_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with prefixes_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in prefix_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    import pandas as pd

    pd.DataFrame(prefix_rows).to_csv(prefixes_csv_path, index=False)
    summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report_path, run_summary, prefix_rows)
    print(json.dumps(run_summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
