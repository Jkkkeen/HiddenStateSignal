#!/usr/bin/env python3
"""Score recoverability candidate options with Qwen3-VL via vLLM prompt logprobs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from recoverability_candidate_scoring import (
        calibrate_scores,
        candidate_token_indices,
        choice_features,
        parse_mathverse_choices,
        sequence_logprob,
    )
except ModuleNotFoundError:
    from scripts.recoverability_candidate_scoring import (
        calibrate_scores,
        candidate_token_indices,
        choice_features,
        parse_mathverse_choices,
        sequence_logprob,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default="/data2/hjk/projects/AI-HiddenState-ER")
    parser.add_argument("--mathverse-metadata", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--hf-home", default="/data2/hjk/cache/huggingface")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260711)
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


def load_mathverse_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("MathVerse metadata must be a JSON list")
    return {
        str(row["sample_index"]): row
        for row in rows
        if isinstance(row, dict) and row.get("sample_index") is not None
    }


def choices_for_record(
    record: dict[str, Any],
    metadata_by_question: dict[str, dict[str, Any]],
) -> dict[str, str]:
    prompt = str(record.get("prompt", ""))
    try:
        return parse_mathverse_choices(prompt)
    except ValueError as prompt_error:
        question_id = str(record.get("question_id"))
        metadata = metadata_by_question.get(question_id)
        if metadata is None:
            raise ValueError(
                f"question {question_id} has no parseable choices or MathVerse metadata"
            ) from prompt_error
        source = str(metadata.get("question_for_eval") or metadata.get("question") or "")
        return parse_mathverse_choices(source)


def resolve_image_path(image_path: str | None, project_root: Path) -> str:
    path = Path(str(image_path or ""))
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def build_candidate_messages(
    record: dict[str, Any],
    project_root: Path,
    option_text: str,
    context: str,
) -> list[dict[str, Any]]:
    image_path = resolve_image_path(record.get("image_path"), project_root)
    prompt = str(record.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("record has no prompt")
    user_content: list[dict[str, Any]] = []
    if image_path:
        user_content.append({"type": "image", "image": image_path})
    user_content.append({"type": "text", "text": prompt})

    if context == "prompt_only":
        final_text = f"Candidate answer: {option_text}"
    elif context == "reasoning":
        prefix = str(record.get("revision_prefix", "")).strip()
        if not prefix:
            raise ValueError("reasoning context requires revision_prefix")
        final_text = f"<think>\n{prefix}\n</think>\n\nCandidate answer: {option_text}"
    else:
        raise ValueError(f"unknown context {context!r}")
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": final_text}]},
    ]


def last_subsequence_indices(prompt_ids: list[int], candidate_token_ids: list[int]) -> list[int]:
    if not candidate_token_ids:
        raise ValueError("candidate token ids must be non-empty")
    width = len(candidate_token_ids)
    for start in range(len(prompt_ids) - width, -1, -1):
        if prompt_ids[start : start + width] == candidate_token_ids:
            return list(range(start, start + width))
    raise ValueError("candidate token subsequence not found in prompt ids")


def candidate_ids_from_rendered(
    rendered_text: str,
    candidate_text: str,
    input_ids: list[int],
    offsets: list[tuple[int, int] | list[int]],
) -> list[int]:
    start = str(rendered_text).rfind(str(candidate_text))
    if start < 0:
        raise ValueError("candidate text not found in rendered prompt")
    indices = candidate_token_indices(offsets, start, start + len(str(candidate_text)))
    return [int(input_ids[index]) for index in indices]


def aggregate_scored_requests(
    records: list[dict[str, Any]],
    scored_requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompt_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    reasoning_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in scored_requests:
        question_id = str(row["question_id"])
        label = str(row["option_label"])
        if row["context"] == "prompt_only":
            prompt_by_key[(question_id, label)] = row
        elif row["context"] == "reasoning":
            reasoning_by_key[(question_id, int(row["rollout_id"]), label)] = row

    outputs: list[dict[str, Any]] = []
    for record in records:
        question_id = str(record["question_id"])
        rollout_id = int(record["rollout_id"])
        gold = str(record.get("answer", "")).strip().upper()
        selected = str(record.get("pred_answer", "")).strip().upper()
        labels = tuple(
            sorted(label for source_question, label in prompt_by_key if source_question == question_id)
        )
        if len(labels) < 2:
            raise ValueError(f"question {question_id} has fewer than two scored options")
        prompt_rows = {label: prompt_by_key[(question_id, label)] for label in labels}
        reasoning_rows = {
            label: reasoning_by_key[(question_id, rollout_id, label)]
            for label in labels
        }
        prompt_scores = {
            label: float(prompt_rows[label]["score_mean"]) for label in labels
        }
        reasoning_scores = {
            label: float(reasoning_rows[label]["score_mean"]) for label in labels
        }
        calibrated_scores = calibrate_scores(reasoning_scores, prompt_scores)
        reasoning_features = choice_features(reasoning_scores, gold, selected)
        prompt_features = choice_features(prompt_scores, gold, selected)
        calibrated_features = choice_features(calibrated_scores, gold, selected)
        output: dict[str, Any] = {
            "question_id": question_id,
            "rollout_id": rollout_id,
            "answer": gold,
            "pred_answer": selected,
            "content_option_labels": "".join(labels),
            "content_option_count": len(labels),
        }
        for label in labels:
            output[f"content_score_{label}"] = reasoning_scores[label]
            output[f"content_prompt_score_{label}"] = prompt_scores[label]
            output[f"content_calibrated_score_{label}"] = calibrated_scores[label]
            output[f"content_score_sum_{label}"] = float(reasoning_rows[label]["score_sum"])
            output[f"content_token_count_{label}"] = int(reasoning_rows[label]["token_count"])
        for name, value in reasoning_features.items():
            output[f"content_{name}"] = value
        for name, value in prompt_features.items():
            output[f"content_prompt_{name}"] = value
        for name, value in calibrated_features.items():
            output[f"content_calibrated_{name}"] = value
        outputs.append(output)
    return outputs


def build_scoring_requests(
    records: list[dict[str, Any]],
    project_root: Path,
    metadata_by_question: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    seen_prompt_only: set[tuple[str, str]] = set()
    metadata_by_question = metadata_by_question or {}
    for record in records:
        choices = choices_for_record(record, metadata_by_question)
        question_id = str(record.get("question_id"))
        rollout_id = int(record.get("rollout_id", -1))
        for label, option_text in choices.items():
            prompt_key = (question_id, label)
            if prompt_key not in seen_prompt_only:
                requests.append(
                    {
                        "question_id": question_id,
                        "rollout_id": None,
                        "context": "prompt_only",
                        "option_label": label,
                        "option_text": option_text,
                        "gold_answer": str(record.get("answer", "")).strip().upper(),
                        "selected_answer": str(record.get("pred_answer", "")).strip().upper(),
                        "messages": build_candidate_messages(
                            record, project_root, option_text, "prompt_only"
                        ),
                    }
                )
                seen_prompt_only.add(prompt_key)
            requests.append(
                {
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "context": "reasoning",
                    "option_label": label,
                    "option_text": option_text,
                    "gold_answer": str(record.get("answer", "")).strip().upper(),
                    "selected_answer": str(record.get("pred_answer", "")).strip().upper(),
                    "messages": build_candidate_messages(
                        record, project_root, option_text, "reasoning"
                    ),
                }
            )
    return requests


def batched(values: list[Any], batch_size: int) -> list[list[Any]]:
    size = max(1, int(batch_size))
    return [values[index : index + size] for index in range(0, len(values), size)]


def render_scoring_request(
    request: dict[str, Any],
    processor: Any,
) -> tuple[dict[str, Any], list[int]]:
    from PIL import Image

    rendered = processor.apply_chat_template(
        request["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = processor.tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    candidate_ids = candidate_ids_from_rendered(
        rendered,
        str(request["option_text"]),
        input_ids,
        offsets,
    )
    image_path = request["messages"][0]["content"][0].get("image")
    image = Image.open(image_path).convert("RGB")
    return {"prompt": rendered, "multi_modal_data": {"image": image}}, candidate_ids


def score_requests(
    llm: Any,
    processor: Any,
    sampling_params: Any,
    requests: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    completed = 0
    started = time.perf_counter()
    for batch in batched(requests, batch_size):
        rendered_batch: list[dict[str, Any]] = []
        candidate_ids_batch: list[list[int]] = []
        for request in batch:
            rendered, candidate_ids = render_scoring_request(request, processor)
            rendered_batch.append(rendered)
            candidate_ids_batch.append(candidate_ids)
        outputs = llm.generate(
            rendered_batch,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        for request, candidate_ids, output in zip(
            batch, candidate_ids_batch, outputs, strict=True
        ):
            prompt_ids = [int(value) for value in output.prompt_token_ids]
            prompt_logprobs = list(output.prompt_logprobs or [])
            indices = last_subsequence_indices(prompt_ids, candidate_ids)
            aggregate = sequence_logprob(prompt_ids, prompt_logprobs, indices)
            scored.append(
                {
                    "question_id": request["question_id"],
                    "rollout_id": request["rollout_id"],
                    "context": request["context"],
                    "option_label": request["option_label"],
                    "option_text": request["option_text"],
                    "score_sum": aggregate["sum"],
                    "score_mean": aggregate["mean"],
                    "token_count": aggregate["token_count"],
                    "prompt_token_count": len(prompt_ids),
                }
            )
        for payload in rendered_batch:
            payload["multi_modal_data"]["image"].close()
        completed += len(batch)
        elapsed = time.perf_counter() - started
        print(
            f"SCORE_PROGRESS completed={completed}/{len(requests)} elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return scored


def main() -> None:
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    rows = load_jsonl(Path(args.input), args.limit)
    metadata_path = (
        Path(args.mathverse_metadata)
        if args.mathverse_metadata
        else Path(args.project_root) / "data" / "mathverse" / "testmini.json"
    )
    metadata_by_question = load_mathverse_metadata(metadata_path)
    requests = build_scoring_requests(
        rows,
        Path(args.project_root),
        metadata_by_question,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    print(
        f"Loading model={args.model} prefixes={len(rows)} requests={len(requests)}",
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    started = time.perf_counter()
    scored_requests = score_requests(
        llm,
        processor,
        sampling,
        requests,
        args.batch_size,
    )
    candidate_rows = aggregate_scored_requests(rows, scored_requests)
    elapsed = time.perf_counter() - started
    summary = {
        "input_rows": len(rows),
        "requests": len(requests),
        "prompt_only_requests": sum(item["context"] == "prompt_only" for item in requests),
        "reasoning_requests": sum(item["context"] == "reasoning" for item in requests),
        "scored_requests": len(scored_requests),
        "output_rows": len(candidate_rows),
        "elapsed_seconds": elapsed,
        "model": args.model,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "mathverse_metadata": str(metadata_path),
        "candidate_count_min": min(row["content_option_count"] for row in candidate_rows),
        "candidate_count_max": max(row["content_option_count"] for row in candidate_rows),
    }
    with (output_dir / "candidate_request_scores.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in scored_requests:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "candidate_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in candidate_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    import pandas as pd

    pd.DataFrame(candidate_rows).to_csv(output_dir / "candidate_scores.csv", index=False)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
