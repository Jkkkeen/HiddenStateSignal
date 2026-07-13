#!/usr/bin/env python3
"""Run the frozen RL03 Stage A teacher-forced MCQ audit with vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from answer_likelihood_scoring import (
        ANSWER_END,
        PROTOCOL_VERSION,
        sequence_logprob,
        target_span_from_rendered,
        target_token_indices,
    )
    from run_rl03_mcq_audit import build_scoring_requests
except ModuleNotFoundError:
    from scripts.answer_likelihood_scoring import (
        ANSWER_END,
        PROTOCOL_VERSION,
        sequence_logprob,
        target_span_from_rendered,
        target_token_indices,
    )
    from scripts.run_rl03_mcq_audit import build_scoring_requests


HEAVY_REQUEST_FIELDS = {"prompt", "reasoning_prefix", "assistant_text"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_contract(
    manifest_path: Path,
    model: str,
    requests: Iterable[Mapping[str, Any]],
    dtype: str,
    max_model_len: int,
    seed: int,
) -> dict[str, Any]:
    """Bind resumable scores to their immutable data, model, and request matrix."""

    request_ids = [str(request["request_id"]) for request in requests]
    model_path = Path(model)
    config_path = model_path / "config.json"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_sha256": file_sha256(manifest_path),
        "model": str(model),
        "model_config_sha256": file_sha256(config_path) if config_path.is_file() else None,
        "request_count": len(request_ids),
        "request_ids_sha256": hashlib.sha256(
            "\n".join(request_ids).encode("utf-8")
        ).hexdigest(),
        "dtype": str(dtype),
        "max_model_len": int(max_model_len),
        "seed": int(seed),
    }


def ensure_run_contract(
    path: Path,
    contract: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Create a run contract or reject an incompatible resume attempt."""

    expected = dict(contract)
    if path.exists() and not overwrite:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != expected:
            raise ValueError("existing run contract does not match this scoring run")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_stage_a0_requests(
    manifest_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize the pre-registered Stage A0 request matrix."""

    return build_scoring_requests(manifest_records)


def target_token_ids_from_rendered(
    rendered_text: str,
    interface: str,
    target_text: str,
    terminator: str,
    tokenizer: Any,
) -> list[int]:
    """Return context-tokenized IDs overlapping the final target span."""

    target_start, target_end = target_span_from_rendered(
        rendered_text,
        interface,
        target_text,
        terminator,
    )
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    indices = target_token_indices(
        encoded["offset_mapping"],
        target_start,
        target_end,
    )
    return [int(encoded["input_ids"][index]) for index in indices]


def _last_subsequence_indices(values: Sequence[int], needle: Sequence[int]) -> list[int]:
    haystack = [int(value) for value in values]
    target = [int(value) for value in needle]
    if not target:
        raise ValueError("target token sequence must be non-empty")
    for start in range(len(haystack) - len(target), -1, -1):
        if haystack[start : start + len(target)] == target:
            return list(range(start, start + len(target)))
    raise ValueError("teacher-forced target token sequence not found in vLLM prompt")


def scored_row(
    request: Mapping[str, Any],
    output: Any,
    target_ids: Sequence[int],
) -> dict[str, Any]:
    """Aggregate prompt logprobs over the final target sequence only."""

    if request.get("representation") == "letter_first" and len(target_ids) != 1:
        raise ValueError("letter_first target must occupy exactly one context token")
    prompt_ids = [int(value) for value in output.prompt_token_ids]
    token_indices = _last_subsequence_indices(prompt_ids, target_ids)
    aggregate = sequence_logprob(
        prompt_ids,
        list(output.prompt_logprobs or []),
        token_indices,
    )
    kept = {
        key: value
        for key, value in request.items()
        if key not in HEAVY_REQUEST_FIELDS
    }
    return {
        **kept,
        "score_sum": float(aggregate["sum"]),
        "score_mean": float(aggregate["mean"]),
        "token_count": int(aggregate["token_count"]),
        "prompt_token_count": len(prompt_ids),
        "target_token_indices": token_indices,
    }


def pending_requests(
    requests: Iterable[Mapping[str, Any]],
    completed_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return requests not already present in a resumable score file."""

    completed_ids = [str(row["request_id"]) for row in completed_rows]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("completed score rows contain duplicate request ids")
    completed = set(completed_ids)
    result = [dict(request) for request in requests if str(request["request_id"]) not in completed]
    known = {str(request["request_id"]) for request in requests}
    unknown = completed - known
    if unknown:
        raise ValueError(f"score file contains {len(unknown)} request ids outside this manifest")
    return result


def _messages_for_request(
    request: Mapping[str, Any],
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[Any]]:
    from PIL import Image

    user_content: list[dict[str, Any]] = []
    opened_images: list[Any] = []
    image_value = str(request.get("image_path") or "").strip()
    if image_value:
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image {image_path}")
        image = Image.open(image_path).convert("RGB")
        opened_images.append(image)
        user_content.append({"type": "image", "image": str(image_path.resolve())})
    user_content.append({"type": "text", "text": str(request["prompt"])})
    messages = [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(request["assistant_text"])}],
        },
    ]
    return messages, opened_images


def render_request(
    request: Mapping[str, Any],
    processor: Any,
    project_root: Path,
) -> tuple[dict[str, Any], list[int], list[Any]]:
    """Render one frozen request and return its vLLM payload and target IDs."""

    messages, opened_images = _messages_for_request(request, project_root)
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    terminator = ANSWER_END if bool(request["include_terminator"]) else ""
    target_ids = target_token_ids_from_rendered(
        rendered,
        interface=str(request["interface"]),
        target_text=str(request["target_text"]),
        terminator=terminator,
        tokenizer=processor.tokenizer,
    )
    payload: dict[str, Any] = {"prompt": rendered}
    if opened_images:
        payload["multi_modal_data"] = {"image": opened_images[0]}
    return payload, target_ids, opened_images


def _batches(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    size = int(batch_size)
    if size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def score_requests(
    llm: Any,
    processor: Any,
    sampling_params: Any,
    requests: list[dict[str, Any]],
    project_root: Path,
    batch_size: int,
    scores_path: Path,
    progress_path: Path,
    completed_count: int = 0,
) -> tuple[list[dict[str, Any]], float]:
    """Score requests in resumable batches and persist each completed batch."""

    scored: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in _batches(requests, batch_size):
        rendered_batch: list[dict[str, Any]] = []
        target_ids_batch: list[list[int]] = []
        opened: list[Any] = []
        try:
            for request in batch:
                payload, target_ids, images = render_request(
                    request,
                    processor,
                    project_root,
                )
                rendered_batch.append(payload)
                target_ids_batch.append(target_ids)
                opened.extend(images)
            outputs = llm.generate(
                rendered_batch,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            batch_rows = [
                scored_row(request, output, target_ids)
                for request, output, target_ids in zip(
                    batch,
                    outputs,
                    target_ids_batch,
                    strict=True,
                )
            ]
            _append_jsonl(scores_path, batch_rows)
            scored.extend(batch_rows)
        finally:
            for image in opened:
                image.close()
        elapsed = time.perf_counter() - started
        progress = {
            "completed": completed_count + len(scored),
            "remaining": len(requests) - len(scored),
            "elapsed_seconds_this_process": elapsed,
            "requests_per_second_this_process": len(scored) / elapsed if elapsed else 0.0,
        }
        progress_path.write_text(
            json.dumps(progress, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "RL03_SCORE_PROGRESS "
            f"completed={progress['completed']} remaining={progress['remaining']} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )
    return scored, time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/data2/hjk/projects/AI-HiddenState-ER"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--request-limit", type=int, default=-1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hf-home", default="/data2/hjk/cache/huggingface")
    return parser.parse_args()


def _configure_environment(hf_home: str) -> None:
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_home) / "hub")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    args = parse_args()
    _configure_environment(args.hf_home)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "request_scores.jsonl"
    progress_path = args.output_dir / "progress.json"
    contract_path = args.output_dir / "run_contract.json"
    if args.overwrite:
        for stale_path in (
            scores_path,
            progress_path,
            contract_path,
            args.output_dir / "run_summary.json",
            args.output_dir / "environment.json",
        ):
            if stale_path.exists():
                stale_path.unlink()

    manifest_records = load_jsonl(args.manifest)
    requests = build_stage_a0_requests(manifest_records)
    if args.request_limit >= 0:
        requests = requests[: args.request_limit]
    contract = build_run_contract(
        args.manifest,
        args.model,
        requests,
        args.dtype,
        args.max_model_len,
        args.seed,
    )
    ensure_run_contract(contract_path, contract, overwrite=args.overwrite)
    existing = load_jsonl(scores_path) if scores_path.exists() else []
    remaining = pending_requests(requests, existing)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
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
    print(
        f"RL03_SCORE_START records={len(manifest_records)} requests={len(requests)} "
        f"completed={len(existing)} remaining={len(remaining)}",
        flush=True,
    )
    new_rows, elapsed = score_requests(
        llm,
        processor,
        sampling,
        remaining,
        args.project_root,
        args.batch_size,
        scores_path,
        progress_path,
        completed_count=len(existing),
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "model": args.model,
        "records": len(manifest_records),
        "requests": len(requests),
        "previously_completed": len(existing),
        "scored_this_process": len(new_rows),
        "scored_total": len(existing) + len(new_rows),
        "score_failures": 0,
        "elapsed_seconds_this_process": elapsed,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "run_contract": contract,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "environment": {
            name: os.environ.get(name, "")
            for name in (
                "HF_HOME",
                "HUGGINGFACE_HUB_CACHE",
                "TRANSFORMERS_CACHE",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
