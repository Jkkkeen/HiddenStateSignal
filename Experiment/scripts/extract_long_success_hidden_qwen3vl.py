#!/usr/bin/env python3
"""Replay fixed long responses and save L24/L36 span-last/span-mean hidden states."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from long_success_trajectory_common import full_span_bounds
from run_long_path_smoke_qwen3vl import (
    build_full_messages,
    build_prompt_messages,
    common_prefix_len,
    encode_messages,
    parse_layers,
    segment_token_span_from_text,
    to_device,
)


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
REQUIRED_SHARD_KEYS = {
    "question_id",
    "layers",
    "span_last",
    "span_mean",
    "rollout_id",
    "is_correct",
    "span_start",
    "span_end",
    "relative_progress",
    "think_length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fixed long-response span hidden states.")
    parser.add_argument("--input", required=True, help="Frozen smoke manifest JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="24,36")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pool_span_representations(
    hidden: np.ndarray,
    spans: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [layers,tokens,dim], got {hidden.shape}")
    if not spans:
        empty = np.empty((0, hidden.shape[0], hidden.shape[-1]), dtype=np.float16)
        return empty, empty.copy()
    last = np.stack([hidden[:, end - 1, :] for _, end in spans], axis=0)
    mean = np.stack([hidden[:, start:end, :].mean(axis=1) for start, end in spans], axis=0)
    return last.astype(np.float16), mean.astype(np.float16)


def pool_torch_span_representations(
    hidden_states: tuple[Any, ...],
    layers: list[int],
    full_segment_start: int,
    spans: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    per_layer_last = []
    per_layer_mean = []
    for layer in layers:
        layer_hidden = hidden_states[layer][0]
        last_indices = torch.tensor(
            [full_segment_start + end - 1 for _, end in spans],
            dtype=torch.long,
            device=layer_hidden.device,
        )
        last = layer_hidden.index_select(0, last_indices).float()
        mean = torch.stack(
            [
                layer_hidden[full_segment_start + start : full_segment_start + end]
                .float()
                .mean(dim=0)
                for start, end in spans
            ],
            dim=0,
        )
        per_layer_last.append(last.cpu().numpy())
        per_layer_mean.append(mean.cpu().numpy())
    span_last = np.stack(per_layer_last, axis=1).astype(np.float16)
    span_mean = np.stack(per_layer_mean, axis=1).astype(np.float16)
    return span_last, span_mean


def shard_name(question_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(question_id)).strip("._")[:40] or "question"
    digest = hashlib.sha1(str(question_id).encode("utf-8")).hexdigest()[:12]
    return f"question_{safe}_{digest}.npz"


def valid_question_shard(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as shard:
            if not REQUIRED_SHARD_KEYS.issubset(set(shard.files)):
                return False
            last_shape = shard["span_last"].shape
            mean_shape = shard["span_mean"].shape
            if last_shape != mean_shape or len(last_shape) != 3:
                return False
            return last_shape[0] == shard["rollout_id"].shape[0]
    except Exception:
        return False


def append_status(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_question(
    question_id: str,
    rows: list[dict[str, Any]],
    processor: Any,
    model: Any,
    device: Any,
    close_tag_ids: list[int],
    layers: list[int],
    window: int,
    stride: int,
) -> dict[str, np.ndarray]:
    span_last_parts = []
    span_mean_parts = []
    rollout_ids = []
    labels = []
    starts = []
    ends = []
    relative_progress = []
    think_lengths = []
    source_lines = []

    for row in sorted(rows, key=lambda item: int(item.get("rollout_id", -1))):
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
        segment_start, segment_end, segment_status = segment_token_span_from_text(
            response_ids, close_tag_ids
        )
        if segment_status == "no_think_close":
            raise ValueError(
                f"token close boundary missing: question={question_id} rollout={row.get('rollout_id')}"
            )
        segment_length = int(segment_end - segment_start)
        spans = full_span_bounds(segment_length, window=window, stride=stride)
        if len(spans) < 2:
            raise ValueError(
                f"too few full spans: question={question_id} rollout={row.get('rollout_id')} "
                f"think_length={segment_length}"
            )

        full_inputs = to_device(full_inputs, device)
        with __import__("torch").inference_mode():
            outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)
        full_segment_start = response_start + segment_start
        span_last, span_mean = pool_torch_span_representations(
            outputs.hidden_states,
            layers=layers,
            full_segment_start=full_segment_start,
            spans=spans,
        )
        span_last_parts.append(span_last)
        span_mean_parts.append(span_mean)
        n_spans = len(spans)
        rollout_id = int(row.get("rollout_id", -1))
        rollout_ids.extend([rollout_id] * n_spans)
        labels.extend([bool(row.get("is_correct"))] * n_spans)
        starts.extend(start for start, _ in spans)
        ends.extend(end for _, end in spans)
        relative_progress.extend(((start + end) / 2.0) / segment_length for start, end in spans)
        think_lengths.extend([segment_length] * n_spans)
        source_lines.extend([int(row.get("_source_line", -1))] * n_spans)

        del outputs, full_inputs, prompt_inputs, span_last, span_mean
        gc.collect()
        if __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()

    unique_rollouts = sorted(set(rollout_ids))
    label_by_rollout = {
        rollout_id: bool(labels[rollout_ids.index(rollout_id)]) for rollout_id in unique_rollouts
    }
    n_correct = int(sum(label_by_rollout.values()))
    n_wrong = int(len(label_by_rollout) - n_correct)
    if n_correct < 3 or n_wrong < 3:
        raise ValueError(
            f"question lost primary eligibility after token checks: {question_id} "
            f"correct={n_correct} wrong={n_wrong}"
        )
    return {
        "question_id": np.asarray(question_id),
        "layers": np.asarray(layers, dtype=np.int16),
        "span_last": np.concatenate(span_last_parts, axis=0),
        "span_mean": np.concatenate(span_mean_parts, axis=0),
        "rollout_id": np.asarray(rollout_ids, dtype=np.int16),
        "is_correct": np.asarray(labels, dtype=bool),
        "span_start": np.asarray(starts, dtype=np.int32),
        "span_end": np.asarray(ends, dtype=np.int32),
        "relative_progress": np.asarray(relative_progress, dtype=np.float32),
        "think_length": np.asarray(think_lengths, dtype=np.int32),
        "source_line": np.asarray(source_lines, dtype=np.int32),
        "window": np.asarray(window, dtype=np.int16),
        "stride": np.asarray(stride, dtype=np.int16),
    }


def main() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    if args.window <= 0 or args.stride <= 0:
        raise ValueError("window and stride must be positive")
    layers = parse_layers(args.layers)
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "hidden"
    shard_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "extraction_status.jsonl"

    rows = load_jsonl(Path(args.input))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("question_id", "")), []).append(row)

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    tokenizer = getattr(processor, "tokenizer", processor)
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

    counts = {"completed": 0, "resumed": 0, "failed": 0}
    for question_id in tqdm(sorted(grouped), desc="long success hidden questions"):
        target = shard_dir / shard_name(question_id)
        if args.resume and valid_question_shard(target):
            counts["resumed"] += 1
            continue
        try:
            payload = extract_question(
                question_id,
                grouped[question_id],
                processor=processor,
                model=model,
                device=device,
                close_tag_ids=close_tag_ids,
                layers=layers,
                window=args.window,
                stride=args.stride,
            )
            temporary = target.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **payload)
            os.replace(temporary, target)
            record = {
                "question_id": question_id,
                "status": "completed",
                "shard": target.name,
                "spans": int(payload["span_last"].shape[0]),
                "rollouts": int(np.unique(payload["rollout_id"]).size),
            }
            counts["completed"] += 1
        except Exception as exc:
            record = {"question_id": question_id, "status": "failed", "error": repr(exc)}
            counts["failed"] += 1
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        append_status(status_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        **counts,
        "input": str(args.input),
        "output_dir": str(output_dir),
        "model": args.model,
        "layers": layers,
        "window": args.window,
        "stride": args.stride,
        "new_generation": False,
    }
    (output_dir / "EXTRACTION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
