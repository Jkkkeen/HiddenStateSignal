#!/usr/bin/env python3
"""Extract per-chunk, per-layer hidden states for Qwen3-VL rollouts.

The output NPZ matches the chunk trajectory plan:
  h_chunk:         (n_chunks, n_layers, hidden_dim) float16
  rollout_idx:     (n_chunks,) int, index into the meta arrays in this NPZ
  chunk_start:     (n_chunks,) int, response-token-relative start
  chunk_end:       (n_chunks,) int, response-token-relative end
  meta_id:         (n_rollouts,) str question id
  meta_roll_index: (n_rollouts,) int rollout id
  meta_is_correct: (n_rollouts,) bool
  meta_skipped:    (n_rollouts,) bool
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Qwen3-VL chunk hidden states.")
    parser.add_argument(
        "--input",
        default="data/rollouts_mathverse_full_cot_labeled_mcq_mixed.jsonl",
        help="Input rollout JSONL.",
    )
    parser.add_argument(
        "--output",
        default="chunklayer/h_smoke16.npz",
        help="Output NPZ path.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=-1, help="Use -1 for no limit.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1, help="Exclusive end; -1 for no end.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--max-records-per-save",
        type=int,
        default=0,
        help="Reserved for future use. Current script saves once at the end.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")

    end = len(rows) if args.end < 0 else min(args.end, len(rows))
    selected = rows[max(args.start, 0) : end]
    if args.num_shards > 1:
        selected = [
            row for pos, row in enumerate(selected) if pos % args.num_shards == args.shard_index
        ]
    if args.limit is not None and args.limit >= 0:
        selected = selected[: args.limit]
    return selected


def image_part(image_path: str | None) -> dict[str, Any] | None:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(path).convert("RGB")
    return {"type": "image", "image": image}


def build_prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = (
        "Solve the following MathVerse problem. Give concise reasoning and the final answer.\n\n"
        f"{row.get('prompt', '')}"
    )
    content: list[dict[str, Any]] = []
    part = image_part(row.get("image_path"))
    if part is not None:
        content.append(part)
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def build_full_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = build_prompt_messages(row)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(row.get("response", ""))}],
        }
    )
    return messages


def to_device(batch: Any, device: torch.device | str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    a_ids = a[0].detach().cpu().tolist()
    b_ids = b[0].detach().cpu().tolist()
    limit = min(len(a_ids), len(b_ids))
    idx = 0
    while idx < limit and a_ids[idx] == b_ids[idx]:
        idx += 1
    return idx


def encode_messages(processor: Any, messages: list[dict[str, Any]], add_generation_prompt: bool) -> Any:
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def chunk_means_from_hidden(
    hidden_states: tuple[torch.Tensor, ...],
    response_start: int,
    response_end: int,
    chunk_size: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    response_len = response_end - response_start
    if response_len <= 0:
        raise ValueError(f"Empty response span: start={response_start}, end={response_end}")

    chunks: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    for chunk_start in range(0, response_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, response_len)
        full_start = response_start + chunk_start
        full_end = response_start + chunk_end

        per_layer = []
        for layer_hidden in hidden_states:
            mean_hidden = layer_hidden[0, full_start:full_end].float().mean(dim=0)
            per_layer.append(mean_hidden.to(torch.float16).cpu().numpy())

        chunks.append(np.stack(per_layer, axis=0))
        spans.append((chunk_start, chunk_end))

    return np.stack(chunks, axis=0), spans


def save_npz(
    output_path: Path,
    h_chunks: list[np.ndarray],
    rollout_idx: list[int],
    chunk_start: list[int],
    chunk_end: list[int],
    meta_id: list[str],
    meta_roll_index: list[int],
    meta_is_correct: list[bool],
    meta_skipped: list[bool],
    meta_source_line: list[int],
    meta_response_len: list[int],
    meta_answer: list[str],
    meta_pred_answer: list[str],
    model_id: str,
    chunk_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if h_chunks:
        h_chunk = np.concatenate(h_chunks, axis=0).astype(np.float16, copy=False)
    else:
        h_chunk = np.empty((0, 0, 0), dtype=np.float16)

    np.savez_compressed(
        output_path,
        h_chunk=h_chunk,
        rollout_idx=np.asarray(rollout_idx, dtype=np.int64),
        chunk_start=np.asarray(chunk_start, dtype=np.int64),
        chunk_end=np.asarray(chunk_end, dtype=np.int64),
        meta_id=np.asarray(meta_id, dtype=str),
        meta_roll_index=np.asarray(meta_roll_index, dtype=np.int64),
        meta_is_correct=np.asarray(meta_is_correct, dtype=bool),
        meta_skipped=np.asarray(meta_skipped, dtype=bool),
        meta_source_line=np.asarray(meta_source_line, dtype=np.int64),
        meta_response_len=np.asarray(meta_response_len, dtype=np.int64),
        meta_answer=np.asarray(meta_answer, dtype=str),
        meta_pred_answer=np.asarray(meta_pred_answer, dtype=str),
        model=np.asarray(model_id),
        chunk_size=np.asarray(chunk_size, dtype=np.int64),
    )


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = select_rows(load_jsonl(input_path), args)
    if not rows:
        raise ValueError("No rows selected.")

    print(f"input: {input_path}")
    print(f"selected rollouts: {len(rows)}")
    print(f"output: {output_path}")
    print(f"model: {args.model}")
    print(f"chunk_size: {args.chunk_size}")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    device = next(model.parameters()).device

    h_chunks: list[np.ndarray] = []
    rollout_idx: list[int] = []
    chunk_start: list[int] = []
    chunk_end: list[int] = []
    meta_id: list[str] = []
    meta_roll_index: list[int] = []
    meta_is_correct: list[bool] = []
    meta_skipped: list[bool] = []
    meta_source_line: list[int] = []
    meta_response_len: list[int] = []
    meta_answer: list[str] = []
    meta_pred_answer: list[str] = []

    for meta_index, row in enumerate(tqdm(rows, desc="extract")):
        meta_id.append(str(row.get("question_id", "")))
        meta_roll_index.append(int(row.get("rollout_id", -1)))
        meta_is_correct.append(bool(row.get("is_correct", False)))
        meta_source_line.append(int(row.get("_source_line", -1)))
        meta_answer.append(str(row.get("answer", "")))
        meta_pred_answer.append(str(row.get("pred_answer", "")))

        skipped = False
        response_len = 0
        try:
            prompt_inputs = encode_messages(
                processor, build_prompt_messages(row), add_generation_prompt=True
            )
            full_inputs = encode_messages(
                processor, build_full_messages(row), add_generation_prompt=False
            )

            response_start = common_prefix_len(prompt_inputs["input_ids"], full_inputs["input_ids"])
            response_end = int(full_inputs["input_ids"].shape[-1])
            response_len = response_end - response_start
            if response_len <= 0:
                raise ValueError(
                    f"No response tokens found for question={row.get('question_id')} "
                    f"rollout={row.get('rollout_id')}"
                )

            full_inputs = to_device(full_inputs, device)
            with torch.inference_mode():
                outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)

            chunk_array, spans = chunk_means_from_hidden(
                outputs.hidden_states,
                response_start=response_start,
                response_end=response_end,
                chunk_size=args.chunk_size,
            )
            h_chunks.append(chunk_array)
            for start, end in spans:
                rollout_idx.append(meta_index)
                chunk_start.append(start)
                chunk_end.append(end)

            del outputs, prompt_inputs, full_inputs, chunk_array
        except Exception as exc:
            skipped = True
            print(
                f"[skip] meta_index={meta_index} question={row.get('question_id')} "
                f"rollout={row.get('rollout_id')} reason={exc}",
                flush=True,
            )
        finally:
            meta_skipped.append(skipped)
            meta_response_len.append(response_len)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_npz(
        output_path=output_path,
        h_chunks=h_chunks,
        rollout_idx=rollout_idx,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        meta_id=meta_id,
        meta_roll_index=meta_roll_index,
        meta_is_correct=meta_is_correct,
        meta_skipped=meta_skipped,
        meta_source_line=meta_source_line,
        meta_response_len=meta_response_len,
        meta_answer=meta_answer,
        meta_pred_answer=meta_pred_answer,
        model_id=args.model,
        chunk_size=args.chunk_size,
    )

    print(f"saved: {output_path}")
    print(f"rollouts: {len(meta_id)}")
    print(f"skipped: {sum(meta_skipped)}")
    print(f"chunks: {len(rollout_idx)}")
    if h_chunks:
        print(f"h_chunk shape: {np.concatenate(h_chunks, axis=0).shape}")


if __name__ == "__main__":
    main()
