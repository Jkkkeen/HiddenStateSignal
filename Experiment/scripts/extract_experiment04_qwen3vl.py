#!/usr/bin/env python3
"""Replay frozen Qwen3-VL rollouts and cache Experiment 04 chunk summaries."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiment04_metrics import REPRESENTATIONS, aggregate_distribution
from run_long_path_smoke_qwen3vl import (
    build_full_messages,
    build_prompt_messages,
    common_prefix_len,
    encode_messages,
    segment_token_span_from_text,
    to_device,
)


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
ENTROPY_FAMILIES = ("raw", "time_diff", "layer_diff")
AGGREGATES = ("mean", "median", "p90", "top25mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Experiment 04 hidden states.")
    parser.add_argument("--input", required=True, help="Frozen rollout manifest JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--last-n", type=int, default=25)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                row.setdefault("_source_line", source_line)
                rows.append(row)
    if not rows:
        raise ValueError("input manifest contains no rows")
    return rows


def question_stem(question_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(question_id)).strip("._")[:40] or "q"
    digest = hashlib.sha1(str(question_id).encode("utf-8")).hexdigest()[:12]
    return f"question_{safe}_{digest}"


def _torch_coordinate_entropy(values: Any, eps: float = 1e-12) -> Any:
    import torch

    array = values.float()
    centered = array - array.mean(dim=-1, keepdim=True)
    energy = centered.square()
    total = energy.sum(dim=-1)
    weighted = (energy * torch.log(energy + eps)).sum(dim=-1)
    raw = torch.log(total + eps) - weighted / torch.clamp(total, min=eps)
    normalized = torch.clamp(raw / np.log(array.shape[-1]), min=0.0, max=1.0)
    return torch.where(total > eps, normalized, torch.zeros_like(normalized))


def _chunk_bounds(token_count: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    full_count = token_count // chunk_size
    full = np.asarray(
        [(idx * chunk_size, (idx + 1) * chunk_size) for idx in range(full_count)],
        dtype=np.int32,
    ).reshape(-1, 2)
    start = full_count * chunk_size
    partial = (
        np.asarray([(start, token_count)], dtype=np.int32)
        if start < token_count
        else np.empty((0, 2), dtype=np.int32)
    )
    return full, partial


def _entropy_table(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    table = np.full((bounds.shape[0], len(AGGREGATES)), np.nan, dtype=np.float32)
    for chunk_id, (start, end) in enumerate(bounds):
        summary = aggregate_distribution(values[int(start) : int(end)])
        table[chunk_id] = [summary[name] for name in AGGREGATES]
    return table


def reduce_hidden_states(
    hidden_states: tuple[Any, ...],
    *,
    segment_start: int,
    segment_end: int,
    chunk_size: int = 256,
    last_n: int = 25,
) -> dict[str, np.ndarray]:
    """Reduce one forward pass into bounded fp16 vectors and entropy scalars."""
    import torch

    if segment_end <= segment_start:
        raise ValueError("empty think segment")
    if chunk_size <= 0 or last_n <= 0:
        raise ValueError("chunk_size and last_n must be positive")
    if not hidden_states:
        raise ValueError("hidden_states is empty")
    token_count = segment_end - segment_start
    layer_count = len(hidden_states)
    hidden_dim = int(hidden_states[0].shape[-1])
    full_bounds, partial_bounds = _chunk_bounds(token_count, chunk_size)
    full_vectors = np.empty(
        (full_bounds.shape[0], len(REPRESENTATIONS), layer_count, hidden_dim),
        dtype=np.float16,
    )
    partial_vectors = np.empty(
        (partial_bounds.shape[0], len(REPRESENTATIONS), layer_count, hidden_dim),
        dtype=np.float16,
    )
    token_entropy_full = np.full(
        (full_bounds.shape[0], layer_count, len(ENTROPY_FAMILIES), len(AGGREGATES)),
        np.nan,
        dtype=np.float32,
    )
    token_entropy_partial = np.full(
        (partial_bounds.shape[0], layer_count, len(ENTROPY_FAMILIES), len(AGGREGATES)),
        np.nan,
        dtype=np.float32,
    )

    for layer, state in enumerate(hidden_states):
        current = state[0, segment_start:segment_end].float()
        if current.shape != (token_count, hidden_dim):
            raise ValueError(f"unexpected hidden shape at layer {layer}: {tuple(current.shape)}")

        for chunk_id, (start, end) in enumerate(full_bounds):
            block = current[int(start) : int(end)]
            full_vectors[chunk_id, 0, layer] = block[-1].to(torch.float16).cpu().numpy()
            full_vectors[chunk_id, 1, layer] = (
                block[-min(last_n, block.shape[0]) :].mean(dim=0).to(torch.float16).cpu().numpy()
            )
            full_vectors[chunk_id, 2, layer] = block.mean(dim=0).to(torch.float16).cpu().numpy()
        for chunk_id, (start, end) in enumerate(partial_bounds):
            block = current[int(start) : int(end)]
            partial_vectors[chunk_id, 0, layer] = block[-1].to(torch.float16).cpu().numpy()
            partial_vectors[chunk_id, 1, layer] = (
                block[-min(last_n, block.shape[0]) :].mean(dim=0).to(torch.float16).cpu().numpy()
            )
            partial_vectors[chunk_id, 2, layer] = block.mean(dim=0).to(torch.float16).cpu().numpy()

        raw_entropy = _torch_coordinate_entropy(current).detach().cpu().numpy()
        time_entropy = np.full(token_count, np.nan, dtype=np.float32)
        if token_count >= 2:
            time_entropy[1:] = (
                _torch_coordinate_entropy(current[1:] - current[:-1]).detach().cpu().numpy()
            )
        layer_entropy = np.full(token_count, np.nan, dtype=np.float32)
        if layer > 0:
            previous = hidden_states[layer - 1][0, segment_start:segment_end].float()
            layer_entropy[:] = (
                _torch_coordinate_entropy(current - previous).detach().cpu().numpy()
            )
            del previous
        for family, values in enumerate((raw_entropy, time_entropy, layer_entropy)):
            token_entropy_full[:, layer, family] = _entropy_table(values, full_bounds)
            token_entropy_partial[:, layer, family] = _entropy_table(values, partial_bounds)
        del current, raw_entropy, time_entropy, layer_entropy

    return {
        "full_vectors": full_vectors,
        "partial_vectors": partial_vectors,
        "full_bounds": full_bounds,
        "partial_bounds": partial_bounds,
        "token_entropy_full": token_entropy_full,
        "token_entropy_partial": token_entropy_partial,
    }


def save_rollout_cache(
    output_path: Path,
    reduced: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    payload: dict[str, Any] = dict(reduced)
    payload.update(
        {
            "complete": np.asarray(True),
            "representation_names": np.asarray(REPRESENTATIONS),
            "entropy_family_names": np.asarray(ENTROPY_FAMILIES),
            "aggregate_names": np.asarray(AGGREGATES),
        }
    )
    for key, value in metadata.items():
        payload[key] = np.asarray(value)
    np.savez(temporary, **payload)
    temporary.replace(output_path)


def cache_is_complete(path: Path, *, expected_layers: int = 37) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as cached:
            vectors = cached["full_vectors"]
            return bool(cached["complete"].item()) and vectors.ndim == 4 and (
                vectors.shape[1] == len(REPRESENTATIONS)
                and vectors.shape[2] == expected_layers
            )
    except Exception:
        return False


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rss_mib() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / 1024.0
    except ImportError:
        return float("nan")


def main() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    if args.chunk_size <= 0 or args.last_n <= 0:
        raise ValueError("chunk-size and last-n must be positive")
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = output_dir / "extraction_telemetry.jsonl"
    progress_path = output_dir / "progress.json"

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    rows = load_jsonl(Path(args.input))
    if args.limit >= 0:
        rows = rows[: args.limit]

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
    expected_layers = int(getattr(model.config, "num_hidden_layers", 36)) + 1

    completed = 0
    failed = 0
    for row in tqdm(rows, desc="experiment04 extract"):
        question_id = str(row["question_id"])
        rollout_id = int(row["rollout_id"])
        cache_path = cache_dir / question_stem(question_id) / f"rollout_{rollout_id}.npz"
        if args.resume and cache_is_complete(cache_path, expected_layers=expected_layers):
            completed += 1
            continue
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            prompt_inputs = encode_messages(
                processor, build_prompt_messages(row), add_generation_prompt=True
            )
            full_inputs = encode_messages(
                processor, build_full_messages(row), add_generation_prompt=False
            )
            response_start = common_prefix_len(
                prompt_inputs["input_ids"], full_inputs["input_ids"]
            )
            response_end = int(full_inputs["input_ids"].shape[-1])
            response_ids = (
                full_inputs["input_ids"][0, response_start:response_end].detach().cpu().tolist()
            )
            local_start, local_end, segment_status = segment_token_span_from_text(
                response_ids, close_tag_ids
            )
            if segment_status != "implicit_think_close_only":
                raise ValueError(f"incomplete think segment: {segment_status}")
            full_inputs = to_device(full_inputs, device)
            with torch.inference_mode():
                outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)
            reduced = reduce_hidden_states(
                outputs.hidden_states,
                segment_start=response_start + local_start,
                segment_end=response_start + local_end,
                chunk_size=args.chunk_size,
                last_n=args.last_n,
            )
            metadata = {
                "question_id": question_id,
                "rollout_id": rollout_id,
                "is_correct": bool(row["is_correct"]),
                "answer": str(row.get("answer", "")),
                "pred_answer": str(row.get("pred_answer", "")),
                "think_length": int(local_end - local_start),
                "response_length": int(response_end - response_start),
                "segment_status": segment_status,
                "source_line": int(row.get("_source_line", -1)),
                "model": args.model,
                "chunk_size": args.chunk_size,
            }
            save_rollout_cache(cache_path, reduced, metadata=metadata)
            elapsed = time.perf_counter() - start_time
            telemetry = {
                **metadata,
                "status": "complete",
                "elapsed_seconds": elapsed,
                "full_chunks": int(reduced["full_vectors"].shape[0]),
                "partial_tokens": int(
                    reduced["partial_bounds"][0, 1] - reduced["partial_bounds"][0, 0]
                )
                if reduced["partial_bounds"].shape[0]
                else 0,
                "cache_bytes": cache_path.stat().st_size,
                "peak_gpu_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20),
                "peak_gpu_reserved_mib": float(torch.cuda.max_memory_reserved() / 2**20),
                "peak_cpu_rss_mib": _rss_mib(),
            }
            _append_jsonl(telemetry_path, telemetry)
            completed += 1
            del outputs, reduced, full_inputs, prompt_inputs
        except Exception as exc:
            failed += 1
            _append_jsonl(
                telemetry_path,
                {
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "status": "failed",
                    "error": repr(exc),
                    "elapsed_seconds": time.perf_counter() - start_time,
                },
            )
            print(f"[failed] {question_id=} {rollout_id=} {exc!r}", flush=True)
        finally:
            _atomic_json(
                progress_path,
                {
                    "input": str(Path(args.input).resolve()),
                    "total": len(rows),
                    "completed": completed,
                    "failed": failed,
                    "updated_unix": time.time(),
                },
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failed and not args.allow_errors:
        raise SystemExit(f"extraction completed with {failed} failed rollouts")


if __name__ == "__main__":
    main()
