#!/usr/bin/env python3
"""Replay frozen long responses and extract Experiment 02 metrics."""

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
import pandas as pd

from experiment02_metrics import path_integrals
from long_success_trajectory_common import full_span_bounds
from run_long_path_smoke_qwen3vl import (
    build_full_messages,
    build_prompt_messages,
    common_prefix_len,
    encode_messages,
    segment_token_span_from_text,
    to_device,
)


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
EPS = 1e-12
PROGRESS_REQUIRED = {
    "question_id",
    "rollout_id",
    "progress_bin",
    "movement_norm_median",
    "activity_norm_p90",
}
ENTROPY_REQUIRED = {
    "question_id",
    "rollout_id",
    "progress_bin",
    "layer",
    "activation_entropy_raw_mean",
    "activation_entropy_z_mean",
}
PATH_REQUIRED = {
    "question_id",
    "rollout_id",
    "representation",
    "layer",
    "scope",
    "path_length",
    "net_displacement",
    "straightness",
    "log_detour",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Experiment 02 metrics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--progress-bins", type=int, default=10)
    parser.add_argument("--activity-layer", type=int, default=15)
    parser.add_argument("--movement-layer", type=int, default=24)
    parser.add_argument("--path-layers", default="24,36")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--primary-stride", type=int, default=64)
    parser.add_argument("--sensitivity-stride", type=int, default=128)
    parser.add_argument("--entropy-chunk-size", type=int, default=512)
    parser.add_argument("--sigma-scale", type=float, default=1e-4)
    parser.add_argument("--z-clip", type=float, default=8.0)
    parser.add_argument("--min-correct", type=int, default=2)
    parser.add_argument("--min-wrong", type=int, default=2)
    parser.add_argument("--save-path-vectors", action="store_true")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_source_line", source_line)
            rows.append(row)
    return rows


def question_stem(question_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(question_id)).strip("._")[:40]
    safe = safe or "question"
    digest = hashlib.sha1(str(question_id).encode("utf-8")).hexdigest()[:12]
    return f"question_{safe}_{digest}"


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not layers or min(layers) < 0 or len(set(layers)) != len(layers):
        raise ValueError("path layers must be unique non-negative integers")
    return layers


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)


def _atomic_npz(target: Path, arrays: dict[str, np.ndarray]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, target)


def _frame_valid(path: Path, required: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    return not frame.empty and required.issubset(frame.columns)


def completed_question(
    output_dir: Path,
    stem: str,
    *,
    require_vectors: bool,
) -> bool:
    marker = output_dir / "completed" / f"{stem}.complete.json"
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    if record.get("status") != "completed":
        return False
    valid = (
        _frame_valid(
            output_dir / "progress_features" / f"{stem}.parquet",
            PROGRESS_REQUIRED,
        )
        and _frame_valid(
            output_dir / "entropy_features" / f"{stem}.parquet",
            ENTROPY_REQUIRED,
        )
        and _frame_valid(
            output_dir / "path_features" / f"{stem}.parquet",
            PATH_REQUIRED,
        )
    )
    if require_vectors:
        valid = valid and (output_dir / "path_vectors" / f"{stem}.npz").is_file()
    return valid


def _summaries(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_count": 0,
        }
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p90": float(np.quantile(finite, 0.9)),
        f"{prefix}_count": int(finite.size),
    }


def _entropy_chunk(values: Any, *, center: bool, eps: float = EPS) -> tuple[Any, Any]:
    import torch

    if center:
        values = values - values.mean(dim=-1, keepdim=True)
    energy = values.square()
    total = energy.sum(dim=-1, keepdim=True)
    probabilities = torch.where(
        total > eps,
        energy / total.clamp_min(eps),
        torch.zeros_like(energy),
    )
    raw = -(probabilities * torch.log(probabilities + eps)).sum(dim=-1)
    return torch.clamp(raw / np.log(values.shape[-1]), 0.0, 1.0), torch.exp(raw)


def _entropy_arrays(
    hidden: Any,
    *,
    chunk_size: int,
    center: bool,
    mean: Any | None = None,
    sigma: Any | None = None,
    clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    entropy_parts = []
    effective_parts = []
    for start in range(0, hidden.shape[0], chunk_size):
        values = hidden[start : start + chunk_size].float()
        if mean is not None and sigma is not None:
            values = (values - mean) / sigma
            if clip is not None:
                values = values.clamp(-clip, clip)
        entropy, effective = _entropy_chunk(values, center=center)
        entropy_parts.append(entropy.cpu().numpy())
        effective_parts.append(effective.cpu().numpy())
        del values, entropy, effective
    return (
        np.concatenate(entropy_parts).astype(np.float32),
        np.concatenate(effective_parts).astype(np.float32),
    )


def _hidden_norm_array(hidden: Any, chunk_size: int) -> np.ndarray:
    import torch

    parts = []
    for start in range(0, hidden.shape[0], chunk_size):
        values = hidden[start : start + chunk_size].float()
        parts.append(torch.linalg.vector_norm(values, dim=-1).cpu().numpy())
        del values
    return np.concatenate(parts).astype(np.float32)


def _vertical_metrics(
    current: Any,
    previous: Any,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    norm_parts = []
    entropy_parts = []
    effective_parts = []
    for start in range(0, current.shape[0], chunk_size):
        update = (
            current[start : start + chunk_size].float()
            - previous[start : start + chunk_size].float()
        )
        norm_parts.append(torch.linalg.vector_norm(update, dim=-1).cpu().numpy())
        entropy, effective = _entropy_chunk(update, center=True)
        entropy_parts.append(entropy.cpu().numpy())
        effective_parts.append(effective.cpu().numpy())
        del update, entropy, effective
    return (
        np.concatenate(norm_parts).astype(np.float32),
        np.concatenate(entropy_parts).astype(np.float32),
        np.concatenate(effective_parts).astype(np.float32),
    )


def _pool_spans(hidden: Any, spans: list[tuple[int, int]]) -> np.ndarray:
    import torch

    starts = torch.tensor([start for start, _ in spans], device=hidden.device)
    ends = torch.tensor([end for _, end in spans], device=hidden.device)
    values = hidden.float()
    prefix = torch.cat(
        [
            torch.zeros(
                (1, values.shape[-1]),
                dtype=values.dtype,
                device=values.device,
            ),
            values.cumsum(dim=0),
        ],
        dim=0,
    )
    pooled = (prefix.index_select(0, ends) - prefix.index_select(0, starts)) / (
        ends - starts
    ).to(values.dtype).unsqueeze(1)
    result = pooled.cpu().numpy().astype(np.float32)
    del values, prefix, pooled
    return result


def _bin_ids(length: int, n_bins: int) -> np.ndarray:
    return np.minimum(
        ((np.arange(length) + 1) * n_bins) // length,
        n_bins - 1,
    ).astype(np.int16)


def reduce_hidden_states(
    hidden_states: tuple[Any, ...],
    *,
    segment_start: int,
    segment_end: int,
    question_id: str,
    rollout_id: int,
    is_correct: bool,
    progress_bins: int,
    activity_layer: int,
    movement_layer: int,
    path_layers: tuple[int, ...],
    window: int,
    primary_stride: int,
    sensitivity_stride: int,
    entropy_chunk_size: int,
    sigma_scale: float,
    z_clip: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    import torch

    segment_length = int(segment_end - segment_start)
    if segment_length < window + primary_stride:
        raise ValueError(
            f"thinking segment too short for path metrics: {segment_length}"
        )
    if max((activity_layer, movement_layer, *path_layers)) >= len(hidden_states):
        raise ValueError(
            f"requested layer exceeds available hidden states: {len(hidden_states)}"
        )
    token_bins = _bin_ids(segment_length, progress_bins)
    entropy_rows: list[dict[str, Any]] = []
    activity_values: np.ndarray | None = None
    pooled_by_key: dict[tuple[str, int], np.ndarray] = {}
    vector_arrays: dict[str, np.ndarray] = {}
    strides = (primary_stride, sensitivity_stride)
    spans_by_stride = {
        stride: full_span_bounds(segment_length, window=window, stride=stride)
        for stride in strides
    }
    previous = None

    for layer, state in enumerate(hidden_states):
        layer_state = state[0] if state.ndim == 3 else state
        hidden = layer_state[segment_start:segment_end]
        raw_entropy, raw_effective = _entropy_arrays(
            hidden,
            chunk_size=entropy_chunk_size,
            center=True,
        )
        hidden_norm = _hidden_norm_array(hidden, entropy_chunk_size)

        float_hidden = hidden.float()
        variance, mean = torch.var_mean(float_hidden, dim=0, correction=1)
        sigma = torch.sqrt(torch.clamp(variance, min=0.0))
        median_sigma = float(torch.median(sigma).item())
        sigma_floor = max(sigma_scale * median_sigma, EPS)
        effective_sigma = sigma.clamp_min(sigma_floor)
        low_variance_fraction = float(
            (sigma <= sigma_floor).float().mean().item()
        )
        z_entropy, z_effective = _entropy_arrays(
            hidden,
            chunk_size=entropy_chunk_size,
            center=False,
            mean=mean,
            sigma=effective_sigma,
            clip=z_clip,
        )
        del float_hidden, variance, mean, sigma, effective_sigma

        vertical_norm = np.full(segment_length, np.nan, dtype=np.float32)
        update_entropy = np.full(segment_length, np.nan, dtype=np.float32)
        update_effective = np.full(segment_length, np.nan, dtype=np.float32)
        if previous is not None:
            vertical_norm, update_entropy, update_effective = _vertical_metrics(
                hidden,
                previous,
                chunk_size=entropy_chunk_size,
            )
            if layer == activity_layer:
                activity_values = vertical_norm.copy()

        for progress_bin in range(progress_bins):
            mask = token_bins == progress_bin
            base = {
                "question_id": str(question_id),
                "rollout_id": int(rollout_id),
                "is_correct": bool(is_correct),
                "think_length": int(segment_length),
                "layer": int(layer),
                "progress_bin": int(progress_bin),
                "token_count": int(mask.sum()),
                "median_sigma": median_sigma,
                "sigma_floor": sigma_floor,
                "low_variance_coord_fraction": low_variance_fraction,
            }
            base.update(_summaries(hidden_norm[mask], "hidden_norm"))
            base.update(
                _summaries(raw_entropy[mask], "activation_entropy_raw")
            )
            base.update(
                _summaries(raw_effective[mask], "activation_effective_raw")
            )
            base.update(_summaries(z_entropy[mask], "activation_entropy_z"))
            base.update(
                _summaries(z_effective[mask], "activation_effective_z")
            )
            base.update(_summaries(update_entropy[mask], "update_entropy"))
            base.update(_summaries(update_effective[mask], "update_effective"))
            entropy_rows.append(base)

        if layer in set(path_layers).union({movement_layer}):
            for stride, spans in spans_by_stride.items():
                if len(spans) < 2:
                    continue
                representation = f"mean_w{window}_s{stride}"
                pooled_by_key[(representation, layer)] = _pool_spans(hidden, spans)
        previous = hidden
        del raw_entropy, raw_effective, hidden_norm, z_entropy, z_effective

    if activity_values is None:
        raise RuntimeError(f"activity layer {activity_layer} was not reduced")

    movement_key = (f"mean_w{window}_s{primary_stride}", movement_layer)
    if movement_key not in pooled_by_key:
        raise RuntimeError("primary movement representation missing")
    primary_spans = spans_by_stride[primary_stride]
    primary_progress = np.asarray(
        [
            (start + end) / (2.0 * segment_length)
            for start, end in primary_spans
        ],
        dtype=np.float32,
    )
    movement_displacements = np.diff(pooled_by_key[movement_key], axis=0)
    movement_norm = np.linalg.norm(movement_displacements, axis=1)
    movement_bins = np.minimum(
        (primary_progress[1:] * progress_bins).astype(np.int16),
        progress_bins - 1,
    )

    progress_rows = []
    for progress_bin in range(progress_bins):
        token_mask = token_bins == progress_bin
        movement_mask = movement_bins == progress_bin
        row = {
            "question_id": str(question_id),
            "rollout_id": int(rollout_id),
            "is_correct": bool(is_correct),
            "think_length": int(segment_length),
            "progress_bin": int(progress_bin),
            "activity_layer": int(activity_layer),
            "movement_layer": int(movement_layer),
            "representation": movement_key[0],
        }
        row.update(_summaries(activity_values[token_mask], "activity_norm"))
        row.update(_summaries(movement_norm[movement_mask], "movement_norm"))
        progress_rows.append(row)

    path_rows = []
    for (representation, layer), pooled in sorted(pooled_by_key.items()):
        if layer not in path_layers:
            continue
        stride = int(representation.rsplit("s", 1)[1])
        spans = spans_by_stride[stride]
        progress = np.asarray(
            [(start + end) / (2.0 * segment_length) for start, end in spans],
            dtype=np.float32,
        )
        displacements = np.diff(pooled, axis=0).astype(np.float32)
        displacement_bins = np.minimum(
            (progress[1:] * progress_bins).astype(np.int16),
            progress_bins - 1,
        )
        for metrics in path_integrals(
            displacements,
            displacement_bins,
            n_bins=progress_bins,
            min_block_steps=3,
            eps=EPS,
        ):
            path_rows.append(
                {
                    "question_id": str(question_id),
                    "rollout_id": int(rollout_id),
                    "is_correct": bool(is_correct),
                    "think_length": int(segment_length),
                    "representation": representation,
                    "layer": int(layer),
                    **metrics,
                }
            )
        vector_key = f"{representation}_L{layer}"
        vector_arrays[f"displacement_{vector_key}"] = displacements.astype(
            np.float16
        )
        vector_arrays[f"progress_bin_{vector_key}"] = displacement_bins
        vector_arrays[f"relative_progress_{vector_key}"] = progress[1:]

    return (
        pd.DataFrame(progress_rows),
        pd.DataFrame(entropy_rows),
        pd.DataFrame(path_rows),
        vector_arrays,
    )


def extract_question(
    question_id: str,
    rows: list[dict[str, Any]],
    *,
    processor: Any,
    model: Any,
    device: Any,
    close_tag_ids: list[int],
    args: argparse.Namespace,
    path_layers: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    import torch

    progress_frames = []
    entropy_frames = []
    path_frames = []
    vector_payload: dict[str, np.ndarray] = {}
    labels: list[bool] = []
    for row in sorted(rows, key=lambda item: int(item.get("rollout_id", -1))):
        started = time.monotonic()
        prompt_inputs = encode_messages(
            processor,
            build_prompt_messages(row),
            add_generation_prompt=True,
        )
        full_inputs = encode_messages(
            processor,
            build_full_messages(row),
            add_generation_prompt=False,
        )
        response_start = common_prefix_len(
            prompt_inputs["input_ids"],
            full_inputs["input_ids"],
        )
        response_ids = full_inputs["input_ids"][
            0, response_start : int(full_inputs["input_ids"].shape[-1])
        ].cpu().tolist()
        segment_start, segment_end, status = segment_token_span_from_text(
            response_ids,
            close_tag_ids,
        )
        if status == "no_think_close":
            raise ValueError(
                f"missing think close: question={question_id} rollout={row.get('rollout_id')}"
            )
        full_segment_start = response_start + segment_start
        full_segment_end = response_start + segment_end
        rollout_id = int(row.get("rollout_id", -1))
        is_correct = bool(row.get("is_correct"))
        labels.append(is_correct)
        full_inputs = to_device(full_inputs, device)
        with torch.inference_mode():
            outputs = model(
                **full_inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            progress, entropy, path, vectors = reduce_hidden_states(
                outputs.hidden_states,
                segment_start=full_segment_start,
                segment_end=full_segment_end,
                question_id=question_id,
                rollout_id=rollout_id,
                is_correct=is_correct,
                progress_bins=args.progress_bins,
                activity_layer=args.activity_layer,
                movement_layer=args.movement_layer,
                path_layers=path_layers,
                window=args.window,
                primary_stride=args.primary_stride,
                sensitivity_stride=args.sensitivity_stride,
                entropy_chunk_size=args.entropy_chunk_size,
                sigma_scale=args.sigma_scale,
                z_clip=args.z_clip,
            )
        progress_frames.append(progress)
        entropy_frames.append(entropy)
        path_frames.append(path)
        for key, value in vectors.items():
            vector_payload[f"r{rollout_id}_{key}"] = value
        print(
            json.dumps(
                {
                    "event": "rollout_reduced",
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "is_correct": is_correct,
                    "think_length": int(full_segment_end - full_segment_start),
                    "layers": len(outputs.hidden_states),
                    "seconds": round(time.monotonic() - started, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del outputs, full_inputs, prompt_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_correct = int(sum(labels))
    n_wrong = int(len(labels) - n_correct)
    if n_correct < args.min_correct or n_wrong < args.min_wrong:
        raise ValueError(
            f"question lost eligibility: {question_id} correct={n_correct} wrong={n_wrong}"
        )
    return (
        pd.concat(progress_frames, ignore_index=True),
        pd.concat(entropy_frames, ignore_index=True),
        pd.concat(path_frames, ignore_index=True),
        vector_payload,
    )


def append_status(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    if args.progress_bins <= 0 or args.entropy_chunk_size <= 0:
        raise ValueError("progress bins and entropy chunk size must be positive")
    path_layers = parse_layers(args.path_layers)
    output_dir = Path(args.output_dir)
    for directory in (
        "progress_features",
        "entropy_features",
        "path_features",
        "path_vectors",
        "completed",
    ):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(Path(args.input))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("question_id", "")), []).append(row)
    question_ids = sorted(grouped)

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
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
    status_path = output_dir / "extraction_status.jsonl"
    for question_id in tqdm(question_ids, desc="experiment02 questions"):
        stem = question_stem(question_id)
        if args.resume and completed_question(
            output_dir,
            stem,
            require_vectors=args.save_path_vectors,
        ):
            counts["resumed"] += 1
            continue
        try:
            progress, entropy, path, vectors = extract_question(
                question_id,
                grouped[question_id],
                processor=processor,
                model=model,
                device=device,
                close_tag_ids=close_tag_ids,
                args=args,
                path_layers=path_layers,
            )
            _atomic_parquet(
                progress,
                output_dir / "progress_features" / f"{stem}.parquet",
            )
            _atomic_parquet(
                entropy,
                output_dir / "entropy_features" / f"{stem}.parquet",
            )
            _atomic_parquet(
                path,
                output_dir / "path_features" / f"{stem}.parquet",
            )
            if args.save_path_vectors:
                _atomic_npz(
                    output_dir / "path_vectors" / f"{stem}.npz",
                    vectors,
                )
            marker = output_dir / "completed" / f"{stem}.complete.json"
            temporary = marker.with_suffix(".tmp.json")
            record = {
                "question_id": question_id,
                "status": "completed",
                "rollouts": int(progress["rollout_id"].nunique()),
                "progress_rows": int(len(progress)),
                "entropy_rows": int(len(entropy)),
                "path_rows": int(len(path)),
            }
            temporary.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, marker)
            counts["completed"] += 1
        except Exception as exc:
            record = {
                "question_id": question_id,
                "status": "failed",
                "error": repr(exc),
            }
            counts["failed"] += 1
        append_status(status_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        **counts,
        "input": str(args.input),
        "output_dir": str(output_dir),
        "model": args.model,
        "progress_bins": args.progress_bins,
        "activity_layer": args.activity_layer,
        "movement_layer": args.movement_layer,
        "path_layers": path_layers,
        "window": args.window,
        "primary_stride": args.primary_stride,
        "sensitivity_stride": args.sensitivity_stride,
        "sigma_scale": args.sigma_scale,
        "z_clip": args.z_clip,
        "save_path_vectors": bool(args.save_path_vectors),
        "new_generation": False,
    }
    (output_dir / "EXTRACTION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
