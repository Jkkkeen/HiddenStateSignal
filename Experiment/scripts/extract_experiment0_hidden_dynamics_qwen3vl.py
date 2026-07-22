#!/usr/bin/env python3
"""Replay fixed long responses and extract bounded-memory Experiment 0 metrics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment0_hidden_dynamics import (
    build_span_direction_records,
    score_cross_rollout_queries,
)
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
TOKEN_REQUIRED_COLUMNS = {"question_id", "rollout_id", "progress_bin", "layer"}
SPAN_REQUIRED_COLUMNS = {
    "question_id",
    "rollout_id",
    "representation",
    "progress_bin",
    "layer",
}
PROTOTYPE_REQUIRED_COLUMNS = SPAN_REQUIRED_COLUMNS | {"kappa_pos", "kappa_neg"}


@dataclass
class ReducedRollout:
    token_features: pd.DataFrame
    span_horizontal: pd.DataFrame
    span_vertical: pd.DataFrame
    audit_vectors: dict[str, np.ndarray]
    audit_metadata: dict[str, dict[str, np.ndarray]]


@dataclass
class AuditRollout:
    rollout_id: int
    vectors: dict[str, np.ndarray]
    metadata: dict[str, dict[str, np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Experiment 0 hidden dynamics.")
    parser.add_argument("--input", required=True, help="Frozen discovery manifest JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--progress-bins", type=int, default=10)
    parser.add_argument("--span-specs", default="64:32,128:64,256:128")
    parser.add_argument("--endpoint-spec", default="128:64")
    parser.add_argument("--primary-spec", default="128:64")
    parser.add_argument("--kappa-min", type=float, default=0.20)
    parser.add_argument("--audit-question-count", type=int, default=1)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_span_spec(value: str) -> tuple[int, int]:
    pieces = value.split(":")
    if len(pieces) != 2:
        raise ValueError(f"invalid span spec: {value}")
    window, stride = (int(piece) for piece in pieces)
    if window <= 0 or stride <= 0:
        raise ValueError("span window and stride must be positive")
    return window, stride


def parse_span_specs(value: str) -> tuple[tuple[int, int], ...]:
    specs = tuple(parse_span_spec(piece.strip()) for piece in value.split(",") if piece.strip())
    if not specs:
        raise ValueError("at least one span spec is required")
    if len(set(specs)) != len(specs):
        raise ValueError("span specs must be unique")
    return specs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle):
            if line.strip():
                row = json.loads(line)
                row.setdefault("_source_line", source_line)
                rows.append(row)
    return rows


def question_stem(question_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(question_id)).strip("._")[:40] or "question"
    digest = hashlib.sha1(str(question_id).encode("utf-8")).hexdigest()[:12]
    return f"question_{safe}_{digest}"


def _frame_has_columns(path: Path, required: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    return not frame.empty and required.issubset(frame.columns)


def valid_completed_question(output_dir: Path, stem: str) -> bool:
    marker = output_dir / "completed" / f"{stem}.complete.json"
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    if record.get("status") != "completed":
        return False
    return (
        _frame_has_columns(
            output_dir / "bin_features" / f"{stem}.parquet", TOKEN_REQUIRED_COLUMNS
        )
        and _frame_has_columns(
            output_dir / "span_features" / f"{stem}.parquet", SPAN_REQUIRED_COLUMNS
        )
        and _frame_has_columns(
            output_dir / "prototype_diagnostics" / f"{stem}.parquet",
            PROTOTYPE_REQUIRED_COLUMNS,
        )
    )


def _array_summary(values: np.ndarray, prefix: str) -> dict[str, float | int]:
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


def _to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    if values.shape[0] != mask.shape[0]:
        raise ValueError("metric values and progress mask length differ")
    return values[mask]


def _coordinate_entropy_torch(update: Any, eps: float = 1e-12) -> tuple[Any, Any]:
    import torch

    centered = update - update.mean(dim=-1, keepdim=True)
    energy = centered.square()
    total = energy.sum(dim=-1, keepdim=True)
    probabilities = torch.where(total > eps, energy / total.clamp_min(eps), torch.zeros_like(energy))
    raw_entropy = -(probabilities * torch.log(probabilities + eps)).sum(dim=-1)
    return raw_entropy / np.log(update.shape[-1]), torch.exp(raw_entropy)


def _pool_layer_torch(
    hidden: Any,
    spans: list[tuple[int, int]],
    mode: str,
) -> np.ndarray:
    import torch

    if mode == "last":
        indices = torch.tensor(
            [end - 1 for _, end in spans], dtype=torch.long, device=hidden.device
        )
        pooled = hidden.index_select(0, indices)
    elif mode == "mean":
        starts = torch.tensor([start for start, _ in spans], dtype=torch.long, device=hidden.device)
        ends = torch.tensor([end for _, end in spans], dtype=torch.long, device=hidden.device)
        prefix = torch.cat(
            [torch.zeros((1, hidden.shape[-1]), dtype=hidden.dtype, device=hidden.device), hidden.cumsum(dim=0)],
            dim=0,
        )
        pooled = (prefix.index_select(0, ends) - prefix.index_select(0, starts)) / (
            ends - starts
        ).to(hidden.dtype).unsqueeze(1)
        del prefix
    else:
        raise ValueError(f"unknown pooling mode: {mode}")
    return pooled.to(dtype=torch.float16).cpu().numpy()


def reduce_rollout_hidden(
    hidden_states: tuple[Any, ...],
    segment_start: int,
    segment_end: int,
    progress_bins: int,
    span_specs: tuple[tuple[int, int], ...],
    endpoint_spec: tuple[int, int],
    question_id: str,
    rollout_id: int,
    is_correct: bool,
    think_length: int,
    primary_spec: tuple[int, int] | None = None,
    retain_audit_vectors: bool = True,
) -> ReducedRollout:
    """Reduce one model forward without stacking all token hidden states."""
    import torch
    import torch.nn.functional as functional

    if progress_bins <= 0:
        raise ValueError("progress_bins must be positive")
    if segment_end <= segment_start:
        raise ValueError("empty hidden segment")
    segment_length = segment_end - segment_start
    if think_length != segment_length:
        raise ValueError("think_length must equal the extracted segment length")
    primary_spec = primary_spec or span_specs[0]
    if primary_spec not in span_specs:
        raise ValueError("primary_spec must be included in span_specs")
    if endpoint_spec not in span_specs:
        raise ValueError("endpoint_spec must be included in span_specs")

    spans_by_representation: dict[str, list[tuple[int, int]]] = {}
    for window, stride in span_specs:
        spans = full_span_bounds(segment_length, window=window, stride=stride)
        if len(spans) < 3:
            raise ValueError(
                f"too few spans for mean_w{window}_s{stride}: think_length={segment_length}"
            )
        spans_by_representation[f"mean_w{window}_s{stride}"] = spans
    endpoint_window, endpoint_stride = endpoint_spec
    spans_by_representation[f"last_w{endpoint_window}_s{endpoint_stride}"] = full_span_bounds(
        segment_length, window=endpoint_window, stride=endpoint_stride
    )
    primary_representation = f"mean_w{primary_spec[0]}_s{primary_spec[1]}"

    pooled_parts: dict[str, list[np.ndarray]] = {
        representation: [] for representation in spans_by_representation
    }
    token_rows: list[dict[str, Any]] = []
    token_bins = np.minimum(
        ((np.arange(segment_length) + 1) * progress_bins) // segment_length,
        progress_bins - 1,
    ).astype(np.int16)
    previous_hidden = None
    previous_vertical_norm = None

    for layer, state in enumerate(hidden_states):
        layer_state = state[0] if state.ndim == 3 else state
        hidden = layer_state[segment_start:segment_end].float()
        hidden_norm = _to_numpy(torch.linalg.vector_norm(hidden, dim=-1))
        horizontal_update = hidden[1:] - hidden[:-1]
        horizontal_norm = _to_numpy(torch.linalg.vector_norm(horizontal_update, dim=-1))
        horizontal_delta = np.diff(horizontal_norm)
        token_turn = _to_numpy(
            functional.cosine_similarity(horizontal_update[1:], horizontal_update[:-1], dim=-1)
        )

        vertical_norm = np.asarray([], dtype=np.float32)
        vertical_delta = np.asarray([], dtype=np.float32)
        coordinate_entropy = np.asarray([], dtype=np.float32)
        effective_dimensions = np.asarray([], dtype=np.float32)
        if previous_hidden is not None:
            vertical_update = hidden - previous_hidden
            vertical_norm_tensor = torch.linalg.vector_norm(vertical_update, dim=-1)
            vertical_norm = _to_numpy(vertical_norm_tensor)
            entropy_tensor, effective_tensor = _coordinate_entropy_torch(vertical_update)
            coordinate_entropy = _to_numpy(entropy_tensor)
            effective_dimensions = _to_numpy(effective_tensor)
            if previous_vertical_norm is not None:
                vertical_delta = vertical_norm - previous_vertical_norm
            previous_vertical_norm = vertical_norm
            del vertical_update, vertical_norm_tensor, entropy_tensor, effective_tensor

        for progress_bin in range(progress_bins):
            token_mask = token_bins == progress_bin
            horizontal_mask = token_bins[1:] == progress_bin
            delta_mask = token_bins[2:] == progress_bin
            row: dict[str, Any] = {
                "question_id": str(question_id),
                "rollout_id": int(rollout_id),
                "is_correct": bool(is_correct),
                "representation": "token",
                "progress_bin": int(progress_bin),
                "layer": int(layer),
                "think_length": int(think_length),
                "token_count": int(token_mask.sum()),
                "horizontal_token_count": int(horizontal_mask.sum()),
                "hidden_norm_mean": float(hidden_norm[token_mask].mean())
                if token_mask.any()
                else float("nan"),
            }
            row.update(_array_summary(horizontal_norm[horizontal_mask], "horizontal_norm"))
            row.update(_array_summary(horizontal_delta[delta_mask], "horizontal_norm_delta"))
            row.update(_array_summary(token_turn[delta_mask], "token_turn_cos"))
            row.update(_array_summary(_masked(vertical_norm, token_mask), "vertical_norm"))
            row.update(
                _array_summary(_masked(vertical_delta, token_mask), "vertical_norm_delta")
            )
            row.update(
                _array_summary(
                    _masked(coordinate_entropy, token_mask), "coordinate_entropy"
                )
            )
            row.update(
                _array_summary(
                    _masked(effective_dimensions, token_mask), "effective_dimensions"
                )
            )
            token_rows.append(row)

        for representation, spans in spans_by_representation.items():
            mode = "last" if representation.startswith("last_") else "mean"
            pooled_parts[representation].append(_pool_layer_torch(hidden, spans, mode=mode))
        previous_hidden = hidden
        del horizontal_update

    horizontal_frames = []
    vertical_frames = []
    audit_vectors: dict[str, np.ndarray] = {}
    audit_metadata: dict[str, dict[str, np.ndarray]] = {}
    layer_values = np.arange(len(hidden_states), dtype=np.int16)
    for representation, layer_parts in pooled_parts.items():
        pooled = np.stack(layer_parts, axis=1).astype(np.float16)
        spans = spans_by_representation[representation]
        starts = np.asarray([start for start, _ in spans], dtype=np.int32)
        ends = np.asarray([end for _, end in spans], dtype=np.int32)
        relative_progress = ((starts.astype(np.float64) + ends) / 2.0) / segment_length
        horizontal, vertical = build_span_direction_records(
            pooled,
            layers=layer_values,
            starts=starts,
            ends=ends,
            relative_progress=relative_progress,
            rollout_id=rollout_id,
            is_correct=is_correct,
            representation=representation,
            progress_bins=progress_bins,
            question_id=question_id,
        )
        horizontal["think_length"] = int(think_length)
        vertical["think_length"] = int(think_length)
        if representation != primary_representation:
            horizontal = horizontal.drop(columns=["displacement"])
        horizontal_frames.append(horizontal)
        vertical_frames.append(vertical)
        if retain_audit_vectors:
            audit_vectors[representation] = pooled
            audit_metadata[representation] = {
                "span_start": starts,
                "span_end": ends,
                "relative_progress": relative_progress.astype(np.float32),
                "layers": layer_values,
            }

    return ReducedRollout(
        token_features=pd.DataFrame(token_rows),
        span_horizontal=pd.concat(horizontal_frames, ignore_index=True),
        span_vertical=pd.concat(vertical_frames, ignore_index=True),
        audit_vectors=audit_vectors,
        audit_metadata=audit_metadata,
    )


def aggregate_span_features(
    horizontal: pd.DataFrame, vertical: pd.DataFrame
) -> pd.DataFrame:
    keys = [
        "question_id",
        "rollout_id",
        "is_correct",
        "representation",
        "progress_bin",
        "layer",
        "think_length",
    ]
    horizontal_values = horizontal.copy()
    horizontal_values["span_turn_cos_split_a"] = horizontal_values["span_turn_cos"].where(
        horizontal_values["span_id"] % 2 == 0
    )
    horizontal_values["span_turn_cos_split_b"] = horizontal_values["span_turn_cos"].where(
        horizontal_values["span_id"] % 2 == 1
    )
    vertical_values = vertical.copy()
    vertical_values["span_layer_turn_cos_split_a"] = vertical_values[
        "span_layer_turn_cos"
    ].where(vertical_values["span_id"] % 2 == 0)
    vertical_values["span_layer_turn_cos_split_b"] = vertical_values[
        "span_layer_turn_cos"
    ].where(vertical_values["span_id"] % 2 == 1)
    horizontal_agg = (
        horizontal_values.groupby(keys, as_index=False, observed=True)
        .agg(
            span_movement_norm_mean=("displacement_norm", "mean"),
            span_movement_norm_median=("displacement_norm", "median"),
            span_movement_norm_p90=("displacement_norm", lambda values: values.quantile(0.9)),
            span_turn_cos_mean=("span_turn_cos", "mean"),
            span_turn_cos_median=("span_turn_cos", "median"),
            span_turn_cos_split_a=("span_turn_cos_split_a", "mean"),
            span_turn_cos_split_b=("span_turn_cos_split_b", "mean"),
            span_count=("span_id", "size"),
        )
    )
    vertical_agg = (
        vertical_values.groupby(keys, as_index=False, observed=True)
        .agg(
            span_layer_update_norm_mean=("span_layer_update_norm", "mean"),
            span_layer_turn_cos_mean=("span_layer_turn_cos", "mean"),
            span_layer_turn_cos_median=("span_layer_turn_cos", "median"),
            span_layer_turn_cos_split_a=("span_layer_turn_cos_split_a", "mean"),
            span_layer_turn_cos_split_b=("span_layer_turn_cos_split_b", "mean"),
            vertical_span_count=("span_id", "size"),
        )
    )
    return horizontal_agg.merge(vertical_agg, on=keys, how="outer", validate="one_to_one")


def aggregate_cross_features(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    keys = [
        "question_id",
        "rollout_id",
        "is_correct",
        "representation",
        "progress_bin",
        "layer",
        "think_length",
    ]
    return (
        scores.groupby(keys, as_index=False, observed=True)
        .agg(
            cross_set_direction=("cross_set_direction", "mean"),
            cross_prototype_direction=("cross_prototype_direction", "mean"),
            cross_length_support=("cross_length_support", "mean"),
            prototype_kappa_pos_mean=("prototype_kappa_pos_mean", "mean"),
            prototype_kappa_pos_min=("prototype_kappa_pos_min", "min"),
            prototype_kappa_neg_mean=("prototype_kappa_neg_mean", "mean"),
            prototype_kappa_neg_min=("prototype_kappa_neg_min", "min"),
            prototype_valid_fraction=("prototype_valid_fraction", "mean"),
            balanced_subset_count_mean=("balanced_subset_count", "mean"),
            cross_query_count=("span_id", "size"),
        )
    )


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)


def _write_audit_npz(
    path: Path,
    payloads: list[AuditRollout],
) -> None:
    if not payloads:
        return
    arrays: dict[str, np.ndarray] = {}
    representations = sorted(payloads[0].vectors)
    for representation in representations:
        safe = representation.replace("-", "_")
        vectors = []
        rollout_ids = []
        starts = []
        ends = []
        progress = []
        for payload in payloads:
            current = payload.vectors[representation]
            metadata = payload.metadata[representation]
            vectors.append(current)
            rollout_ids.extend([payload.rollout_id] * current.shape[0])
            starts.append(metadata["span_start"])
            ends.append(metadata["span_end"])
            progress.append(metadata["relative_progress"])
        arrays[f"vectors_{safe}"] = np.concatenate(vectors, axis=0)
        arrays[f"rollout_id_{safe}"] = np.asarray(rollout_ids, dtype=np.int16)
        arrays[f"span_start_{safe}"] = np.concatenate(starts)
        arrays[f"span_end_{safe}"] = np.concatenate(ends)
        arrays[f"relative_progress_{safe}"] = np.concatenate(progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def extract_question(
    question_id: str,
    rows: list[dict[str, Any]],
    processor: Any,
    model: Any,
    device: Any,
    close_tag_ids: list[int],
    progress_bins: int,
    span_specs: tuple[tuple[int, int], ...],
    endpoint_spec: tuple[int, int],
    primary_spec: tuple[int, int],
    kappa_min: float,
    retain_audit_vectors: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[AuditRollout]]:
    import torch

    token_frames = []
    span_aggregate_frames = []
    primary_horizontal_frames = []
    audit_payloads: list[AuditRollout] = []
    label_by_rollout: dict[int, bool] = {}
    for row in sorted(rows, key=lambda item: int(item.get("rollout_id", -1))):
        prompt_inputs = encode_messages(
            processor, build_prompt_messages(row), add_generation_prompt=True
        )
        full_inputs = encode_messages(
            processor, build_full_messages(row), add_generation_prompt=False
        )
        response_start = common_prefix_len(prompt_inputs["input_ids"], full_inputs["input_ids"])
        response_end = int(full_inputs["input_ids"].shape[-1])
        response_ids = full_inputs["input_ids"][0, response_start:response_end].cpu().tolist()
        segment_start, segment_end, segment_status = segment_token_span_from_text(
            response_ids, close_tag_ids
        )
        if segment_status == "no_think_close":
            raise ValueError(
                f"token close boundary missing: question={question_id} rollout={row.get('rollout_id')}"
            )
        full_segment_start = response_start + segment_start
        full_segment_end = response_start + segment_end
        segment_length = int(segment_end - segment_start)
        rollout_id = int(row.get("rollout_id", -1))
        is_correct = bool(row.get("is_correct"))
        label_by_rollout[rollout_id] = is_correct

        full_inputs = to_device(full_inputs, device)
        with torch.inference_mode():
            outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)
        reduced = reduce_rollout_hidden(
            outputs.hidden_states,
            segment_start=full_segment_start,
            segment_end=full_segment_end,
            progress_bins=progress_bins,
            span_specs=span_specs,
            endpoint_spec=endpoint_spec,
            primary_spec=primary_spec,
            question_id=question_id,
            rollout_id=rollout_id,
            is_correct=is_correct,
            think_length=segment_length,
            retain_audit_vectors=retain_audit_vectors,
        )
        token_frames.append(reduced.token_features)
        span_aggregate_frames.append(
            aggregate_span_features(reduced.span_horizontal, reduced.span_vertical)
        )
        primary_representation = f"mean_w{primary_spec[0]}_s{primary_spec[1]}"
        primary_horizontal_frames.append(
            reduced.span_horizontal[
                reduced.span_horizontal["representation"] == primary_representation
            ].copy()
        )
        if retain_audit_vectors:
            audit_payloads.append(
                AuditRollout(
                    rollout_id=rollout_id,
                    vectors=reduced.audit_vectors,
                    metadata=reduced.audit_metadata,
                )
            )
        del outputs, full_inputs, prompt_inputs, reduced
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_correct = int(sum(label_by_rollout.values()))
    n_wrong = int(len(label_by_rollout) - n_correct)
    if n_correct < 3 or n_wrong < 3:
        raise ValueError(
            f"question lost 3+3 eligibility: {question_id} correct={n_correct} wrong={n_wrong}"
        )
    primary_horizontal = pd.concat(primary_horizontal_frames, ignore_index=True)
    cross_scores, prototype_diagnostics = score_cross_rollout_queries(
        primary_horizontal, kappa_min=kappa_min
    )
    span_features = pd.concat(span_aggregate_frames, ignore_index=True)
    cross_aggregate = aggregate_cross_features(cross_scores)
    if not cross_aggregate.empty:
        merge_keys = [
            "question_id",
            "rollout_id",
            "is_correct",
            "representation",
            "progress_bin",
            "layer",
            "think_length",
        ]
        span_features = span_features.merge(
            cross_aggregate, on=merge_keys, how="left", validate="one_to_one"
        )
    return (
        pd.concat(token_frames, ignore_index=True),
        span_features,
        prototype_diagnostics,
        audit_payloads,
    )


def append_status(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    span_specs = parse_span_specs(args.span_specs)
    endpoint_spec = parse_span_spec(args.endpoint_spec)
    primary_spec = parse_span_spec(args.primary_spec)
    if endpoint_spec not in span_specs or primary_spec not in span_specs:
        raise ValueError("endpoint and primary specs must be present in --span-specs")
    if not 0.0 <= args.kappa_min <= 1.0:
        raise ValueError("kappa-min must be in [0, 1]")
    output_dir = Path(args.output_dir)
    for directory in (
        "bin_features",
        "span_features",
        "prototype_diagnostics",
        "audit_spans",
        "completed",
    ):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(Path(args.input))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("question_id", "")), []).append(row)
    question_ids = sorted(grouped)
    audit_ids = set(question_ids[: max(args.audit_question_count, 0)])
    (output_dir / "AUDIT_QUESTIONS.json").write_text(
        json.dumps(sorted(audit_ids), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
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
    for question_id in tqdm(question_ids, desc="experiment zero questions"):
        stem = question_stem(question_id)
        if args.resume and valid_completed_question(output_dir, stem):
            counts["resumed"] += 1
            continue
        try:
            token_frame, span_frame, prototype_frame, audit_payloads = extract_question(
                question_id,
                grouped[question_id],
                processor=processor,
                model=model,
                device=device,
                close_tag_ids=close_tag_ids,
                progress_bins=args.progress_bins,
                span_specs=span_specs,
                endpoint_spec=endpoint_spec,
                primary_spec=primary_spec,
                kappa_min=args.kappa_min,
                retain_audit_vectors=question_id in audit_ids,
            )
            _atomic_parquet(token_frame, output_dir / "bin_features" / f"{stem}.parquet")
            _atomic_parquet(span_frame, output_dir / "span_features" / f"{stem}.parquet")
            _atomic_parquet(
                prototype_frame,
                output_dir / "prototype_diagnostics" / f"{stem}.parquet",
            )
            if audit_payloads:
                _write_audit_npz(output_dir / "audit_spans" / f"{stem}.npz", audit_payloads)
            marker = output_dir / "completed" / f"{stem}.complete.json"
            temporary_marker = marker.with_suffix(".tmp.json")
            record = {
                "question_id": question_id,
                "status": "completed",
                "rollouts": int(token_frame["rollout_id"].nunique()),
                "layers": int(token_frame["layer"].nunique()),
                "token_rows": int(len(token_frame)),
                "span_rows": int(len(span_frame)),
                "prototype_rows": int(len(prototype_frame)),
            }
            temporary_marker.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary_marker, marker)
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
        "progress_bins": args.progress_bins,
        "span_specs": span_specs,
        "endpoint_spec": endpoint_spec,
        "primary_spec": primary_spec,
        "kappa_min": args.kappa_min,
        "audit_questions": sorted(audit_ids),
        "new_generation": False,
    }
    (output_dir / "EXTRACTION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
