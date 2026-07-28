#!/usr/bin/env python3
"""Replay frozen rollouts and extract only raw entropy at L14-L19/bin6."""

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

from entropy_band_metrics import EPS, frozen_band_score, progress_bin_ids
from run_long_path_smoke_qwen3vl import (
    build_full_messages,
    build_prompt_messages,
    common_prefix_len,
    encode_messages,
    segment_token_span_from_text,
    to_device,
)


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_LAYERS = tuple(range(14, 20))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="14,15,16,17,18,19")
    parser.add_argument("--progress-bins", type=int, default=10)
    parser.add_argument("--progress-bin", type=int, default=6)
    parser.add_argument("--entropy-chunk-size", type=int, default=512)
    parser.add_argument("--min-correct", type=int, default=2)
    parser.add_argument("--min-wrong", type=int, default=2)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data2/hjk/models/huggingface")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if layers != DEFAULT_LAYERS:
        raise ValueError(f"confirmatory layers are frozen to {DEFAULT_LAYERS}")
    return layers


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
    digest = hashlib.sha1(str(question_id).encode("utf-8")).hexdigest()[:12]
    return f"question_{safe or 'question'}_{digest}"


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)


def _atomic_json(target: Path, payload: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def completed_question(output_dir: Path, stem: str) -> bool:
    feature_path = output_dir / "features" / f"{stem}.parquet"
    marker_path = output_dir / "completed" / f"{stem}.complete.json"
    if not feature_path.is_file() or not marker_path.is_file():
        return False
    try:
        frame = pd.read_parquet(feature_path)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = {
        "question_id",
        "rollout_id",
        "is_correct",
        "think_length",
        "raw_entropy_L14_19_bin6",
    }.union({f"raw_entropy_L{layer}" for layer in DEFAULT_LAYERS})
    return bool(not frame.empty and required.issubset(frame.columns) and marker.get("status") == "completed")


def _torch_entropy_mean(hidden: Any, *, chunk_size: int) -> float:
    import torch

    total = 0.0
    count = 0
    for start in range(0, int(hidden.shape[0]), chunk_size):
        values = hidden[start : start + chunk_size].float()
        values = values - values.mean(dim=-1, keepdim=True)
        energy = values.square()
        denominator = energy.sum(dim=-1, keepdim=True)
        probabilities = torch.where(
            denominator > EPS,
            energy / denominator.clamp_min(EPS),
            torch.zeros_like(energy),
        )
        entropy = -(probabilities * torch.log(probabilities + EPS)).sum(dim=-1)
        entropy = torch.clamp(entropy / np.log(values.shape[-1]), 0.0, 1.0)
        total += float(entropy.sum().item())
        count += int(entropy.numel())
        del values, energy, denominator, probabilities, entropy
    if count == 0:
        raise ValueError("frozen progress bin has no tokens")
    return total / count


def reduce_frozen_band(
    hidden_states: tuple[Any, ...],
    *,
    segment_start: int,
    segment_end: int,
    layers: tuple[int, ...],
    n_bins: int,
    progress_bin: int,
    chunk_size: int,
) -> tuple[dict[int, float], float, int]:
    import torch

    segment_length = int(segment_end - segment_start)
    if segment_length <= 0:
        raise ValueError("empty thinking segment")
    if progress_bin != 6 or n_bins != 10:
        raise ValueError("confirmatory progress specification is frozen to 10 bins and bin6")
    if max(layers) >= len(hidden_states):
        raise ValueError(f"requested L{max(layers)} but only {len(hidden_states)} states exist")
    relative = np.flatnonzero(progress_bin_ids(segment_length, n_bins) == progress_bin)
    absolute = torch.as_tensor(relative + segment_start, dtype=torch.long)
    layer_means: dict[int, float] = {}
    for layer in layers:
        state = hidden_states[layer]
        state = state[0] if state.ndim == 3 else state
        indices = absolute.to(state.device)
        selected = state.index_select(0, indices)
        layer_means[layer] = _torch_entropy_mean(selected, chunk_size=chunk_size)
        del selected, indices
    return layer_means, frozen_band_score(layer_means, layers=layers), int(relative.size)


def extract_question(
    question_id: str,
    rows: list[dict[str, Any]],
    *,
    processor: Any,
    model: Any,
    device: Any,
    close_tag_ids: list[int],
    layers: tuple[int, ...],
    args: argparse.Namespace,
) -> pd.DataFrame:
    import torch

    output_rows = []
    labels = []
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
        response_start = common_prefix_len(prompt_inputs["input_ids"], full_inputs["input_ids"])
        response_ids = full_inputs["input_ids"][0, response_start:].cpu().tolist()
        segment_start, segment_end, status = segment_token_span_from_text(response_ids, close_tag_ids)
        if status == "no_think_close":
            raise ValueError(f"missing think close: question={question_id} rollout={row.get('rollout_id')}")
        full_start = response_start + segment_start
        full_end = response_start + segment_end
        full_inputs = to_device(full_inputs, device)
        with torch.inference_mode():
            outputs = model(
                **full_inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            layer_means, band_score, token_count = reduce_frozen_band(
                outputs.hidden_states,
                segment_start=full_start,
                segment_end=full_end,
                layers=layers,
                n_bins=args.progress_bins,
                progress_bin=args.progress_bin,
                chunk_size=args.entropy_chunk_size,
            )
        rollout_id = int(row.get("rollout_id", -1))
        is_correct = bool(row.get("is_correct"))
        labels.append(is_correct)
        output_row = {
            "question_id": str(question_id),
            "rollout_id": rollout_id,
            "is_correct": is_correct,
            "answer": str(row.get("answer", "")),
            "pred_answer": str(row.get("pred_answer", "")),
            "think_length": int(full_end - full_start),
            "progress_bin": args.progress_bin,
            "bin_token_count": token_count,
            "raw_entropy_L14_19_bin6": band_score,
            "feature_direction": "higher_is_correct",
        }
        output_row.update({f"raw_entropy_L{layer}": value for layer, value in layer_means.items()})
        output_rows.append(output_row)
        print(
            json.dumps(
                {
                    "event": "entropy_band_rollout",
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "is_correct": is_correct,
                    "think_length": output_row["think_length"],
                    "band_score": round(band_score, 8),
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
        raise ValueError(f"question lost 2+2 eligibility: {question_id} correct={n_correct} wrong={n_wrong}")
    return pd.DataFrame(output_rows)


def append_status(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    layers = parse_layers(args.layers)
    if args.entropy_chunk_size <= 0:
        raise ValueError("entropy_chunk_size must be positive")
    output_dir = Path(args.output_dir)
    for name in ("features", "completed"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.input))
    grouped: dict[str, list[dict[str, Any]]] = {}
    question_order = []
    for row in rows:
        qid = str(row.get("question_id", ""))
        if qid not in grouped:
            grouped[qid] = []
            question_order.append(qid)
        grouped[qid].append(row)

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
    for qid in tqdm(question_order, desc="entropy confirm120 questions"):
        stem = question_stem(qid)
        if args.resume and completed_question(output_dir, stem):
            counts["resumed"] += 1
            continue
        try:
            frame = extract_question(
                qid,
                grouped[qid],
                processor=processor,
                model=model,
                device=device,
                close_tag_ids=close_tag_ids,
                layers=layers,
                args=args,
            )
            _atomic_parquet(frame, output_dir / "features" / f"{stem}.parquet")
            record = {
                "question_id": qid,
                "status": "completed",
                "rollouts": len(frame),
                "correct": int(frame["is_correct"].sum()),
                "wrong": int((~frame["is_correct"]).sum()),
            }
            _atomic_json(output_dir / "completed" / f"{stem}.complete.json", record)
            counts["completed"] += 1
        except Exception as exc:
            record = {"question_id": qid, "status": "failed", "error": repr(exc)}
            counts["failed"] += 1
        append_status(status_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        **counts,
        "expected_questions": len(question_order),
        "available_feature_shards": len(list((output_dir / "features").glob("question_*.parquet"))),
        "layers": list(layers),
        "progress_bins": args.progress_bins,
        "progress_bin": args.progress_bin,
        "feature": "raw_entropy_L14_19_bin6",
    }
    _atomic_json(output_dir / "EXTRACTION_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if counts["failed"]:
        raise RuntimeError(f"entropy extraction failed for {counts['failed']} questions")


if __name__ == "__main__":
    main()
