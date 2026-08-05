from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibrators import (
    REPRESENTATIONS,
    cache_name,
    fit_base_calibrators,
    load_calibrators,
    load_pooled_cache,
    write_pooled_cache,
)
from .common import sha256_text, write_json_atomic
from .formal_manifest import CHECKPOINT_PROGRESS
from .metrics import STAGES, stage_mask
from .reduction import pool_token_hidden, reduce_h8, reduce_horizontal, reduce_vertical, trajectory_endpoints
from .rollout_audit import load_rollouts


def _bundle_path(bundle_dir: Path, rollout_id: str) -> Path:
    return bundle_dir / f"{sha256_text(rollout_id)[:20]}.json"


def _write_bundle(path: Path, value: dict[str, Any]) -> None:
    write_json_atomic(path, value)


def stage_controls(
    *,
    token_logprob: np.ndarray,
    policy_entropy: np.ndarray,
    hidden_norm: np.ndarray,
    response_token_count: int,
    trajectory_point_count: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    progress = np.arange(1, response_token_count + 1, dtype=float) / response_token_count
    rows = []
    for stage in range(len(STAGES)):
        selection = stage_mask(progress, stage)
        rows.append(
            {
                **metadata,
                "stage": stage,
                "response_token_count": response_token_count,
                "stage_token_count": int(selection.sum()),
                "trajectory_point_count": trajectory_point_count,
                "policy_entropy": float(np.mean(policy_entropy[selection])) if selection.any() else np.nan,
                "token_logprob_mean": float(np.mean(token_logprob[selection])) if selection.any() else np.nan,
                "hidden_norm_mean": float(np.mean(hidden_norm[selection])) if selection.any() else np.nan,
            }
        )
    return rows


def enrich_metric_rows(rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> None:
    by_stage = {int(row["stage"]): row for row in controls}
    for row in rows:
        control = by_stage[int(row["stage"])]
        row["response_length"] = control["response_token_count"]
        row["policy_entropy"] = control["policy_entropy"]
        row["token_logprob_mean"] = control["token_logprob_mean"]
        row["hidden_norm_mean"] = control["hidden_norm_mean"]


def _metadata(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    checkpoint = str(record["checkpoint"])
    return {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "training_progress": CHECKPOINT_PROGRESS[checkpoint],
        "question_id": str(record["question_id"]),
        "rollout_id": str(record["rollout_id"]),
        "rollout_slot": int(record["rollout_slot"]),
        "is_correct": bool(float(record.get("answer_reward", 0.0)) > 0.5),
        "answer_reward": float(record.get("answer_reward", 0.0)),
        "format_reward": float(record.get("format_reward", 0.0)),
        "truncated": bool(record.get("truncated", False)),
        "finish_reason": record.get("finish_reason"),
        "level": record.get("level"),
        "subject": record.get("subject"),
    }


def reduce_cached_rollout(
    cache: dict[str, Any],
    *,
    calibrator_path: Path,
    h8_rows: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = cache["metadata"]
    horizontal: list[dict[str, Any]] = list(h8_rows)
    vertical: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        common, coordinate_mean, coordinate_sigma = load_calibrators(calibrator_path, representation)
        trajectory = cache[representation]
        horizontal.extend(
            reduce_horizontal(
                trajectory,
                cache["progress"],
                representation=representation,
                metadata=metadata,
            )
        )
        vertical.extend(
            reduce_vertical(
                trajectory,
                cache["progress"],
                representation=representation,
                metadata=metadata,
                base_common=common,
                coordinate_common=coordinate_mean,
                coordinate_sigma=coordinate_sigma,
            )
        )
    enrich_metric_rows(horizontal, controls)
    enrich_metric_rows(vertical, controls)
    return {"metadata": metadata, "horizontal": horizontal, "vertical": vertical, "controls": controls}


def _model_forward(model: Any, tokenizer: Any, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    prompt_ids = tokenizer.encode(record["prompt"], add_special_tokens=False)
    response_ids = [int(token) for token in record["response_token_ids"]]
    if not prompt_ids or not response_ids:
        raise ValueError(f"empty prompt or response token IDs for {record['rollout_id']}")
    model_device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=model_device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
    start = len(prompt_ids)
    stop = start + len(response_ids)
    token_hidden = np.stack(
        [state[0, start:stop].float().cpu().numpy() for state in output.hidden_states],
        axis=1,
    )
    prediction_logits = output.logits[0, start - 1 : stop - 1].float()
    log_normalizer = torch.logsumexp(prediction_logits, dim=-1)
    probabilities = torch.softmax(prediction_logits, dim=-1)
    policy_entropy = (log_normalizer - (probabilities * prediction_logits).sum(dim=-1)).cpu().numpy()
    targets = torch.tensor(response_ids, dtype=torch.long, device=prediction_logits.device)
    token_logprob = (prediction_logits.gather(1, targets[:, None]).squeeze(1) - log_normalizer).cpu().numpy()
    del output, prediction_logits, probabilities, input_ids, targets
    return token_hidden, token_logprob, policy_entropy


def _load_model(base_model_path: Path, adapter_path: Path | None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda"},
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def _forward_and_cache(
    record: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    run_id: str,
    cache_dir: Path,
    partial_dir: Path,
) -> dict[str, Any]:
    metadata = _metadata(record, run_id)
    token_hidden, token_logprob, policy_entropy = _model_forward(model, tokenizer, record)
    endpoints = trajectory_endpoints(token_hidden.shape[0])
    progress = endpoints / token_hidden.shape[0]
    pooled = pool_token_hidden(token_hidden, endpoints)
    h8_rows = reduce_h8(token_hidden, endpoints, progress, metadata=metadata)
    hidden_norm = np.linalg.norm(token_hidden[:, -1, :], axis=1)
    controls = stage_controls(
        token_logprob=token_logprob,
        policy_entropy=policy_entropy,
        hidden_norm=hidden_norm,
        response_token_count=token_hidden.shape[0],
        trajectory_point_count=len(endpoints),
        metadata=metadata,
    )
    cache_path = cache_dir / cache_name(metadata["rollout_id"])
    write_pooled_cache(
        cache_path,
        pooled=pooled,
        endpoints=endpoints,
        progress=progress,
        metadata=metadata,
    )
    partial_path = _bundle_path(partial_dir, metadata["rollout_id"])
    _write_bundle(partial_path, {"h8": h8_rows, "controls": controls})
    del token_hidden, pooled
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return {"cache": cache_path, "partial": partial_path}


def _finalize_checkpoint(bundle_dir: Path, output_dir: Path, checkpoint: str) -> dict[str, Any]:
    bundles = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(bundle_dir.glob("*.json"))]
    if not bundles:
        raise ValueError(f"no reduced bundles found under {bundle_dir}")
    horizontal = pd.DataFrame([row for bundle in bundles for row in bundle["horizontal"]])
    vertical = pd.DataFrame([row for bundle in bundles for row in bundle["vertical"]])
    controls = pd.DataFrame([row for bundle in bundles for row in bundle["controls"]])
    output_dir.mkdir(parents=True, exist_ok=True)
    horizontal.to_parquet(output_dir / f"horizontal_metrics_{checkpoint}.parquet", index=False)
    vertical.to_parquet(output_dir / f"vertical_metrics_{checkpoint}.parquet", index=False)
    controls.to_parquet(output_dir / f"controls_{checkpoint}.parquet", index=False)
    audit = {
        "checkpoint": checkpoint,
        "n_bundles": len(bundles),
        "n_horizontal_rows": len(horizontal),
        "n_vertical_rows": len(vertical),
        "n_control_rows": len(controls),
        "n_unique_rollouts": int(controls["rollout_id"].nunique()),
        "h8_not_duplicated_by_representation": bool(
            set(horizontal.loc[horizontal["family_id"] == "H8", "representation"]) <= {"token"}
        ),
    }
    write_json_atomic(output_dir / f"extraction_audit_{checkpoint}.json", audit)
    return audit


def run_extraction(args: argparse.Namespace) -> None:
    records = load_rollouts(args.rollout_dir)
    records = records.loc[records["checkpoint"] == args.checkpoint].copy()
    if args.question_limit:
        question_ids = records["question_id"].drop_duplicates().head(args.question_limit)
        records = records.loc[records["question_id"].isin(question_ids)]
    if args.rollouts_per_question_limit:
        records = records.sort_values(["question_id", "rollout_slot"]).groupby(
            "question_id", as_index=False, sort=False
        ).head(args.rollouts_per_question_limit)
    if args.limit:
        records = records.head(args.limit)
    if records.empty:
        raise ValueError(f"no rollout records for checkpoint={args.checkpoint}")
    adapter_path = None if args.checkpoint == "base" else args.adapter_path
    model, tokenizer = _load_model(args.base_model_path, adapter_path)
    cache_dir = args.work_dir / "pooled_cache" / args.checkpoint
    partial_dir = args.work_dir / "partial" / args.checkpoint
    bundle_dir = args.work_dir / "bundles" / args.checkpoint
    for directory in (cache_dir, partial_dir, bundle_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    for index, record in enumerate(records.to_dict("records"), start=1):
        final_path = _bundle_path(bundle_dir, record["rollout_id"])
        cache_path = cache_dir / cache_name(record["rollout_id"])
        partial_path = _bundle_path(partial_dir, record["rollout_id"])
        if final_path.exists():
            continue
        if not cache_path.exists() or not partial_path.exists():
            _forward_and_cache(
                record,
                model=model,
                tokenizer=tokenizer,
                run_id=args.run_id,
                cache_dir=cache_dir,
                partial_dir=partial_dir,
            )
        print(f"checkpoint={args.checkpoint} forward={index}/{len(records)}", flush=True)

    del model
    gc.collect()
    if args.checkpoint == "base":
        fit_base_calibrators(sorted(cache_dir.glob("*.npz")), args.calibrator_path)
    if not args.calibrator_path.exists():
        raise FileNotFoundError(f"missing base calibrator: {args.calibrator_path}")

    for cache_path in sorted(cache_dir.glob("*.npz")):
        cache = load_pooled_cache(cache_path)
        final_path = _bundle_path(bundle_dir, cache["metadata"]["rollout_id"])
        if final_path.exists():
            continue
        partial = json.loads(_bundle_path(partial_dir, cache["metadata"]["rollout_id"]).read_text(encoding="utf-8"))
        bundle = reduce_cached_rollout(
            cache,
            calibrator_path=args.calibrator_path,
            h8_rows=partial["h8"],
            controls=partial["controls"],
        )
        _write_bundle(final_path, bundle)

    audit = _finalize_checkpoint(bundle_dir, args.output_dir, args.checkpoint)
    audit["elapsed_seconds"] = time.time() - started
    audit["peak_gpu_memory_bytes"] = None
    try:
        import torch

        audit["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        pass
    write_json_atomic(args.output_dir / f"extraction_audit_{args.checkpoint}.json", audit)
    if not args.keep_pooled_cache:
        for path in cache_dir.glob("*.npz"):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline hidden-state extraction and online scalar reduction.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINT_PROGRESS), required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibrator-path", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--rollouts-per-question-limit", type=int)
    parser.add_argument("--keep-pooled-cache", action="store_true")
    args = parser.parse_args()
    if args.checkpoint != "base" and args.adapter_path is None:
        parser.error("--adapter-path is required for trained checkpoints")
    if args.question_limit is not None and args.question_limit < 1:
        parser.error("--question-limit must be positive")
    if args.rollouts_per_question_limit is not None and args.rollouts_per_question_limit < 1:
        parser.error("--rollouts-per-question-limit must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run_extraction(args)


if __name__ == "__main__":
    main()
