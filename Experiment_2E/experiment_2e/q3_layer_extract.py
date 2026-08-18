from __future__ import annotations

import argparse
import gc
import hashlib
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
from .common import sha256_file, write_json_atomic
from .hidden_extract import enrich_metric_rows, stage_controls
from .q3_layer_forward import teacher_forced_pooled_forward
from .q3_layer_records import load_q3_generation_records
from .reduction import reduce_vertical
from .vertical_profiles import PROFILE_COLUMNS, reduce_vertical_profiles


CONTROL_FIELDS = (
    "response_token_count",
    "stage_token_count",
    "trajectory_point_count",
    "policy_entropy",
    "token_logprob_mean",
    "hidden_norm_mean",
)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def parameter_sample_sha256(model: Any) -> str:
    named = list(model.named_parameters())
    if not named:
        raise ValueError("model contains no parameters")
    digest = hashlib.sha256()
    indices = sorted({0, len(named) // 3, 2 * len(named) // 3, len(named) - 1})
    for index in indices:
        name, parameter = named[index]
        sample = parameter.detach().reshape(-1)[:4096].float().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(sample.tobytes())
    return digest.hexdigest()


def validate_trained_parameter_sample(
    sample: str,
    base_identity_path: Path,
    *,
    global_step: int,
) -> None:
    if global_step <= 0:
        return
    if not base_identity_path.is_file():
        raise FileNotFoundError(f"missing base model identity: {base_identity_path}")
    base = json.loads(base_identity_path.read_text(encoding="utf-8"))
    if sample == base.get("parameter_sample_sha256"):
        raise ValueError(f"trained checkpoint step={global_step} parameter sample matches base")


def _hash_manifest(paths: list[Path], *, root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def write_model_identity(
    model: Any,
    *,
    model_path: Path,
    source_checkpoint_dir: Path | None,
    global_step: int,
    expected_hidden_states: int,
    base_identity_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    sample = parameter_sample_sha256(model)
    config = model.config
    source_hashes: list[dict[str, Any]] = []
    merged_files = sorted(model_path.glob("*.safetensors"))
    if not merged_files:
        raise FileNotFoundError(f"no safetensors under model path: {model_path}")
    if global_step > 0:
        if source_checkpoint_dir is None:
            raise ValueError("trained checkpoint requires source_checkpoint_dir")
        source_files = sorted(source_checkpoint_dir.glob("model_world_size_*_rank_*.pt"))
        if not source_files:
            raise FileNotFoundError(f"no FSDP model shards under {source_checkpoint_dir}")
        source_hashes = _hash_manifest(source_files, root=source_checkpoint_dir)
    identity: dict[str, Any] = {
        "global_step": int(global_step),
        "model_type": str(config.model_type),
        "num_hidden_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "expected_hidden_states": int(expected_hidden_states),
        "parameter_sample_sha256": sample,
        "source_model_shards": source_hashes,
        "model_safetensors": _hash_manifest(merged_files, root=model_path),
    }
    gates = {
        "hidden_state_count_from_config": int(config.num_hidden_layers) + 1
        == expected_hidden_states,
        "source_shards_present": global_step == 0 or bool(source_hashes),
        "safetensors_present": bool(identity["model_safetensors"]),
        "trained_sample_differs_from_base": True,
    }
    try:
        if global_step > 0:
            if base_identity_path is None:
                raise ValueError("trained checkpoint requires base identity")
            validate_trained_parameter_sample(sample, base_identity_path, global_step=global_step)
    except (FileNotFoundError, ValueError) as exc:
        gates["trained_sample_differs_from_base"] = False
        identity["error"] = str(exc)
    identity["gates"] = gates
    identity["passed"] = bool(all(gates.values()))
    write_json_atomic(output_path, identity)
    if not identity["passed"]:
        raise ValueError(f"model identity audit failed: {identity}")
    return identity


def _controls_path(controls_dir: Path, cache_path: Path) -> Path:
    return controls_dir / f"{cache_path.stem}.json"


def _cache_complete(cache_path: Path, controls_path: Path, rollout_id: str) -> bool:
    if not cache_path.is_file() or not controls_path.is_file():
        return False
    try:
        cache = load_pooled_cache(cache_path)
        controls = json.loads(controls_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return (
        cache["metadata"].get("rollout_id") == rollout_id
        and controls.get("rollout_id") == rollout_id
        and len(controls.get("controls", [])) == 4
    )


def _record_metadata(
    record: dict[str, Any],
    *,
    run_id: str,
    model_name: str,
    forward: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model": model_name,
        "checkpoint": str(record["checkpoint"]),
        "global_step": int(record["global_step"]),
        "training_progress": float(record["training_progress"]),
        "question_id": str(record["question_id"]),
        "rollout_id": str(record["rollout_id"]),
        "rollout_slot": int(record["rollout_slot"]),
        "is_correct": bool(record["is_correct"]),
        "answer_reward": float(record["answer_reward"]),
        "format_correct": bool(record["format_correct"]),
        "parse_correct": bool(record["parse_correct"]),
        "difficulty": record.get("difficulty"),
        "prompt_hash": record.get("prompt_hash"),
        "prompt_token_count": int(forward["prompt_token_count"]),
        "response_token_count": int(forward["response_token_count"]),
        "response_length": int(forward["response_token_count"]),
        "trajectory_point_count": int(len(forward["endpoints"])),
        "n_hidden_states": int(forward["n_hidden_states"]),
    }


def forward_and_cache(
    record: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    run_id: str,
    model_name: str,
    cache_dir: Path,
    controls_dir: Path,
    expected_hidden_states: int,
) -> Path:
    cache_path = cache_dir / cache_name(str(record["rollout_id"]))
    controls_path = _controls_path(controls_dir, cache_path)
    if _cache_complete(cache_path, controls_path, str(record["rollout_id"])):
        return cache_path
    forward = teacher_forced_pooled_forward(
        model,
        tokenizer,
        str(record["prompt"]),
        str(record["response"]),
    )
    if int(forward["n_hidden_states"]) != expected_hidden_states:
        raise ValueError(
            f"expected {expected_hidden_states} hidden states, found {forward['n_hidden_states']}"
        )
    metadata = _record_metadata(record, run_id=run_id, model_name=model_name, forward=forward)
    controls = stage_controls(
        token_logprob=np.asarray(forward["token_logprob"]),
        policy_entropy=np.asarray(forward["policy_entropy"]),
        hidden_norm=np.asarray(forward["hidden_norm"]),
        response_token_count=int(forward["response_token_count"]),
        trajectory_point_count=int(len(forward["endpoints"])),
        metadata=metadata,
    )
    write_pooled_cache(
        cache_path,
        pooled={representation: forward[representation] for representation in REPRESENTATIONS},
        endpoints=np.asarray(forward["endpoints"]),
        progress=np.asarray(forward["progress"]),
        metadata=metadata,
    )
    write_json_atomic(
        controls_path,
        {"rollout_id": metadata["rollout_id"], "controls": controls},
    )
    del forward
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return cache_path


def _attach_stage_controls(frame: pd.DataFrame, controls: list[dict[str, Any]]) -> pd.DataFrame:
    by_stage = {int(row["stage"]): row for row in controls}
    result = frame.copy()
    for field in CONTROL_FIELDS:
        result[field] = result["stage"].map(lambda stage: by_stage[int(stage)][field])
    result["response_length"] = result["response_token_count"]
    return result


def reduce_checkpoint_caches(
    cache_paths: list[Path],
    *,
    controls_dir: Path,
    calibrator_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not cache_paths:
        raise ValueError("checkpoint reduction requires pooled caches")
    profile_frames: list[pd.DataFrame] = []
    aggregate_rows: list[dict[str, Any]] = []
    all_controls: list[dict[str, Any]] = []
    for cache_path in sorted(cache_paths):
        cache = load_pooled_cache(cache_path)
        payload = json.loads(
            _controls_path(controls_dir, cache_path).read_text(encoding="utf-8")
        )
        if payload.get("rollout_id") != cache["metadata"].get("rollout_id"):
            raise ValueError(f"cache/control rollout mismatch: {cache_path}")
        controls = list(payload["controls"])
        all_controls.extend(controls)
        for representation in REPRESENTATIONS:
            common, coordinate_mean, coordinate_sigma = load_calibrators(
                calibrator_path,
                representation,
            )
            profiles = reduce_vertical_profiles(
                cache[representation],
                cache["progress"],
                representation=representation,
                metadata=cache["metadata"],
                base_common=common,
            )
            profile_frames.append(_attach_stage_controls(profiles, controls))
            rows = reduce_vertical(
                cache[representation],
                cache["progress"],
                representation=representation,
                metadata=cache["metadata"],
                base_common=common,
                coordinate_common=coordinate_mean,
                coordinate_sigma=coordinate_sigma,
            )
            enrich_metric_rows(rows, controls)
            aggregate_rows.extend(rows)
    return (
        pd.concat(profile_frames, ignore_index=True),
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(all_controls),
    )


def checkpoint_audit(
    *,
    profiles: pd.DataFrame,
    controls: pd.DataFrame,
    expected_rollouts: int,
    expected_hidden_states: int,
    output_root: Path,
    identity_passed: bool,
    calibrator_path: Path,
) -> dict[str, Any]:
    expected_rows = expected_rollouts * len(REPRESENTATIONS) * 4 * expected_hidden_states
    layer_indices = sorted(profiles["layer_index"].dropna().astype(int).unique().tolist())
    expected_layers = list(range(expected_hidden_states))
    expected_depths = np.arange(expected_hidden_states, dtype=float) / (expected_hidden_states - 1)
    actual_depths = (
        profiles[["layer_index", "relative_depth"]]
        .drop_duplicates()
        .sort_values("layer_index")["relative_depth"]
        .to_numpy(float)
    )
    structural_masks = {
        "v1_raw_update_norm": profiles["layer_index"] == 0,
        "v1_relative_update_norm": profiles["layer_index"] == 0,
        "v3_demean_state_angle": profiles["layer_index"] == 0,
        "v4_layer_update_turning_angle": profiles["layer_index"] < 2,
        "v6_raw_activation_entropy": np.zeros(len(profiles), dtype=bool),
        "v7_layer_difference_entropy": profiles["layer_index"] == 0,
    }
    structural_ok = True
    defined_coverage_ok = True
    for metric in PROFILE_COLUMNS:
        coverage_column = f"profile_coverage_count_{metric}"
        structural = np.asarray(structural_masks[metric], dtype=bool)
        if structural.any():
            structural_ok = bool(
                structural_ok
                and profiles.loc[structural, metric].isna().all()
                and (profiles.loc[structural, coverage_column] == 0).all()
            )
        defined = ~structural
        defined_coverage_ok = bool(
            defined_coverage_ok
            and (profiles.loc[defined, coverage_column] > 0).any()
            and profiles.loc[defined & (profiles[coverage_column] > 0), metric]
            .notna()
            .all()
        )
    unique_profile_keys = not profiles.duplicated(
        ["rollout_id", "representation", "stage", "layer_index"]
    ).any()
    unique_control_keys = not controls.duplicated(["rollout_id", "stage"]).any()
    calibrator_audit = calibrator_path.with_suffix(".audit.json")
    token_hidden_files = [path for path in output_root.rglob("*token_hidden*") if path.is_file()]
    gates = {
        "profile_row_count": len(profiles) == expected_rows,
        "profile_rollout_count": profiles["rollout_id"].nunique() == expected_rollouts,
        "control_row_count": len(controls) == expected_rollouts * 4,
        "control_rollout_count": controls["rollout_id"].nunique() == expected_rollouts,
        "layer_indices": layer_indices == expected_layers,
        "relative_depths": len(actual_depths) == len(expected_depths)
        and bool(np.allclose(actual_depths, expected_depths)),
        "unique_profile_keys": bool(unique_profile_keys),
        "unique_control_keys": bool(unique_control_keys),
        "structural_nan_semantics": bool(structural_ok),
        "defined_profile_coverage": bool(defined_coverage_ok),
        "model_identity": bool(identity_passed),
        "calibrator_audit": calibrator_audit.is_file(),
        "no_full_token_hidden_files": not token_hidden_files,
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "n_profile_rows": int(len(profiles)),
        "n_control_rows": int(len(controls)),
        "n_unique_rollouts": int(controls["rollout_id"].nunique()),
        "n_hidden_states": int(expected_hidden_states),
        "token_hidden_files": [str(path) for path in token_hidden_files],
    }


def _load_model(model_path: Path, tokenizer_path: Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def run_extraction(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = f"step{args.global_step:03d}"
    profile_path = args.output_dir / f"layer_profiles_{checkpoint}.parquet"
    aggregate_path = args.output_dir / f"vertical_metrics_{checkpoint}.parquet"
    controls_path = args.output_dir / f"controls_{checkpoint}.parquet"
    identity_path = args.output_dir / f"model_identity_{checkpoint}.json"
    audit_path = args.output_dir / f"extraction_audit_{checkpoint}.json"
    if (
        audit_path.is_file()
        and profile_path.is_file()
        and aggregate_path.is_file()
        and controls_path.is_file()
        and identity_path.is_file()
    ):
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing.get("passed"):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return existing

    records = load_q3_generation_records(
        args.generation_file,
        args.heldout_file,
        global_step=args.global_step,
        total_steps=args.total_steps,
        question_limit=args.question_limit,
    )
    expected_rollouts = len(records)
    cache_dir = args.work_dir / "pooled_cache" / checkpoint
    control_cache_dir = args.work_dir / "controls" / checkpoint
    for directory in (args.output_dir, cache_dir, control_cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    model, tokenizer = _load_model(args.model_path, args.tokenizer_path)
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass
    identity = write_model_identity(
        model,
        model_path=args.model_path,
        source_checkpoint_dir=args.source_checkpoint_dir,
        global_step=args.global_step,
        expected_hidden_states=args.expected_hidden_states,
        base_identity_path=args.base_parameter_sample,
        output_path=identity_path,
    )
    for index, record in enumerate(records.to_dict("records"), start=1):
        forward_and_cache(
            record,
            model=model,
            tokenizer=tokenizer,
            run_id=args.run_id,
            model_name=args.model_name,
            cache_dir=cache_dir,
            controls_dir=control_cache_dir,
            expected_hidden_states=args.expected_hidden_states,
        )
        print(f"checkpoint={checkpoint} forward={index}/{expected_rollouts}", flush=True)
    peak_gpu_memory = None
    try:
        import torch

        peak_gpu_memory = int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        pass
    del model, tokenizer
    gc.collect()

    cache_paths = sorted(cache_dir.glob("*.npz"))
    if args.global_step == 0 and not args.calibrator_path.is_file():
        fit_base_calibrators(cache_paths, args.calibrator_path)
    if not args.calibrator_path.is_file():
        raise FileNotFoundError(f"missing base calibrator: {args.calibrator_path}")
    profiles, aggregate, controls = reduce_checkpoint_caches(
        cache_paths,
        controls_dir=control_cache_dir,
        calibrator_path=args.calibrator_path,
    )
    atomic_parquet(profiles, profile_path)
    atomic_parquet(aggregate, aggregate_path)
    atomic_parquet(controls, controls_path)
    audit = checkpoint_audit(
        profiles=profiles,
        controls=controls,
        expected_rollouts=expected_rollouts,
        expected_hidden_states=args.expected_hidden_states,
        output_root=args.output_dir,
        identity_passed=bool(identity["passed"]),
        calibrator_path=args.calibrator_path,
    )
    elapsed = time.time() - started
    audit.update(
        {
            "run_id": args.run_id,
            "checkpoint": checkpoint,
            "global_step": int(args.global_step),
            "elapsed_seconds": elapsed,
            "seconds_per_rollout": elapsed / expected_rollouts,
            "peak_gpu_memory_bytes": peak_gpu_memory,
            "generation_sha256": sha256_file(args.generation_file),
            "heldout_sha256": sha256_file(args.heldout_file),
            "calibrator_sha256": sha256_file(args.calibrator_path),
            "profile_sha256": sha256_file(profile_path),
            "aggregate_sha256": sha256_file(aggregate_path),
            "controls_sha256": sha256_file(controls_path),
        }
    )
    write_json_atomic(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise RuntimeError(f"checkpoint extraction audit failed: {checkpoint}")
    if not args.keep_pooled_cache:
        for path in cache_paths:
            path.unlink()
        for path in control_cache_dir.glob("*.json"):
            path.unlink()
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Q3 layer-resolved hidden profiles.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation-file", type=Path, required=True)
    parser.add_argument("--heldout-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=250)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibrator-path", type=Path, required=True)
    parser.add_argument("--source-checkpoint-dir", type=Path)
    parser.add_argument("--base-parameter-sample", type=Path)
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--expected-hidden-states", type=int, default=29)
    parser.add_argument("--keep-pooled-cache", action="store_true")
    args = parser.parse_args()
    if args.global_step > 0 and args.source_checkpoint_dir is None:
        parser.error("trained checkpoint requires --source-checkpoint-dir")
    if args.global_step > 0 and args.base_parameter_sample is None:
        parser.error("trained checkpoint requires --base-parameter-sample")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run_extraction(args)


if __name__ == "__main__":
    main()
