from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .common import sha256_file, write_json_atomic, write_jsonl
from .data_prep import CHECKPOINT_NAMES, stable_seed
from .math_reward import score_response


def _as_python(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _mapping(value: Any) -> dict[str, Any]:
    value = _as_python(value)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping, received {type(value).__name__}")
    return value


def _messages(value: Any) -> list[dict[str, str]]:
    value = _as_python(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise TypeError(f"prompt must be a message list, received {type(value).__name__}")
    return [dict(message) for message in value]


def expected_rollout_keys(
    question_ids: list[str],
    *,
    checkpoints: tuple[str, ...] = CHECKPOINT_NAMES,
    rollouts_per_question: int = 8,
) -> set[tuple[str, str, int]]:
    return {
        (checkpoint, question_id, slot)
        for checkpoint in checkpoints
        for question_id in question_ids
        for slot in range(rollouts_per_question)
    }


def _completed_keys(output_dir: Path) -> set[tuple[str, str, int]]:
    completed: set[tuple[str, str, int]] = set()
    for part in sorted(output_dir.glob("**/parts/part_*.jsonl")):
        with part.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                completed.add((row["checkpoint"], row["question_id"], int(row["rollout_slot"])))
    return completed


def rollout_id(checkpoint: str, question_id: str, rollout_slot: int) -> str:
    return f"{checkpoint}/{question_id}/r{rollout_slot:02d}"


def audit_rollout_records(
    records: pd.DataFrame,
    *,
    expected_keys: set[tuple[str, str, int]],
) -> dict[str, Any]:
    required = {"checkpoint", "question_id", "rollout_slot", "response", "truncated"}
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"Missing rollout columns: {missing}")
    keys = list(
        zip(
            records["checkpoint"].astype(str),
            records["question_id"].astype(str),
            records["rollout_slot"].astype(int),
            strict=True,
        )
    )
    unique_keys = set(keys)
    duplicates = len(keys) - len(unique_keys)
    missing_keys = sorted(expected_keys - unique_keys)
    unexpected_keys = sorted(unique_keys - expected_keys)
    per_checkpoint = records.groupby("checkpoint").size().astype(int).to_dict()
    return {
        "passed": duplicates == 0 and not missing_keys and not unexpected_keys,
        "n_records": int(len(records)),
        "n_expected": int(len(expected_keys)),
        "n_duplicate_keys": int(duplicates),
        "n_missing_keys": int(len(missing_keys)),
        "missing_keys_sample": [list(key) for key in missing_keys[:10]],
        "unexpected_keys_sample": [list(key) for key in unexpected_keys[:10]],
        "records_by_checkpoint": {str(key): int(value) for key, value in per_checkpoint.items()},
        "truncation_rate": float(records["truncated"].astype(bool).mean()) if len(records) else None,
        "correct_rate": float(records["answer_reward"].mean()) if "answer_reward" in records and len(records) else None,
    }


def _checkpoint_adapter_path(checkpoint_root: Path, checkpoint: str, global_step: int) -> Path:
    return checkpoint_root / f"global_step_{global_step}" / "actor" / "huggingface"


def generate_checkpoint(args: argparse.Namespace, checkpoint: str, global_step: int) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    frame = pd.read_parquet(args.eval_file)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, local_files_only=True)
    checkpoint_dir = args.output_dir / checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "parts").mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(args.output_dir)
    rows = []
    prompts: list[str] = []
    params: list[SamplingParams] = []
    metadata: list[dict[str, Any]] = []
    adapter_path = None if checkpoint == "base" else _checkpoint_adapter_path(args.checkpoint_root, checkpoint, global_step)
    if adapter_path is not None and not adapter_path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter for {checkpoint}: {adapter_path}")

    llm = LLM(
        model=str(args.base_model_path),
        tokenizer=str(args.base_model_path),
        trust_remote_code=True,
        enable_lora=adapter_path is not None,
        max_lora_rank=8,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_tokens + args.max_new_tokens,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
    )
    lora_request = None
    if adapter_path is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(checkpoint, global_step or 1, str(adapter_path))

    for _, row in frame.iterrows():
        extra = _mapping(row["extra_info"])
        reward_model = _mapping(row["reward_model"])
        question_id = str(extra["question_id"])
        rendered = tokenizer.apply_chat_template(
            _messages(row["prompt"]), tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = tokenizer.encode(rendered, add_special_tokens=False)
        if len(prompt_tokens) > args.max_prompt_tokens:
            raise ValueError(f"Prompt {question_id} exceeds max_prompt_tokens={args.max_prompt_tokens}")
        for slot in range(args.rollouts_per_question):
            key = (checkpoint, question_id, slot)
            if key in completed:
                continue
            seed = args.seed_lookup.get(key, stable_seed(args.seed, question_id, checkpoint, slot))
            prompts.append(rendered)
            params.append(
                SamplingParams(
                    n=1,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=-1,
                    max_tokens=args.max_new_tokens,
                    seed=seed,
                )
            )
            metadata.append(
                {
                    "run_id": args.run_id,
                    "checkpoint": checkpoint,
                    "global_step": global_step,
                    "question_id": question_id,
                    "rollout_slot": slot,
                    "rollout_id": rollout_id(checkpoint, question_id, slot),
                    "rollout_seed": seed,
                    "prompt": rendered,
                    "prompt_token_count": len(prompt_tokens),
                    "ground_truth": reward_model["ground_truth"],
                    "level": extra.get("level"),
                    "subject": extra.get("subject"),
                }
            )
            if len(prompts) >= args.batch_rollouts:
                _write_batch(
                    llm, prompts, params, metadata, checkpoint_dir, tokenizer, args, lora_request
                )
                prompts, params, metadata = [], [], []
    if prompts:
        _write_batch(llm, prompts, params, metadata, checkpoint_dir, tokenizer, args, lora_request)


def _write_batch(llm: Any, prompts: list[str], params: list[Any], metadata: list[dict[str, Any]], checkpoint_dir: Path,
                 tokenizer: Any, args: argparse.Namespace, lora_request: Any) -> None:
    outputs = llm.generate(
        prompts,
        sampling_params=params,
        use_tqdm=True,
        lora_request=lora_request,
    )
    part_rows = []
    for meta, request_output in zip(metadata, outputs, strict=True):
        sample = request_output.outputs[0]
        response = sample.text
        reward = score_response(response, meta["ground_truth"])
        token_ids = list(sample.token_ids or tokenizer.encode(response, add_special_tokens=False))
        part_rows.append(
            {
                **meta,
                "response": response,
                "response_token_ids": token_ids,
                "response_token_count": len(token_ids),
                "finish_reason": sample.finish_reason,
                "stop_reason": str(sample.stop_reason) if sample.stop_reason is not None else None,
                "truncated": sample.finish_reason == "length",
                **reward,
            }
        )
    existing = [int(path.stem.split("_")[-1]) for path in checkpoint_dir.joinpath("parts").glob("part_*.jsonl")]
    part = max(existing, default=-1) + 1
    write_jsonl(checkpoint_dir / "parts" / f"part_{part:05d}.jsonl", part_rows)
    print(f"Wrote {checkpoint_dir.name} part={part} n={len(part_rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the frozen six-checkpoint roll8 evaluation cohort.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", choices=CHECKPOINT_NAMES, default=list(CHECKPOINT_NAMES))
    parser.add_argument("--global-steps", nargs="+", type=int, required=True)
    parser.add_argument("--rollouts-per-question", type=int, default=8)
    parser.add_argument("--batch-rollouts", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.global_steps):
        raise ValueError("--checkpoints and --global-steps must have equal lengths")
    args.seed_lookup = {}
    if args.seed_manifest is not None:
        seed_frame = pd.read_parquet(args.seed_manifest)
        required = {"checkpoint", "question_id", "rollout_slot", "seed"}
        if required - set(seed_frame.columns):
            raise ValueError("seed manifest is missing required columns")
        if seed_frame.duplicated(["checkpoint", "question_id", "rollout_slot"]).any():
            raise ValueError("seed manifest contains duplicate rollout keys")
        args.seed_lookup = {
            (str(row.checkpoint), str(row.question_id), int(row.rollout_slot)): int(row.seed)
            for row in seed_frame.itertuples(index=False)
        }
        eval_frame = pd.read_parquet(args.eval_file)
        question_ids = [str(_mapping(value)["question_id"]) for value in eval_frame["extra_info"]]
        expected = expected_rollout_keys(
            question_ids,
            checkpoints=tuple(args.checkpoints),
            rollouts_per_question=args.rollouts_per_question,
        )
        missing_seed_keys = expected - set(args.seed_lookup)
        if missing_seed_keys:
            raise ValueError(f"seed manifest is missing {len(missing_seed_keys)} requested rollout keys")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output_dir / "generation_config.json",
        {
            "run_id": args.run_id,
            "eval_file_sha256": sha256_file(args.eval_file),
            "checkpoints": args.checkpoints,
            "global_steps": args.global_steps,
            "rollouts_per_question": args.rollouts_per_question,
            "batch_rollouts": args.batch_rollouts,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "seed_manifest_sha256": sha256_file(args.seed_manifest) if args.seed_manifest else None,
        },
    )
    for checkpoint, step in zip(args.checkpoints, args.global_steps, strict=True):
        generate_checkpoint(args, checkpoint, step)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
