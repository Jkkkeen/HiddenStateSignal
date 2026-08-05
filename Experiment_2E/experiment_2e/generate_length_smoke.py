from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .common import sha256_file, write_json_atomic, write_jsonl
from .data_prep import stable_seed
from .math_reward import score_response


def _as_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _extract_prompt(row: pd.Series) -> list[dict[str, str]]:
    prompt = _as_python(row["prompt"])
    if isinstance(prompt, dict):
        prompt = [prompt]
    return [dict(message) for message in prompt]


def _extract_mapping(value: Any) -> dict[str, Any]:
    value = _as_python(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected mapping, received {type(value).__name__}")


def _completed_question_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    for part in sorted((output_dir / "parts").glob("part_*.jsonl")):
        counts: dict[str, int] = {}
        with part.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                counts[row["question_id"]] = counts.get(row["question_id"], 0) + 1
        completed.update(question_id for question_id, count in counts.items() if count == 4)
    return completed


def generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "parts").mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.eval_file).head(args.num_questions).copy()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    config = {
        "model_path": str(args.model_path),
        "eval_file": str(args.eval_file),
        "eval_file_sha256": sha256_file(args.eval_file),
        "num_questions": args.num_questions,
        "rollouts_per_question": 4,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "base_seed": args.seed,
    }
    write_json_atomic(args.output_dir / "generation_config.json", config)
    llm = LLM(
        model=str(args.model_path),
        tokenizer=str(args.model_path),
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_tokens + args.max_new_tokens,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
    )

    completed = _completed_question_ids(args.output_dir)
    pending = []
    for _, row in frame.iterrows():
        extra = _extract_mapping(row["extra_info"])
        if extra["question_id"] not in completed:
            pending.append(row)
    for batch_index, start in enumerate(range(0, len(pending), args.batch_questions)):
        rows = pending[start : start + args.batch_questions]
        prompts: list[str] = []
        params: list[SamplingParams] = []
        metadata: list[dict[str, Any]] = []
        for row in rows:
            extra = _extract_mapping(row["extra_info"])
            reward_model = _extract_mapping(row["reward_model"])
            messages = _extract_prompt(row)
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_tokens = tokenizer.encode(rendered, add_special_tokens=False)
            if len(prompt_tokens) > args.max_prompt_tokens:
                raise ValueError(f"Prompt {extra['question_id']} has {len(prompt_tokens)} tokens, above frozen limit")
            for rollout_slot in range(4):
                rollout_seed = stable_seed(args.seed, extra["question_id"], "base", rollout_slot)
                prompts.append(rendered)
                params.append(
                    SamplingParams(
                        n=1,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=-1,
                        max_tokens=args.max_new_tokens,
                        seed=rollout_seed,
                    )
                )
                metadata.append(
                    {
                        "question_id": extra["question_id"],
                        "rollout_slot": rollout_slot,
                        "rollout_seed": rollout_seed,
                        "prompt_token_count": len(prompt_tokens),
                        "ground_truth": reward_model["ground_truth"],
                    }
                )
        outputs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
        part_rows = []
        for meta, request_output in zip(metadata, outputs, strict=True):
            sample = request_output.outputs[0]
            response = sample.text
            reward = score_response(response, meta["ground_truth"])
            part_rows.append(
                {
                    **meta,
                    "response": response,
                    "token_count": len(sample.token_ids),
                    "finish_reason": sample.finish_reason,
                    "stop_reason": str(sample.stop_reason) if sample.stop_reason is not None else None,
                    "truncated": sample.finish_reason == "length",
                    **reward,
                }
            )
        existing_indices = [int(path.stem.split("_")[-1]) for path in (args.output_dir / "parts").glob("part_*.jsonl")]
        part_number = max(existing_indices, default=-1) + 1
        write_jsonl(args.output_dir / "parts" / f"part_{part_number:05d}.jsonl", part_rows)
        print(f"Wrote batch {batch_index + 1}: {len(part_rows)} rollouts", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 100 x 4 base-model length-only smoke.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--batch-questions", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    generate(args)


if __name__ == "__main__":
    main()
