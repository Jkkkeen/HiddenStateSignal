from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd

from .common import sha256_file, write_json_atomic
from .q3_layer_forward import teacher_forced_pooled_forward
from .q3_layer_records import load_q3_generation_records
from .q3_layer_extract import _load_model
from .h2_h10 import reduce_forward_output


def run(args: argparse.Namespace) -> dict[str, object]:
    records = load_q3_generation_records(
        args.generation_file,
        args.heldout_file,
        global_step=args.global_step,
        total_steps=args.total_steps,
        question_limit=args.question_limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"h2_h10_step{args.global_step:03d}.parquet"
    audit_path = args.output_dir / f"h2_h10_audit_step{args.global_step:03d}.json"
    if output_path.is_file() and audit_path.is_file():
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing.get("passed"):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return existing

    model, tokenizer = _load_model(args.model_path, args.tokenizer_path)
    started = time.time()
    rows: list[dict[str, object]] = []
    try:
        for index, record in enumerate(records.to_dict("records"), start=1):
            forward = teacher_forced_pooled_forward(
                model, tokenizer, str(record["prompt"]), str(record["response"])
            )
            metadata = {
                "run_id": args.run_id,
                "model": args.model_name,
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
                "response_length": int(forward["response_token_count"]),
                "trajectory_point_count": int(len(forward["endpoints"])),
                "n_hidden_states": int(forward["n_hidden_states"]),
            }
            rows.extend(reduce_forward_output(forward, metadata=metadata))
            if index == 1 or index % 64 == 0 or index == len(records):
                print(
                    f"checkpoint=step{args.global_step:03d} forward={index}/{len(records)}",
                    flush=True,
                )
            del forward
            gc.collect()
    finally:
        del model, tokenizer
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    frame = pd.DataFrame(rows)
    frame.to_parquet(output_path, index=False)
    coverage = frame.loc[frame["coverage_ok"].astype(bool)]
    total_rollouts = int(len(records))
    covered_rollouts = int(coverage["rollout_id"].nunique())
    coverage_rate = covered_rollouts / max(total_rollouts, 1)
    audit = {
        "passed": bool(
            len(frame) == len(records) * 2 * 4
            and covered_rollouts > 0
            and coverage["h2_cumulative"].notna().all()
        ),
        "checkpoint": f"step{args.global_step:03d}",
        "global_step": int(args.global_step),
        "n_rollouts": int(len(records)),
        "n_rows": int(len(frame)),
        "n_covered_rows": int(len(coverage)),
        "n_unique_rollouts": covered_rollouts,
        "coverage_rate": float(coverage_rate),
        "elapsed_seconds": float(time.time() - started),
        "seconds_per_rollout": float((time.time() - started) / max(len(records), 1)),
        "generation_sha256": sha256_file(args.generation_file),
        "heldout_sha256": sha256_file(args.heldout_file),
        "output_sha256": sha256_file(output_path),
    }
    write_json_atomic(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise RuntimeError(f"H2/H10 audit failed: {audit}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation-file", type=Path, required=True)
    parser.add_argument("--heldout-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-limit", type=int)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
