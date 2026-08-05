from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import canonical_json, sha256_file, sha256_text, write_json_atomic
from .data_prep import CHECKPOINT_NAMES, stable_seed


CHECKPOINT_PROGRESS = {
    "base": 0.0,
    "20pct": 0.2,
    "40pct": 0.4,
    "60pct": 0.6,
    "80pct": 0.8,
    "final": 1.0,
}
HORIZONTAL_ANCHORS = (3, 12, 24, "final")
STAGE_BOUNDS = ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0))
REPRESENTATIVES = {
    "H1": "median_relative_movement",
    "H2": "straightness",
    "H3": "median_turn_angle",
    "H4": "p90_abs_angular_velocity",
    "H5": "centered_ERV",
    "H6": "directional_ER",
    "H7": "weighted_turning",
    "H8": "median_token_time_diff_coordinate_entropy",
    "V1": "p90_relative_layer_update_norm",
    "V2": "median_raw_state_angle",
    "V3": "median_state_angle_demean",
    "V4": "median_layer_update_turning_angle",
    "V5": "vertical_straightness",
    "V6": "median_raw_activation_entropy",
    "V7": "median_layer_difference_entropy",
    "V8": "layer_update_ER",
}


def checkpoint_schedule(total_training_steps: int) -> list[dict[str, Any]]:
    """Return base plus five equally spaced frozen checkpoints.

    veRL's launcher uses one save frequency, so an exactly reproducible
    20/40/60/80/final schedule requires five equal integer intervals.
    """
    if total_training_steps < 5 or total_training_steps % 5:
        raise ValueError("total_training_steps must be a positive multiple of 5")
    interval = total_training_steps // 5
    rows = [{"checkpoint": "base", "training_progress": 0.0, "global_step": 0}]
    for index, checkpoint in enumerate(CHECKPOINT_NAMES[1:], start=1):
        rows.append(
            {
                "checkpoint": checkpoint,
                "training_progress": CHECKPOINT_PROGRESS[checkpoint],
                "global_step": interval * index,
            }
        )
    return rows


def _extract_extra(value: Any) -> dict[str, Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, dict):
        raise TypeError(f"extra_info must be a mapping, received {type(value).__name__}")
    return value


def build_seed_manifest(
    eval_frame: pd.DataFrame,
    *,
    base_seed: int,
    rollouts_per_question: int = 8,
) -> pd.DataFrame:
    if rollouts_per_question < 1:
        raise ValueError("rollouts_per_question must be positive")
    question_ids = [_extract_extra(value)["question_id"] for value in eval_frame["extra_info"]]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("evaluation question_id values must be unique")
    rows = []
    for checkpoint in CHECKPOINT_NAMES:
        for question_id in question_ids:
            for rollout_slot in range(rollouts_per_question):
                rows.append(
                    {
                        "question_id": question_id,
                        "checkpoint": checkpoint,
                        "rollout_slot": rollout_slot,
                        "seed": stable_seed(base_seed, question_id, checkpoint, rollout_slot),
                    }
                )
    return pd.DataFrame(rows)


def frozen_analysis_spec(*, source_length_spec: Path, rollouts_per_question: int) -> dict[str, Any]:
    length_spec = json.loads(source_length_spec.read_text(encoding="utf-8"))
    if (length_spec.get("window"), length_spec.get("stride")) != (128, 32):
        raise ValueError("source length spec must freeze window=128 and stride=32")
    primary_tests = 8 + 1 + 7 * 4
    return {
        "schema_version": 2,
        "decision_source": "Experiment_2E RL_discovery_plan.md; updated before formal hidden extraction",
        "source_length_spec_sha256": sha256_file(source_length_spec),
        "window": 128,
        "stride": 32,
        "representations": ["mean_w128_s32", "last_s32", "token"],
        "primary_representation": "mean_w128_s32",
        "sensitivity_representation": "last_s32",
        "horizontal_anchors": list(HORIZONTAL_ANCHORS),
        "stages": [list(bounds) for bounds in STAGE_BOUNDS],
        "stage_boundary_rule": "left-closed/right-open except final stage right-closed",
        "rollouts_per_question": rollouts_per_question,
        "representatives": REPRESENTATIVES,
        "primary_aggregation_modes": {
            "H1": "local",
            "H2": "local",
            "H3": "local",
            "H4": "local",
            "H5": "cumulative",
            "H6": "cumulative",
            "H7": "local",
            "H8": "local",
            "V1": "local",
            "V2": "local",
            "V3": "local",
            "V4": "local",
            "V5": "local",
            "V6": "local",
            "V7": "local",
            "V8": "local",
        },
        "n_primary_training_stage_tests": primary_tests,
        "multiple_testing": {"method": "benjamini-hochberg", "q": 0.10},
        "max_new_tokens": 1536,
        "exclude_truncated_from_primary": True,
    }


def freeze_formal_run(
    *,
    eval_file: Path,
    train_file: Path,
    source_length_spec: Path,
    output_dir: Path,
    run_id: str,
    total_training_steps: int,
    base_seed: int = 20260805,
    rollouts_per_question: int = 8,
    expected_eval_questions: int = 256,
) -> dict[str, Any]:
    schedule = checkpoint_schedule(total_training_steps)
    eval_frame = pd.read_parquet(eval_file)
    if len(eval_frame) != expected_eval_questions:
        raise ValueError(f"expected {expected_eval_questions} evaluation questions, found {len(eval_frame)}")
    seed_frame = build_seed_manifest(
        eval_frame,
        base_seed=base_seed,
        rollouts_per_question=rollouts_per_question,
    )
    expected_rollouts = len(CHECKPOINT_NAMES) * expected_eval_questions * rollouts_per_question
    if len(seed_frame) != expected_rollouts:
        raise AssertionError("rollout seed manifest size is inconsistent")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "rollout_seed_manifest.parquet"
    seed_frame.to_parquet(seed_path, index=False)
    checkpoint_path = output_dir / "checkpoint_manifest.json"
    write_json_atomic(checkpoint_path, {"run_id": run_id, "checkpoints": schedule})
    analysis_spec = frozen_analysis_spec(
        source_length_spec=source_length_spec,
        rollouts_per_question=rollouts_per_question,
    )
    analysis_path = output_dir / "frozen_analysis_spec.json"
    write_json_atomic(analysis_path, analysis_spec)

    run_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "base_seed": base_seed,
        "total_training_steps": total_training_steps,
        "save_frequency": total_training_steps // 5,
        "train_batch_size": 1,
        "grpo_group_size": 8,
        "rollouts_per_eval_question": rollouts_per_question,
        "n_eval_questions": expected_eval_questions,
        "n_evaluation_rollouts": expected_rollouts,
        "max_prompt_tokens": 2048,
        "max_new_tokens": 1536,
        "temperature": 1.0,
        "top_p": 1.0,
        "train_file": str(train_file),
        "train_file_sha256": sha256_file(train_file),
        "eval_file": str(eval_file),
        "eval_file_sha256": sha256_file(eval_file),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
        "rollout_seed_manifest_sha256": sha256_file(seed_path),
        "frozen_analysis_spec_sha256": sha256_file(analysis_path),
    }
    run_manifest["manifest_id"] = sha256_text(canonical_json(run_manifest))
    run_path = output_dir / "formal_run_manifest.json"
    write_json_atomic(run_path, run_manifest)
    (output_dir / "formal_run_manifest.sha256").write_text(
        f"{sha256_file(run_path)}  {run_path.name}\n", encoding="ascii"
    )
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the formal Experiment 2E run and roll8 manifests.")
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--source-length-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--total-training-steps", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=20260805)
    parser.add_argument("--rollouts-per-question", type=int, default=8)
    parser.add_argument("--expected-eval-questions", type=int, default=256)
    args = parser.parse_args()
    manifest = freeze_formal_run(
        eval_file=args.eval_file,
        train_file=args.train_file,
        source_length_spec=args.source_length_spec,
        output_dir=args.output_dir,
        run_id=args.run_id,
        total_training_steps=args.total_training_steps,
        base_seed=args.base_seed,
        rollouts_per_question=args.rollouts_per_question,
        expected_eval_questions=args.expected_eval_questions,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
