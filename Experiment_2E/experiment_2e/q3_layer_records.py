from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROLLOUTS_PER_QUESTION = 8


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    try:
        return dict(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest {field} must be a mapping") from exc


def render_manifest_prompt(row: pd.Series) -> str:
    messages = list(row["prompt"])
    if len(messages) != 1:
        raise ValueError("Q3 held-out manifest must contain one user message")
    message = _mapping(messages[0], field="prompt message")
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise ValueError("Q3 held-out manifest must contain one user message")
    return f"user\n{message['content']}\nassistant\n"


def load_q3_generation_records(
    generation_file: Path,
    heldout_file: Path,
    *,
    global_step: int,
    total_steps: int,
    question_limit: int | None = None,
) -> pd.DataFrame:
    if total_steps < 1 or not 0 <= global_step <= total_steps:
        raise ValueError("global_step must be in [0, total_steps]")
    manifest = pd.read_parquet(heldout_file)
    full_question_count = len(manifest)
    if question_limit is not None:
        if question_limit < 1:
            raise ValueError("question_limit must be positive")
        manifest = manifest.head(question_limit)

    with generation_file.open("r", encoding="utf-8") as handle:
        generated = [json.loads(line) for line in handle if line.strip()]
    expected = len(manifest) * ROLLOUTS_PER_QUESTION
    full_expected = full_question_count * ROLLOUTS_PER_QUESTION
    valid_count = len(generated) == full_expected if question_limit is not None else len(generated) == expected
    if not valid_count:
        target = full_expected if question_limit is not None else expected
        raise ValueError(f"expected {target} generation rows, found {len(generated)}")
    generated = generated[:expected]

    rows: list[dict[str, Any]] = []
    checkpoint = f"step{global_step:03d}"
    for question_index, manifest_row in manifest.reset_index(drop=True).iterrows():
        extra = _mapping(manifest_row["extra_info"], field="extra_info")
        reward_model = _mapping(manifest_row["reward_model"], field="reward_model")
        expected_prompt = render_manifest_prompt(manifest_row)
        expected_ground_truth = str(reward_model["ground_truth"])
        begin = question_index * ROLLOUTS_PER_QUESTION
        group = generated[begin : begin + ROLLOUTS_PER_QUESTION]
        for slot, generated_row in enumerate(group):
            if generated_row.get("input") != expected_prompt:
                raise ValueError(f"prompt mismatch question={question_index} slot={slot}")
            if int(generated_row.get("step", global_step)) != global_step:
                raise ValueError(f"generation step mismatch question={question_index} slot={slot}")
            ground_truth = str(generated_row.get("gts", expected_ground_truth))
            if ground_truth != expected_ground_truth:
                raise ValueError(f"ground truth mismatch question={question_index} slot={slot}")
            question_id = str(extra["question_id"])
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "global_step": int(global_step),
                    "training_progress": float(global_step / total_steps),
                    "question_id": question_id,
                    "rollout_id": f"{checkpoint}/{question_id}/r{slot:02d}",
                    "rollout_slot": slot,
                    "prompt": expected_prompt,
                    "response": str(generated_row["output"]),
                    "ground_truth": ground_truth,
                    "is_correct": bool(generated_row.get("acc", False)),
                    "answer_reward": float(
                        generated_row.get("reward", generated_row.get("score", 0.0))
                    ),
                    "format_correct": bool(generated_row.get("format_correct", False)),
                    "parse_correct": bool(generated_row.get("parse_correct", False)),
                    "difficulty": extra.get("difficulty"),
                    "prompt_hash": extra.get("prompt_hash"),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["question_id", "rollout_slot"]).any():
        raise ValueError("duplicate question/rollout slots")
    return frame
