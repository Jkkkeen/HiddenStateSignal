from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Experiment_RL" / "scripts"))

from rl03_mcq_audit import (
    DENSE_THINK_FRACS,
    INTERFACES,
    aggregate_probe_scores,
    build_manifest,
    build_scoring_requests,
    neutralize_letter_instruction,
    probe_prefixes,
    summarize_step_trajectory,
)


def _rollout(question_id: str, rollout_id: int, correct: bool) -> dict:
    return {
        "question_id": question_id,
        "rollout_id": rollout_id,
        "answer": "B",
        "pred_answer": "B" if correct else "A",
        "is_correct": correct,
        "prompt": (
            "Please first conduct reasoning, and then answer the question and provide "
            "the correct option letter, e.g., A, B, C, D, at the end.\n"
            "Question: Pick the equal value.\nChoices:\nA:1/3\nB:1/2\nC:2/3\nD:3/4"
        ),
        "image_path": "image.png",
        "response": "first establish the denominator then compare the fractions</think>\nB",
        "truncated": False,
        "has_think_close": True,
    }


def test_build_manifest_selects_deterministic_mixed_question_groups(tmp_path: Path) -> None:
    raw = tmp_path / "rollouts.jsonl"
    rows = [
        _rollout(f"q{question}", rollout, correct=rollout % 2 == 0)
        for question in range(6)
        for rollout in range(4)
    ]
    raw.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest, summary = build_manifest(raw, question_count=3, seed=17)

    assert summary["eligible_mixed_questions"] == 6
    assert summary["selected_questions"] == 3
    assert summary["selected_rollouts"] == 12
    assert manifest.groupby("question_id")["is_correct"].nunique().eq(2).all()
    assert manifest["question_id"].drop_duplicates().tolist() == summary["chosen_question_ids"]


def test_neutral_prompt_removes_only_the_letter_output_instruction() -> None:
    prompt = _rollout("q1", 0, True)["prompt"]

    neutral = neutralize_letter_instruction(prompt)

    assert neutral.startswith("Question: Pick the equal value.")
    assert "provide the correct option letter" not in neutral
    assert "Choices:\nA:1/3\nB:1/2" in neutral


def test_probe_prefixes_include_dense_25_percent_and_keep_trimmed_separate() -> None:
    thinking = "abcdefghij" * 20

    probes = probe_prefixes(thinking)

    numeric = [row for row in probes if row["probe_kind"] in {"prompt", "think"}]
    assert [row["frac"] for row in numeric] == [0.0, *DENSE_THINK_FRACS]
    assert 0.25 in [row["frac"] for row in numeric]
    assert numeric[0]["prefix_text"] == ""
    assert numeric[-1]["prefix_text"] == thinking
    assert probes[-1]["probe_kind"] == "trimmed"
    assert probes[-1]["is_adjacent"] is False


def test_build_scoring_requests_controls_interface_and_representation(tmp_path: Path) -> None:
    record = _rollout("q1", 7, False)
    manifest = pd.DataFrame([record])
    metadata = {
        "q1": {
            "question_for_eval": (
                "Pick the equal value.\nChoices:\nA:1/3\nB:1/2\nC:2/3\nD:3/4"
            )
        }
    }

    requests = build_scoring_requests(
        manifest,
        project_root=tmp_path,
        metadata_by_question=metadata,
        interface_names=("I0", "I1"),
        probe_mode="sparse",
    )

    i0_letter = next(
        row for row in requests
        if row["interface"] == "I0" and row["representation"] == "letter"
    )
    i1_content = next(
        row for row in requests
        if row["interface"] == "I1" and row["representation"] == "content"
    )
    assert i0_letter["target_text"] == "B"
    assert i0_letter["prompt_mode"] == "original"
    assert i0_letter["cue"] == INTERFACES["I0"]
    assert i1_content["target_text"] == "1/2"
    assert i1_content["prompt_mode"] == "neutral"
    assert "provide the correct option letter" not in i1_content["messages"][0]["content"][-1]["text"]


def test_aggregate_probe_scores_takes_median_after_per_surface_difference() -> None:
    scored = pd.DataFrame(
        [
            {"question_id": "q1", "rollout_id": 1, "interface": "I1", "representation": "content", "surface": "canonical", "probe_kind": "think", "frac": 0.25, "score_mean": -4.0},
            {"question_id": "q1", "rollout_id": 1, "interface": "I1", "representation": "content", "surface": "canonical", "probe_kind": "think", "frac": 0.90, "score_mean": -1.0},
            {"question_id": "q1", "rollout_id": 1, "interface": "I1", "representation": "content", "surface": "alternate", "probe_kind": "think", "frac": 0.25, "score_mean": -1.0},
            {"question_id": "q1", "rollout_id": 1, "interface": "I1", "representation": "content", "surface": "alternate", "probe_kind": "think", "frac": 0.90, "score_mean": -3.0},
        ]
    )

    row = aggregate_probe_scores(scored).iloc[0]

    assert math.isclose(row["gold_level_25"], -2.5)
    assert math.isclose(row["gold_level_90"], -2.0)
    assert math.isclose(row["gold_gain_25_90"], 0.5)


def test_summarize_step_trajectory_normalizes_unequal_intervals_and_excludes_trimmed() -> None:
    levels = pd.DataFrame(
        [
            {"question_id": "q1", "rollout_id": 1, "is_correct": True, "probe_kind": "prompt", "frac": 0.0, "gold_level": -4.0},
            {"question_id": "q1", "rollout_id": 1, "is_correct": True, "probe_kind": "think", "frac": 0.25, "gold_level": -3.0},
            {"question_id": "q1", "rollout_id": 1, "is_correct": True, "probe_kind": "think", "frac": 0.50, "gold_level": -2.0},
            {"question_id": "q1", "rollout_id": 1, "is_correct": True, "probe_kind": "think", "frac": 0.90, "gold_level": -1.0},
            {"question_id": "q1", "rollout_id": 1, "is_correct": True, "probe_kind": "trimmed", "frac": 1.0, "gold_level": -1.5},
        ]
    )

    steps = summarize_step_trajectory(levels)

    assert steps["frac_end"].tolist() == [0.25, 0.50, 0.90]
    assert steps["step_gain"].tolist() == [1.0, 1.0, 1.0]
    assert steps["step_gain_rate"].tolist() == [4.0, 4.0, 2.5]
    assert "trimmed" not in set(steps["probe_kind_end"])
