from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from semantic_step_basin import (
    compute_step_basin_metrics,
    split_semantic_steps,
    summarize_rollout_basin_features,
)


def test_split_semantic_steps_uses_discourse_and_rethink_boundaries() -> None:
    text = (
        "First, compute angle 1. Then compare it with the diagram.\n"
        "Wait, actually angle 2 is supplementary. Therefore answer A."
    )

    steps = split_semantic_steps(text, min_chars=8)

    assert [step["text"] for step in steps] == [
        "First, compute angle 1.",
        "Then compare it with the diagram.",
        "Wait, actually angle 2 is supplementary.",
        "Therefore answer A.",
    ]
    assert [step["start_char"] for step in steps] == sorted(step["start_char"] for step in steps)
    assert steps[2]["has_rethink_trigger"] is True


def test_compute_step_basin_metrics_measures_correct_margin_and_turns() -> None:
    states = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    option_vectors = {
        "A": np.asarray([1.0, 0.0], dtype=np.float32),
        "B": np.asarray([0.0, 1.0], dtype=np.float32),
    }

    rows = compute_step_basin_metrics(
        states,
        option_vectors,
        correct_label="A",
        chosen_label="B",
        rethink_flags=[False, False, True],
    )

    assert len(rows) == 3
    assert math.isclose(rows[0]["state_margin"], 1.0, abs_tol=1e-6)
    assert math.isclose(rows[0]["step_correct_gain"], 1.0, abs_tol=1e-6)
    assert math.isclose(rows[0]["turn_to_correct"], 1.0, abs_tol=1e-6)

    assert math.isclose(rows[1]["turn_angle"], 0.0, abs_tol=1e-6)
    assert math.isclose(rows[1]["productive_turn"], 0.0, abs_tol=1e-6)

    assert rows[2]["turn_to_correct"] < -0.99
    assert math.isclose(rows[2]["turn_angle"], math.pi / 2.0, abs_tol=1e-6)
    assert rows[2]["productive_turn"] == 0.0
    assert rows[2]["has_rethink_trigger"] is True
    assert rows[2]["chosen_wrong_margin"] < 0.0


def test_productive_turn_is_large_only_when_turn_improves_correct_margin() -> None:
    states = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    option_vectors = {
        "A": np.asarray([1.0, 0.0], dtype=np.float32),
        "B": np.asarray([0.0, 1.0], dtype=np.float32),
    }

    rows = compute_step_basin_metrics(states, option_vectors, correct_label="A")

    assert rows[0]["state_margin"] < 0.0
    assert rows[1]["step_correct_gain"] > 0.9
    assert math.isclose(rows[1]["turn_angle"], math.pi / 2.0, abs_tol=1e-6)
    assert rows[1]["productive_turn"] > 1.4


def test_summarize_rollout_basin_features_aggregates_late_and_rethink_signals() -> None:
    rows = [
        {
            "step_correct_gain": -1.0,
            "state_margin": -1.0,
            "turn_to_correct": -1.0,
            "productive_turn": 0.0,
            "chosen_wrong_margin": 1.0,
            "has_rethink_trigger": False,
        },
        {
            "step_correct_gain": 1.0,
            "state_margin": 0.0,
            "turn_to_correct": 1.0,
            "productive_turn": math.pi / 2.0,
            "chosen_wrong_margin": 0.0,
            "has_rethink_trigger": True,
        },
        {
            "step_correct_gain": 0.5,
            "state_margin": 0.5,
            "turn_to_correct": 0.5,
            "productive_turn": 0.1,
            "chosen_wrong_margin": -0.5,
            "has_rethink_trigger": False,
        },
        {
            "step_correct_gain": 0.25,
            "state_margin": 0.75,
            "turn_to_correct": 0.25,
            "productive_turn": 0.0,
            "chosen_wrong_margin": -0.75,
            "has_rethink_trigger": False,
        },
    ]

    summary = summarize_rollout_basin_features(rows)

    assert math.isclose(summary["mean_step_correct_gain"], 0.1875, abs_tol=1e-6)
    assert math.isclose(summary["positive_gain_rate"], 0.75, abs_tol=1e-6)
    assert math.isclose(summary["late_margin_mean"], 0.75, abs_tol=1e-6)
    assert summary["early_to_late_margin_gain"] > 1.0
    assert summary["first_cross_pos"] == 0.5
    assert summary["rethink_step_count"] == 1
    assert math.isclose(summary["rethink_gain_mean"], 1.0, abs_tol=1e-6)
