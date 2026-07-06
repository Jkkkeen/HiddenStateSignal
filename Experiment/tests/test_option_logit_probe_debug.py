from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_option_logit_probe_debug_qwen3vl import (
    entropy_from_logits,
    logit_metrics,
    probe_char_positions,
)


def test_probe_char_positions_deduplicates_and_labels() -> None:
    probes = probe_char_positions(
        think_text="0123456789",
        answer_text="abcdef",
        think_fracs=[0.05, 0.5, 1.0],
        answer_fracs=[0.0, 0.5, 0.75],
    )

    assert [(p.segment, p.frac) for p in probes] == [
        ("think", 0.05),
        ("think", 0.5),
        ("think", 1.0),
        ("answer", 0.0),
        ("answer", 0.5),
        ("answer", 0.75),
    ]
    assert [p.char_end for p in probes[:3]] == [1, 5, 10]
    assert [p.char_end for p in probes[3:]] == [1, 3, 4]


def test_logit_metrics_computes_max_and_mean_margins() -> None:
    labels = ["A", "B", "C", "D"]
    logits = {"A": 1.0, "B": 4.0, "C": 2.0, "D": 3.0}

    metrics = logit_metrics(logits, labels, correct="B", chosen="D")

    assert metrics["top_option"] == "B"
    assert np.isclose(metrics["correct_margin_max"], 1.0)
    assert np.isclose(metrics["correct_margin_mean"], 2.0)
    assert np.isclose(metrics["chosen_margin_max"], -1.0)
    assert metrics["is_top_correct"] is True
    assert metrics["is_top_chosen"] is False


def test_entropy_from_logits_is_positive_and_lower_when_peaked() -> None:
    flat = entropy_from_logits(np.array([0.0, 0.0, 0.0, 0.0]))
    peaked = entropy_from_logits(np.array([10.0, 0.0, 0.0, 0.0]))

    assert flat > peaked
    assert peaked >= 0.0
