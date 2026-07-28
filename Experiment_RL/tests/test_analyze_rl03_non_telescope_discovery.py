from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_rl03_non_telescope_discovery import (
    PRIMARY_FRACTIONS,
    analyze_features,
    build_non_telescope_features,
)


def _probe_rows(question_id: str, rollout_id: int, is_correct: bool, scores: list[float]) -> list[dict[str, object]]:
    return [
        {
            "question_id": question_id,
            "rollout_id": rollout_id,
            "is_correct": is_correct,
            "probe_kind": "think",
            "frac": float(frac),
            "interface_id": "I2",
            "representation": "content_sequence",
            "prompt_mode": "neutralized_letter_instruction",
            "gold_level": float(score),
        }
        for frac, score in zip(PRIMARY_FRACTIONS, scores, strict=True)
    ]


def test_build_non_telescope_features_handles_monotonic_and_reversing_paths() -> None:
    monotonic = np.arange(len(PRIMARY_FRACTIONS), dtype=float).tolist()
    reversing = [0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0]
    probes = pd.DataFrame(
        _probe_rows("q1", 0, True, monotonic)
        + _probe_rows("q1", 1, False, reversing)
    )

    features = build_non_telescope_features(probes).sort_values("rollout_id")
    clean = features.iloc[0]
    unstable = features.iloc[1]

    assert clean["excess_total_variation"] == 0.0
    assert clean["maximum_drawdown"] == 0.0
    assert clean["primary_score"] == 0.0
    assert unstable["excess_total_variation"] > 0.0
    assert unstable["maximum_drawdown"] == 1.0
    assert unstable["primary_score"] < clean["primary_score"]


def test_discovery_gate_stops_when_primary_feature_is_uninformative() -> None:
    rows = []
    for question_index in range(4):
        for rollout_id, is_correct in enumerate((False, True)):
            rows.append(
                {
                    "question_id": f"q{question_index}",
                    "rollout_id": rollout_id,
                    "is_correct": is_correct,
                    "gold_level_90": float(is_correct),
                    "primary_score": 0.0,
                    "negative_max_drawdown": 0.0,
                    "negative_sign_reversal": 0.0,
                    "negative_curvature": 0.0,
                }
            )
    summary = analyze_features(pd.DataFrame(rows), bootstrap_samples=20, seed=9)

    assert summary["decision"] == {
        "label": "STOP-NO-CONFIRMATION",
        "authorize_new_h200_confirmation": False,
    }
