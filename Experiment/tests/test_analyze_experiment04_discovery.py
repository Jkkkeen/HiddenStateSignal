from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_experiment04_discovery import (  # noqa: E402
    cluster_components,
    label_permutation_cluster_test,
    pairwise_auc,
    question_equal_curve,
)


def test_pairwise_auc_handles_ties() -> None:
    assert pairwise_auc(np.asarray([2.0, 1.0]), np.asarray([1.0, 0.0])) == 0.875


def test_question_equal_curve_does_not_weight_extra_rollouts() -> None:
    frame = pd.DataFrame(
        [
            {"question_id": "q1", "is_correct": True, "x": 0, "score": 10.0},
            {"question_id": "q1", "is_correct": False, "x": 0, "score": 0.0},
            {"question_id": "q2", "is_correct": True, "x": 0, "score": 2.0},
            {"question_id": "q2", "is_correct": True, "x": 0, "score": 2.0},
            {"question_id": "q2", "is_correct": True, "x": 0, "score": 2.0},
            {"question_id": "q2", "is_correct": False, "x": 0, "score": 0.0},
        ]
    )
    curve = question_equal_curve(frame, "score", "x", bootstrap=200, seed=7)

    correct = curve.loc[curve["is_correct"], "mean"].item()
    assert np.isclose(correct, 6.0)
    assert curve["questions"].min() == 2


def test_cluster_components_use_four_neighbor_connectivity() -> None:
    statistic = np.asarray(
        [
            [3.0, 3.0, 0.0],
            [0.0, 3.0, -3.0],
            [0.0, 0.0, -3.0],
        ]
    )
    clusters = cluster_components(statistic, threshold=2.0)

    masses = sorted(round(cluster["mass"], 6) for cluster in clusters)
    assert masses == [6.0, 9.0]


def test_label_permutation_cluster_is_deterministic() -> None:
    rows = []
    for question_id in ("q1", "q2", "q3"):
        for rollout_id, label in enumerate((True, True, False, False)):
            for layer in (0, 1):
                for progress_bin in (0, 1):
                    rows.append(
                        {
                            "question_id": question_id,
                            "rollout_id": rollout_id,
                            "is_correct": label,
                            "representation": "mean",
                            "layer": layer,
                            "progress_bin": progress_bin,
                            "score": float(label) + 0.1 * rollout_id,
                        }
                    )
    frame = pd.DataFrame(rows)
    first = label_permutation_cluster_test(
        frame, "score", "mean", "progress_bin", permutations=20, seed=3
    )
    second = label_permutation_cluster_test(
        frame, "score", "mean", "progress_bin", permutations=20, seed=3
    )

    assert np.array_equal(first[1], second[1])
    assert first[2:] == ([0, 1], [0, 1])
