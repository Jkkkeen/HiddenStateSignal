from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_experiment_f_subset import select_question_ids
from analyze_training_step_dynamics import load_step_rollouts, summarize_step_metrics


def test_select_question_ids_prefers_mixed_and_is_deterministic() -> None:
    rows = []
    for qid, labels in {
        "q1": [True, False],
        "q2": [True, True],
        "q3": [False, True],
        "q4": [False, False],
    }.items():
        for rid, label in enumerate(labels):
            rows.append({"question_id": qid, "rollout_id": rid, "is_correct": label})

    selected_a = select_question_ids(rows, limit=2, seed=7, require_mixed=True)
    selected_b = select_question_ids(rows, limit=2, seed=7, require_mixed=True)

    assert selected_a == selected_b
    assert set(selected_a) == {"q1", "q3"}


def test_summarize_step_metrics_computes_raw_means_and_auc() -> None:
    frame = pd.DataFrame(
        {
            "step": [0, 0, 0, 0, 10, 10, 10, 10],
            "question_id": ["q1", "q1", "q2", "q2"] * 2,
            "rollout_id": [0, 1, 0, 1] * 2,
            "is_correct": [True, False, True, False] * 2,
            "path_score": [2.0, 1.0, 4.0, 3.0, 3.0, 1.0, 5.0, 2.0],
            "response_length": [100, 110, 120, 130, 90, 100, 95, 105],
        }
    )

    summary = summarize_step_metrics(
        frame,
        score_specs={"path_score": "pos"},
        behavior_cols=["response_length"],
        bootstrap=10,
        seed=2026,
    )

    auc0 = summary[(summary["step"] == 0) & (summary["metric"] == "path_score")]["mean_auc"].iloc[0]
    auc10 = summary[(summary["step"] == 10) & (summary["metric"] == "path_score")]["mean_auc"].iloc[0]
    correct_mean = summary[
        (summary["step"] == 10)
        & (summary["metric"] == "path_score")
        & (summary["group"] == "correct")
    ]["mean_value"].iloc[0]

    assert np.isclose(auc0, 1.0)
    assert np.isclose(auc10, 1.0)
    assert np.isclose(correct_mean, 4.0)


def test_load_step_rollouts_does_not_treat_unlabeled_rollouts_as_labels(tmp_path: Path) -> None:
    raw = tmp_path / "rollouts.jsonl"
    raw.write_text(
        '{"question_id":"q1","rollout_id":0,"response":"Answer: A"}\n',
        encoding="utf-8",
    )

    loaded = load_step_rollouts(tmp_path, step=0)

    assert loaded.empty
