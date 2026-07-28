from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Experiment_RL" / "scripts"))

from build_recoverability_manifest import (
    build_manifest,
    explicit_answer_claims,
    evenly_spaced_rows,
)


def test_evenly_spaced_rows_cover_low_and_high_signal() -> None:
    frame = pd.DataFrame(
        {
            "rollout_id": list(range(8)),
            "trimmed_margin_mean": [8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0],
        }
    )

    selected = evenly_spaced_rows(frame, count=4, feature="trimmed_margin_mean")

    assert selected["trimmed_margin_mean"].tolist() == [1.0, 3.0, 6.0, 8.0]
    assert selected["signal_rank"].tolist() == [0, 1, 2, 3]


def test_explicit_answer_claims_detects_answer_leakage() -> None:
    assert explicit_answer_claims("After checking, the answer is C.") == ["C"]
    assert explicit_answer_claims("So the answer should be B:25 degrees.") == ["B"]
    assert explicit_answer_claims("The final answer would be D.") == ["D"]
    assert explicit_answer_claims("Option B is the correct answer.") == ["B"]
    assert explicit_answer_claims("Compare choices A and B before calculating.") == []


def test_build_manifest_selects_wrong_rollouts_with_question_groups(tmp_path: Path) -> None:
    raw_path = tmp_path / "rollouts.jsonl"
    trimmed_path = tmp_path / "trimmed.parquet"
    output_path = tmp_path / "manifest.jsonl"
    summary_path = tmp_path / "summary.json"

    raw_rows = []
    feature_rows = []
    for question_id in ("q1", "q2", "q3"):
        for rollout_id in range(5):
            raw_rows.append(
                {
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "answer": "A",
                    "pred_answer": "B",
                    "is_correct": False,
                    "prompt": f"Question {question_id}",
                    "image_path": "image.png",
                    "response": (
                        "First inspect the diagram. Then compute the angle carefully. "
                        f"Intermediate value {rollout_id}. Therefore, the answer is B."
                    ),
                }
            )
            feature_rows.append(
                {
                    "question_id": question_id,
                    "rollout_id": rollout_id,
                    "source_line": rollout_id,
                    "is_correct": False,
                    "answer": "A",
                    "pred_answer": "B",
                    "trim_status": "trigger",
                    "trimmed_margin_mean": float(rollout_id),
                    "trimmed_margin_max": float(rollout_id) - 0.5,
                    "trimmed_entropy": 0.5,
                }
            )

    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8"
    )
    pd.DataFrame(feature_rows).to_parquet(trimmed_path, index=False)

    summary = build_manifest(
        raw_path=raw_path,
        trimmed_features_path=trimmed_path,
        trajectory_features_path=None,
        output_path=output_path,
        summary_path=summary_path,
        question_count=2,
        errors_per_question=4,
        seed=7,
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 8
    assert len({row["question_id"] for row in records}) == 2
    assert all(row["is_correct"] is False for row in records)
    assert all(row["split_role"] == "discovery_smoke" for row in records)
    assert all("Therefore, the answer" not in row["revision_prefix"] for row in records)
    assert all(row["answer_leakage_flag"] is False for row in records)
    assert summary["selected_rollouts"] == 8
    assert summary["selected_questions"] == 2
    assert summary["answer_leakage_count"] == 0
