from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_experiment04 import prepare  # noqa: E402


def _rows(question_id: str, correct: int, wrong: int, length: int) -> list[dict]:
    rows = []
    for rollout_id in range(correct + wrong):
        rows.append(
            {
                "question_id": question_id,
                "rollout_id": rollout_id,
                "is_correct": rollout_id < correct,
                "think_token_count": length + rollout_id,
                "response": "<think>x</think>",
            }
        )
    return rows


def test_prepare_freezes_disjoint_question_split_and_longest_smoke(tmp_path: Path) -> None:
    input_path = tmp_path / "rollouts.jsonl"
    rows = (
        _rows("ineligible", 1, 3, 10)
        + _rows("q1", 2, 2, 20)
        + _rows("q2", 3, 3, 30)
        + _rows("q3", 4, 4, 100)
    )
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = prepare(input_path, tmp_path / "manifest", seed=20260726)

    assert summary["eligible_questions"] == 3
    assert summary["strict_3plus3_questions"] == 2
    assert summary["discovery_questions"] == 1
    assert summary["confirm_questions"] == 2
    discovery = {
        json.loads(line)["question_id"]
        for line in (tmp_path / "manifest" / "manifest_discovery.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    }
    confirm = {
        json.loads(line)["question_id"]
        for line in (tmp_path / "manifest" / "manifest_confirm.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    }
    assert discovery.isdisjoint(confirm)
    assert discovery | confirm == {"q1", "q2", "q3"}
    smoke_rows = [
        json.loads(line)
        for line in (tmp_path / "manifest" / "manifest_smoke.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {row["question_id"] for row in smoke_rows} <= discovery
    assert len(smoke_rows) in {4, 6, 8}
    assert all(row["experiment04_split"] == "discovery" for row in smoke_rows)


def test_prepare_is_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "rollouts.jsonl"
    rows = _rows("q1", 2, 2, 20) + _rows("q2", 2, 2, 30)
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    first = prepare(input_path, tmp_path / "first", seed=7)
    second = prepare(input_path, tmp_path / "second", seed=7)

    assert first["ordered_question_ids_sha256"] == second["ordered_question_ids_sha256"]
    assert (tmp_path / "first" / "manifest_discovery.jsonl").read_bytes() == (
        tmp_path / "second" / "manifest_discovery.jsonl"
    ).read_bytes()
