from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_entropy_band_confirm120 import (  # noqa: E402
    freeze_candidate_extension,
    freeze_candidates,
    run_extend,
    select_formal_cohort,
)


def _record(question_id: str, answer: str = "A") -> dict:
    return {
        "sample_index": question_id,
        "answer": answer,
        "query_wo": f"Question {question_id}",
    }


def _rollout(question_id: str, rollout_id: int, correct: bool) -> dict:
    return {
        "question_id": question_id,
        "rollout_id": rollout_id,
        "answer": "A",
        "pred_answer": "A" if correct else "B",
        "is_correct": correct,
        "truncated": False,
        "has_think_close": True,
        "segment_status": "ok",
    }


def test_freeze_candidates_excludes_old_and_non_mcq_deterministically() -> None:
    records = [_record(f"q{i}") for i in range(10)]
    records.extend([_record("free_response", "12.5"), _record("old")])

    first = freeze_candidates(
        records,
        excluded_question_ids={"old"},
        candidate_count=8,
        seed=20260725,
    )
    second = freeze_candidates(
        list(reversed(records)),
        excluded_question_ids={"old"},
        candidate_count=8,
        seed=20260725,
    )

    assert first == second
    assert len(first) == 8
    assert "old" not in first
    assert "free_response" not in first


def test_freeze_candidate_extension_preserves_prefix_and_takes_next_ranks() -> None:
    records = [_record(f"q{i:02d}") for i in range(20)]
    existing = freeze_candidates(
        records,
        excluded_question_ids=set(),
        candidate_count=8,
        seed=20260725,
    )

    combined, extension = freeze_candidate_extension(
        records,
        excluded_question_ids=set(),
        existing_candidate_ids=existing,
        extension_count=5,
        seed=20260725,
    )

    assert combined[:8] == existing
    assert extension == combined[8:13]
    assert len(extension) == 5
    assert not set(existing).intersection(extension)


def test_freeze_candidate_extension_rejects_changed_existing_prefix() -> None:
    records = [_record(f"q{i:02d}") for i in range(20)]
    existing = freeze_candidates(
        records,
        excluded_question_ids=set(),
        candidate_count=8,
        seed=20260725,
    )
    existing[0], existing[1] = existing[1], existing[0]

    with pytest.raises(ValueError, match="existing candidate prefix does not match"):
        freeze_candidate_extension(
            records,
            excluded_question_ids=set(),
            existing_candidate_ids=existing,
            extension_count=5,
            seed=20260725,
        )


def test_run_extend_writes_combined_extension_batches_and_audit(tmp_path: Path) -> None:
    mathverse = tmp_path / "testmini.json"
    excluded = tmp_path / "old.jsonl"
    existing = tmp_path / "candidate_ids_all8.txt"
    output = tmp_path / "extension"
    records = [_record(f"q{i:02d}") for i in range(20)]
    mathverse.write_text(json.dumps(records), encoding="utf-8")
    excluded.write_text("", encoding="utf-8")
    frozen = freeze_candidates(
        records,
        excluded_question_ids=set(),
        candidate_count=8,
        seed=20260725,
    )
    existing.write_text("".join(f"{qid}\n" for qid in frozen), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "mathverse_json": str(mathverse),
            "exclude_rollouts": str(excluded),
            "existing_candidate_ids": str(existing),
            "output_dir": str(output),
            "extension_count": 6,
            "batch_size": 2,
            "seed": 20260725,
        },
    )()

    run_extend(args)

    assert len((output / "candidate_ids_all14.txt").read_text().splitlines()) == 14
    assert len((output / "candidate_ids_extension6.txt").read_text().splitlines()) == 6
    assert len(list(output.glob("candidate_ids_extension2_*.txt"))) == 3
    audit = json.loads((output / "candidate_extension_audit.json").read_text())
    assert audit["existing_count"] == 8
    assert audit["extension_count"] == 6
    assert audit["entropy_used_for_selection"] is False


def test_select_formal_cohort_uses_frozen_order_and_clean_two_by_two() -> None:
    order = ["q_bad", "q2", "q1", "q3"]
    rows = []
    rows.extend(_rollout("q_bad", rid, label) for rid, label in enumerate([1, 0, 0]))
    rows.extend(_rollout("q2", rid, label) for rid, label in enumerate([1, 1, 0, 0]))
    rows.extend(_rollout("q1", rid, label) for rid, label in enumerate([1, 1, 1, 0, 0, 0]))
    rows.extend(_rollout("q3", rid, label) for rid, label in enumerate([1, 1, 0, 0]))
    rows[-1]["truncated"] = True

    selected = select_formal_cohort(rows, order, target_questions=2)

    assert selected.selected_questions == ["q2", "q1"]
    assert selected.strict_questions == ["q1"]
    assert {row["question_id"] for row in selected.selected_rows} == {"q2", "q1"}
    assert all(not row["truncated"] for row in selected.selected_rows)


def test_extension50_launcher_is_bounded_and_does_not_extract_entropy() -> None:
    script = (SCRIPTS / "launch_entropy_band_extension50_tmux.sh").read_text(
        encoding="utf-8"
    )

    assert "--extension-count 500" in script
    assert "--batch-size 50" in script
    assert "candidate_ids_extension50_01.txt" in script
    assert "a6be58071e4cba09297c7d2446d0fc632958f5759e7c0c8ca491d02e8da4eb80" in script
    assert 'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"' in script
    assert "--rollouts 8" in script
    assert "--max-tokens 16384" in script
    assert "--resume" in script
    assert "extract_entropy_band_qwen3vl.py" not in script
    assert "analyze_entropy_band_confirm120.py" not in script
