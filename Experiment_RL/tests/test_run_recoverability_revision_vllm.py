from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Experiment_RL" / "scripts"))

from run_recoverability_revision_vllm import (
    DEFAULT_REVISION_INSTRUCTION,
    build_revision_messages,
    extract_revision_choice,
    summarize_prefixes,
)


def test_build_revision_messages_preserves_image_and_prefix(tmp_path: Path) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"not-opened-in-this-unit-test")
    record = {
        "question_id": "q1",
        "answer": "C",
        "prompt": "Question text\nA: one\nB: two\nC: three\nD: four",
        "image_path": str(image),
        "revision_prefix": "Compute the two angles and compare the expressions.",
    }

    messages = build_revision_messages(record, project_root=tmp_path)

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"][0] == {"type": "image", "image": str(image)}
    assert "Question text" in messages[0]["content"][1]["text"]
    assert "Compute the two angles" in messages[1]["content"][0]["text"]
    assert messages[2]["content"][0]["text"] == DEFAULT_REVISION_INSTRUCTION
    assert "correct option is C" not in DEFAULT_REVISION_INSTRUCTION


def test_summarize_prefixes_aggregates_k_revisions() -> None:
    records = []
    predictions = ["A", "B", None, "A"]
    for sample_index, prediction in enumerate(predictions):
        records.append(
            {
                "question_id": "q1",
                "rollout_id": 3,
                "sample_index": sample_index,
                "answer": "A",
                "prediction": prediction,
                "correct": prediction == "A",
                "output_token_count": 100 + sample_index,
                "finish_reason": "stop",
                "truncated_by_length": False,
                "trimmed_margin_mean": 1.25,
                "signal_rank": 2,
            }
        )

    summary = summarize_prefixes(records)

    assert len(summary) == 1
    row = summary[0]
    assert row["revision_samples"] == 4
    assert row["recovered_count"] == 2
    assert row["recovery_rate"] == 0.5
    assert row["any_recovery"] is True
    assert row["invalid_revision_count"] == 1
    assert row["mean_output_tokens"] == 101.5
    assert row["trimmed_margin_mean"] == 1.25


def test_extract_revision_choice_requires_explicit_answer_marker() -> None:
    assert extract_revision_choice("I compared options A and B but need more time.") is None
    assert extract_revision_choice("Reasoning complete. Final answer: C") == "C"
    assert extract_revision_choice("</think>\nAnswer is D.") == "D"
