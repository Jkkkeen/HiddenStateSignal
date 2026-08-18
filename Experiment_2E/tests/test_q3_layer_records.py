import json

import pandas as pd
import pytest

from experiment_2e.q3_layer_records import load_q3_generation_records


def _write_manifest(path):
    rows = []
    for index, content in enumerate(("question zero", "question one")):
        rows.append(
            {
                "prompt": [{"role": "user", "content": content}],
                "reward_model": {"ground_truth": str(index), "style": "rule"},
                "extra_info": {
                    "question_id": f"q{index}",
                    "difficulty": float(index + 1),
                    "prompt_hash": f"hash{index}",
                },
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _generation_rows():
    rows = []
    for question, word in enumerate(("zero", "one")):
        prompt = f"user\nquestion {word}\nassistant\n"
        for slot in range(8):
            rows.append(
                {
                    "input": prompt,
                    "output": f"response {question}-{slot}",
                    "gts": str(question),
                    "score": float(slot == 0),
                    "reward": float(slot == 0),
                    "acc": slot == 0,
                    "format_correct": True,
                    "parse_correct": True,
                    "step": 50,
                }
            )
    return rows


def test_adapter_maps_manifest_order_and_slots(tmp_path):
    manifest = tmp_path / "heldout.parquet"
    generation = tmp_path / "50.jsonl"
    _write_manifest(manifest)
    generation.write_text(
        "".join(json.dumps(row) + "\n" for row in _generation_rows()),
        encoding="utf-8",
    )

    frame = load_q3_generation_records(
        generation,
        manifest,
        global_step=50,
        total_steps=250,
    )

    assert len(frame) == 16
    assert frame["question_id"].tolist()[:8] == ["q0"] * 8
    assert frame["rollout_slot"].tolist() == list(range(8)) * 2
    assert frame.iloc[8]["rollout_id"] == "step050/q1/r00"
    assert frame["training_progress"].unique().tolist() == [0.2]


def test_adapter_rejects_prompt_mismatch(tmp_path):
    manifest = tmp_path / "heldout.parquet"
    generation = tmp_path / "50.jsonl"
    _write_manifest(manifest)
    rows = _generation_rows()
    rows[4]["input"] = "wrong"
    generation.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt mismatch"):
        load_q3_generation_records(
            generation,
            manifest,
            global_step=50,
            total_steps=250,
        )


def test_adapter_rejects_full_cohort_count_mismatch(tmp_path):
    manifest = tmp_path / "heldout.parquet"
    generation = tmp_path / "50.jsonl"
    _write_manifest(manifest)
    generation.write_text(
        "".join(json.dumps(row) + "\n" for row in _generation_rows()[:-1]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 16 generation rows"):
        load_q3_generation_records(
            generation,
            manifest,
            global_step=50,
            total_steps=250,
        )


def test_question_limit_reads_prefix_of_full_generation_file(tmp_path):
    manifest = tmp_path / "heldout.parquet"
    generation = tmp_path / "50.jsonl"
    _write_manifest(manifest)
    generation.write_text(
        "".join(json.dumps(row) + "\n" for row in _generation_rows()),
        encoding="utf-8",
    )

    frame = load_q3_generation_records(
        generation,
        manifest,
        global_step=50,
        total_steps=250,
        question_limit=1,
    )
    assert len(frame) == 8
    assert frame["question_id"].unique().tolist() == ["q0"]
