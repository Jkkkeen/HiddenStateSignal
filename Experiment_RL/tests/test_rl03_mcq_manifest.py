import pandas as pd
import json
import pytest

import scripts.build_rl03_mcq_audit_manifest as manifest_module

from scripts.build_rl03_mcq_audit_manifest import (
    build_manifest_records,
    build_probe_prefixes,
    extract_thinking_text,
    parseable_question_ids,
    select_mixed_question_ids,
    trim_final_conclusion,
    write_manifest_bundle,
)


def test_select_mixed_question_ids_is_deterministic_and_excludes_constant_questions():
    features = pd.DataFrame(
        [
            {"question_id": "q1", "rollout_id": 0, "is_correct": False},
            {"question_id": "q1", "rollout_id": 1, "is_correct": True},
            {"question_id": "q2", "rollout_id": 0, "is_correct": True},
            {"question_id": "q2", "rollout_id": 1, "is_correct": True},
            {"question_id": "q3", "rollout_id": 0, "is_correct": False},
            {"question_id": "q3", "rollout_id": 1, "is_correct": True},
            {"question_id": "q4", "rollout_id": 0, "is_correct": False},
            {"question_id": "q4", "rollout_id": 1, "is_correct": False},
            {"question_id": "q5", "rollout_id": 0, "is_correct": True},
            {"question_id": "q5", "rollout_id": 1, "is_correct": False},
        ]
    )

    first = select_mixed_question_ids(features, question_count=2, seed=17)
    second = select_mixed_question_ids(features, question_count=2, seed=17)

    assert first == second
    assert len(first) == 2
    assert set(first) <= {"q1", "q3", "q5"}


def test_select_mixed_question_ids_parses_string_booleans_without_truthiness():
    features = pd.DataFrame(
        [
            {"question_id": "mixed", "is_correct": "False"},
            {"question_id": "mixed", "is_correct": "True"},
            {"question_id": "wrong", "is_correct": "False"},
            {"question_id": "wrong", "is_correct": "False"},
        ]
    )

    assert select_mixed_question_ids(features, question_count=1, seed=3) == ["mixed"]
    with pytest.raises(ValueError, match="is_correct"):
        select_mixed_question_ids(
            pd.DataFrame(
                [
                    {"question_id": "q1", "is_correct": True},
                    {"question_id": "q1", "is_correct": float("nan")},
                ]
            ),
            question_count=1,
            seed=3,
        )


def test_select_all_question_ids_returns_every_unique_question_in_sorted_order():
    features = pd.DataFrame(
        [
            {"question_id": "q2", "is_correct": True},
            {"question_id": "q1", "is_correct": False},
            {"question_id": "q2", "is_correct": False},
            {"question_id": 3, "is_correct": True},
        ]
    )

    assert manifest_module.select_all_question_ids(features) == ["3", "q1", "q2"]


def test_build_probe_prefixes_keeps_dense_grid_and_separate_trimmed_control():
    probes = build_probe_prefixes("x" * 100, trimmed_text="x" * 73)

    assert len(probes) == 15
    assert probes[0]["probe_id"] == "prompt_0.00"
    assert probes[0]["reasoning_prefix"] == ""
    by_id = {probe["probe_id"]: probe for probe in probes}
    assert len(by_id["think_0.25"]["reasoning_prefix"]) == 25
    assert len(by_id["think_0.90"]["reasoning_prefix"]) == 90
    assert by_id["think_0.25"]["is_sparse"] is True
    assert by_id["think_0.20"]["is_sparse"] is False
    assert by_id["trimmed_final"]["reasoning_prefix"] == "x" * 73
    assert by_id["trimmed_final"]["frac"] is None


def test_extract_and_trim_thinking_removes_explicit_final_conclusion():
    response = "<think>Compute the height. Check the diagram. Therefore the answer is B.</think>\nB"

    think_text, status = extract_thinking_text(response)
    trimmed, trim_status = trim_final_conclusion(think_text, min_keep_chars=10)

    assert status == "explicit_think_tags"
    assert think_text.endswith("answer is B.")
    assert trimmed == "Compute the height. Check the diagram."
    assert trim_status == "trigger"


def test_extract_thinking_accepts_implicit_open_tag_used_by_qwen_rollouts():
    think_text, status = extract_thinking_text("work through it</think>\nFinal answer: C")

    assert think_text == "work through it"
    assert status == "implicit_think_open"


def test_extract_thinking_preserves_legacy_whitespace_boundaries():
    explicit, explicit_status = extract_thinking_text(
        "<think>\n  work through it  \n</think>\nC"
    )
    implicit, implicit_status = extract_thinking_text(
        "\n  work through it  \n</think>\nC"
    )

    assert explicit == "\n  work through it  \n"
    assert implicit == "\n  work through it  \n"
    assert explicit_status == "explicit_think_tags"
    assert implicit_status == "implicit_think_open"


def test_build_manifest_records_joins_exact_rollouts_and_freezes_gold_content():
    raw_rows = [
        {
            "question_id": "q1",
            "rollout_id": 0,
            "prompt": "Question?\nChoices:\nA: 3\nB: 4 sqrt(2)\nC: 8\nD: 9",
            "image_path": "data/q1.png",
            "response": "<think>Compute carefully. Therefore the answer is B.</think>\nB",
            "answer": "B",
            "pred_answer": "B",
        },
        {
            "question_id": "q1",
            "rollout_id": 1,
            "prompt": "Question?\nChoices:\nA: 3\nB: 4 sqrt(2)\nC: 8\nD: 9",
            "image_path": "data/q1.png",
            "response": "work another way</think>\nA",
            "answer": "B",
            "pred_answer": "A",
        },
        {
            "question_id": "q2",
            "rollout_id": 0,
            "prompt": "Unused\nA: one\nB: two",
            "response": "unused</think>",
            "answer": "A",
            "pred_answer": "A",
        },
    ]
    features = pd.DataFrame(
        [
            {"question_id": "q1", "rollout_id": 0, "is_correct": True, "source_line": 11},
            {"question_id": "q1", "rollout_id": 1, "is_correct": False, "source_line": 12},
        ]
    )

    records = build_manifest_records(raw_rows, features, selected_question_ids=["q1"])

    assert len(records) == 2
    assert [record["rollout_id"] for record in records] == [0, 1]
    assert records[0]["gold_letter"] == "B"
    assert records[0]["gold_content_surfaces"] == ["4 sqrt(2)", "4*sqrt(2)"]
    assert records[0]["gold_surface_status"] == "parsed_semantic"
    assert records[0]["choices"]["D"] == "9"
    assert len(records[0]["probes"]) == 15
    assert len(records[0]["think_sha256"]) == 64
    assert records[0]["is_correct"] is True
    assert records[1]["is_correct"] is False


def test_write_manifest_bundle_records_checksums_and_selected_ids(tmp_path):
    records = [
        {
            "protocol_version": "rl03_stage_a_v2",
            "question_id": "q2",
            "rollout_id": 1,
            "is_correct": False,
        },
        {
            "protocol_version": "rl03_stage_a_v2",
            "question_id": "q1",
            "rollout_id": 0,
            "is_correct": True,
        },
    ]
    raw_path = tmp_path / "raw.jsonl"
    feature_path = tmp_path / "features.parquet"
    metadata_path = tmp_path / "metadata.json"
    raw_path.write_text("{}\n", encoding="utf-8")
    feature_path.write_bytes(b"parquet-placeholder")
    metadata_path.write_text("[]\n", encoding="utf-8")

    summary = write_manifest_bundle(
        records,
        selected_question_ids=["q2", "q1"],
        output_dir=tmp_path / "out",
        seed=17,
        raw_path=raw_path,
        features_path=feature_path,
        metadata_path=metadata_path,
    )

    output_dir = tmp_path / "out"
    manifest_rows = [
        json.loads(line)
        for line in (output_dir / "stage_a0_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    stored_summary = json.loads(
        (output_dir / "stage_a0_manifest_summary.json").read_text(encoding="utf-8")
    )
    assert [(row["question_id"], row["rollout_id"]) for row in manifest_rows] == [
        ("q1", 0),
        ("q2", 1),
    ]
    assert (output_dir / "stage_a0_selected_question_ids.txt").read_text(
        encoding="utf-8"
    ) == "q1\nq2\n"
    assert summary == stored_summary
    assert summary["selected_questions"] == 2
    assert summary["selected_rollouts"] == 2
    assert len(summary["manifest_sha256"]) == 64
    assert set(summary["source_sha256"]) == {
        "raw_rollouts",
        "features",
        "mathverse_metadata",
    }
    protocol = summary["protocol"]
    assert protocol["request_matrix"]["I0_legacy_letter_first"] == "dense"
    assert protocol["request_matrix"]["I1_I2_neutral_content_sequence"] == "dense"
    assert protocol["primary_process_features"] == [
        "gold_gain_25_90",
        "gold_trajectory_slope",
    ]
    assert protocol["matched_level_controls"] == [
        "gold_level_90",
        "gold_level_trimmed",
    ]
    assert protocol["decision_thresholds"]["process_auc_min"] == 0.70
    assert protocol["gate_order"][2] == "LEVEL-EQUIVALENT"


def test_write_manifest_bundle_supports_stage_a1_names_without_creating_a0_files(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    feature_path = tmp_path / "features.parquet"
    metadata_path = tmp_path / "metadata.json"
    raw_path.write_text("{}\n", encoding="utf-8")
    feature_path.write_bytes(b"parquet-placeholder")
    metadata_path.write_text("[]\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    summary = write_manifest_bundle(
        [
            {
                "protocol_version": "rl03_stage_a_v2",
                "question_id": "q1",
                "rollout_id": 0,
                "is_correct": True,
            }
        ],
        selected_question_ids=["q1"],
        output_dir=output_dir,
        seed=17,
        raw_path=raw_path,
        features_path=feature_path,
        metadata_path=metadata_path,
        stage_name="stage_a1",
    )

    assert (output_dir / "stage_a1_manifest.jsonl").is_file()
    assert (output_dir / "stage_a1_selected_question_ids.txt").is_file()
    assert (output_dir / "stage_a1_manifest_summary.json").is_file()
    assert not list(output_dir.glob("stage_a0_*"))
    assert summary["stage_name"] == "stage_a1"


def test_write_manifest_bundle_refuses_to_overwrite_changed_frozen_manifest(tmp_path):
    source_paths = [tmp_path / name for name in ("raw.jsonl", "features.csv", "meta.json")]
    for path in source_paths:
        path.write_text("source\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    record = {
        "protocol_version": "rl03_stage_a_v2",
        "question_id": "q1",
        "rollout_id": 0,
        "is_correct": True,
    }
    kwargs = {
        "selected_question_ids": ["q1"],
        "output_dir": output_dir,
        "seed": 17,
        "raw_path": source_paths[0],
        "features_path": source_paths[1],
        "metadata_path": source_paths[2],
    }
    write_manifest_bundle([record], **kwargs)
    write_manifest_bundle([record], **kwargs)

    changed = {**record, "is_correct": False}
    with pytest.raises(ValueError, match="frozen manifest"):
        write_manifest_bundle([changed], **kwargs)


def test_parseable_question_ids_excludes_vision_only_rows_without_text_choices():
    raw_rows = [
        {
            "question_id": "text",
            "prompt": "Question?\nA: one\nB: two\nC: three\nD: four",
        },
        {"question_id": "vision", "prompt": "According to the image, choose A, B, C, or D."},
    ]
    metadata = {
        "vision": {"question_for_eval": "The options are visible only in the image."}
    }

    parseable, excluded = parseable_question_ids(raw_rows, metadata)

    assert parseable == {"text"}
    assert excluded == {"vision": "choices_not_recoverable_from_text"}


def test_manifest_requires_complete_mathverse_abcd_choice_matrix():
    raw_rows = [
        {
            "question_id": "q1",
            "rollout_id": 0,
            "prompt": "Question?\nA: one\nB: two\nC: three",
            "response": "reasoning</think>\nB",
            "answer": "B",
            "pred_answer": "B",
        }
    ]
    features = pd.DataFrame(
        [{"question_id": "q1", "rollout_id": 0, "is_correct": True}]
    )

    with pytest.raises(ValueError, match="exactly A/B/C/D"):
        build_manifest_records(raw_rows, features, selected_question_ids=["q1"])
