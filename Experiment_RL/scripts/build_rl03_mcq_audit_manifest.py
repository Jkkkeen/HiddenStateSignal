#!/usr/bin/env python3
"""Freeze RL03 MathVerse questions, rollouts, prefixes, interfaces, and targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from answer_likelihood_scoring import (
        ANSWER_END,
        DENSE_THINK_FRACS,
        INTERFACES,
        PROTOCOL_VERSION,
        SPARSE_THINK_FRACS,
    )
except ModuleNotFoundError:
    from scripts.answer_likelihood_scoring import (
        ANSWER_END,
        DENSE_THINK_FRACS,
        INTERFACES,
        PROTOCOL_VERSION,
        SPARSE_THINK_FRACS,
    )

try:
    from answer_normalization import (
        accepted_mcq_surface_forms,
        normalize_mcq_content,
        parse_mcq_choices,
    )
except ModuleNotFoundError:
    from scripts.answer_normalization import (
        accepted_mcq_surface_forms,
        normalize_mcq_content,
        parse_mcq_choices,
    )


TRIGGER_RE = re.compile(
    r"(?i)\b(?:therefore|thus|hence|so[, ]+the answer|so[, ]+answer|"
    r"final answer|answer is|the answer is|correct answer|option [ABCD]|"
    r"answer should be|answer would be|answer must be)\b"
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _strict_correctness(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError(f"invalid is_correct value {value!r}")
    if pd.isna(value):
        raise ValueError("invalid is_correct value: missing")
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"invalid is_correct value {value!r}")


def select_mixed_question_ids(
    features: pd.DataFrame,
    question_count: int,
    seed: int,
) -> list[str]:
    """Select a deterministic question-equal subset with both outcomes present."""

    required = {"question_id", "is_correct"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"features missing columns: {sorted(missing)}")
    grouped = features.assign(
        question_id=features["question_id"].astype(str),
        is_correct=features["is_correct"].map(_strict_correctness),
    ).groupby("question_id", sort=True)["is_correct"]
    eligible = sorted(question_id for question_id, values in grouped if values.nunique() == 2)
    count = int(question_count)
    if count <= 0:
        raise ValueError("question_count must be positive")
    if len(eligible) < count:
        raise ValueError(f"need {count} mixed questions, found {len(eligible)}")
    rng = np.random.default_rng(int(seed))
    return sorted(str(value) for value in rng.choice(eligible, size=count, replace=False))


def select_all_question_ids(features: pd.DataFrame) -> list[str]:
    """Select every question represented by an already-filtered feature table."""

    if "question_id" not in features.columns:
        raise ValueError("features missing columns: ['question_id']")
    return sorted(features["question_id"].astype(str).unique().tolist())


def build_probe_prefixes(think_text: str, trimmed_text: str) -> list[dict[str, object]]:
    """Materialize the frozen prompt, dense thinking, and trimmed probe prefixes."""

    think_text = str(think_text or "")
    if not think_text:
        raise ValueError("thinking text must be non-empty")
    probes: list[dict[str, object]] = [
        {
            "probe_id": "prompt_0.00",
            "probe_kind": "prompt",
            "frac": 0.0,
            "char_end": 0,
            "reasoning_prefix": "",
            "is_sparse": True,
            "is_dense": True,
        }
    ]
    sparse = set(SPARSE_THINK_FRACS)
    for frac in DENSE_THINK_FRACS:
        char_end = max(1, min(len(think_text), int(round(len(think_text) * frac))))
        probes.append(
            {
                "probe_id": f"think_{frac:.2f}",
                "probe_kind": "think",
                "frac": float(frac),
                "char_end": char_end,
                "reasoning_prefix": think_text[:char_end],
                "is_sparse": frac in sparse,
                "is_dense": True,
            }
        )
    probes.append(
        {
            "probe_id": "trimmed_final",
            "probe_kind": "trimmed",
            "frac": None,
            "char_end": len(str(trimmed_text or "")),
            "reasoning_prefix": str(trimmed_text or ""),
            "is_sparse": True,
            "is_dense": True,
        }
    )
    return probes


def extract_thinking_text(response: str) -> tuple[str, str]:
    """Extract Qwen explicit or implicit thinking content from one rollout."""

    text = str(response or "")
    close = text.find("</think>")
    open_index = text.find("<think>")
    if open_index >= 0 and close > open_index:
        return text[open_index + len("<think>") : close], "explicit_think_tags"
    if close >= 0:
        return text[:close], "implicit_think_open"
    return text, "missing_think_close"


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def trim_final_conclusion(
    text: str,
    min_keep_chars: int = 40,
    tail_window_chars: int = 1800,
) -> tuple[str, str]:
    """Remove the final explicit conclusion using the legacy frozen rule."""

    text = str(text or "").strip()
    if len(text) <= min_keep_chars:
        return text, "too_short"
    tail_start = max(0, len(text) - int(tail_window_chars))
    matches = list(TRIGGER_RE.finditer(text[tail_start:]))
    spans = _sentence_spans(text)
    if matches:
        trigger_pos = tail_start + matches[-1].start()
        starts = [start for start, end in spans if start <= trigger_pos < end]
        cut = starts[-1] if starts else 0
        if cut >= min_keep_chars:
            return text[:cut].strip(), "trigger"
    if len(spans) >= 2 and spans[-1][0] >= min_keep_chars:
        return text[: spans[-1][0]].strip(), "last_sentence"
    return text, "no_trim"


def _choices_for_row(
    row: Mapping[str, Any],
    metadata_by_question: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    try:
        choices = parse_mcq_choices(str(row.get("prompt", "")))
    except ValueError as prompt_error:
        question_id = str(row.get("question_id"))
        metadata = metadata_by_question.get(question_id)
        if metadata is None:
            raise ValueError(f"question {question_id} has no parseable choices") from prompt_error
        source = str(metadata.get("question_for_eval") or metadata.get("question") or "")
        choices = parse_mcq_choices(source)
    if tuple(choices) != ("A", "B", "C", "D"):
        raise ValueError("MathVerse choices must contain exactly A/B/C/D")
    return choices


def parseable_question_ids(
    raw_rows: Iterable[Mapping[str, Any]],
    metadata_by_question: Mapping[str, Mapping[str, Any]],
) -> tuple[set[str], dict[str, str]]:
    """Identify questions whose complete option contents are recoverable from text."""

    first_by_question: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        first_by_question.setdefault(str(row.get("question_id")), row)
    parseable: set[str] = set()
    excluded: dict[str, str] = {}
    for question_id, row in first_by_question.items():
        try:
            _choices_for_row(row, metadata_by_question)
        except ValueError:
            excluded[question_id] = "choices_not_recoverable_from_text"
        else:
            parseable.add(question_id)
    return parseable, excluded


def build_manifest_records(
    raw_rows: Iterable[Mapping[str, Any]],
    features: pd.DataFrame,
    selected_question_ids: Iterable[str],
    metadata_by_question: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Join the frozen feature rows to raw rollouts and materialize protocol metadata."""

    metadata_by_question = metadata_by_question or {}
    selected = {str(question_id) for question_id in selected_question_ids}
    feature_rows = features.copy()
    required = {"question_id", "rollout_id", "is_correct"}
    missing = required - set(feature_rows.columns)
    if missing:
        raise ValueError(f"features missing columns: {sorted(missing)}")
    feature_rows["question_id"] = feature_rows["question_id"].astype(str)
    feature_rows["rollout_id"] = feature_rows["rollout_id"].astype(int)
    feature_rows["is_correct"] = feature_rows["is_correct"].map(_strict_correctness)
    feature_rows = feature_rows[feature_rows["question_id"].isin(selected)].copy()

    raw_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in raw_rows:
        key = (str(row.get("question_id")), int(row.get("rollout_id", -1)))
        if key in raw_by_key:
            raise ValueError(f"duplicate raw rollout key {key}")
        raw_by_key[key] = row

    records: list[dict[str, Any]] = []
    for feature in feature_rows.sort_values(["question_id", "rollout_id"]).to_dict("records"):
        key = (str(feature["question_id"]), int(feature["rollout_id"]))
        raw = raw_by_key.get(key)
        if raw is None:
            raise ValueError(f"missing raw rollout for key {key}")
        response = str(raw.get("response") or raw.get("original_response") or "")
        think_text, think_status = extract_thinking_text(response)
        if not think_text.strip():
            raise ValueError(f"empty thinking text for key {key}")
        trimmed_text, trim_status = trim_final_conclusion(think_text)
        choices = _choices_for_row(raw, metadata_by_question)
        gold = str(raw.get("answer", "")).strip().upper()
        if gold not in choices:
            raise ValueError(f"gold option {gold!r} is absent for key {key}")
        gold_surfaces = accepted_mcq_surface_forms(choices[gold])
        probes = build_probe_prefixes(think_text, trimmed_text)
        frozen_probes = [
            {name: value for name, value in probe.items() if name != "reasoning_prefix"}
            for probe in probes
        ]
        source_line = feature.get("source_line", raw.get("_source_line", raw.get("source_line", -1)))
        records.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "question_id": key[0],
                "rollout_id": key[1],
                "source_line": int(source_line),
                "is_correct": bool(feature["is_correct"]),
                "gold_letter": gold,
                "pred_letter": str(raw.get("pred_answer", "")).strip().upper(),
                "prompt": str(raw.get("prompt", "")),
                "image_path": str(raw.get("image_path", "")),
                "choices": {label: normalize_mcq_content(text) for label, text in choices.items()},
                "gold_content_surfaces": list(gold_surfaces.forms),
                "gold_surface_status": gold_surfaces.status,
                "think_text": think_text,
                "trimmed_text": trimmed_text,
                "think_status": think_status,
                "trim_status": trim_status,
                "think_sha256": hashlib.sha256(think_text.encode("utf-8")).hexdigest(),
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "probes": frozen_probes,
            }
        )
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest_bundle(
    records: Iterable[Mapping[str, Any]],
    selected_question_ids: Iterable[str],
    output_dir: Path,
    seed: int,
    raw_path: Path,
    features_path: Path,
    metadata_path: Path,
    extra_summary: Mapping[str, Any] | None = None,
    stage_name: str = "stage_a0",
) -> dict[str, Any]:
    """Write an immutable Stage A manifest, selection, and checksum summary."""

    if stage_name not in {"stage_a0", "stage_a1"}:
        raise ValueError(f"unsupported stage name {stage_name!r}")

    frozen_records = sorted(
        (dict(record) for record in records),
        key=lambda row: (str(row["question_id"]), int(row["rollout_id"])),
    )
    if not frozen_records:
        raise ValueError("manifest records must be non-empty")
    keys = [
        (str(record["question_id"]), int(record["rollout_id"]))
        for record in frozen_records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate question/rollout keys")
    chosen = sorted({str(question_id) for question_id in selected_question_ids})
    observed = sorted({str(record["question_id"]) for record in frozen_records})
    if chosen != observed:
        raise ValueError("selected question ids do not match manifest records")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{stage_name}_manifest.jsonl"
    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in frozen_records
    )
    selected_path = output_dir / f"{stage_name}_selected_question_ids.txt"
    selected_text = "".join(f"{question_id}\n" for question_id in chosen)
    for path, expected in ((manifest_path, manifest_text), (selected_path, selected_text)):
        if path.exists() and path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"refusing to overwrite changed frozen manifest file {path}")
        if not path.exists():
            path.write_text(expected, encoding="utf-8")

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_seed": int(seed),
        "selected_questions": len(chosen),
        "selected_rollouts": len(frozen_records),
        "correct_rollouts": sum(bool(record["is_correct"]) for record in frozen_records),
        "incorrect_rollouts": sum(not bool(record["is_correct"]) for record in frozen_records),
        "rollouts_per_question": {
            question_id: sum(str(record["question_id"]) == question_id for record in frozen_records)
            for question_id in chosen
        },
        "selected_question_ids": chosen,
        "manifest_sha256": _file_sha256(manifest_path),
        "source_paths": {
            "raw_rollouts": str(raw_path),
            "features": str(features_path),
            "mathverse_metadata": str(metadata_path),
        },
        "source_sha256": {
            "raw_rollouts": _file_sha256(raw_path),
            "features": _file_sha256(features_path),
            "mathverse_metadata": _file_sha256(metadata_path),
        },
        "protocol": {
            "answer_end": ANSWER_END,
            "interfaces": dict(INTERFACES),
            "dense_think_fracs": list(DENSE_THINK_FRACS),
            "sparse_think_fracs": list(SPARSE_THINK_FRACS),
            "adjacent_sequence": "prompt_then_numeric_thinking",
            "trimmed_final_is_control": True,
            "request_matrix": {
                "I0_legacy_letter_first": "dense",
                "I1_I2_legacy_letter_first": "sparse_interface_control",
                "I1_I2_neutral_letter_first": "sparse_representation_control",
                "I1_I2_neutral_letter_sequence": "sparse",
                "I1_I2_neutral_content_sequence": "dense",
                "wrong_content_targets": "sparse_only",
            },
            "primary_process_features": [
                "gold_gain_25_90",
                "gold_trajectory_slope",
            ],
            "matched_level_controls": [
                "gold_level_90",
                "gold_level_trimmed",
            ],
            "bootstrap_unit": "question_id",
            "bootstrap_seed": int(seed),
            "decision_thresholds": {
                "process_auc_min": 0.70,
                "process_ci_low_min_exclusive": 0.60,
                "positive_correlation_fraction_min": 0.65,
                "interface_auc_difference_max": 0.03,
                "trimmed_or_preclaim_auc_min": 0.65,
                "label_permutation_auc_change_max": 0.02,
                "borderline_auc_min": 0.65,
                "verifier_level_auc_min": 0.75,
            },
            "gate_order": [
                "PIPELINE-FAIL",
                "INTERFACE-FAIL_OR_REPRESENTATION-FAIL",
                "LEVEL-EQUIVALENT",
                "PASS-PROCESS",
                "CONTRASTIVE-ONLY",
                "BORDERLINE-PROCESS",
                "VERIFIER-ONLY",
                "NO-SIGNAL",
            ],
            "label_permutation": {
                "status": "required_full_confirmation_control",
                "seed": int(seed),
            },
        },
    }
    if extra_summary:
        overlap = set(summary) & set(extra_summary)
        if overlap:
            raise ValueError(f"extra summary cannot replace frozen keys: {sorted(overlap)}")
        summary.update(dict(extra_summary))
    if stage_name != "stage_a0":
        summary["stage_name"] = stage_name
    summary_path = output_dir / f"{stage_name}_manifest_summary.json"
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if summary_path.exists() and summary_path.read_text(encoding="utf-8") != summary_text:
        raise ValueError(f"refusing to overwrite changed frozen manifest file {summary_path}")
    if not summary_path.exists():
        summary_path.write_text(summary_text, encoding="utf-8")
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{source_line} is not a JSON object")
            row.setdefault("_source_line", source_line)
            rows.append(row)
    return rows


def _load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("MathVerse metadata must be a JSON list")
    return {
        str(row["sample_index"]): row
        for row in rows
        if isinstance(row, dict) and row.get("sample_index") is not None
    }


def _load_features(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"unsupported feature file {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-rollouts", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--mathverse-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--question-count", type=int, default=32)
    parser.add_argument(
        "--selection-mode",
        choices=("mixed-smoke", "all-parseable"),
        default="mixed-smoke",
    )
    parser.add_argument("--stage-name", choices=("stage_a0", "stage_a1"), default="stage_a0")
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_rows = _load_jsonl(args.raw_rollouts)
    metadata = _load_metadata(args.mathverse_metadata)
    features = _load_features(args.features)
    parseable, excluded = parseable_question_ids(raw_rows, metadata)
    features = features[features["question_id"].astype(str).isin(parseable)].copy()
    if args.selection_mode == "all-parseable":
        selected = select_all_question_ids(features)
    else:
        selected = select_mixed_question_ids(
            features,
            question_count=args.question_count,
            seed=args.seed,
        )
    records = build_manifest_records(
        raw_rows,
        features,
        selected,
        metadata,
    )
    summary = write_manifest_bundle(
        records,
        selected,
        args.output_dir,
        args.seed,
        args.raw_rollouts,
        args.features,
        args.mathverse_metadata,
        extra_summary={
            "parseable_questions": len(parseable),
            "excluded_unparseable_questions": excluded,
        },
        stage_name=args.stage_name,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
