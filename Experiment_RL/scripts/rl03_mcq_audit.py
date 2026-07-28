#!/usr/bin/env python3
"""Core protocol utilities for the RL03 MathVerse scoring audit."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_SCRIPTS = ROOT / "Experiment" / "scripts"
if EXPERIMENT_SCRIPTS.exists() and str(EXPERIMENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SCRIPTS))

from run_option_logit_trimmed_conclusion_qwen3vl import trim_final_conclusion
from run_semantic_step_basin_qwen3vl import think_text_and_char_span

try:
    from recoverability_candidate_scoring import parse_mathverse_choices
except ModuleNotFoundError:
    from scripts.recoverability_candidate_scoring import parse_mathverse_choices


DENSE_THINK_FRACS = (0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)
SPARSE_THINK_FRACS = (0.25, 0.50, 0.90)
INTERFACES = {
    "I0": "\n\nGiven the reasoning so far, the answer is (",
    "I1": "\n\nFinal answer: ",
    "I2": "\n\nGiven the reasoning so far, the final answer is: ",
}
LETTER_INSTRUCTION_RE = re.compile(
    r"(?im)^Please first conduct reasoning, and then answer the question and provide "
    r"the correct option letter, e\.g\., A, B, C, D, at the end\.\s*\n?"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_source_line", source_line)
            rows.append(row)
    return rows


def neutralize_letter_instruction(prompt: str) -> str:
    """Remove only the known rollout-time instruction that mandates A/B/C/D output."""

    return LETTER_INSTRUCTION_RE.sub("", str(prompt or ""), count=1).strip()


def _eligible_rollout(row: Mapping[str, Any]) -> bool:
    return (
        not bool(row.get("truncated", False))
        and bool(row.get("has_think_close", True))
        and row.get("pred_answer") is not None
    )


def build_manifest(
    raw_path: Path,
    question_count: int = 32,
    seed: int = 20260713,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select complete rollout groups for deterministic mixed-correctness questions."""

    if question_count <= 0:
        raise ValueError("question_count must be positive")
    rows = [row for row in load_jsonl(raw_path) if _eligible_rollout(row)]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no eligible rollouts")
    frame["question_id"] = frame["question_id"].astype(str)
    frame["rollout_id"] = frame["rollout_id"].astype(int)
    frame["is_correct"] = frame["is_correct"].astype(bool)
    mixed = sorted(
        question_id
        for question_id, group in frame.groupby("question_id", sort=True)
        if group["is_correct"].any() and not group["is_correct"].all()
    )
    if len(mixed) < question_count:
        raise ValueError(f"need {question_count} mixed questions, found {len(mixed)}")
    rng = np.random.default_rng(seed)
    chosen = sorted(str(value) for value in rng.choice(mixed, size=question_count, replace=False))
    selected = frame[frame["question_id"].isin(chosen)].copy()
    selected = selected.sort_values(["question_id", "rollout_id"], kind="stable").reset_index(drop=True)
    selected["experiment"] = "rl03_mcq_stage_a0"
    selected["split_role"] = "scoring_smoke"
    selected["manifest_seed"] = int(seed)
    summary = {
        "protocol_version": "rl03_mcq_stage_a0_v1",
        "source_rollouts": str(raw_path),
        "seed": int(seed),
        "eligible_rollouts": int(len(frame)),
        "eligible_questions": int(frame["question_id"].nunique()),
        "eligible_mixed_questions": int(len(mixed)),
        "selected_questions": int(len(chosen)),
        "selected_rollouts": int(len(selected)),
        "rollouts_per_question": {
            str(key): int(value)
            for key, value in selected.groupby("question_id").size().items()
        },
        "correct_rollouts": int(selected["is_correct"].sum()),
        "incorrect_rollouts": int((~selected["is_correct"]).sum()),
        "chosen_question_ids": chosen,
    }
    return selected, summary


def write_manifest(
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
    output_path: Path,
    summary_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(dict(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _char_end(text: str, frac: float) -> int:
    if not text:
        return 0
    return max(1, min(len(text), int(round(float(frac) * len(text)))))


def probe_prefixes(
    thinking: str,
    think_fracs: Sequence[float] = DENSE_THINK_FRACS,
) -> list[dict[str, Any]]:
    """Return prompt, numeric-thinking, and separate trimmed prefixes."""

    text = str(thinking or "")
    probes: list[dict[str, Any]] = [
        {
            "probe_kind": "prompt",
            "frac": 0.0,
            "char_end": 0,
            "prefix_text": "",
            "is_adjacent": True,
        }
    ]
    seen: set[int] = set()
    for frac in think_fracs:
        end = _char_end(text, float(frac))
        if end <= 0 or end in seen:
            continue
        probes.append(
            {
                "probe_kind": "think",
                "frac": float(frac),
                "char_end": int(end),
                "prefix_text": text[:end],
                "is_adjacent": True,
            }
        )
        seen.add(end)
    trimmed, trim_status = trim_final_conclusion(text)
    probes.append(
        {
            "probe_kind": "trimmed",
            "frac": 1.0,
            "char_end": len(trimmed),
            "prefix_text": trimmed,
            "is_adjacent": False,
            "trim_status": trim_status,
        }
    )
    return probes


def _metadata_choices(
    record: Mapping[str, Any],
    metadata_by_question: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    try:
        return parse_mathverse_choices(str(record.get("prompt", "")))
    except ValueError as prompt_error:
        question_id = str(record.get("question_id"))
        metadata = metadata_by_question.get(question_id)
        if metadata is None:
            raise ValueError(f"question {question_id} has no parseable choices") from prompt_error
        source = str(metadata.get("question_for_eval") or metadata.get("question") or "")
        return parse_mathverse_choices(source)


def _messages(
    record: Mapping[str, Any],
    project_root: Path,
    prompt_text: str,
    assistant_text: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    image_raw = str(record.get("image_path") or "").strip()
    if image_raw:
        image_path = Path(image_raw)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        content.append({"type": "image", "image": str(image_path.resolve())})
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def _probe_rows(thinking: str, probe_mode: str) -> list[dict[str, Any]]:
    if probe_mode == "dense":
        return probe_prefixes(thinking, DENSE_THINK_FRACS)
    if probe_mode == "sparse":
        return probe_prefixes(thinking, SPARSE_THINK_FRACS)
    raise ValueError(f"unknown probe_mode {probe_mode!r}")


def build_scoring_requests(
    manifest: pd.DataFrame | Iterable[Mapping[str, Any]],
    project_root: Path,
    metadata_by_question: Mapping[str, Mapping[str, Any]],
    interface_names: Sequence[str] = ("I0", "I1", "I2"),
    probe_mode: str = "sparse",
) -> list[dict[str, Any]]:
    """Build controlled letter/content teacher-forced scoring requests."""

    records = manifest.to_dict(orient="records") if isinstance(manifest, pd.DataFrame) else list(manifest)
    requests: list[dict[str, Any]] = []
    for record in records:
        question_id = str(record["question_id"])
        rollout_id = int(record["rollout_id"])
        gold = str(record["answer"]).strip().upper()
        choices = _metadata_choices(record, metadata_by_question)
        if gold not in choices:
            raise ValueError(f"question {question_id} has invalid gold label {gold!r}")
        thinking, _, _, think_status = think_text_and_char_span(str(record.get("response", "")))
        probes = _probe_rows(thinking, probe_mode)
        for interface in interface_names:
            if interface not in INTERFACES:
                raise ValueError(f"unknown interface {interface!r}")
            representations = ("letter",) if interface == "I0" else ("letter", "content")
            prompt_mode = "original" if interface == "I0" else "neutral"
            prompt_text = (
                str(record.get("prompt", "")).strip()
                if prompt_mode == "original"
                else neutralize_letter_instruction(str(record.get("prompt", "")))
            )
            for representation in representations:
                target = gold if representation == "letter" else choices[gold]
                terminator = "" if interface == "I0" and representation == "letter" else "\n"
                score_text = f"{target}{terminator}"
                for probe in probes:
                    assistant_text = f"{probe['prefix_text']}{INTERFACES[interface]}{score_text}"
                    requests.append(
                        {
                            "question_id": question_id,
                            "rollout_id": rollout_id,
                            "is_correct": bool(record["is_correct"]),
                            "answer": gold,
                            "pred_answer": str(record.get("pred_answer", "")).strip().upper(),
                            "interface": interface,
                            "representation": representation,
                            "prompt_mode": prompt_mode,
                            "surface": "canonical",
                            "target_text": target,
                            "score_text": score_text,
                            "cue": INTERFACES[interface],
                            "think_status": think_status,
                            **probe,
                            "messages": _messages(record, project_root, prompt_text, assistant_text),
                        }
                    )
    return requests


def aggregate_probe_levels(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accepted surfaces at each probe into a robust gold level."""

    keys = [
        column
        for column in ("question_id", "rollout_id", "is_correct", "interface", "representation", "probe_kind", "frac")
        if column in scored.columns
    ]
    levels = (
        scored.groupby(keys, as_index=False, dropna=False)["score_mean"]
        .median()
        .rename(columns={"score_mean": "gold_level"})
    )
    return levels


def aggregate_probe_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Create rollout features while computing endpoint gain per surface first."""

    base_keys = [
        column
        for column in ("question_id", "rollout_id", "is_correct", "interface", "representation")
        if column in scored.columns
    ]
    rows: list[dict[str, Any]] = []
    for key, group in scored.groupby(base_keys, sort=False, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        out = dict(zip(base_keys, values, strict=True))
        for frac, suffix in ((0.25, "25"), (0.50, "50"), (0.90, "90"), (1.0, "100")):
            probe = group[(group["probe_kind"] == "think") & np.isclose(group["frac"].astype(float), frac)]
            if not probe.empty:
                out[f"gold_level_{suffix}"] = float(probe["score_mean"].median())
        trimmed = group[group["probe_kind"] == "trimmed"]
        if not trimmed.empty:
            out["gold_level_trimmed"] = float(trimmed["score_mean"].median())
        surface_keys = ["surface"] if "surface" in group else []
        by_surface = group.groupby(surface_keys, dropna=False) if surface_keys else [("canonical", group)]
        gains_25_90: list[float] = []
        gains_25_trimmed: list[float] = []
        for _, surface in by_surface:
            early = surface[(surface["probe_kind"] == "think") & np.isclose(surface["frac"].astype(float), 0.25)]
            late = surface[(surface["probe_kind"] == "think") & np.isclose(surface["frac"].astype(float), 0.90)]
            trim = surface[surface["probe_kind"] == "trimmed"]
            if len(early) == 1 and len(late) == 1:
                gains_25_90.append(float(late.iloc[0]["score_mean"] - early.iloc[0]["score_mean"]))
            if len(early) == 1 and len(trim) == 1:
                gains_25_trimmed.append(float(trim.iloc[0]["score_mean"] - early.iloc[0]["score_mean"]))
        if gains_25_90:
            out["gold_gain_25_90"] = float(np.median(gains_25_90))
        if gains_25_trimmed:
            out["gold_gain_25_trimmed"] = float(np.median(gains_25_trimmed))
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_step_trajectory(levels: pd.DataFrame) -> pd.DataFrame:
    """Return adjacent gains over prompt and numeric thinking probes only."""

    base_keys = [
        column
        for column in ("question_id", "rollout_id", "is_correct", "interface", "representation")
        if column in levels.columns
    ]
    rows: list[dict[str, Any]] = []
    for key, group in levels.groupby(base_keys, sort=False, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(base_keys, values, strict=True))
        numeric = group[group["probe_kind"].isin(["prompt", "think"])].copy()
        numeric = numeric.sort_values("frac", kind="stable").drop_duplicates("frac", keep="last")
        for (_, left), (_, right) in zip(numeric.iloc[:-1].iterrows(), numeric.iloc[1:].iterrows()):
            span = float(right["frac"] - left["frac"])
            if span <= 0:
                continue
            gain = float(right["gold_level"] - left["gold_level"])
            rows.append(
                {
                    **base,
                    "probe_kind_start": str(left["probe_kind"]),
                    "probe_kind_end": str(right["probe_kind"]),
                    "frac_start": float(left["frac"]),
                    "frac_end": float(right["frac"]),
                    "frac_mid": float((left["frac"] + right["frac"]) / 2.0),
                    "step_gain": gain,
                    "step_gain_rate": gain / span,
                }
            )
    return pd.DataFrame(rows)
