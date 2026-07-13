#!/usr/bin/env python3
"""Pure protocol and aggregation helpers for RL03 answer-likelihood probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from statistics import median
from typing import Final
from typing import Any


PROTOCOL_VERSION: Final = "rl03_stage_a_v2"
ANSWER_END: Final = "\n"
INTERFACES: Final = {
    "I0": "\n\nGiven the reasoning so far, the answer is (",
    "I1": "\n\nFinal answer: ",
    "I2": "\n\nGiven the reasoning so far, the final answer is: ",
}
SPARSE_THINK_FRACS: Final = (0.0, 0.25, 0.50, 0.90)
DENSE_THINK_FRACS: Final = (
    0.05,
    0.10,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    1.00,
)


def actual_prompt_token_logprob(row: Any, expected_token_id: int) -> float:
    """Extract the actual token's log probability from a vLLM prompt row."""

    if row is None or not isinstance(row, Mapping):
        raise ValueError("prompt logprob row must be a mapping")
    expected = int(expected_token_id)
    value = row.get(expected)
    if value is None:
        value = row.get(str(expected))
    if value is None:
        raise ValueError(f"prompt logprob row is missing actual token {expected}")
    return float(getattr(value, "logprob", value))


def target_token_indices(
    offsets: Sequence[tuple[int, int] | list[int]],
    target_start: int,
    target_end: int,
) -> list[int]:
    """Return token indices whose character offsets overlap the target span."""

    if target_end <= target_start:
        raise ValueError("target span must be non-empty")
    indices = [
        index
        for index, offset in enumerate(offsets)
        if int(offset[1]) > int(offset[0])
        and int(offset[1]) > int(target_start)
        and int(offset[0]) < int(target_end)
    ]
    if not indices:
        raise ValueError("target span does not overlap any prompt tokens")
    return indices


def sequence_logprob(
    prompt_token_ids: Sequence[int],
    prompt_logprobs: Sequence[Any],
    token_indices: Sequence[int],
) -> dict[str, float | int]:
    """Aggregate actual-token prompt log probabilities over a target span."""

    if len(prompt_token_ids) != len(prompt_logprobs):
        raise ValueError("prompt token ids and logprobs must have equal length")
    indices = [int(index) for index in token_indices]
    if not indices:
        raise ValueError("target token indices must be non-empty")
    values = [
        actual_prompt_token_logprob(prompt_logprobs[index], int(prompt_token_ids[index]))
        for index in indices
    ]
    total = float(sum(values))
    return {"sum": total, "mean": total / len(values), "token_count": len(values)}


def build_assistant_scoring_text(
    reasoning_prefix: str,
    interface: str,
    target: str,
    terminator: str = ANSWER_END,
) -> tuple[str, int, int]:
    """Build assistant text and return the target-plus-terminator character span."""

    target = str(target)
    if not target:
        raise ValueError("target must be non-empty")
    prefix = f"{reasoning_prefix}{interface}"
    rendered = f"{prefix}{target}{terminator}"
    return rendered, len(prefix), len(rendered)


def target_span_from_rendered(
    rendered_text: str,
    interface: str,
    target: str,
    terminator: str = ANSWER_END,
) -> tuple[int, int]:
    """Locate the final answer target and terminator in a rendered chat prompt."""

    suffix = f"{interface}{target}{terminator}"
    suffix_start = str(rendered_text).rfind(suffix)
    if suffix_start < 0:
        raise ValueError("scoring suffix not found in rendered prompt")
    target_start = suffix_start + len(interface)
    return target_start, target_start + len(str(target)) + len(terminator)


def summarize_surface_trajectory(
    surface_scores: Mapping[str, Mapping[float | str, float]],
) -> dict[str, object]:
    """Aggregate levels and gains without switching surface form across probes."""

    if not surface_scores:
        raise ValueError("surface scores must be non-empty")
    surfaces = {str(name): dict(scores) for name, scores in surface_scores.items()}
    key_sets = {frozenset(scores) for scores in surfaces.values()}
    if len(key_sets) != 1:
        raise ValueError("all surface forms must contain the same probes")
    probe_keys = next(iter(key_sets))
    numeric_probes = sorted(
        float(probe) for probe in probe_keys if isinstance(probe, Real) and not isinstance(probe, bool)
    )
    if len(numeric_probes) < 2:
        raise ValueError("at least two numeric probes are required")
    for required in (0.25, 0.90):
        if required not in numeric_probes:
            raise ValueError(f"required probe {required:.2f} is missing")

    level_by_probe: dict[float | str, float] = {
        probe: float(median(float(scores[probe]) for scores in surfaces.values()))
        for probe in numeric_probes
    }
    if "trimmed" in probe_keys:
        level_by_probe["trimmed"] = float(
            median(float(scores["trimmed"]) for scores in surfaces.values())
        )

    def aggregate_gain(start: float | str, end: float | str) -> float:
        return float(
            median(
                float(scores[end]) - float(scores[start])
                for scores in surfaces.values()
            )
        )

    step_gain_by_interval: dict[tuple[float, float], float] = {}
    step_gain_rate_by_interval: dict[tuple[float, float], float] = {}
    for start, end in zip(numeric_probes[:-1], numeric_probes[1:], strict=True):
        interval = (start, end)
        gain = aggregate_gain(start, end)
        step_gain_by_interval[interval] = gain
        step_gain_rate_by_interval[interval] = gain / (end - start)

    result: dict[str, object] = {
        "level_by_probe": level_by_probe,
        "step_gain_by_interval": step_gain_by_interval,
        "step_gain_rate_by_interval": step_gain_rate_by_interval,
        "gold_gain_25_90": aggregate_gain(0.25, 0.90),
    }
    process_intervals = [
        interval
        for interval in step_gain_by_interval
        if interval[0] >= 0.25 and interval[1] <= 0.90
    ]
    process_gains = [step_gain_by_interval[interval] for interval in process_intervals]
    if not process_gains:
        raise ValueError("no adjacent process intervals exist between 0.25 and 0.90")
    result["gold_positive_gain_rate"] = sum(gain > 0 for gain in process_gains) / len(
        process_gains
    )
    result["gold_monotonicity_violation_rate"] = sum(
        gain < 0 for gain in process_gains
    ) / len(process_gains)

    slope_probes = [probe for probe in numeric_probes if 0.25 <= probe <= 0.90]
    x_mean = sum(slope_probes) / len(slope_probes)
    denominator = sum((probe - x_mean) ** 2 for probe in slope_probes)
    slopes: list[float] = []
    for scores in surfaces.values():
        values = [float(scores[probe]) for probe in slope_probes]
        y_mean = sum(values) / len(values)
        numerator = sum(
            (probe - x_mean) * (value - y_mean)
            for probe, value in zip(slope_probes, values, strict=True)
        )
        slopes.append(numerator / denominator)
    result["gold_trajectory_slope"] = float(median(slopes))
    if "trimmed" in probe_keys:
        result["gold_gain_25_trimmed"] = aggregate_gain(0.25, "trimmed")
    return result
