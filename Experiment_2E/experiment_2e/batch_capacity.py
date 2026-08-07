from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Iterable


NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
STEP_RE = re.compile(r"training/global_step:" + NUMBER)
METRIC_RES = {
    "step_time_seconds": re.compile(r"timing_s/step:(?:np\.float64\()?" + NUMBER),
    "throughput_tokens_per_second": re.compile(r"perf/throughput:(?:np\.float64\()?" + NUMBER),
    "actor_allocated_gib": re.compile(
        r"actor/perf/max_memory_allocated_gb:(?:np\.float64\()?" + NUMBER
    ),
    "actor_reserved_gib": re.compile(
        r"actor/perf/max_memory_reserved_gb:(?:np\.float64\()?" + NUMBER
    ),
}
FAILURE_PATTERNS = {
    "cuda_oom": re.compile(r"CUDA out of memory|OutOfMemoryError", re.IGNORECASE),
    "too_many_open_files": re.compile(r"Too many open files", re.IGNORECASE),
    "vllm_preemption": re.compile(r"preempt", re.IGNORECASE),
    "ray_failure": re.compile(r"RayTaskError|RayActorError", re.IGNORECASE),
}


def _float(match: re.Match[str] | None) -> float | None:
    return None if match is None else float(match.group(1))


def _parse_training_log(path: Path) -> tuple[dict[int, dict[str, float]], list[str]]:
    if not path.exists():
        return {}, ["missing_train_log"]
    text = path.read_text(encoding="utf-8", errors="replace")
    records: dict[int, dict[str, float]] = {}
    for line in text.splitlines():
        step_match = STEP_RE.search(line)
        if step_match is None:
            continue
        step = int(float(step_match.group(1)))
        values: dict[str, float] = {}
        for name, pattern in METRIC_RES.items():
            value = _float(pattern.search(line))
            if value is not None:
                values[name] = value
        if values:
            records[step] = values
    failures = [name for name, pattern in FAILURE_PATTERNS.items() if pattern.search(text)]
    return records, failures


def _parse_telemetry(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    totals: list[float] = []
    used: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                totals.append(float(row["memory.total [MiB]"]))
                used.append(float(row["memory.used [MiB]"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not totals or not used:
        return None, None
    return max(totals), max(used)


def _mean_metric(records: Iterable[dict[str, float]], name: str) -> float | None:
    values = [record[name] for record in records if name in record]
    return None if not values else float(mean(values))


def _read_status(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def summarize_candidate(
    candidate_root: Path,
    *,
    source_step: int,
    warmup_steps: int,
    measured_steps: int,
    minimum_headroom: float,
) -> dict[str, object]:
    records, failures = _parse_training_log(candidate_root / "train.log")
    expected = list(
        range(
            source_step + warmup_steps + 1,
            source_step + warmup_steps + measured_steps + 1,
        )
    )
    measured = [step for step in expected if step in records]
    measured_records = [records[step] for step in measured]
    total_mib, peak_used_mib = _parse_telemetry(candidate_root / "gpu_telemetry.csv")
    headroom = None
    if total_mib is not None and peak_used_mib is not None and total_mib > 0:
        headroom = (total_mib - peak_used_mib) / total_mib
    status = _read_status(candidate_root / "exit_status.txt")
    stable = (
        status == 0
        and len(measured) == measured_steps
        and not failures
        and headroom is not None
        and headroom >= minimum_headroom
    )
    return {
        "exit_status": status,
        "completed_steps": sorted(records),
        "measured_steps": measured,
        "mean_step_time_seconds": _mean_metric(measured_records, "step_time_seconds"),
        "mean_throughput_tokens_per_second": _mean_metric(
            measured_records, "throughput_tokens_per_second"
        ),
        "max_actor_allocated_gib": max(
            (record.get("actor_allocated_gib", math.nan) for record in records.values()),
            default=math.nan,
        ),
        "max_actor_reserved_gib": max(
            (record.get("actor_reserved_gib", math.nan) for record in records.values()),
            default=math.nan,
        ),
        "gpu_total_mib": total_mib,
        "gpu_peak_used_mib": peak_used_mib,
        "gpu_headroom_fraction": headroom,
        "failure_signatures": failures,
        "stable": stable,
    }


def build_audit(
    run_root: Path,
    *,
    candidates: tuple[int, ...] = (2, 4, 8),
    source_step: int = 3348,
    warmup_steps: int = 2,
    measured_steps: int = 5,
    minimum_headroom: float = 0.10,
) -> dict[str, object]:
    summaries = {
        str(batch): summarize_candidate(
            run_root / f"batch_{batch}",
            source_step=source_step,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            minimum_headroom=minimum_headroom,
        )
        for batch in candidates
    }
    stable_batches = [batch for batch in candidates if summaries[str(batch)]["stable"]]
    useful_batches: list[int] = []
    preceding_throughput: float | None = None
    for batch in candidates:
        summary = summaries[str(batch)]
        throughput = summary["mean_throughput_tokens_per_second"]
        if summary["stable"] and throughput is not None:
            if preceding_throughput is None or throughput >= preceding_throughput:
                useful_batches.append(batch)
            preceding_throughput = float(throughput)
    return {
        "run_root": str(run_root),
        "source_step": source_step,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "minimum_headroom": minimum_headroom,
        "candidates": summaries,
        "maximum_stable_batch": max(stable_batches) if stable_batches else None,
        "maximum_useful_batch": max(useful_batches) if useful_batches else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an Experiment 2E batch-capacity smoke.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
