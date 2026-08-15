from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


_STEP_PATTERNS = (
    re.compile(r"['\"]timing_s/step['\"]\s*[:=]\s*([0-9.eE+-]+)"),
    re.compile(r"\btiming_s/step\s*:\s*([0-9.eE+-]+)"),
    re.compile(r"['\"]perf/step_time['\"]\s*[:=]\s*([0-9.eE+-]+)"),
    re.compile(r"\bperf/step_time\s*:\s*([0-9.eE+-]+)"),
)
_FATAL = ("CUDA out of memory", "Too many open files", "Traceback (most recent call last)")


def _step_times(log: Path) -> list[float]:
    text = log.read_text(encoding="utf-8", errors="replace")
    values = []
    for line in text.splitlines():
        for pattern in _STEP_PATTERNS:
            match = pattern.search(line)
            if match:
                values.append(float(match.group(1)))
                break
    return [value for value in values if np.isfinite(value) and value > 0]


def _history(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            return []
    return records


def audit(
    baseline_log: Path,
    probe_log: Path,
    probe_result: Path,
    checkpoint_root: Path,
    output: Path,
    reduced_probe_log: Path | None = None,
    reduced_probe_result: Path | None = None,
) -> dict:
    baseline_times = _step_times(baseline_log)
    probe_times = _step_times(probe_log)
    baseline_median = float(np.median(baseline_times[-5:])) if baseline_times else None
    probe_median = float(np.median(probe_times[-5:])) if probe_times else None
    full_ratio = probe_median / baseline_median if baseline_median and probe_median else None
    middle_branch = full_ratio is not None and 1.30 < full_ratio <= 1.50
    if full_ratio is None:
        group_limit = None
        interval = None
    elif full_ratio <= 1.30:
        group_limit = 32
        interval = 1
    elif full_ratio > 1.50:
        group_limit = 32
        interval = 5
    else:
        group_limit = 16
        interval = 1

    reduced_times = _step_times(reduced_probe_log) if reduced_probe_log and reduced_probe_log.exists() else []
    reduced_median = float(np.median(reduced_times[-5:])) if reduced_times else None
    reduced_ratio = reduced_median / baseline_median if baseline_median and reduced_median else None

    probe_text = probe_log.read_text(encoding="utf-8", errors="replace")
    figures = list((probe_result / "online_hidden" / "latest_figures").glob("*.png"))
    online_history = probe_result / "online_hidden" / "per_step_summary.jsonl"
    heldout_history = probe_result / "heldout" / "hidden_summary.jsonl"
    exit_status = probe_result / "exit_status.txt"
    restore_passed = "Setting global step to 5" in probe_text and (checkpoint_root / "global_step_20").exists()
    checks = {
        "baseline_step_times_present": bool(baseline_times),
        "probe_step_times_present": bool(probe_times),
        "probe_settings_resolved": interval is not None and group_limit is not None,
        "no_fatal_log_pattern": not any(token in probe_text for token in _FATAL),
        "exit_status_zero": exit_status.exists() and exit_status.read_text().strip() == "0",
        "checkpoint_step5_exists": (checkpoint_root / "global_step_5").exists(),
        "checkpoint_step20_exists": (checkpoint_root / "global_step_20").exists(),
        "checkpoint_restore_passed": restore_passed,
        "online_history_nonempty": online_history.exists() and online_history.stat().st_size > 0,
        "heldout_history_nonempty": heldout_history.exists() and heldout_history.stat().st_size > 0,
        "dashboard_image_count_141": len(figures) == 141,
    }
    reduced_figure_count = 0
    reduced_history: list[dict] = []
    if middle_branch:
        reduced_text = (
            reduced_probe_log.read_text(encoding="utf-8", errors="replace")
            if reduced_probe_log and reduced_probe_log.exists()
            else ""
        )
        reduced_history_path = (
            reduced_probe_result / "online_hidden" / "per_step_summary.jsonl"
            if reduced_probe_result
            else None
        )
        reduced_history = _history(reduced_history_path)
        if reduced_probe_result:
            reduced_figure_count = len(
                list((reduced_probe_result / "online_hidden" / "latest_figures").glob("*.png"))
            )
        reduced_exit = reduced_probe_result / "exit_status.txt" if reduced_probe_result else None
        checks.update(
            {
                "reduced_step_times_present": len(reduced_times) >= 5,
                "reduced_overhead_at_most_50_percent": reduced_ratio is not None and reduced_ratio <= 1.50,
                "reduced_no_fatal_log_pattern": not any(token in reduced_text for token in _FATAL),
                "reduced_exit_status_zero": bool(
                    reduced_exit and reduced_exit.exists() and reduced_exit.read_text().strip() == "0"
                ),
                "reduced_history_has_five_steps": len(reduced_history) == 5,
                "reduced_probed_questions_16": len(reduced_history) == 5
                and all(record.get("hidden/_probed_questions") == 16 for record in reduced_history),
                "reduced_dashboard_image_count_141": reduced_figure_count == 141,
            }
        )
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "baseline_step_time_median_s": baseline_median,
        "probe_step_time_median_s": probe_median,
        "probe_overhead_ratio": full_ratio,
        "full_probe_overhead_ratio": full_ratio,
        "reduced_probe_step_time_median_s": reduced_median,
        "reduced_probe_overhead_ratio": reduced_ratio,
        "hidden_probe_group_limit": group_limit,
        "hidden_probe_interval": interval,
        "checkpoint_restore_passed": restore_passed,
        "dashboard_image_count": len(figures),
        "baseline_step_time_samples": len(baseline_times),
        "probe_step_time_samples": len(probe_times),
        "reduced_probe_step_time_samples": len(reduced_times),
        "reduced_dashboard_image_count": reduced_figure_count,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--probe-log", type=Path, required=True)
    parser.add_argument("--probe-result", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--reduced-probe-log", type=Path)
    parser.add_argument("--reduced-probe-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
