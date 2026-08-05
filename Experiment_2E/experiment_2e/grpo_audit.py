from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from .common import sha256_file, write_json_atomic


ANSI = re.compile(r"\x1b\[[0-9;]*m")
METRIC = re.compile(
    r"(?P<key>[A-Za-z0-9_./-]+):(?:np\.(?:float64|int64)\()?"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)?"
)


def parse_step_metrics(log_paths: list[Path]) -> list[dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for log_path in log_paths:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = ANSI.sub("", raw_line)
                if "training/global_step:" not in line:
                    continue
                metrics = {match.group("key"): float(match.group("value")) for match in METRIC.finditer(line)}
                if "training/global_step" in metrics:
                    by_step[int(metrics["training/global_step"])] = metrics
    return [by_step[step] for step in sorted(by_step)]


def audit_grpo(log_paths: list[Path], checkpoint_dir: Path) -> dict[str, Any]:
    steps = parse_step_metrics(log_paths)
    mixed = [
        step.get("critic/score/max", 0.0) > step.get("critic/score/min", 0.0)
        for step in steps
        if "critic/score/max" in step and "critic/score/min" in step
    ]
    nonzero_advantage = [
        max(abs(step.get("critic/advantages/max", 0.0)), abs(step.get("critic/advantages/min", 0.0))) > 0.0
        for step in steps
    ]
    finite = all(math.isfinite(value) for step in steps for value in step.values())
    checkpoints = sorted(path.name for path in checkpoint_dir.glob("global_step_*") if path.is_dir())
    latest_file = checkpoint_dir / "latest_checkpointed_iteration.txt"
    latest = latest_file.read_text(encoding="utf-8").strip() if latest_file.exists() else None
    mixed_rate = sum(mixed) / len(mixed) if mixed else None
    return {
        "n_logged_steps": len(steps),
        "logged_global_steps": [int(step["training/global_step"]) for step in steps],
        "all_logged_metrics_finite": finite,
        "n_mixed_reward_groups": sum(mixed),
        "mixed_reward_group_rate": mixed_rate,
        "n_nonzero_advantage_steps": sum(nonzero_advantage),
        "hard_subset_triggered": bool(mixed_rate is not None and mixed_rate < 0.20),
        "checkpoint_directories": checkpoints,
        "latest_checkpointed_iteration": latest,
        "resume_verified": "global_step_5" in checkpoints and "global_step_10" in checkpoints and latest == "10",
        "logs": {str(path): sha256_file(path) for path in log_paths},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit finite metrics, mixed groups, and checkpoint resume in GRPO smoke.")
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_grpo(args.log, args.checkpoint_dir)
    write_json_atomic(args.output, audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_logged_metrics_finite"] or not audit["resume_verified"] or audit["n_nonzero_advantage_steps"] < 1:
        raise SystemExit("GRPO smoke audit failed")


if __name__ == "__main__":
    main()
