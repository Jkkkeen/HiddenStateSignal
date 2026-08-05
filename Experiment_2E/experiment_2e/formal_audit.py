from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .common import sha256_file, write_json_atomic
from .formal_manifest import checkpoint_schedule
from .grpo_audit import parse_step_metrics


def audit_formal_training(
    *,
    run_manifest_path: Path,
    checkpoint_dir: Path,
    log_paths: list[Path],
) -> dict[str, Any]:
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    total_steps = int(manifest["total_training_steps"])
    schedule = checkpoint_schedule(total_steps)
    required_steps = [int(row["global_step"]) for row in schedule if row["checkpoint"] != "base"]
    found_steps = sorted(
        int(path.name.removeprefix("global_step_"))
        for path in checkpoint_dir.glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
    )
    metrics = parse_step_metrics(log_paths)
    finite = all(math.isfinite(value) for row in metrics for value in row.values())
    logged_steps = [int(row["training/global_step"]) for row in metrics]
    latest_path = checkpoint_dir / "latest_checkpointed_iteration.txt"
    latest = int(latest_path.read_text(encoding="utf-8").strip()) if latest_path.exists() else None
    missing_steps = sorted(set(required_steps) - set(found_steps))
    passed = not missing_steps and latest == total_steps and finite and total_steps in logged_steps
    return {
        "passed": passed,
        "run_id": manifest["run_id"],
        "required_checkpoint_steps": required_steps,
        "found_checkpoint_steps": found_steps,
        "missing_checkpoint_steps": missing_steps,
        "latest_checkpointed_iteration": latest,
        "n_logged_steps": len(logged_steps),
        "max_logged_step": max(logged_steps, default=None),
        "all_logged_metrics_finite": finite,
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "logs": {str(path): sha256_file(path) for path in log_paths},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit formal Experiment 2E checkpoints and GRPO logs.")
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_formal_training(
        run_manifest_path=args.run_manifest,
        checkpoint_dir=args.checkpoint_dir,
        log_paths=args.log,
    )
    write_json_atomic(args.output, audit)
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit("Formal GRPO audit failed")


if __name__ == "__main__":
    main()
