from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import write_json_atomic


DEPENDENCIES = ("torch", "transformers", "vllm", "verl", "datasets", "pyarrow", "math-verify", "peft")


def _run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(error)}


def collect_environment_audit(model_path: Path | None, data_paths: list[Path]) -> dict[str, Any]:
    dependency_versions = {}
    for dependency in DEPENDENCIES:
        try:
            dependency_versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[dependency] = None
    try:
        hf_addresses = sorted({item[4][0] for item in socket.getaddrinfo("huggingface.co", 443)})
    except OSError as error:
        hf_addresses = [f"ERROR: {error}"]

    disk = {}
    for path in {Path("/"), Path("/data2")}:
        if path.exists():
            usage = shutil.disk_usage(path)
            disk[str(path)] = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}

    gpu_query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    compute_processes = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    assets = {
        "model": {"path": str(model_path) if model_path else None, "exists": bool(model_path and model_path.exists())},
        "data": [{"path": str(path), "exists": path.exists()} for path in data_paths],
    }
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "dependencies": dependency_versions,
        "gpu": gpu_query,
        "compute_processes": compute_processes,
        "disk": disk,
        "huggingface_dns": hf_addresses,
        "assets": assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the H200 environment before Experiment 2E starts.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-path", type=Path, action="append", default=[])
    parser.add_argument("--strict", action="store_true", help="Fail when GPU, dependencies, model, or data are missing")
    args = parser.parse_args()
    audit = collect_environment_audit(args.model_path, args.data_path)
    write_json_atomic(args.output, audit)
    print(json.dumps(audit, indent=2))
    if args.strict:
        missing_dependencies = [name for name, version in audit["dependencies"].items() if version is None]
        missing_assets = [asset["path"] for asset in audit["assets"]["data"] if not asset["exists"]]
        if args.model_path and not audit["assets"]["model"]["exists"]:
            missing_assets.append(str(args.model_path))
        if audit["gpu"]["returncode"] != 0 or missing_dependencies or missing_assets:
            raise SystemExit(f"Strict environment audit failed: dependencies={missing_dependencies}, assets={missing_assets}")


if __name__ == "__main__":
    main()
