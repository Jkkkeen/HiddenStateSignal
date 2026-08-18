from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .profiles import METRIC_REGISTRY


FORBIDDEN_DATA_PATTERNS = (
    "*.npz",
    "*.npy",
    "*.safetensors",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "pytorch_model*.bin",
    "model*.bin",
)
REQUIRED_README_MARKERS = (
    "preflight",
    "smoke",
    "tmux",
    "base calibration",
    "bootstrap",
    "cluster permutation",
    "do not push",
)


def _forbidden_candidates(repo_root: Path) -> list[Path]:
    git_dir = repo_root / ".git"
    if git_dir.exists():
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if tracked.returncode == 0:
            names = [name for name in tracked.stdout.decode().split("\0") if name]
            return [
                repo_root / name
                for name in names
                if any(fnmatch.fnmatch(name, pattern) for pattern in FORBIDDEN_DATA_PATTERNS)
            ]
    return [
        path
        for path in repo_root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and any(fnmatch.fnmatch(path.name, pattern) for pattern in FORBIDDEN_DATA_PATTERNS)
    ]


def _run_tests(repo_root: Path) -> dict[str, Any]:
    tests_dir = repo_root / "tests"
    if not tests_dir.is_dir():
        return {"passed": True, "status": "skipped_no_tests"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-q",
            "--ignore",
            "tests/test_release_gate.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "passed": result.returncode == 0,
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "tail": result.stdout[-2000:] + result.stderr[-2000:],
    }


def _shell_syntax(repo_root: Path) -> dict[str, Any]:
    scripts = sorted((repo_root / "scripts").glob("*.sh"))
    bash = shutil.which("bash")
    if bash is None:
        return {
            "passed": True,
            "status": "skipped_bash_unavailable",
            "required_on": "Linux formal host",
        }
    result = subprocess.run(
        [bash, "-n", *(str(path) for path in scripts)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "passed": result.returncode == 0,
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "stderr": result.stderr[-2000:],
    }


def _fixture_size(repo_root: Path) -> dict[str, Any]:
    generator = repo_root / "examples" / "synthetic_hidden" / "make_fixture.py"
    if not generator.is_file():
        return {"passed": False, "status": "generator_missing"}
    with tempfile.TemporaryDirectory(prefix="vertical_fixture_") as temporary:
        result = subprocess.run(
            [sys.executable, str(generator), "--output", temporary],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        files = list(Path(temporary).rglob("*.npz"))
        max_bytes = max((path.stat().st_size for path in files), default=0)
        passed = (
            result.returncode == 0
            and (Path(temporary) / "qwen3_base_raw").is_dir()
            and (Path(temporary) / "pooled_example").is_dir()
            and max_bytes < 1 << 20
        )
        return {
            "passed": passed,
            "status": "passed" if passed else "failed",
            "returncode": int(result.returncode),
            "npz_count": int(len(files)),
            "max_npz_bytes": int(max_bytes),
            "stderr": result.stderr[-2000:],
        }


def run_release_gate(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    forbidden = _forbidden_candidates(repo_root)
    tracked_data_gate = not forbidden
    readme = repo_root / "vertical_readme.md"
    readme_text = readme.read_text(encoding="utf-8").lower() if readme.is_file() else ""
    readme_command_gate = all(marker in readme_text for marker in REQUIRED_README_MARKERS)
    metric_registry_gate = bool(METRIC_REGISTRY) and all(
        entry.get("status") in {"stable", "experimental"}
        and entry.get("release") in {"v0.1", "v0.2"}
        for entry in METRIC_REGISTRY.values()
    )
    tests = _run_tests(repo_root)
    shell = _shell_syntax(repo_root)
    fixture = _fixture_size(repo_root)
    diff_check = subprocess.run(
        ["git", "diff", "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    ) if (repo_root / ".git").exists() else None
    diff_gate = diff_check is None or diff_check.returncode == 0
    result: dict[str, Any] = {
        "passed": bool(
            tracked_data_gate
            and tests["passed"]
            and shell["passed"]
            and fixture["passed"]
            and readme_command_gate
            and metric_registry_gate
            and diff_gate
        ),
        "tracked_data_gate": tracked_data_gate,
        "forbidden_data_candidates": [str(path) for path in forbidden],
        "tests_gate": tests["passed"],
        "tests": tests,
        "shell_syntax_gate": shell["passed"],
        "shell_syntax": shell,
        "fixture_size_gate": fixture["passed"],
        "fixture": fixture,
        "readme_command_gate": readme_command_gate,
        "metric_registry_gate": metric_registry_gate,
        "diff_check_gate": diff_gate,
    }
    return result


def write_release_gate(result: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a vertical-toolkit release")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_release_gate(args.repo_root)
    if args.output is not None:
        write_release_gate(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
