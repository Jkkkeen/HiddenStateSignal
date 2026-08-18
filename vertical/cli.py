from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .audit import write_json_atomic
from .config import VerticalConfig
from .inspect import inspect_source


def validate_smoke_approval(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"smoke approval file is missing: {path}")
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"smoke approval is unreadable: {path}") from exc
    passed = approval.get("status") == "passed" or approval.get("passed") is True
    if not passed:
        raise ValueError(f"smoke approval did not pass: {path}")
    return approval


def render_tmux_command(
    session_name: str,
    runner: Path,
    *,
    config: Path,
    output_root: Path,
    smoke_approval: Path,
    log_path: Path,
    project_root: Path | None = None,
) -> list[str]:
    if not session_name or any(character.isspace() for character in session_name):
        raise ValueError("tmux session_name must be nonempty and contain no whitespace")
    root = Path(project_root or Path.cwd()).resolve()
    runner_path = Path(runner)
    if not runner_path.is_absolute():
        runner_path = root / runner_path
    shell_command = " ".join(
        [
            "set -o pipefail;",
            "cd",
            shlex.quote(str(root)),
            "&& bash",
            shlex.quote(str(runner_path)),
            "--config",
            shlex.quote(str(Path(config).resolve())),
            "--output-root",
            shlex.quote(str(Path(output_root).resolve())),
            "--smoke-approval",
            shlex.quote(str(Path(smoke_approval).resolve())),
            "2>&1 | tee",
            shlex.quote(str(Path(log_path).resolve())),
        ]
    )
    return [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session_name,
        "bash",
        "-lc",
        shell_command,
    ]


def _inspect_command(args: argparse.Namespace) -> int:
    config = VerticalConfig.from_yaml(args.config)
    reports: dict[str, dict[str, Any]] = {}
    for dataset in config.datasets:
        single = replace(config, datasets=(dataset,))
        report = inspect_source(dataset.source, single, sample_limit=args.sample_limit)
        reports[dataset.name] = report.to_dict()
    payload = {
        "run_id": config.run_id,
        "passed": bool(reports) and all(report["passed"] for report in reports.values()),
        "datasets": reports,
    }
    write_json_atomic(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def _pipeline_command(args: argparse.Namespace, *, mode: str) -> int:
    if mode == "formal":
        validate_smoke_approval(args.smoke_approval)
    from .pipeline import run_pipeline

    result = run_pipeline(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        mode=mode,
        smoke_approval=(
            None if mode == "smoke" else Path(args.smoke_approval)
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") else 1


def _config_value_command(args: argparse.Namespace) -> int:
    config = VerticalConfig.from_yaml(args.config)
    values = {
        "run_id": config.run_id,
        "output_root": str(config.output_root),
    }
    print(values[args.field])
    return 0


def _validate_approval_command(args: argparse.Namespace) -> int:
    approval = validate_smoke_approval(args.path)
    print(json.dumps(approval, indent=2, sort_keys=True))
    return 0


def _launch_tmux_command(args: argparse.Namespace) -> int:
    config = VerticalConfig.from_yaml(args.config)
    output_root = Path(args.output_root or config.output_root)
    approval = Path(args.smoke_approval or output_root / "smoke_approval.json")
    validate_smoke_approval(approval)
    if shutil.which("tmux") is None:
        raise ValueError("tmux is not installed or not on PATH")
    session = args.session or f"vertical_formal_{config.run_id}"
    session = "".join(character if character.isalnum() or character in "_-" else "_" for character in session)
    exists = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
        check=False,
    )
    if exists.returncode == 0:
        raise ValueError(f"tmux session already exists: {session}")
    log_path = Path(args.log or output_root / "logs" / "formal.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = render_tmux_command(
        session,
        Path(args.runner),
        config=Path(args.config),
        output_root=output_root,
        smoke_approval=approval,
        log_path=log_path,
        project_root=Path(args.project_root),
    )
    subprocess.run(command, check=True)
    print(json.dumps({"session": session, "log": str(log_path)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline vertical hidden-state toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--config", type=Path, required=True)
    inspect_parser.add_argument("--output", type=Path, required=True)
    inspect_parser.add_argument("--sample-limit", type=int, default=16)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--config", type=Path, required=True)
    smoke_parser.add_argument("--output-root", type=Path, required=True)

    formal_parser = subparsers.add_parser("formal")
    formal_parser.add_argument("--config", type=Path, required=True)
    formal_parser.add_argument("--output-root", type=Path, required=True)
    formal_parser.add_argument("--smoke-approval", type=Path, required=True)

    value_parser = subparsers.add_parser("config-value")
    value_parser.add_argument("--config", type=Path, required=True)
    value_parser.add_argument("--field", choices=("run_id", "output_root"), required=True)

    approval_parser = subparsers.add_parser("validate-approval")
    approval_parser.add_argument("--path", type=Path, required=True)

    tmux_parser = subparsers.add_parser("launch-tmux")
    tmux_parser.add_argument("--config", type=Path, required=True)
    tmux_parser.add_argument("--output-root", type=Path)
    tmux_parser.add_argument("--smoke-approval", type=Path)
    tmux_parser.add_argument("--session")
    tmux_parser.add_argument("--log", type=Path)
    tmux_parser.add_argument(
        "--runner",
        default="scripts/run_vertical_formal.sh",
    )
    tmux_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect_command(args)
        if args.command == "smoke":
            return _pipeline_command(args, mode="smoke")
        if args.command == "formal":
            return _pipeline_command(args, mode="formal")
        if args.command == "config-value":
            return _config_value_command(args)
        if args.command == "validate-approval":
            return _validate_approval_command(args)
        if args.command == "launch-tmux":
            return _launch_tmux_command(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
