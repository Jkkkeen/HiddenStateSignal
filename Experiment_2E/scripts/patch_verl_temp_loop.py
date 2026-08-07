#!/usr/bin/env python3
"""Apply or restore the Experiment 2E veRL temporary-event-loop fix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


TARGET_RELATIVE = Path("verl/utils/transferqueue_utils.py")
BACKUP_SUFFIX = ".experiment2e.orig"
PATCH_MARKER = "# Experiment 2E: explicitly release epoll and wakeup-pipe FDs."

OLD_BLOCK = '''    async def stop_loop():
        tmp_event_loop.stop()

    try:
        return run_coroutine(async_func(*args, **kwargs))
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()
'''

NEW_BLOCK = f'''    try:
        return run_coroutine(async_func(*args, **kwargs))
    finally:
        if thread.is_alive():
            tmp_event_loop.call_soon_threadsafe(tmp_event_loop.stop)
            thread.join()
        {PATCH_MARKER}
        tmp_event_loop.close()
'''


@dataclass(frozen=True)
class PatchResult:
    action: str
    target: str
    backup: str
    before_sha256: str
    after_sha256: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def target_from_root(verl_root: Path) -> Path:
    target = verl_root / TARGET_RELATIVE
    if not target.is_file():
        raise FileNotFoundError(f"veRL source not found: {target}")
    return target


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def apply_patch(verl_root: Path) -> PatchResult:
    target = target_from_root(verl_root)
    backup = Path(f"{target}{BACKUP_SUFFIX}")
    before = target.read_bytes()
    source = before.decode("utf-8")

    if PATCH_MARKER in source:
        if OLD_BLOCK in source:
            raise RuntimeError("Patched marker and original event-loop block both present")
        original_hash = sha256(backup.read_bytes()) if backup.is_file() else sha256(before)
        return PatchResult("already_patched", str(target), str(backup), original_hash, sha256(before))

    count = source.count(OLD_BLOCK)
    if count != 1:
        raise RuntimeError(f"Expected exactly one known event-loop block, found {count}; refusing to patch")
    if backup.exists() and backup.read_bytes() != before:
        raise RuntimeError(f"Existing backup does not match current unpatched source: {backup}")
    if not backup.exists():
        _atomic_write(backup, before)

    after = source.replace(OLD_BLOCK, NEW_BLOCK, 1).encode("utf-8")
    _atomic_write(target, after)
    return PatchResult("patched", str(target), str(backup), sha256(before), sha256(after))


def restore_patch(verl_root: Path) -> PatchResult:
    target = target_from_root(verl_root)
    backup = Path(f"{target}{BACKUP_SUFFIX}")
    if not backup.is_file():
        raise FileNotFoundError(f"Patch backup not found: {backup}")
    before = target.read_bytes()
    original = backup.read_bytes()
    if PATCH_MARKER not in before.decode("utf-8") and before != original:
        raise RuntimeError("Target is neither the Experiment 2E patch nor its saved original")
    _atomic_write(target, original)
    return PatchResult("restored", str(target), str(backup), sha256(before), sha256(original))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", type=Path, default=Path("/data2/hjk/projects/verl"))
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args()

    result = restore_patch(args.verl_root) if args.restore else apply_patch(args.verl_root)
    payload = asdict(result)
    if args.audit is not None:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
