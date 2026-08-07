from __future__ import annotations

from pathlib import Path

import pytest

from scripts.patch_verl_temp_loop import (
    BACKUP_SUFFIX,
    NEW_BLOCK,
    OLD_BLOCK,
    PATCH_MARKER,
    apply_patch,
    restore_patch,
)


def fake_verl(tmp_path: Path, block: str = OLD_BLOCK) -> tuple[Path, Path, bytes]:
    target = tmp_path / "verl" / "utils" / "transferqueue_utils.py"
    target.parent.mkdir(parents=True)
    original = ("prefix\n" + block + "suffix\n").encode()
    target.write_bytes(original)
    return tmp_path, target, original


def test_apply_is_idempotent_and_restore_is_exact(tmp_path: Path) -> None:
    root, target, original = fake_verl(tmp_path)
    first = apply_patch(root)
    assert first.action == "patched"
    assert PATCH_MARKER in target.read_text()
    assert OLD_BLOCK not in target.read_text()
    assert Path(f"{target}{BACKUP_SUFFIX}").read_bytes() == original

    second = apply_patch(root)
    assert second.action == "already_patched"
    assert second.before_sha256 == first.before_sha256
    assert second.after_sha256 == first.after_sha256

    restored = restore_patch(root)
    assert restored.action == "restored"
    assert target.read_bytes() == original


def test_refuses_unknown_source_shape(tmp_path: Path) -> None:
    root, target, _ = fake_verl(tmp_path, "unknown lifecycle\n")
    with pytest.raises(RuntimeError, match="refusing to patch"):
        apply_patch(root)
    assert not Path(f"{target}{BACKUP_SUFFIX}").exists()


def test_refuses_mismatched_existing_backup(tmp_path: Path) -> None:
    root, target, _ = fake_verl(tmp_path)
    Path(f"{target}{BACKUP_SUFFIX}").write_text("different")
    with pytest.raises(RuntimeError, match="backup does not match"):
        apply_patch(root)


def test_new_block_stops_joins_and_closes_loop() -> None:
    assert "call_soon_threadsafe(tmp_event_loop.stop)" in NEW_BLOCK
    assert "thread.join()" in NEW_BLOCK
    assert "tmp_event_loop.close()" in NEW_BLOCK
    assert "run_coroutine_threadsafe(stop_loop" not in NEW_BLOCK
