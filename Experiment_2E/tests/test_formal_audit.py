import json
from pathlib import Path

from experiment_2e.formal_audit import audit_formal_training


def test_formal_training_audit_requires_all_five_checkpoints(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"run_id": "formal", "total_training_steps": 100}),
        encoding="utf-8",
    )
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for step in (20, 40, 60, 80, 100):
        (checkpoints / f"global_step_{step}").mkdir()
    (checkpoints / "latest_checkpointed_iteration.txt").write_text("100\n", encoding="utf-8")
    log = tmp_path / "train.log"
    log.write_text(
        "step:100 - training/global_step:100 - actor/loss:np.float64(0.1)\n",
        encoding="utf-8",
    )
    audit = audit_formal_training(
        run_manifest_path=manifest,
        checkpoint_dir=checkpoints,
        log_paths=[log],
    )
    assert audit["passed"] is True
    assert audit["missing_checkpoint_steps"] == []


def test_formal_training_audit_fails_when_checkpoint_is_missing(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"run_id": "formal", "total_training_steps": 100}),
        encoding="utf-8",
    )
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for step in (20, 40, 80, 100):
        (checkpoints / f"global_step_{step}").mkdir()
    (checkpoints / "latest_checkpointed_iteration.txt").write_text("100\n", encoding="utf-8")
    log = tmp_path / "train.log"
    log.write_text("step:100 - training/global_step:100 - actor/loss:0.1\n", encoding="utf-8")
    audit = audit_formal_training(
        run_manifest_path=manifest,
        checkpoint_dir=checkpoints,
        log_paths=[log],
    )
    assert audit["passed"] is False
    assert audit["missing_checkpoint_steps"] == [60]
