import json

from experiment_2e.q3_smoke_audit import audit


def _write_probe_artifacts(result, *, questions=32, rows=1):
    (result / "online_hidden" / "latest_figures").mkdir(parents=True)
    for index in range(141):
        (result / "online_hidden" / "latest_figures" / f"{index:03d}.png").write_bytes(b"png")
    history = "".join(
        json.dumps({"training_step": index + 1, "hidden/_probed_questions": questions}) + "\n"
        for index in range(rows)
    )
    (result / "online_hidden" / "per_step_summary.jsonl").write_text(history, encoding="utf-8")
    (result / "heldout").mkdir()
    (result / "heldout" / "hidden_summary.jsonl").write_text("{}\n", encoding="utf-8")
    (result / "exit_status.txt").write_text("0\n", encoding="utf-8")


def _write_restore_artifacts(tmp_path):
    result = tmp_path / "result"
    _write_probe_artifacts(result)
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "global_step_5").mkdir(parents=True)
    (checkpoints / "global_step_20").mkdir()
    return result, checkpoints


def test_smoke_audit_freezes_every_five_steps_for_large_probe_overhead(tmp_path):
    baseline_log = tmp_path / "baseline.log"
    probe_log = tmp_path / "probe.log"
    baseline_log.write_text("{'timing_s/step': 10}\n" * 5, encoding="utf-8")
    probe_log.write_text("Setting global step to 5\n" + "{'timing_s/step': 16}\n" * 5, encoding="utf-8")
    result, checkpoints = _write_restore_artifacts(tmp_path)

    payload = audit(baseline_log, probe_log, result, checkpoints, tmp_path / "approval.json")
    assert payload["status"] == "passed"
    assert payload["hidden_probe_interval"] == 5
    assert payload["hidden_probe_group_limit"] == 32
    assert payload["checkpoint_restore_passed"] is True
    assert payload["dashboard_image_count"] == 141
    assert json.loads((tmp_path / "approval.json").read_text())["status"] == "passed"


def test_smoke_audit_approves_console_timing_and_reduced_middle_branch(tmp_path):
    baseline_log = tmp_path / "baseline.log"
    full_log = tmp_path / "full.log"
    reduced_log = tmp_path / "reduced.log"
    baseline_log.write_text("timing_s/step:100\n" * 5, encoding="utf-8")
    full_log.write_text("Setting global step to 5\n" + "timing_s/step:149\n" * 5, encoding="utf-8")
    reduced_log.write_text("timing_s/step:135\n" * 5, encoding="utf-8")
    full_result, checkpoints = _write_restore_artifacts(tmp_path)
    reduced_result = tmp_path / "reduced"
    _write_probe_artifacts(reduced_result, questions=16, rows=5)

    payload = audit(
        baseline_log,
        full_log,
        full_result,
        checkpoints,
        tmp_path / "approval.json",
        reduced_probe_log=reduced_log,
        reduced_probe_result=reduced_result,
    )

    assert payload["status"] == "passed"
    assert payload["baseline_step_time_samples"] == 5
    assert payload["full_probe_overhead_ratio"] == 1.49
    assert payload["reduced_probe_overhead_ratio"] == 1.35
    assert payload["hidden_probe_group_limit"] == 16
    assert payload["hidden_probe_interval"] == 1
    assert payload["checks"]["reduced_history_has_five_steps"] is True
    assert payload["checks"]["reduced_probed_questions_16"] is True


def test_smoke_audit_rejects_reduced_history_with_wrong_question_count(tmp_path):
    baseline_log = tmp_path / "baseline.log"
    full_log = tmp_path / "full.log"
    reduced_log = tmp_path / "reduced.log"
    baseline_log.write_text("timing_s/step:100\n" * 5, encoding="utf-8")
    full_log.write_text("Setting global step to 5\n" + "timing_s/step:149\n" * 5, encoding="utf-8")
    reduced_log.write_text("timing_s/step:135\n" * 5, encoding="utf-8")
    full_result, checkpoints = _write_restore_artifacts(tmp_path)
    reduced_result = tmp_path / "reduced"
    _write_probe_artifacts(reduced_result, questions=15, rows=5)

    payload = audit(
        baseline_log,
        full_log,
        full_result,
        checkpoints,
        tmp_path / "approval.json",
        reduced_probe_log=reduced_log,
        reduced_probe_result=reduced_result,
    )

    assert payload["status"] == "failed"
    assert payload["checks"]["reduced_probed_questions_16"] is False
