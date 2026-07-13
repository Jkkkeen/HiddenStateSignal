from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_stage_a1_runner_selects_all_parseable_rollouts_and_reuses_safe_chain():
    text = (SCRIPTS / "run_rl03_mcq_audit_stage_a1.sh").read_text(encoding="utf-8")

    assert "STAGE_NAME=stage_a1" in text
    assert "SELECTION_MODE=all-parseable" in text
    assert "RUN_NAME=${RUN_NAME:-rl03_stage_a1_v2_full_seed20260713}" in text
    assert "run_rl03_mcq_audit_stage_a0.sh" in text
    assert "QUESTION_COUNT" not in text
    assert "REQUEST_LIMIT" not in text
    assert "sudo " not in text
    assert "ray stop" not in text
    assert "pkill " not in text
    assert "killall " not in text


def test_shared_stage_runner_supports_a1_manifest_without_changing_a0_defaults():
    text = (SCRIPTS / "run_rl03_mcq_audit_stage_a0.sh").read_text(encoding="utf-8")

    assert "STAGE_NAME=${STAGE_NAME:-stage_a0}" in text
    assert "SELECTION_MODE=${SELECTION_MODE:-mixed-smoke}" in text
    assert '"${MANIFEST_DIR}/${STAGE_NAME}_manifest.jsonl"' in text
    assert 'builder_args+=(--question-count 32)' in text
    assert 'builder_args+=(--selection-mode "${SELECTION_MODE}")' in text
    assert 'builder_args+=(--stage-name "${STAGE_NAME}")' in text


def test_stage_a1_tmux_launcher_is_idempotent_and_uses_distinct_names():
    text = (SCRIPTS / "launch_rl03_mcq_audit_stage_a1_tmux.sh").read_text(
        encoding="utf-8"
    )

    assert "SESSION_NAME=${SESSION_NAME:-rl03_stage_a1_v2_full}" in text
    assert "RUN_NAME=${RUN_NAME:-rl03_stage_a1_v2_full_seed20260713}" in text
    assert "tmux has-session" in text
    assert "tmux new-session -d" in text
    assert "tmux attach -t" in text
    assert "tail -f" in text
    assert "run_rl03_mcq_audit_stage_a1.sh" in text
    assert "require_data2_path" in text
    assert "realpath -m" in text
    assert "tmux kill-session" not in text
