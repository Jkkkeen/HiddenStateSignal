from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_q3_1p7b_base_grpo.sh"
)


def test_q3_runner_freezes_remove_padding_true_by_default() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}" in text
    assert '"remove_padding":${USE_REMOVE_PADDING,,}' in text
    assert (
        'actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}"'
        in text
    )


def test_q3_runner_rejects_unknown_remove_padding_values() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'case "${USE_REMOVE_PADDING}" in' in text
    assert "True|False)" in text
    assert "USE_REMOVE_PADDING must be True or False" in text


def test_q3_runner_freezes_hidden_probe_group_limit() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "HIDDEN_PROBE_GROUP_LIMIT=${HIDDEN_PROBE_GROUP_LIMIT:-32}" in text
    assert "HIDDEN_PROBE_GROUP_LIMIT must be an integer in 1..32" in text
    assert '"hidden_probe_group_limit":${HIDDEN_PROBE_GROUP_LIMIT}' in text


def test_q3_formal_runner_reads_probe_settings_from_approval() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "p.get('hidden_probe_group_limit') in (16, 32)" in text
    assert "p['hidden_probe_group_limit']" in text
    assert "p['hidden_probe_interval']" in text
    assert "export HIDDEN_PROBE_GROUP_LIMIT HIDDEN_PROBE_INTERVAL" in text


def test_q3_behavior_reward_switch_and_lambda_are_forwarded() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "BEHAVIOR_REWARD_ENABLED=${BEHAVIOR_REWARD_ENABLED:-0}" in text
    assert "BEHAVIOR_LAMBDA=${BEHAVIOR_LAMBDA:-0.2}" in text
    assert "EXPERIMENT_2E_BEHAVIOR_REWARD" in text
    assert "EXPERIMENT_2E_BEHAVIOR_LAMBDA" in text
    assert '"behavior_reward_enabled":${BEHAVIOR_REWARD_ENABLED}' in text
    assert '"behavior_lambda":${BEHAVIOR_LAMBDA}' in text
