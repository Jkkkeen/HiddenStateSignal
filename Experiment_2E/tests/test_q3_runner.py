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
