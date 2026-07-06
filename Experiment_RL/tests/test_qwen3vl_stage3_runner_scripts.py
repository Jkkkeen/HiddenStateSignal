from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def read_script(name: str) -> str:
    path = SCRIPTS / name
    assert path.exists(), f"missing runner script: {path}"
    return path.read_text(encoding="utf-8")


def test_stage3_a1_and_c2_use_matched_core_settings() -> None:
    a1 = read_script("run_verl_qwen3vl_stage3_a1_long_pilot.sh")
    c2 = read_script("run_verl_qwen3vl_stage3_c2_zscore_pilot.sh")

    common_needles = [
        "rl_data/mathverse_qwen3vl_stage3_pilot200_128",
        "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}",
        "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}",
        "ROLLOUT_N=${ROLLOUT_N:-2}",
        "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}",
        "TOTAL_STEPS=${TOTAL_STEPS:-50}",
        "TEST_FREQ=${TEST_FREQ:--1}",
        "VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}",
        "trainer.val_before_train=\"${VAL_BEFORE_TRAIN}\"",
        "actor_rollout_ref.rollout.max_model_len=\"${MAX_MODEL_LEN}\"",
        "data.image_key=images",
    ]
    for script in (a1, c2):
        for needle in common_needles:
            assert needle in script


def test_stage3_reward_difference_is_only_reward_manager_path() -> None:
    a1 = read_script("run_verl_qwen3vl_stage3_a1_long_pilot.sh")
    c2 = read_script("run_verl_qwen3vl_stage3_c2_zscore_pilot.sh")

    assert "scripts/mathverse_answer_reward.py" in a1
    assert "scripts/mathverse_qwen3vl_option_gain_reward.py" in c2
    assert "Qwen3VLC2ZscoreRewardManager" not in a1
    assert "Qwen3VLC2ZscoreRewardManager" in c2
    assert "OPTION_GAIN_LAMBDA=${OPTION_GAIN_LAMBDA:-0.2}" in c2
    assert "OPTION_GAIN_REWARD_START_INDEX=${OPTION_GAIN_REWARD_START_INDEX:-1}" in c2
    assert "OPTION_GAIN_MAX_RESPONSE_CHARS=${OPTION_GAIN_MAX_RESPONSE_CHARS:-4096}" in c2


def test_stage3_scripts_keep_runtime_state_on_data2() -> None:
    for name in (
        "run_verl_qwen3vl_stage3_a1_long_pilot.sh",
        "run_verl_qwen3vl_stage3_c2_zscore_pilot.sh",
    ):
        script = read_script(name)
        for needle in (
            "HF_HOME=${HF_HOME:-/data2/hjk/cache/huggingface}",
            "PIP_CACHE_DIR=${PIP_CACHE_DIR:-/data2/hjk/cache/pip}",
            "TMPDIR=${TMPDIR:-/data2/hjk/tmp}",
            "RAY_TMPDIR=${RAY_TMPDIR:-/data2/hjk/cache/ray}",
            "WANDB_DIR=${WANDB_DIR:-/data2/hjk/checkpoints/wandb}",
            "CKPT_DIR=${CKPT_DIR:-/data2/hjk/checkpoints/verl_qwen3vl/${RUN_NAME}}",
        ):
            assert needle in script


def test_stage3_tmux_launcher_dispatches_a1_and_c2() -> None:
    launcher = read_script("launch_qwen3vl_stage3_tmux.sh")

    assert "EXPERIMENT=${EXPERIMENT:-a1}" in launcher
    assert "run_verl_qwen3vl_stage3_a1_long_pilot.sh" in launcher
    assert "run_verl_qwen3vl_stage3_c2_zscore_pilot.sh" in launcher
    assert "TMUX_A1_EXIT_CODE" in launcher
    assert "TMUX_C2_EXIT_CODE" in launcher
    assert "code=\\$?" in launcher
    assert "sleep ${KEEP_OPEN_SECONDS}" in launcher
