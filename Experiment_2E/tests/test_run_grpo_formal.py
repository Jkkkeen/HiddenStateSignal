from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_grpo_formal.sh"
LAUNCHER = ROOT / "scripts" / "launch_grpo_formal_tmux.sh"


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_formal_runner_has_an_opt_in_swanlab_configuration() -> None:
    text = script_text()

    assert 'ENABLE_SWANLAB=${ENABLE_SWANLAB:-0}' in text
    assert 'SWANLAB_ENV_FILE=${SWANLAB_ENV_FILE:-/data2/hjk/secrets/experiment_2e_swanlab.env}' in text
    assert 'source "${SWANLAB_ENV_FILE}"' in text
    assert 'SWANLAB_API_KEY must be defined' in text
    assert 'SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-/data2/hjk/logs/experiment_2e/swanlab/${RUN_NAME}}' in text
    assert 'SWANLAB_MODE=${SWANLAB_MODE:-online}' in text
    assert 'SWANLAB_RUN_ID=${SWANLAB_RUN_ID:-${RUN_NAME}}' in text
    assert 'SWANLAB_RESUME=${SWANLAB_RESUME:-allow}' in text


def test_formal_runner_keeps_batch_one_rollout_eight_and_uses_native_logger() -> None:
    text = script_text()

    assert 'TRAIN_BATCH_SIZE=1' in text
    assert 'ROLLOUT_N=8' in text
    assert 'trainer.logger="${TRAINER_LOGGER}"' in text
    assert "TRAINER_LOGGER='[\"console\",\"swanlab\"]'" in text
    assert "TRAINER_LOGGER='[\"console\"]'" in text


def test_tmux_launcher_forwards_the_opt_in_swanlab_flag_without_a_secret() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'ENABLE_SWANLAB=${ENABLE_SWANLAB:-0}' in text
    assert "ENABLE_SWANLAB='${ENABLE_SWANLAB}'" in text
    assert "SWANLAB_API_KEY" not in text
