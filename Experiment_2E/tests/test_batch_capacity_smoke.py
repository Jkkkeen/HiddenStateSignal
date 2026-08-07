from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_batch_capacity_smoke.sh"


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_runner_uses_frozen_candidates_and_seven_steps() -> None:
    text = script_text()
    assert 'BATCHES=(2 4 8)' in text
    assert 'MEASURED_STEPS=${MEASURED_STEPS:-5}' in text
    assert 'WARMUP_STEPS=${WARMUP_STEPS:-2}' in text
    assert 'TOTAL_TRAINING_STEPS=$((SOURCE_STEP + WARMUP_STEPS + MEASURED_STEPS))' in text


def test_runner_isolates_checkpoint_and_ray_paths() -> None:
    text = script_text()
    assert 'ln -s "${SOURCE_CKPT}" "${CANDIDATE_CKPT}/global_step_${SOURCE_STEP}"' in text
    assert 'trainer.default_local_dir="${CANDIDATE_CKPT}"' in text
    assert 'local RAY_TMPDIR="${RAY_BASE}/s${SOURCE_STEP}_b${BATCH_SIZE}_$$"' in text
    assert '"${CANDIDATE_ROOT}/ray_tmpdir.txt"' in text
    assert 'test "$(tr -d \'[:space:]\' < "${SOURCE_CKPT_ROOT}/latest_checkpointed_iteration.txt")" = "${SOURCE_STEP}"' in text


def test_runner_keeps_rollout_n_and_parameterizes_train_batch() -> None:
    text = script_text()
    assert 'TRAIN_BATCH_SIZE="${BATCH_SIZE}"' in text
    assert 'ROLLOUT_N=8' in text
    assert 'data.train_batch_size="${BATCH_SIZE}"' in text


def test_runner_guards_gpu_and_records_telemetry() -> None:
    text = script_text()
    assert 'assert_gpu_idle' in text
    assert '--query-compute-apps=pid' in text
    assert '--query-gpu=timestamp,memory.total,memory.used,memory.free,utilization.gpu' in text
    assert 'exit_status.txt' in text
