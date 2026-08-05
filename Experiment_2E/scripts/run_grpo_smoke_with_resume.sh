#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
SEED=${SEED:-20260805}
RUN_NAME=${RUN_NAME:-qwen25_7b_math_grpo_lora_smoke_seed${SEED}}
CKPT_DIR=${CKPT_DIR:-/data2/hjk/checkpoints/experiment_2e/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-/data2/hjk/logs/experiment_2e}

export RUN_NAME CKPT_DIR SEED

if pgrep -af 'verl.trainer.main_ppo' >/dev/null; then
  echo "Another veRL trainer is active; refusing to launch." >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "GPU has an active compute process; refusing to launch." >&2
  exit 1
fi

TOTAL_TRAINING_STEPS=5 SAVE_FREQ=5 RESUME_MODE=disable \
LOG="${LOG_DIR}/${RUN_NAME}_phase1.log" \
bash "${PROJECT_ROOT}/scripts/run_grpo_smoke_phase.sh"

test -d "${CKPT_DIR}/global_step_5"
"${ENV_ROOT}/bin/ray" stop -f || true

TOTAL_TRAINING_STEPS=10 SAVE_FREQ=5 RESUME_MODE=auto \
LOG="${LOG_DIR}/${RUN_NAME}_phase2_resume.log" \
bash "${PROJECT_ROOT}/scripts/run_grpo_smoke_phase.sh"

test -d "${CKPT_DIR}/global_step_10"
test "$(tr -d '[:space:]' < "${CKPT_DIR}/latest_checkpointed_iteration.txt")" = "10"
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${ENV_ROOT}/bin/python" -m experiment_2e.grpo_audit \
  --log "${LOG_DIR}/${RUN_NAME}_phase1.log" \
  --log "${LOG_DIR}/${RUN_NAME}_phase2_resume.log" \
  --checkpoint-dir "${CKPT_DIR}" \
  --output "${CKPT_DIR}/grpo_smoke_audit.json"
echo "GRPO_SMOKE_AND_RESUME_COMPLETE"
