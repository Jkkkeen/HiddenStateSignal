#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814}
RUN_NAME=${RUN_NAME:-qwen3_1p7b_base_simplerl_behavior_reward_formal_seed20260824}
MODE=${MODE:-formal}
SESSION=${SESSION:-q3_behavior_reward_formal_20260824}
RESULT_ROOT=${RESULT_ROOT:-/data2/hjk/results/experiment_2e/${RUN_NAME}}
CKPT_ROOT=${CKPT_ROOT:-/data2/hjk/checkpoints/experiment_2e/${RUN_NAME}}
LOG=${LOG:-/data2/hjk/logs/experiment_2e/${RUN_NAME}.log}
BEHAVIOR_LAMBDA=${BEHAVIOR_LAMBDA:-0.2}

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

if pgrep -af 'recipe.dapo.main_dapo' >/dev/null 2>&1; then
  echo "an existing veRL trainer process is running" >&2
  exit 4
fi

GPU_STATE=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
if [[ -z "${GPU_STATE}" ]]; then
  echo "no visible GPU" >&2
  exit 5
fi
GPU_USED_MB=$(printf '%s\n' "${GPU_STATE}" | head -n 1 | cut -d, -f2 | tr -d '[:space:]')
if [[ "${GPU_USED_MB}" =~ ^[0-9]+$ ]] && (( GPU_USED_MB > 8192 )); then
  echo "GPU memory is already in use: ${GPU_STATE}" >&2
  exit 6
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${PROJECT_ROOT}' && MODE='${MODE}' BEHAVIOR_REWARD_ENABLED=1 BEHAVIOR_LAMBDA='${BEHAVIOR_LAMBDA}' RUN_NAME='${RUN_NAME}' RESULT_ROOT='${RESULT_ROOT}' CKPT_ROOT='${CKPT_ROOT}' LOG='${LOG}' bash scripts/run_q3_1p7b_base_grpo.sh"

echo "started tmux session ${SESSION}"
echo "run=${RUN_NAME}"
echo "log=${LOG}"
