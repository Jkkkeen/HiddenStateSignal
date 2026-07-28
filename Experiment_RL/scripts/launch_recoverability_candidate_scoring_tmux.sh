#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
SESSION_NAME=${SESSION_NAME:-recoverability_candidate_scoring_v1}
RUN_NAME=${RUN_NAME:-${SESSION_NAME}}
LOG_PATH=${LOG_PATH:-${PROJECT_ROOT}/logs/${RUN_NAME}.log}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/rl_recoverability/${RUN_NAME}}

LIMIT=${LIMIT:--1}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}

mkdir -p "$(dirname "${LOG_PATH}")" "${OUTPUT_DIR}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 2
fi

command="cd '${PROJECT_ROOT}' && \
LIMIT='${LIMIT}' BATCH_SIZE='${BATCH_SIZE}' \
MAX_MODEL_LEN='${MAX_MODEL_LEN}' \
GPU_MEMORY_UTILIZATION='${GPU_MEMORY_UTILIZATION}' \
RUN_NAME='${RUN_NAME}' OUTPUT_DIR='${OUTPUT_DIR}' \
bash scripts/run_recoverability_candidate_scoring.sh > '${LOG_PATH}' 2>&1"

tmux new-session -d -s "${SESSION_NAME}" "${command}"
echo "session=${SESSION_NAME}"
echo "log=${LOG_PATH}"
echo "output=${OUTPUT_DIR}"
tmux list-panes -t "${SESSION_NAME}" -F '#{session_name} pid=#{pane_pid} cmd=#{pane_current_command}'
