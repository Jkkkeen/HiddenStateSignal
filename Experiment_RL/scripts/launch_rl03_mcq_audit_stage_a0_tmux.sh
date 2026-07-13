#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}
RAW_ROLLOUTS=${RAW_ROLLOUTS:-${PROJECT_ROOT}/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl}
FEATURES_PATH=${FEATURES_PATH:-${PROJECT_ROOT}/option_logit_trimmed_conclusion_smoke500/trimmed_conclusion_features.parquet}
MATHVERSE_METADATA=${MATHVERSE_METADATA:-${PROJECT_ROOT}/data/mathverse/testmini.json}
SESSION_NAME=${SESSION_NAME:-rl03_stage_a0_smoke32}
RUN_NAME=${RUN_NAME:-rl03_stage_a0_smoke32_seed20260713}
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/rl03_results/${RUN_NAME}}
LOG_PATH=${LOG_PATH:-${PROJECT_ROOT}/logs/${RUN_NAME}.log}
SEED=${SEED:-20260713}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-10000}
OVERWRITE=${OVERWRITE:-0}

require_data2_path() {
  resolved=$(realpath -m "$1")
  case "${resolved}" in
    /data2|/data2/*) ;;
    *)
      echo "RL03 paths must stay under /data2: ${resolved}" >&2
      exit 2
      ;;
  esac
}
for data2_path in \
  "${PROJECT_ROOT}" \
  "${ENV_ROOT}" \
  "${MODEL_PATH}" \
  "${RAW_ROLLOUTS}" \
  "${FEATURES_PATH}" \
  "${MATHVERSE_METADATA}" \
  "${RUN_ROOT}" \
  "${LOG_PATH}"; do
  require_data2_path "${data2_path}"
done

mkdir -p "$(dirname "${LOG_PATH}")" "${RUN_ROOT}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 2
fi

command="cd '${PROJECT_ROOT}' && \
PROJECT_ROOT='${PROJECT_ROOT}' RUN_NAME='${RUN_NAME}' RUN_ROOT='${RUN_ROOT}' \
ENV_ROOT='${ENV_ROOT}' MODEL_PATH='${MODEL_PATH}' \
RAW_ROLLOUTS='${RAW_ROLLOUTS}' FEATURES_PATH='${FEATURES_PATH}' \
MATHVERSE_METADATA='${MATHVERSE_METADATA}' SEED='${SEED}' \
BATCH_SIZE='${BATCH_SIZE}' MAX_MODEL_LEN='${MAX_MODEL_LEN}' \
GPU_MEMORY_UTILIZATION='${GPU_MEMORY_UTILIZATION}' \
BOOTSTRAP_SAMPLES='${BOOTSTRAP_SAMPLES}' OVERWRITE='${OVERWRITE}' \
bash scripts/run_rl03_mcq_audit_stage_a0.sh > '${LOG_PATH}' 2>&1"

tmux new-session -d -s "${SESSION_NAME}" "${command}"
echo "session=${SESSION_NAME}"
echo "log=${LOG_PATH}"
echo "output=${RUN_ROOT}"
echo "attach: tmux attach -t ${SESSION_NAME}"
echo "progress: tail -f ${LOG_PATH}"
tmux list-panes -t "${SESSION_NAME}" -F '#{session_name} pid=#{pane_pid} cmd=#{pane_current_command}'
