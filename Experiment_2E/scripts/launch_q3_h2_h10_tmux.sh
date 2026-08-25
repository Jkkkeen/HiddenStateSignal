#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814}
RUN_ROOT=${RUN_ROOT:-/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814}
LOG_ROOT=${LOG_ROOT:-/data2/hjk/logs/experiment_2e}
MODE=${MODE:-formal}
SESSION=${SESSION:-q3_h2_h10_formal}
LOG=${LOG:-${LOG_ROOT}/q3_h2_h10_formal_20260824.log}

test -f "${PROJECT_ROOT}/scripts/run_q3_h2_h10.sh"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi
if pgrep -af 'experiment_2e.q3_h2_h10_extract|verl.model_merger' >/dev/null; then
  echo "a Q3 H2/H10 extractor or model merger is active; refusing to launch" >&2
  pgrep -af 'experiment_2e.q3_h2_h10_extract|verl.model_merger' >&2
  exit 4
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[0-9]+'; then
  echo "H200 has an active compute process; refusing to launch" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
  exit 5
fi

mkdir -p "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "set -o pipefail; cd '${PROJECT_ROOT}' && MODE='${MODE}' PROJECT_ROOT='${PROJECT_ROOT}' RUN_ROOT='${RUN_ROOT}' bash scripts/run_q3_h2_h10.sh 2>&1 | tee '${LOG}'"
tmux has-session -t "${SESSION}"
echo "Started tmux session=${SESSION}"
echo "Log=${LOG}"
