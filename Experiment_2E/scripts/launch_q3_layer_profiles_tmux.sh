#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-smoke}
PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814}
RUN_ROOT=${RUN_ROOT:-/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814}
LOG_ROOT=${LOG_ROOT:-/data2/hjk/logs/experiment_2e}

case "${MODE}" in
  smoke)
    SESSION=${SESSION:-q3_layer_profiles_smoke}
    LOG=${LOG:-${LOG_ROOT}/q3_layer_profiles_smoke_20260818.log}
    ;;
  formal)
    SESSION=${SESSION:-q3_layer_profiles_formal}
    LOG=${LOG:-${LOG_ROOT}/q3_layer_profiles_formal_20260818.log}
    APPROVAL="${RUN_ROOT}/layer_profiles_smoke/smoke_approval.json"
    python3 - "${APPROVAL}" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get('status') == 'passed', p
assert p.get("expected_steps") == [0, 50, 250], p
assert p.get("expected_questions") == 16, p
PY
    ;;
  *)
    echo "MODE must be smoke or formal" >&2
    exit 2
    ;;
esac

test -f "${PROJECT_ROOT}/scripts/run_q3_layer_profiles.sh"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi
if pgrep -af 'experiment_2e.q3_layer_extract|verl.model_merger|verl.trainer.main_ppo' >/dev/null; then
  echo "a trainer, model merger, or Q3 layer extractor is active; refusing to launch" >&2
  pgrep -af 'experiment_2e.q3_layer_extract|verl.model_merger|verl.trainer.main_ppo' >&2
  exit 4
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[0-9]+'; then
  echo "H200 has an active compute process; refusing to launch" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
  exit 5
fi

mkdir -p "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "cd '${PROJECT_ROOT}' && MODE='${MODE}' PROJECT_ROOT='${PROJECT_ROOT}' RUN_ROOT='${RUN_ROOT}' bash scripts/run_q3_layer_profiles.sh 2>&1 | tee '${LOG}'"
tmux has-session -t "${SESSION}"
echo "Started tmux session=${SESSION}"
echo "Log=${LOG}"
echo "Attach with: tmux attach -t ${SESSION}"
