#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
SESSION=${SESSION:-exp2e_grpo_smoke}
LOG=${LOG:-/data2/hjk/logs/experiment_2e/grpo_smoke_launcher.log}

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if pgrep -af 'verl.trainer.main_ppo' >/dev/null; then
  echo "Another veRL trainer is active; refusing to launch." >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "GPU has an active compute process; refusing to launch." >&2
  exit 1
fi
mkdir -p "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "cd '${PROJECT_ROOT}' && bash scripts/run_grpo_smoke_with_resume.sh 2>&1 | tee '${LOG}'"
echo "Started ${SESSION}; inspect with: tmux attach -t ${SESSION}"
