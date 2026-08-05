#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
SESSION=${SESSION:-exp2e_assets}
LOG=${LOG:-/data2/hjk/logs/experiment_2e/assets.log}

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "cd '${PROJECT_ROOT}' && bash scripts/download_assets.sh 2>&1 | tee '${LOG}'"
echo "Started ${SESSION}; inspect with: tmux attach -t ${SESSION}"
