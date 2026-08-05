#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
SESSION=${SESSION:-exp2e_rollouts_formal}
LOG=${LOG:-/data2/hjk/logs/experiment_2e/rollouts_formal_launcher.log}
: "${MANIFEST:?Set MANIFEST to frozen formal_run_manifest.json}"
: "${CKPT_DIR:?Set CKPT_DIR to the audited formal checkpoint root}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if pgrep -af 'experiment_2e.rollouts|verl.trainer.main_ppo' >/dev/null; then
  echo "A rollout generator or veRL trainer is active; refusing to launch." >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "GPU has an active compute process; refusing to launch." >&2
  exit 1
fi
mkdir -p "$(dirname "${LOG}")"
tmux new-session -d -s "${SESSION}" \
  "cd '${PROJECT_ROOT}' && MANIFEST='${MANIFEST}' CKPT_DIR='${CKPT_DIR}' bash scripts/run_rollouts_formal.sh 2>&1 | tee '${LOG}'"
echo "Started ${SESSION}; inspect with: tmux attach -t ${SESSION}"
