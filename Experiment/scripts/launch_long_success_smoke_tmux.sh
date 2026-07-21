#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"
PYTHON="${PYTHON:-/data2/hjk/envs/hs_er/bin/python}"
SESSION="${SESSION:-two_dim_e1_smoke32}"
RUN_NAME="${RUN_NAME:-smoke32_20260721}"
RUN_DIR="${ROOT}/long_success_trajectory/${RUN_NAME}"
INPUT="${ROOT}/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl"
LOG="${ROOT}/logs/two_dim_e1_${RUN_NAME}.log"

run_pipeline() {
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="/data2/hjk/models/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false

  mkdir -p "${RUN_DIR}/manifest" "${RUN_DIR}/results" "${ROOT}/logs"
  "${PYTHON}" scripts/prepare_long_success_smoke.py \
    --input "${INPUT}" \
    --output-dir "${RUN_DIR}/manifest" \
    --smoke-questions 32 \
    --discovery-fraction 0.7 \
    --seed 20260721

  manifests=("${RUN_DIR}"/manifest/manifest_smoke*.jsonl)
  if [[ ! -f "${manifests[0]}" ]]; then
    echo "No frozen manifest was produced" >&2
    exit 1
  fi
  manifest="${manifests[0]}"

  "${PYTHON}" scripts/extract_long_success_hidden_qwen3vl.py \
    --input "${manifest}" \
    --output-dir "${RUN_DIR}" \
    --model Qwen/Qwen3-VL-8B-Thinking \
    --layers 24,36 \
    --window 128 \
    --stride 64 \
    --local-files-only \
    --resume

  "${PYTHON}" scripts/analyze_long_success_trajectory.py \
    --input-glob "${RUN_DIR}/hidden/question_*.npz" \
    --output-dir "${RUN_DIR}/results" \
    --progress-bins 10 \
    --bootstrap 1000 \
    --permutations 100 \
    --seed 20260721

  echo "PIPELINE_COMPLETE ${RUN_DIR}"
}

if [[ "${1:-}" == "--run" ]]; then
  run_pipeline
  exit 0
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "${gpu_pids}" ]]; then
  echo "GPU has active compute processes; refusing to launch: ${gpu_pids}" >&2
  exit 2
fi

mkdir -p "${ROOT}/logs"
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && bash scripts/launch_long_success_smoke_tmux.sh --run 2>&1 | tee '${LOG}'"
echo "launched tmux=${SESSION} log=${LOG} run_dir=${RUN_DIR}"
