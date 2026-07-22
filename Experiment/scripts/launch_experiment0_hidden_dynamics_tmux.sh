#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"
PYTHON="${PYTHON:-/data2/hjk/envs/hs_er/bin/python}"
SESSION="${SESSION:-two_dim_e0_smoke2}"
RUN_NAME="${RUN_NAME:-extraction_smoke2_20260722}"
QUESTION_LIMIT="${QUESTION_LIMIT:-2}"
RUN_LABEL="${RUN_LABEL:-Extraction Smoke}"
BOOTSTRAP="${BOOTSTRAP:-200}"
PERMUTATIONS="${PERMUTATIONS:-20}"
AUDIT_QUESTION_COUNT="${AUDIT_QUESTION_COUNT:-1}"
RUN_DIR="${ROOT}/experiment0_hidden_dynamics/${RUN_NAME}"
INPUT="${ROOT}/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl"
LOG="${ROOT}/logs/two_dim_e0_${RUN_NAME}.log"

run_pipeline() {
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="/data2/hjk/models/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false

  mkdir -p "${RUN_DIR}/manifest" "${RUN_DIR}/extraction" "${RUN_DIR}/results" "${ROOT}/logs"
  "${PYTHON}" scripts/prepare_long_success_smoke.py \
    --input "${INPUT}" \
    --output-dir "${RUN_DIR}/manifest" \
    --smoke-questions "${QUESTION_LIMIT}" \
    --discovery-fraction 0.7 \
    --seed 20260721

  manifests=("${RUN_DIR}"/manifest/manifest_smoke*.jsonl)
  if [[ ! -f "${manifests[0]}" ]]; then
    echo "No frozen manifest was produced" >&2
    exit 1
  fi
  manifest="${manifests[0]}"

  "${PYTHON}" scripts/extract_experiment0_hidden_dynamics_qwen3vl.py \
    --input "${manifest}" \
    --output-dir "${RUN_DIR}/extraction" \
    --model Qwen/Qwen3-VL-8B-Thinking \
    --progress-bins 10 \
    --span-specs 64:32,128:64,256:128 \
    --endpoint-spec 128:64 \
    --primary-spec 128:64 \
    --kappa-min 0.20 \
    --audit-question-count "${AUDIT_QUESTION_COUNT}" \
    --local-files-only \
    --resume

  "${PYTHON}" scripts/analyze_experiment0_hidden_dynamics.py \
    --input-dir "${RUN_DIR}/extraction" \
    --output-dir "${RUN_DIR}/results" \
    --primary-representation mean_w128_s64 \
    --bootstrap "${BOOTSTRAP}" \
    --permutations "${PERMUTATIONS}" \
    --run-label "${RUN_LABEL}" \
    --seed 20260721

  required=(
    LONG_EXPERIMENT_0_RESULTS.md
    long_experiment_0_bin_features.parquet
    long_experiment_0_question_effects.csv
    long_experiment_0_predictor_comparisons.csv
    long_experiment_0_prototype_diagnostics.parquet
    long_experiment_0_pairwise_geometry.parquet
    long_experiment_0_permutation_null.csv
    analysis_meta.json
  )
  for artifact in "${required[@]}"; do
    if [[ ! -s "${RUN_DIR}/results/${artifact}" ]]; then
      echo "Missing required result: ${artifact}" >&2
      exit 1
    fi
  done
  for figure_id in {1..7}; do
    if ! compgen -G "${RUN_DIR}/results/figures/E0_F${figure_id}_*.png" >/dev/null; then
      echo "Missing E0-F${figure_id} figure" >&2
      exit 1
    fi
  done
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
printf -v tmux_command \
  "cd %q && env ROOT=%q PYTHON=%q SESSION=%q RUN_NAME=%q QUESTION_LIMIT=%q RUN_LABEL=%q BOOTSTRAP=%q PERMUTATIONS=%q AUDIT_QUESTION_COUNT=%q bash scripts/launch_experiment0_hidden_dynamics_tmux.sh --run 2>&1 | tee %q" \
  "${ROOT}" "${ROOT}" "${PYTHON}" "${SESSION}" "${RUN_NAME}" "${QUESTION_LIMIT}" "${RUN_LABEL}" "${BOOTSTRAP}" "${PERMUTATIONS}" "${AUDIT_QUESTION_COUNT}" "${LOG}"
tmux new-session -d -s "${SESSION}" "${tmux_command}"
echo "launched tmux=${SESSION} log=${LOG} run_dir=${RUN_DIR}"
