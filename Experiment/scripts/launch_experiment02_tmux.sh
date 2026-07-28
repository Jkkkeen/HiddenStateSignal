#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"
PYTHON="${PYTHON:-/data2/hjk/envs/hs_er/bin/python}"
SESSION="${SESSION:-two_dim02_replication96}"
RUN_NAME="${RUN_NAME:-experiment02_replication96_20260723}"
QUESTION_LIMIT="${QUESTION_LIMIT:-96}"
SMOKE_QUESTIONS="${SMOKE_QUESTIONS:-4}"
BOOTSTRAP="${BOOTSTRAP:-4000}"
SEED="${SEED:-20260724}"
RUN_DIR="${ROOT}/experiment0_hidden_dynamics/${RUN_NAME}"
SMOKE_DIR="${ROOT}/experiment0_hidden_dynamics/${RUN_NAME}_smoke4"
SOURCE="${ROOT}/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl"
EXCLUDE="${ROOT}/experiment0_hidden_dynamics/formal_discovery24_20260722/manifest/manifest_smoke24.jsonl"
LOG="${ROOT}/logs/two_dim02_${RUN_NAME}.log"

run_extract() {
  local manifest="$1"
  local output_dir="$2"
  "${PYTHON}" scripts/extract_experiment02_qwen3vl.py \
    --input "${manifest}" \
    --output-dir "${output_dir}" \
    --model Qwen/Qwen3-VL-8B-Thinking \
    --progress-bins 10 \
    --activity-layer 15 \
    --movement-layer 24 \
    --path-layers 24,36 \
    --window 128 \
    --primary-stride 64 \
    --sensitivity-stride 128 \
    --entropy-chunk-size 512 \
    --sigma-scale 0.0001 \
    --z-clip 8 \
    --min-correct 2 \
    --min-wrong 2 \
    --save-path-vectors \
    --local-files-only \
    --resume
}

run_analyze() {
  local extraction_dir="$1"
  local output_dir="$2"
  local bootstrap="$3"
  local label="$4"
  "${PYTHON}" scripts/analyze_experiment02.py \
    --input-dir "${extraction_dir}" \
    --output-dir "${output_dir}" \
    --bootstrap "${bootstrap}" \
    --seed "${SEED}" \
    --run-label "${label}"
}

validate_outputs() {
  local extraction_dir="$1"
  local result_dir="$2"
  test -s "${extraction_dir}/EXTRACTION_SUMMARY.json"
  test -s "${result_dir}/EXPERIMENT02_RESULTS.md"
  test -s "${result_dir}/experiment02_confirmatory.csv"
  test -s "${result_dir}/experiment02_entropy_effects.csv"
  test -s "${result_dir}/experiment02_path_effects.csv"
  for figure in E02_F1_progress_spaghetti.png E02_F2_confirmatory_auc_curves.png E02_F3_entropy_layer_progress.png E02_F4_path_geometry_auc.png; do
    test -s "${result_dir}/figures/${figure}"
  done
}

run_pipeline() {
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="/data2/hjk/models/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false

  mkdir -p "${RUN_DIR}/manifest" "${RUN_DIR}/extraction" "${RUN_DIR}/results"
  mkdir -p "${SMOKE_DIR}/extraction" "${SMOKE_DIR}/results" "${ROOT}/logs"
  "${PYTHON}" scripts/prepare_experiment02_replication.py \
    --input "${SOURCE}" \
    --exclude-manifest "${EXCLUDE}" \
    --output-dir "${RUN_DIR}/manifest" \
    --question-limit "${QUESTION_LIMIT}" \
    --smoke-questions "${SMOKE_QUESTIONS}" \
    --seed "${SEED}"

  cp "${RUN_DIR}/manifest/manifest_smoke${SMOKE_QUESTIONS}.jsonl" "${SMOKE_DIR}/manifest_smoke${SMOKE_QUESTIONS}.jsonl"
  run_extract \
    "${SMOKE_DIR}/manifest_smoke${SMOKE_QUESTIONS}.jsonl" \
    "${SMOKE_DIR}/extraction"
  run_analyze "${SMOKE_DIR}/extraction" "${SMOKE_DIR}/results" 200 "Smoke4"
  validate_outputs "${SMOKE_DIR}/extraction" "${SMOKE_DIR}/results"
  echo "SMOKE_COMPLETE ${SMOKE_DIR}"

  run_extract \
    "${RUN_DIR}/manifest/manifest_replication96.jsonl" \
    "${RUN_DIR}/extraction"
  run_analyze "${RUN_DIR}/extraction" "${RUN_DIR}/results" "${BOOTSTRAP}" "Replication96"
  validate_outputs "${RUN_DIR}/extraction" "${RUN_DIR}/results"
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

mkdir -p "${ROOT}/logs"
printf -v tmux_command \
  "cd %q && env ROOT=%q PYTHON=%q SESSION=%q RUN_NAME=%q QUESTION_LIMIT=%q SMOKE_QUESTIONS=%q BOOTSTRAP=%q SEED=%q bash scripts/launch_experiment02_tmux.sh --run 2>&1 | tee %q" \
  "${ROOT}" "${ROOT}" "${PYTHON}" "${SESSION}" "${RUN_NAME}" "${QUESTION_LIMIT}" "${SMOKE_QUESTIONS}" "${BOOTSTRAP}" "${SEED}" "${LOG}"
tmux new-session -d -s "${SESSION}" "${tmux_command}"
echo "launched tmux=${SESSION} log=${LOG} run_dir=${RUN_DIR}"
