#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/Qwen3-1.7B-Base}
DATA_ROOT=${DATA_ROOT:-/data2/hjk/data/experiment_2e/q3_1p7b_base_seed20260814}
RESULT_ROOT=${RESULT_ROOT:-/data2/hjk/results/experiment_2e/q3_1p7b_base_calibration_seed20260814}

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${RESULT_ROOT}"

run_dataset() {
  local name=$1
  local manifest=$2
  local rollout_dir="${RESULT_ROOT}/${name}_rollouts"
  "${ENV_ROOT}/bin/python" -m experiment_2e.rollouts \
    --run-id "q3_1p7b_base_${name}_calibration_seed20260814" \
    --eval-file "${manifest}" \
    --base-model-path "${MODEL_PATH}" \
    --output-dir "${rollout_dir}" \
    --checkpoints base \
    --global-steps 0 \
    --rollouts-per-question 8 \
    --batch-rollouts 16 \
    --max-prompt-tokens 2048 \
    --max-new-tokens 12288 \
    --temperature 0.8 \
    --top-p 0.95 \
    --top-k 20 \
    --seed 20260814 \
    --gpu-memory-utilization 0.72

  "${ENV_ROOT}/bin/python" -m experiment_2e.q3_1p7b_calibration \
    --dataset "${name}" \
    --model-path "${MODEL_PATH}" \
    --rollout-dir "${rollout_dir}" \
    --output "${RESULT_ROOT}/calibration_${name}.json"
}

run_dataset deepmath "${DATA_ROOT}/deepmath_calibration_256.parquet"
run_dataset simplerl "${DATA_ROOT}/simplerl_calibration_256.parquet"

echo Q3_1P7B_DUAL_CALIBRATION_COMPLETE
