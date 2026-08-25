#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/Qwen3-1.7B-Base}
DATA_ROOT=${DATA_ROOT:-/data2/hjk/data/experiment_2e/q3_1p7b_base_seed20260814}
RESULT_ROOT=${RESULT_ROOT:-/data2/hjk/results/experiment_2e/q3_1p7b_base_prompt_calibration_full_seed20260814}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-12288}
PROMPT_START=${PROMPT_START:-0}
PROMPT_SIZE=${PROMPT_SIZE:-256}
PROMPT_TAG=${PROMPT_TAG:-full256}

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

mkdir -p "${RESULT_ROOT}"
"${ENV_ROOT}/bin/python" -m experiment_2e.q3_prompt_calibration \
  --input "${DATA_ROOT}/simplerl_calibration_256.parquet" \
  --output "${DATA_ROOT}/simplerl_prompt_calibration_strict_v2_${PROMPT_TAG}.parquet" \
  --audit "${RESULT_ROOT}/prompt_manifest_audit.json" \
  --size "${PROMPT_SIZE}" \
  --start "${PROMPT_START}"

"${ENV_ROOT}/bin/python" -m experiment_2e.rollouts \
  --run-id q3_1p7b_base_simplerl_strict_v2_prompt_calibration_seed20260814 \
  --eval-file "${DATA_ROOT}/simplerl_prompt_calibration_strict_v2_${PROMPT_TAG}.parquet" \
  --base-model-path "${MODEL_PATH}" \
  --output-dir "${RESULT_ROOT}/rollouts" \
  --checkpoints base \
  --global-steps 0 \
  --rollouts-per-question 8 \
  --batch-rollouts 16 \
  --max-prompt-tokens 2048 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 20 \
  --seed 20260814 \
  --gpu-memory-utilization 0.65

"${ENV_ROOT}/bin/python" -m experiment_2e.q3_1p7b_calibration \
  --dataset simplerl_strict_v2 \
  --model-path "${MODEL_PATH}" \
  --rollout-dir "${RESULT_ROOT}/rollouts" \
  --output "${RESULT_ROOT}/calibration_simplerl_strict_v2.json" \
  --questions "${PROMPT_SIZE}" \
  --caps 4096,8192,12288

echo Q3_1P7B_PROMPT_CALIBRATION_COMPLETE
