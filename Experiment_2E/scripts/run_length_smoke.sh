#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/Qwen2.5-7B-Instruct}
DATA_DIR=${DATA_DIR:-/data2/hjk/data/experiment_2e/math_prepared_seed20260805}
OUTPUT_DIR=${OUTPUT_DIR:-/data2/hjk/results/experiment_2e/length_smoke_seed20260805}

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

test -f "${MODEL_PATH}/config.json"
test -f "${DATA_DIR}/eval_256.parquet"
mkdir -p "${OUTPUT_DIR}"

python -m experiment_2e.generate_length_smoke \
  --model-path "${MODEL_PATH}" \
  --eval-file "${DATA_DIR}/eval_256.parquet" \
  --output-dir "${OUTPUT_DIR}" \
  --num-questions 100 \
  --batch-questions 16 \
  --max-prompt-tokens 2048 \
  --max-new-tokens 1536 \
  --temperature 1.0 \
  --top-p 1.0 \
  --seed 20260805 \
  --gpu-memory-utilization 0.80

python -m experiment_2e.length_smoke \
  --input "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}/analysis"

echo "LENGTH_SMOKE_COMPLETE"
