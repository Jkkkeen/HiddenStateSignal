#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}
INPUT_PATH=${INPUT_PATH:-${PROJECT_ROOT}/rl_recoverability/discovery_smoke_v1/recoverability_manifest.jsonl}
MATHVERSE_METADATA=${MATHVERSE_METADATA:-${PROJECT_ROOT}/data/mathverse/testmini.json}
RUN_NAME=${RUN_NAME:-recoverability_candidate_scoring_v1}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/rl_recoverability/${RUN_NAME}}

LIMIT=${LIMIT:--1}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
SEED=${SEED:-20260711}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}

export HF_HOME=/data2/hjk/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data2/hjk/cache/huggingface/hub
export TRANSFORMERS_CACHE=/data2/hjk/cache/huggingface
export HF_DATASETS_CACHE=/data2/hjk/cache/huggingface/datasets
export TORCH_HOME=/data2/hjk/cache/torch
export PIP_CACHE_DIR=/data2/hjk/cache/pip
export TMPDIR=/data2/hjk/tmp
export RAY_TMPDIR=/data2/hjk/cache/ray
export XDG_CACHE_HOME=/data2/hjk/cache
export VLLM_CACHE_ROOT=/data2/hjk/cache/vllm
export VLLM_CONFIG_ROOT=/data2/hjk/cache/vllm/config
export TORCHINDUCTOR_CACHE_DIR=/data2/hjk/cache/torchinductor
export TRITON_CACHE_DIR=/data2/hjk/cache/triton
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p \
  "${OUTPUT_DIR}" \
  /data2/hjk/tmp \
  /data2/hjk/cache/ray \
  /data2/hjk/cache/vllm/config \
  /data2/hjk/cache/torchinductor \
  /data2/hjk/cache/triton
cd "${PROJECT_ROOT}"

"${ENV_ROOT}/bin/python" scripts/score_recoverability_candidates_vllm.py \
  --input "${INPUT_PATH}" \
  --model "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}" \
  --mathverse-metadata "${MATHVERSE_METADATA}" \
  --limit "${LIMIT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --seed "${SEED}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

status=$?
echo "RECOVERABILITY_CANDIDATE_SCORING_EXIT_CODE=${status}"
df -h / /data2
nvidia-smi
exit "${status}"
