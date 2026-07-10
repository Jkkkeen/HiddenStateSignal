#!/usr/bin/env bash
# Run answer-only vLLM evaluation for A1 and C2 global_step_20 checkpoints.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
FULL_ROOT=${FULL_ROOT:-/data2/hjk/checkpoints/verl_qwen3vl_merged_full}
DATA_FILE=${DATA_FILE:-${PROJECT_ROOT}/rl_data/mathverse_qwen3vl_stage3_pilot200_128/mathverse_grpo_val.parquet}
OUT_ROOT=${OUT_ROOT:-${PROJECT_ROOT}/rl_audit/offline_answer_eval_vllm_a1_c2_20260707}
LOG=${LOG:-${PROJECT_ROOT}/logs/goal5_offline_answer_eval_vllm_a1_c2_20260707.log}

MAX_TOKENS=${MAX_TOKENS:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20480}
BATCH_SIZE=${BATCH_SIZE:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
LIMIT=${LIMIT:--1}
SEED=${SEED:-20260707}

export PATH="${ENV_ROOT}/bin:${PATH}"
export HF_HOME=${HF_HOME:-/data2/hjk/cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/data2/hjk/cache/huggingface/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/hjk/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data2/hjk/cache/huggingface/datasets}
export TORCH_HOME=${TORCH_HOME:-/data2/hjk/cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/data2/hjk/cache/pip}
export TMPDIR=${TMPDIR:-/data2/hjk/tmp}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/data2/hjk/cache/vllm}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/hjk/cache/xdg}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${PROJECT_ROOT}/logs" "${OUT_ROOT}" "${TMPDIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"
exec > >(tee "${LOG}") 2>&1

date
python --version
df -h / /data2
free -h
nvidia-smi
ls -lh "${DATA_FILE}"
du -sh \
    "${FULL_ROOT}/a1_answer_memsafe20_n4_seed42_global_step_20_full" \
    "${FULL_ROOT}/c2_actor_forward_memsafe20_n4_seed42_v2_global_step_20_full"

cd "${PROJECT_ROOT}"

python scripts/offline_answer_eval_qwen3vl_vllm.py \
    --run-name a1_answer_memsafe20_n4_seed42_step20_vllm_answer_only \
    --input "${DATA_FILE}" \
    --model "${FULL_ROOT}/a1_answer_memsafe20_n4_seed42_global_step_20_full" \
    --output-dir "${OUT_ROOT}/a1_answer_memsafe20_n4_seed42_step20" \
    --limit "${LIMIT}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --seed "${SEED}"

date
nvidia-smi

python scripts/offline_answer_eval_qwen3vl_vllm.py \
    --run-name c2_actor_forward_memsafe20_n4_seed42_step20_vllm_answer_only \
    --input "${DATA_FILE}" \
    --model "${FULL_ROOT}/c2_actor_forward_memsafe20_n4_seed42_v2_global_step_20_full" \
    --output-dir "${OUT_ROOT}/c2_actor_forward_memsafe20_n4_seed42_v2_step20" \
    --limit "${LIMIT}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --seed "${SEED}"

date
cat "${OUT_ROOT}/a1_answer_memsafe20_n4_seed42_step20/summary.json"
cat "${OUT_ROOT}/c2_actor_forward_memsafe20_n4_seed42_v2_step20/summary.json"
df -h / /data2
free -h
nvidia-smi
echo "GOAL5_VLLM_OFFLINE_ANSWER_EVAL_DONE"
