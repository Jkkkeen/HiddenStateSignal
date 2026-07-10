#!/usr/bin/env bash
# Merge A1/C2 LoRA adapters into standalone HF model directories for vLLM eval.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MERGED_ROOT=${MERGED_ROOT:-/data2/hjk/checkpoints/verl_qwen3vl_merged}
FULL_ROOT=${FULL_ROOT:-/data2/hjk/checkpoints/verl_qwen3vl_merged_full}
LOG=${LOG:-${PROJECT_ROOT}/logs/goal5_merge_lora_full_models_20260707.log}

export PATH="${ENV_ROOT}/bin:${PATH}"
export HF_HOME=${HF_HOME:-/data2/hjk/cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/data2/hjk/cache/huggingface/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/hjk/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data2/hjk/cache/huggingface/datasets}
export TORCH_HOME=${TORCH_HOME:-/data2/hjk/cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/data2/hjk/cache/pip}
export TMPDIR=${TMPDIR:-/data2/hjk/tmp}
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PROJECT_ROOT}/logs" "${FULL_ROOT}" "${TMPDIR}"
exec > >(tee "${LOG}") 2>&1

date
python --version
df -h / /data2
free -h
nvidia-smi

cd "${PROJECT_ROOT}"
python scripts/merge_qwen3vl_lora_adapters.py \
    --spec "{\"name\":\"a1\",\"model_dir\":\"${MERGED_ROOT}/a1_answer_memsafe20_n4_seed42_global_step_20\",\"adapter_dir\":\"${MERGED_ROOT}/a1_answer_memsafe20_n4_seed42_global_step_20/lora_adapter\",\"output_dir\":\"${FULL_ROOT}/a1_answer_memsafe20_n4_seed42_global_step_20_full\"}" \
    --spec "{\"name\":\"c2\",\"model_dir\":\"${MERGED_ROOT}/c2_actor_forward_memsafe20_n4_seed42_v2_global_step_20\",\"adapter_dir\":\"${MERGED_ROOT}/c2_actor_forward_memsafe20_n4_seed42_v2_global_step_20/lora_adapter\",\"output_dir\":\"${FULL_ROOT}/c2_actor_forward_memsafe20_n4_seed42_v2_global_step_20_full\"}"

date
du -sh "${FULL_ROOT}"/* || true
df -h / /data2
free -h
nvidia-smi
echo "GOAL5_MERGE_LORA_FULL_DONE"
