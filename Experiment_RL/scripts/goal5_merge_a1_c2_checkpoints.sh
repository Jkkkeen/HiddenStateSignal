#!/usr/bin/env bash
# Export the A1/C2 VERL FSDP actor checkpoints to HuggingFace format for
# answer-only offline evaluation.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
VERL_ROOT=${VERL_ROOT:-/data2/hjk/projects/verl}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}

A1_ACTOR_DIR=${A1_ACTOR_DIR:-/data2/hjk/checkpoints/verl_qwen3vl/qwen3vl8b_stage3_a1_answer_memsafe20_n4_seed42/global_step_20/actor}
C2_ACTOR_DIR=${C2_ACTOR_DIR:-/data2/hjk/checkpoints/verl_qwen3vl/qwen3vl8b_stage3_c2_actor_forward_memsafe20_n4_seed42_v2_20260706/global_step_20/actor}

MERGED_ROOT=${MERGED_ROOT:-/data2/hjk/checkpoints/verl_qwen3vl_merged}
A1_TARGET_DIR=${A1_TARGET_DIR:-${MERGED_ROOT}/a1_answer_memsafe20_n4_seed42_global_step_20}
C2_TARGET_DIR=${C2_TARGET_DIR:-${MERGED_ROOT}/c2_actor_forward_memsafe20_n4_seed42_v2_global_step_20}
LOG=${LOG:-${PROJECT_ROOT}/logs/goal5_merge_a1_c2_20260707.log}

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/data2/hjk/cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/data2/hjk/cache/huggingface/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/hjk/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data2/hjk/cache/huggingface/datasets}
export TORCH_HOME=${TORCH_HOME:-/data2/hjk/cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/data2/hjk/cache/pip}
export TMPDIR=${TMPDIR:-/data2/hjk/tmp}

mkdir -p "${PROJECT_ROOT}/logs" "${MERGED_ROOT}" "${TMPDIR}"
exec > >(tee "${LOG}") 2>&1

date
python --version
df -h / /data2
free -h

if [ ! -d "${A1_ACTOR_DIR}" ]; then
    echo "Missing A1 actor checkpoint: ${A1_ACTOR_DIR}" >&2
    exit 1
fi
if [ ! -d "${C2_ACTOR_DIR}" ]; then
    echo "Missing C2 actor checkpoint: ${C2_ACTOR_DIR}" >&2
    exit 1
fi

cd "${VERL_ROOT}"

if [ -f "${A1_TARGET_DIR}/config.json" ] && find "${A1_TARGET_DIR}" -maxdepth 1 -name "*.safetensors" | grep -q .; then
    echo "A1 merged model already exists: ${A1_TARGET_DIR}"
else
    echo "Merging A1 actor checkpoint to ${A1_TARGET_DIR}"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${A1_ACTOR_DIR}" \
        --target_dir "${A1_TARGET_DIR}"
fi

date

if [ -f "${C2_TARGET_DIR}/config.json" ] && find "${C2_TARGET_DIR}" -maxdepth 1 -name "*.safetensors" | grep -q .; then
    echo "C2 merged model already exists: ${C2_TARGET_DIR}"
else
    echo "Merging C2 actor checkpoint to ${C2_TARGET_DIR}"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${C2_ACTOR_DIR}" \
        --target_dir "${C2_TARGET_DIR}"
fi

date
find "${MERGED_ROOT}" -maxdepth 2 -type f \( -name "*.safetensors" -o -name "config.json" \) | sort
df -h / /data2
free -h
echo "GOAL5_MERGE_DONE"
