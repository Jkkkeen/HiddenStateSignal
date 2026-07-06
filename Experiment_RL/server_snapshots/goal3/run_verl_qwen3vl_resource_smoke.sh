#!/usr/bin/env bash
# Qwen3-VL-8B-Thinking GRPO answer-only resource smoke on one H200.
# Intended to live under /data2/hjk/projects/AI-HiddenState-ER/scripts.

set -xeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
VERL_ROOT=${VERL_ROOT:-/data2/hjk/projects/verl}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}

TRAIN_FILE=${TRAIN_FILE:-${PROJECT_ROOT}/rl_data/mathverse_verl_smoke16/mathverse_grpo_train.parquet}
TEST_FILE=${TEST_FILE:-${PROJECT_ROOT}/rl_data/mathverse_verl_smoke16/mathverse_grpo_val.parquet}
REWARD_PATH=${REWARD_PATH:-${PROJECT_ROOT}/scripts/mathverse_answer_reward.py}

MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
ROLLOUT_N=${ROLLOUT_N:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
TOTAL_STEPS=${TOTAL_STEPS:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.50}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}

RUN_NAME=${RUN_NAME:-qwen3vl8b_answer_resource_smoke_len${MAX_RESPONSE_LENGTH}_n${ROLLOUT_N}}
CKPT_DIR=${CKPT_DIR:-/data2/hjk/checkpoints/verl_qwen3vl/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/logs}

if [ -f "${ENV_ROOT}/bin/activate" ]; then
    source "${ENV_ROOT}/bin/activate"
else
    export CONDA_PREFIX="${ENV_ROOT}"
    export CONDA_DEFAULT_ENV="$(basename "${ENV_ROOT}")"
    export PATH="${ENV_ROOT}/bin:${PATH}"
fi

export HF_HOME=${HF_HOME:-/data2/hjk/cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/data2/hjk/cache/huggingface/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data2/hjk/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data2/hjk/cache/huggingface/datasets}
export TORCH_HOME=${TORCH_HOME:-/data2/hjk/cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/data2/hjk/cache/pip}
export TMPDIR=${TMPDIR:-/data2/hjk/tmp}
export RAY_TMPDIR=${RAY_TMPDIR:-/data2/hjk/cache/ray}
export WANDB_DIR=${WANDB_DIR:-/data2/hjk/checkpoints/wandb}
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${RAY_TMPDIR}" "${WANDB_DIR}" "${CKPT_DIR}" "${LOG_DIR}"

python --version
df -h / /data2
free -h
nvidia-smi
ls -lh "${TRAIN_FILE}" "${TEST_FILE}" "${REWARD_PATH}"

cd "${VERL_ROOT}"

MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}

DEVICE=gpu \
MODEL_PATH="${MODEL_PATH}" \
TRAIN_FILE="${TRAIN_FILE}" \
TEST_FILE="${TEST_FILE}" \
NDEVICES_PER_NODE=1 \
NNODES=1 \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
ROLLOUT_TP=1 \
ROLLOUT_N="${ROLLOUT_N}" \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL}" \
TOTAL_EPOCHS=1 \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
PROJECT_NAME=qwen3vl_resource_smoke \
EXPERIMENT_NAME="${RUN_NAME}" \
bash examples/grpo_trainer/run_qwen3_vl_8b_fsdp.sh \
    trainer.logger='["console"]' \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.val_before_train=False \
    trainer.default_local_dir="${CKPT_DIR}" \
    reward.num_workers=1 \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.model.use_fused_kernels=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts=True \
    data.truncation=error

free -h
nvidia-smi
df -h / /data2
echo "QWEN3VL_RESOURCE_SMOKE_EXIT_CODE=0"
