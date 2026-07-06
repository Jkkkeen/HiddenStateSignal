#!/usr/bin/env bash
# RL02 Stage 3 A1-long pilot: Qwen3-VL-8B-Thinking GRPO answer-only baseline.

set -xeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
VERL_ROOT=${VERL_ROOT:-/data2/hjk/projects/verl}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}

DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/rl_data/mathverse_qwen3vl_stage3_pilot200_128}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/mathverse_grpo_train.parquet}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/mathverse_grpo_val.parquet}
REWARD_PATH=${REWARD_PATH:-${PROJECT_ROOT}/scripts/mathverse_answer_reward.py}

SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-qwen3vl8b_stage3_a1_long_answer_pilot_seed${SEED}}
CKPT_DIR=${CKPT_DIR:-/data2/hjk/checkpoints/verl_qwen3vl/${RUN_NAME}}
LOG=${LOG:-${PROJECT_ROOT}/logs/${RUN_NAME}.log}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${PROJECT_ROOT}/rl_audit/${RUN_NAME}/validation_generations}

MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
ROLLOUT_N=${ROLLOUT_N:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
TOTAL_STEPS=${TOTAL_STEPS:-50}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.50}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-16}

if [ -d "${ENV_ROOT}/bin" ]; then
    export CONDA_PREFIX="${ENV_ROOT}"
    export CONDA_DEFAULT_ENV="$(basename "${ENV_ROOT}")"
    export PATH="${ENV_ROOT}/bin:${PATH}"
else
    echo "Missing env: ${ENV_ROOT}" >&2
    exit 1
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
export WANDB_MODE=${WANDB_MODE:-disabled}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/data2/hjk/cache/vllm}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data2/hjk/cache/xdg}
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export RAY_idle_worker_killing_memory_threshold_bytes=${RAY_idle_worker_killing_memory_threshold_bytes:-100000000}

mkdir -p \
    "${PROJECT_ROOT}/logs" \
    "${CKPT_DIR}" \
    "${HF_HOME}" \
    "${PIP_CACHE_DIR}" \
    "${TMPDIR}" \
    "${RAY_TMPDIR}" \
    "${WANDB_DIR}" \
    "${VLLM_CACHE_ROOT}" \
    "${XDG_CACHE_HOME}" \
    "${VALIDATION_DATA_DIR}"

cd "${PROJECT_ROOT}"
"${ENV_ROOT}/bin/ray" stop -f || true
exec > >(tee "${LOG}") 2>&1

python --version
df -h / /data2
free -h
nvidia-smi
ls -lh "${TRAIN_FILE}" "${TEST_FILE}" "${REWARD_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "SEED=${SEED}"
echo "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "ROLLOUT_N=${ROLLOUT_N}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "TOTAL_STEPS=${TOTAL_STEPS}"
echo "VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR}"

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
SAVE_FREQ="${SAVE_FREQ}" \
TEST_FREQ="${TEST_FREQ}" \
PROJECT_NAME=qwen3vl_stage3_pilot \
EXPERIMENT_NAME="${RUN_NAME}" \
bash examples/grpo_trainer/run_qwen3_vl_8b_fsdp.sh \
    trainer.logger='["console"]' \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    reward.num_workers=1 \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.model.use_fused_kernels=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.seed="${SEED}" \
    actor_rollout_ref.actor.data_loader_seed="${SEED}" \
    actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
    actor_rollout_ref.ref.fsdp_config.seed="${SEED}" \
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
    data.image_key=images \
    data.dataloader_num_workers=0 \
    data.shuffle=True \
    data.seed="${SEED}" \
    data.filter_overlong_prompts=True \
    data.truncation=error

status=$?
echo QWEN3VL_STAGE3_A1_LONG_PILOT_EXIT_CODE=${status}
free -h
nvidia-smi
df -h / /data2
exit ${status}
