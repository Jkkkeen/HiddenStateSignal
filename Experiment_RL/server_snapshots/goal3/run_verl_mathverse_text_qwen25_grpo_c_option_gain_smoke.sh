#!/usr/bin/env bash
# Experiment C smoke: GRPO answer reward plus frozen option-logit margin-gain bonus.

set -xeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
VERL_ROOT=${VERL_ROOT:-/data2/hjk/projects/verl}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/rl_data/mathverse_text_a0_clean_qwen25_500}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/mathverse_text_a0_clean_train.parquet}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/mathverse_text_a0_clean_val.parquet}
REWARD_PATH=${REWARD_PATH:-${PROJECT_ROOT}/scripts/mathverse_option_logit_gain_reward.py}
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-mathverse_text_qwen25_1p5b_grpo_c_option_gain_smoke_seed${SEED}}
CKPT_DIR=${CKPT_DIR:-/data2/hjk/checkpoints/verl_qwen3vl/${RUN_NAME}}
LOG=${LOG:-${PROJECT_ROOT}/logs/${RUN_NAME}.log}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${PROJECT_ROOT}/rl_audit/${RUN_NAME}/validation_generations}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.30}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}

OPTION_GAIN_LAMBDA=${OPTION_GAIN_LAMBDA:-0.2}
OPTION_GAIN_CLIP=${OPTION_GAIN_CLIP:-0.5}
OPTION_GAIN_RESPONSE_FRACS=${OPTION_GAIN_RESPONSE_FRACS:-0.0,0.33,0.67,0.90}
OPTION_GAIN_MAX_RESPONSE_CHARS=${OPTION_GAIN_MAX_RESPONSE_CHARS:-2048}

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
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export RAY_idle_worker_killing_memory_threshold_bytes=${RAY_idle_worker_killing_memory_threshold_bytes:-100000000}

export OPTION_GAIN_MODEL_PATH=${OPTION_GAIN_MODEL_PATH:-${MODEL_PATH}}
export OPTION_GAIN_LAMBDA
export OPTION_GAIN_CLIP
export OPTION_GAIN_RESPONSE_FRACS
export OPTION_GAIN_MAX_RESPONSE_CHARS
export OPTION_GAIN_ATTN_IMPLEMENTATION=${OPTION_GAIN_ATTN_IMPLEMENTATION:-sdpa}
export OPTION_GAIN_DTYPE=${OPTION_GAIN_DTYPE:-bfloat16}

mkdir -p "${PROJECT_ROOT}/logs" "${CKPT_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${RAY_TMPDIR}" "${WANDB_DIR}" "${VALIDATION_DATA_DIR}"

cd "${PROJECT_ROOT}"
"${ENV_ROOT}/bin/ray" stop -f || true
exec > >(tee "${LOG}") 2>&1

python --version
df -h / /data2
nvidia-smi
ls -lh "${TRAIN_FILE}" "${TEST_FILE}" "${REWARD_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "SEED=${SEED}"
echo "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "ROLLOUT_N=${ROLLOUT_N}"
echo "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "OPTION_GAIN_LAMBDA=${OPTION_GAIN_LAMBDA}"
echo "OPTION_GAIN_CLIP=${OPTION_GAIN_CLIP}"
echo "OPTION_GAIN_RESPONSE_FRACS=${OPTION_GAIN_RESPONSE_FRACS}"

cd "${VERL_ROOT}"
DEVICE=gpu \
INFER_BACKEND=vllm \
MODEL_PATH="${MODEL_PATH}" \
NGPUS_PER_NODE=1 \
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
PROJECT_NAME=mathverse_text_grpo_c_option_gain \
EXPERIMENT_NAME="${RUN_NAME}" \
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.dataloader_num_workers=0 \
  data.shuffle=True \
  data.seed="${SEED}" \
  trainer.logger='["console"]' \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
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
  data.filter_overlong_prompts=True \
  data.truncation=error

status=$?
echo C_OPTION_GAIN_SMOKE_EXIT_CODE=${status}
df -h / /data2
exit ${status}

