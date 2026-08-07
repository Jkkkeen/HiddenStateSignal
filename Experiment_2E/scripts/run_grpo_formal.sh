#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
VERL_ROOT=${VERL_ROOT:-/data2/hjk/projects/verl}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/Qwen2.5-7B-Instruct}
MANIFEST=${MANIFEST:?Set MANIFEST to frozen formal_run_manifest.json}
RESUME_MODE=${RESUME_MODE:-auto}
NOFILE_LIMIT=${NOFILE_LIMIT:-65535}

if [[ "$(ulimit -n)" != "unlimited" ]]; then
  ulimit -n "${NOFILE_LIMIT}"
fi
CURRENT_NOFILE=$(ulimit -n)
if [[ "${CURRENT_NOFILE}" != "unlimited" ]] && (( CURRENT_NOFILE < NOFILE_LIMIT )); then
  echo "Unable to raise file descriptor limit to ${NOFILE_LIMIT}; current=${CURRENT_NOFILE}" >&2
  exit 1
fi

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/data2/hjk/cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export EXPERIMENT_2E_FORCE_SHM=1
export RAY_memory_usage_threshold=0.99
export RAY_idle_worker_killing_memory_threshold_bytes=100000000
export RAY_TMPDIR=${RAY_TMPDIR:-/data2/hjk/cache/ray/experiment_2e}
export TMPDIR=${TMPDIR:-/data2/hjk/tmp}

test -f "${MANIFEST}"
test -f "$(dirname "${MANIFEST}")/formal_run_manifest.sha256"
(
  cd "$(dirname "${MANIFEST}")"
  sha256sum -c formal_run_manifest.sha256
)

mapfile -t FROZEN < <("${ENV_ROOT}/bin/python" - "${MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["run_id"])
print(manifest["total_training_steps"])
print(manifest["save_frequency"])
print(manifest["train_file"])
print(manifest["eval_file"])
print(manifest["base_seed"])
PY
)
RUN_NAME=${FROZEN[0]}
TOTAL_TRAINING_STEPS=${FROZEN[1]}
SAVE_FREQ=${FROZEN[2]}
TRAIN_FILE=${FROZEN[3]}
VAL_FILE=${FROZEN[4]}
SEED=${FROZEN[5]}

if (( TOTAL_TRAINING_STEPS != SAVE_FREQ * 5 )); then
  echo "Frozen checkpoint schedule is not five equal intervals" >&2
  exit 1
fi

CKPT_DIR=${CKPT_DIR:-/data2/hjk/checkpoints/experiment_2e/${RUN_NAME}}
LOG=${LOG:-/data2/hjk/logs/experiment_2e/${RUN_NAME}.log}
AUDIT=${AUDIT:-/data2/hjk/results/experiment_2e/${RUN_NAME}/training/formal_training_audit.json}
PATCH_AUDIT=${PATCH_AUDIT:-/data2/hjk/results/experiment_2e/${RUN_NAME}/training/verl_temp_loop_patch_audit.json}

test -f "${MODEL_PATH}/config.json"
test -f "${TRAIN_FILE}"
test -f "${VAL_FILE}"
test -f "${PROJECT_ROOT}/experiment_2e/math_reward.py"
mkdir -p "${CKPT_DIR}" "$(dirname "${LOG}")" "$(dirname "${AUDIT}")" "${RAY_TMPDIR}" "${TMPDIR}"

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/patch_verl_temp_loop.py" \
  --verl-root "${VERL_ROOT}" \
  --audit "${PATCH_AUDIT}"

exec > >(tee -a "${LOG}") 2>&1
echo "RUN_NAME=${RUN_NAME} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} SAVE_FREQ=${SAVE_FREQ} RESUME_MODE=${RESUME_MODE}"
nvidia-smi

cd "${VERL_ROOT}"
DEVICE=gpu \
INFER_BACKEND=vllm \
MODEL_PATH="${MODEL_PATH}" \
NGPUS_PER_NODE=1 \
NNODES=1 \
TRAIN_BATCH_SIZE=1 \
PPO_MINI_BATCH_SIZE=8 \
MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=1536 \
PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
ROLLOUT_TP=1 \
ROLLOUT_N=8 \
ROLLOUT_GPU_MEM_UTIL=0.30 \
TOTAL_EPOCHS=1 \
SAVE_FREQ="${SAVE_FREQ}" \
TEST_FREQ=-1 \
PROJECT_NAME=experiment_2e_math_grpo \
EXPERIMENT_NAME="${RUN_NAME}" \
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.dataloader_num_workers=0 \
  data.shuffle=True \
  data.seed="${SEED}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  trainer.logger='["console"]' \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.val_before_train=False \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.log_val_generations=0 \
  ray_kwargs.ray_init.num_cpus=16 \
  reward.num_workers=1 \
  reward.custom_reward_function.path="${PROJECT_ROOT}/experiment_2e/math_reward.py" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.use_fused_kernels=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2560 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.seed="${SEED}" \
  actor_rollout_ref.actor.data_loader_seed="${SEED}" \
  actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
  actor_rollout_ref.ref.fsdp_config.seed="${SEED}" \
  algorithm.use_kl_in_reward=False

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${ENV_ROOT}/bin/python" -m experiment_2e.formal_audit \
  --run-manifest "${MANIFEST}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --log "${LOG}" \
  --output "${AUDIT}"
echo "EXPERIMENT_2E_FORMAL_GRPO_COMPLETE"
