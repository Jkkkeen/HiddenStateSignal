#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
VERL_ROOT=${VERL_ROOT:-/data2/hjk/projects/verl}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/Qwen2.5-7B-Instruct}
DATA_DIR=${DATA_DIR:-/data2/hjk/data/experiment_2e/math_prepared_seed20260805}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train_level3_5.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/eval_256.parquet}
SOURCE_CKPT_ROOT=${SOURCE_CKPT_ROOT:-/data2/hjk/checkpoints/experiment_2e/qwen25_7b_math_grpo_formal_seed20260805}
SOURCE_STEP=${SOURCE_STEP:-3348}
WARMUP_STEPS=${WARMUP_STEPS:-2}
MEASURED_STEPS=${MEASURED_STEPS:-5}
TOTAL_TRAINING_STEPS=$((SOURCE_STEP + WARMUP_STEPS + MEASURED_STEPS))
SMOKE_EPOCHS=${SMOKE_EPOCHS:-2}
SEED=${SEED:-20260805}
BATCHES=(2 4 8)
RUN_ROOT=${RUN_ROOT:-/data2/hjk/results/experiment_2e/batch_capacity_step${SOURCE_STEP}_$(date -u +%Y%m%dT%H%M%SZ)}
RAY_BASE=${RAY_BASE:-/data2/hjk/r}
TMP_BASE=${TMP_BASE:-/data2/hjk/t}
NOFILE_LIMIT=${NOFILE_LIMIT:-65535}

if [[ "$(ulimit -n)" != "unlimited" ]]; then
  ulimit -n "${NOFILE_LIMIT}"
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

SOURCE_CKPT="${SOURCE_CKPT_ROOT}/global_step_${SOURCE_STEP}"
test -d "${SOURCE_CKPT}"
test -f "${SOURCE_CKPT_ROOT}/latest_checkpointed_iteration.txt"
test "$(tr -d '[:space:]' < "${SOURCE_CKPT_ROOT}/latest_checkpointed_iteration.txt")" = "${SOURCE_STEP}"
test -f "${SOURCE_CKPT}/actor/model_world_size_1_rank_0.pt"
test -f "${SOURCE_CKPT}/actor/optim_world_size_1_rank_0.pt"
test -f "${SOURCE_CKPT}/data.pt"
test -f "${MODEL_PATH}/config.json"
test -f "${TRAIN_FILE}"
test -f "${VAL_FILE}"
test -f "${PROJECT_ROOT}/experiment_2e/math_reward.py"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT already exists: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" > "${RUN_ROOT}/run_root.txt"

assert_gpu_idle() {
  local compute_pids
  compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
  if [[ -n "${compute_pids}" ]]; then
    echo "GPU is not idle; compute PIDs: ${compute_pids}" >&2
    return 1
  fi
}

wait_for_gpu_idle() {
  local attempt
  for attempt in $(seq 1 120); do
    if assert_gpu_idle >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  assert_gpu_idle
}

monitor_gpu() {
  local output=$1
  printf '%s\n' 'timestamp,memory.total [MiB],memory.used [MiB],memory.free [MiB],utilization.gpu [%]' > "${output}"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits >> "${output}" || true
    sleep 1
  done
}

run_candidate() {
  local BATCH_SIZE=$1
  local CANDIDATE_ROOT="${RUN_ROOT}/batch_${BATCH_SIZE}"
  local CANDIDATE_CKPT="${CANDIDATE_ROOT}/checkpoints"
  local LOG="${CANDIDATE_ROOT}/train.log"
  local TELEMETRY="${CANDIDATE_ROOT}/gpu_telemetry.csv"
  local RAY_TMPDIR="${RAY_BASE}/b${BATCH_SIZE}_$$"
  local TMPDIR="${TMP_BASE}/b${BATCH_SIZE}_$$"

  assert_gpu_idle
  if [[ -e "${RAY_TMPDIR}" || -e "${TMPDIR}" ]]; then
    echo "Candidate temporary path already exists: ${RAY_TMPDIR} or ${TMPDIR}" >&2
    return 2
  fi
  mkdir -p "${CANDIDATE_CKPT}" "${RAY_TMPDIR}" "${TMPDIR}"
  printf '%s\n' "${RAY_TMPDIR}" > "${CANDIDATE_ROOT}/ray_tmpdir.txt"
  printf '%s\n' "${TMPDIR}" > "${CANDIDATE_ROOT}/tmpdir.txt"
  ln -s "${SOURCE_CKPT}" "${CANDIDATE_CKPT}/global_step_${SOURCE_STEP}"
  printf '%s\n' "${SOURCE_STEP}" > "${CANDIDATE_CKPT}/latest_checkpointed_iteration.txt"

  monitor_gpu "${TELEMETRY}" &
  local monitor_pid=$!
  local status=0

  set +e
  (
    export RAY_TMPDIR TMPDIR
    cd "${VERL_ROOT}"
    DEVICE=gpu \
    INFER_BACKEND=vllm \
    MODEL_PATH="${MODEL_PATH}" \
    NGPUS_PER_NODE=1 \
    NNODES=1 \
    TRAIN_BATCH_SIZE="${BATCH_SIZE}" \
    PPO_MINI_BATCH_SIZE=8 \
    MAX_PROMPT_LENGTH=2048 \
    MAX_RESPONSE_LENGTH=1536 \
    PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
    ROLLOUT_TP=1 \
    ROLLOUT_N=8 \
    ROLLOUT_GPU_MEM_UTIL=0.30 \
    TOTAL_EPOCHS="${SMOKE_EPOCHS}" \
    SAVE_FREQ=-1 \
    TEST_FREQ=-1 \
    PROJECT_NAME=experiment_2e_batch_capacity \
    EXPERIMENT_NAME="batch_${BATCH_SIZE}_from_${SOURCE_STEP}" \
    bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
      data.train_files="${TRAIN_FILE}" \
      data.val_files="${VAL_FILE}" \
      data.train_batch_size="${BATCH_SIZE}" \
      data.dataloader_num_workers=0 \
      data.shuffle=True \
      data.seed="${SEED}" \
      data.filter_overlong_prompts=True \
      data.truncation=error \
      trainer.logger='["console"]' \
      trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
      trainer.total_epochs="${SMOKE_EPOCHS}" \
      trainer.val_before_train=False \
      trainer.resume_mode=auto \
      trainer.default_local_dir="${CANDIDATE_CKPT}" \
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
  ) > >(tee "${LOG}") 2>&1
  status=$?
  set -e

  kill "${monitor_pid}" >/dev/null 2>&1 || true
  wait "${monitor_pid}" >/dev/null 2>&1 || true
  printf '%s\n' "${status}" > "${CANDIDATE_ROOT}/exit_status.txt"
  wait_for_gpu_idle
  return "${status}"
}

assert_gpu_idle
printf '%s\n' "source_checkpoint=${SOURCE_CKPT}" "total_training_steps=${TOTAL_TRAINING_STEPS}" "batches=${BATCHES[*]}" > "${RUN_ROOT}/run_config.txt"

for batch in "${BATCHES[@]}"; do
  echo "BATCH_CAPACITY_START batch=${batch} source_step=${SOURCE_STEP}"
  if ! run_candidate "${batch}"; then
    echo "BATCH_CAPACITY_FAILED batch=${batch}; later candidates skipped" >&2
    exit 1
  fi
  echo "BATCH_CAPACITY_COMPLETE batch=${batch}"
done

test "$(tr -d '[:space:]' < "${SOURCE_CKPT_ROOT}/latest_checkpointed_iteration.txt")" = "${SOURCE_STEP}"
echo "BATCH_CAPACITY_SWEEP_COMPLETE run_root=${RUN_ROOT}"
