#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-smoke5}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
VERL_ROOT=${VERL_ROOT:-/data2/hjk/projects/verl_q3_1p7b_base_20260814}
PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/Qwen3-1.7B-Base}
DATA_ROOT=${DATA_ROOT:-/data2/hjk/data/experiment_2e/q3_1p7b_base_seed20260814}
DECISION=${DECISION:-/data2/hjk/results/experiment_2e/q3_1p7b_base_calibration_seed20260814/dataset_selection_decision.json}
BASE_COMMON=${BASE_COMMON:-/data2/hjk/results/experiment_2e/q3_1p7b_base_calibration_seed20260814/base_common.npz}
SEED=${SEED:-20260814}
NOFILE_LIMIT=${NOFILE_LIMIT:-65535}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
case "${USE_REMOVE_PADDING}" in
  True|False) ;;
  *) echo "USE_REMOVE_PADDING must be True or False" >&2; exit 2 ;;
esac

case "${MODE}" in
  smoke5) TOTAL_STEPS=5; SAVE_FREQ=5; TEST_FREQ=5; RUN_SUFFIX=smoke; RESUME_MODE=disable; VAL_BEFORE=True ;;
  smoke20) TOTAL_STEPS=20; SAVE_FREQ=20; TEST_FREQ=20; RUN_SUFFIX=smoke; RESUME_MODE=auto; VAL_BEFORE=False ;;
  formal) TOTAL_STEPS=300; SAVE_FREQ=25; TEST_FREQ=25; RUN_SUFFIX=formal; RESUME_MODE=disable; VAL_BEFORE=True ;;
  *) echo "MODE must be smoke5, smoke20, or formal" >&2; exit 2 ;;
esac

ulimit -n "${NOFILE_LIMIT}"
export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export RAY_TMPDIR=${RAY_TMPDIR:-/data2/hjk/r/q3b17}
export TMPDIR=${TMPDIR:-/data2/hjk/t/q3b17}
export RAY_memory_usage_threshold=0.99
export HIDDEN_PROBE_RATE=${HIDDEN_PROBE_RATE:-1.0}
export HIDDEN_PROBE_INTERVAL=${HIDDEN_PROBE_INTERVAL:-1}
export HIDDEN_PROBE_STRIDE=128
export HIDDEN_PROBE_V3_MODE=base
export HIDDEN_PROBE_BASE_COMMON="${BASE_COMMON}"

test -f "${DECISION}"; test -f "${BASE_COMMON}"; test -f "${MODEL_PATH}/config.json"
mapfile -t FROZEN < <("${ENV_ROOT}/bin/python" - "${DECISION}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
assert p['status']=='go', p
for key in ('selected_dataset','selected_cap','train_file','calibration_file','heldout_file','smoke_file','smoke_val_file','formal_steps'):
    print(p[key])
PY
)
DATASET=${FROZEN[0]}; MAX_RESPONSE_LENGTH=${FROZEN[1]}
FORMAL_TRAIN_FILE=${FROZEN[2]}; CALIBRATION_FILE=${FROZEN[3]}; HELDOUT_FILE=${FROZEN[4]}; SMOKE_FILE=${FROZEN[5]}; SMOKE_VAL_FILE=${FROZEN[6]}
FORMAL_STEPS=${FROZEN[7]}
if [[ "${MODE}" == "formal" ]]; then
  TOTAL_STEPS=${FORMAL_STEPS}
  TRAIN_FILE=${FORMAL_TRAIN_FILE}
  VAL_FILE=${HELDOUT_FILE}
  : "${SMOKE_APPROVAL:?formal mode requires SMOKE_APPROVAL}"
  "${ENV_ROOT}/bin/python" - "${SMOKE_APPROVAL}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
assert p.get('status')=='passed', p
assert p.get('checkpoint_restore_passed') is True, p
assert p.get('dashboard_image_count') == 141, p
assert p.get('hidden_probe_interval') in (1, 5), p
PY
  HIDDEN_PROBE_INTERVAL=$("${ENV_ROOT}/bin/python" - "${SMOKE_APPROVAL}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1],encoding='utf-8'))['hidden_probe_interval'])
PY
)
  export HIDDEN_PROBE_INTERVAL
else
  TRAIN_FILE=${SMOKE_FILE}
  VAL_FILE=${SMOKE_VAL_FILE}
fi
test -f "${TRAIN_FILE}"; test -f "${VAL_FILE}"

RUN_NAME=${RUN_NAME:-qwen3_1p7b_base_${DATASET}_grpo_${RUN_SUFFIX}_seed${SEED}}
RESULT_ROOT=${RESULT_ROOT:-/data2/hjk/results/experiment_2e/${RUN_NAME}}
CKPT_ROOT=${CKPT_ROOT:-/data2/hjk/checkpoints/experiment_2e/${RUN_NAME}}
LOG=${LOG:-/data2/hjk/logs/experiment_2e/${RUN_NAME}.log}
export HIDDEN_PROBE_HISTORY_JSONL="${RESULT_ROOT}/online_hidden/per_step_summary.jsonl"
export HIDDEN_PROBE_HELDOUT_JSONL="${RESULT_ROOT}/heldout/hidden_summary.jsonl"
FIGURE_DIR="${RESULT_ROOT}/online_hidden/latest_figures"
SIDECAR_STATE="${RESULT_ROOT}/online_hidden/sidecar_state.json"
DONE_FILE="${RESULT_ROOT}/training.done"
SWANLAB_ENV_FILE=${SWANLAB_ENV_FILE:-/data2/hjk/secrets/experiment_2e_swanlab.env}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-experiment_2e_q3_1p7b_base_grpo}
TRAINER_LOGGER='["console"]'
SIDECAR_PID=""

mkdir -p "${RESULT_ROOT}" "${CKPT_ROOT}" "$(dirname "${LOG}")" "${FIGURE_DIR}" "${RAY_TMPDIR}" "${TMPDIR}"
if [[ "${RESUME_MODE}" == "disable" && -f "${CKPT_ROOT}/latest_checkpointed_iteration.txt" ]]; then
  echo "refusing to overwrite existing non-resume run at ${CKPT_ROOT}" >&2
  exit 3
fi
if [[ "${RESUME_MODE}" == "disable" ]]; then
  rm -f "${HIDDEN_PROBE_HISTORY_JSONL}" "${HIDDEN_PROBE_HELDOUT_JSONL}" "${SIDECAR_STATE}"
fi
rm -f "${DONE_FILE}"
if [[ -r "${SWANLAB_ENV_FILE}" ]]; then
  test "$(stat -c '%a' "${SWANLAB_ENV_FILE}")" = 600
  # shellcheck source=/dev/null
  source "${SWANLAB_ENV_FILE}"
  : "${SWANLAB_API_KEY:?missing SwanLab key}"
  export SWANLAB_RUN_ID=${SWANLAB_RUN_ID:-${RUN_NAME}}
  export SWANLAB_RESUME=allow SWANLAB_MODE=cloud
  export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-/data2/hjk/logs/experiment_2e/swanlab/${RUN_NAME}}
  mkdir -p "${SWANLAB_LOG_DIR}"
  TRAINER_LOGGER='["console","swanlab"]'
  "${ENV_ROOT}/bin/python" -m verl.hidden_probe.dashboard_sidecar \
    --history "${HIDDEN_PROBE_HISTORY_JSONL}" \
    --heldout-history "${HIDDEN_PROBE_HELDOUT_JSONL}" \
    --output-dir "${FIGURE_DIR}" \
    --state-file "${SIDECAR_STATE}" \
    --done-file "${DONE_FILE}" \
    --project "${SWANLAB_PROJECT}" \
    --experiment-name "${RUN_NAME}" \
    --run-id "${SWANLAB_RUN_ID}" \
    --log-dir "${SWANLAB_LOG_DIR}/hidden_sidecar" \
    --total-steps "${TOTAL_STEPS}" &
  SIDECAR_PID=$!
fi

finish_sidecar() {
  touch "${DONE_FILE}"
  if [[ -n "${SIDECAR_PID}" ]]; then wait "${SIDECAR_PID}" || true; fi
}
trap finish_sidecar EXIT

cat > "${RESULT_ROOT}/resolved_shell_config.json" <<JSON
{"mode":"${MODE}","dataset":"${DATASET}","steps":${TOTAL_STEPS},"train_batch_size":32,"rollout_n":8,"ppo_mini_batch_size_prompts":8,"max_response_length":${MAX_RESPONSE_LENGTH},"attention_implementation":"sdpa","remove_padding":${USE_REMOVE_PADDING,,},"dataloader_num_workers":0,"hidden_schema":"hidden_dashboard_v1_141x4","hidden_stride":128,"hidden_probe_rate":${HIDDEN_PROBE_RATE},"hidden_probe_interval":${HIDDEN_PROBE_INTERVAL},"model":"${MODEL_PATH}"}
JSON

cd "${VERL_ROOT}"
exec > >(tee -a "${LOG}") 2>&1
echo "Q3_1P7B_GRPO_START mode=${MODE} run=${RUN_NAME} dataset=${DATASET} cap=${MAX_RESPONSE_LENGTH}"
nvidia-smi

set +e
"${ENV_ROOT}/bin/python" -m recipe.dapo.main_dapo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.max_prompt_length=2048 \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.gen_batch_size=32 \
  data.train_batch_size=32 \
  data.truncation=error \
  data.filter_overlong_prompts=True \
  data.dataloader_num_workers=0 \
  data.shuffle=True \
  data.seed="${SEED}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  '+actor_rollout_ref.model.override_config={attn_implementation: sdpa}' \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2048 + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.0001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.optim.lr=0.000001 \
  actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((2048 + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((2048 + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=20 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_num_gen_batches=8 \
  reward.custom_reward_function.path="${PROJECT_ROOT}/experiment_2e/q3_math_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=dapo \
  reward.reward_kwargs.overlong_buffer_cfg.enable=False \
  reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name="${SWANLAB_PROJECT}" \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=100 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.val_before_train="${VAL_BEFORE}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.default_local_dir="${CKPT_ROOT}" \
  trainer.validation_data_dir="${RESULT_ROOT}/heldout/generations" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.log_val_generations=0 \
  trainer.balance_batch=True
STATUS=$?
set -e
echo "${STATUS}" > "${RESULT_ROOT}/exit_status.txt"
finish_sidecar
trap - EXIT
if (( STATUS != 0 )); then exit "${STATUS}"; fi
echo Q3_1P7B_GRPO_COMPLETE
