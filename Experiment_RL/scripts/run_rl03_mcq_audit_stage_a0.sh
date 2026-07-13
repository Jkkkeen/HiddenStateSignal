#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}
RAW_ROLLOUTS=${RAW_ROLLOUTS:-${PROJECT_ROOT}/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl}
FEATURES_PATH=${FEATURES_PATH:-${PROJECT_ROOT}/option_logit_trimmed_conclusion_smoke500/trimmed_conclusion_features.parquet}
MATHVERSE_METADATA=${MATHVERSE_METADATA:-${PROJECT_ROOT}/data/mathverse/testmini.json}

RUN_NAME=${RUN_NAME:-rl03_stage_a0_smoke32_seed20260713}
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/rl03_results/${RUN_NAME}}
MANIFEST_DIR=${MANIFEST_DIR:-${RUN_ROOT}/manifest}
SCORE_DIR=${SCORE_DIR:-${RUN_ROOT}/scores}
AUDIT_DIR=${AUDIT_DIR:-${RUN_ROOT}/audit}
SEED=${SEED:-20260713}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-10000}
OVERWRITE=${OVERWRITE:-0}

require_data2_path() {
  resolved=$(realpath -m "$1")
  case "${resolved}" in
    /data2|/data2/*) ;;
    *)
      echo "RL03 artifacts and inputs must stay under /data2: ${resolved}" >&2
      exit 2
      ;;
  esac
}
for data2_path in \
  "${PROJECT_ROOT}" \
  "${ENV_ROOT}" \
  "${MODEL_PATH}" \
  "${RAW_ROLLOUTS}" \
  "${FEATURES_PATH}" \
  "${MATHVERSE_METADATA}" \
  "${RUN_ROOT}" \
  "${MANIFEST_DIR}" \
  "${SCORE_DIR}" \
  "${AUDIT_DIR}"; do
  require_data2_path "${data2_path}"
done

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
  "${MANIFEST_DIR}" \
  "${SCORE_DIR}" \
  "${AUDIT_DIR}" \
  /data2/hjk/tmp \
  /data2/hjk/cache/huggingface/hub \
  /data2/hjk/cache/huggingface/datasets \
  /data2/hjk/cache/torch \
  /data2/hjk/cache/pip \
  /data2/hjk/cache/ray \
  /data2/hjk/cache/vllm/config \
  /data2/hjk/cache/torchinductor \
  /data2/hjk/cache/triton

monitor_pid=""
finish() {
  status=$?
  set +e
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ -f "${RUN_ROOT}/resource_samples.csv" ]]; then
    awk -F, '
      NR > 1 {
        if ($2 > cpu) cpu = $2
        if ($3 > gpu_mem) gpu_mem = $3
        if ($4 > gpu_util) gpu_util = $4
      }
      END {
        print "peak_cpu_used_bytes=" cpu
        print "peak_gpu_memory_mib=" gpu_mem
        print "peak_gpu_utilization_percent=" gpu_util
      }
    ' "${RUN_ROOT}/resource_samples.csv" > "${RUN_ROOT}/resource_peaks.txt"
  fi
  echo "RL03_STAGE_A0_EXIT_CODE=${status}" | tee "${RUN_ROOT}/exit_code.txt"
  date --iso-8601=seconds | tee "${RUN_ROOT}/finished_at.txt"
  free -h > "${RUN_ROOT}/memory_after.txt"
  df -h / /data2 > "${RUN_ROOT}/disk_after.txt"
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_after.txt" 2>&1
  trap - EXIT
  exit "${status}"
}
trap finish EXIT

for required in \
  "${ENV_ROOT}/bin/python" \
  "${MODEL_PATH}/config.json" \
  "${RAW_ROLLOUTS}" \
  "${FEATURES_PATH}" \
  "${MATHVERSE_METADATA}" \
  "${PROJECT_ROOT}/scripts/answer_normalization.py" \
  "${PROJECT_ROOT}/scripts/answer_likelihood_scoring.py" \
  "${PROJECT_ROOT}/scripts/build_rl03_mcq_audit_manifest.py" \
  "${PROJECT_ROOT}/scripts/run_rl03_mcq_audit.py" \
  "${PROJECT_ROOT}/scripts/run_rl03_mcq_audit_vllm.py" \
  "${PROJECT_ROOT}/scripts/analyze_rl03_mcq_audit.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 3
  fi
done

gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if [[ -n "${gpu_pids}" ]]; then
  echo "GPU is busy; refusing to disturb existing compute processes: ${gpu_pids}" >&2
  exit 4
fi

busy_workers=$(pgrep -u "${USER}" -af 'ray::|raylet|gcs_server|vllm\.entrypoints|vllm\.worker|EngineCore' || true)
if [[ -n "${busy_workers}" ]]; then
  echo "Ray/vLLM workers are active; refusing to clean them automatically:" >&2
  echo "${busy_workers}" >&2
  exit 5
fi

sample_resources() {
  echo "timestamp,cpu_used_bytes,gpu_memory_mib,gpu_utilization_percent"
  while true; do
    timestamp=$(date --iso-8601=seconds)
    cpu_used=$(free -b | awk '/^Mem:/ {print $3}')
    gpu_values=$(nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    echo "${timestamp},${cpu_used},${gpu_values}"
    sleep 5
  done
}
sample_resources > "${RUN_ROOT}/resource_samples.csv" 2>&1 &
monitor_pid=$!

cd "${PROJECT_ROOT}"
{
  date --iso-8601=seconds
  hostname
  pwd
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  "${ENV_ROOT}/bin/python" --version
  "${ENV_ROOT}/bin/python" -m pip freeze
  sha256sum \
    scripts/answer_normalization.py \
    scripts/answer_likelihood_scoring.py \
    scripts/build_rl03_mcq_audit_manifest.py \
    scripts/run_rl03_mcq_audit.py \
    scripts/run_rl03_mcq_audit_vllm.py \
    scripts/analyze_rl03_mcq_audit.py
} > "${RUN_ROOT}/environment_before.txt" 2>&1
free -h > "${RUN_ROOT}/memory_before.txt"
df -h / /data2 > "${RUN_ROOT}/disk_before.txt"
nvidia-smi > "${RUN_ROOT}/nvidia_smi_before.txt" 2>&1

"${ENV_ROOT}/bin/python" scripts/build_rl03_mcq_audit_manifest.py \
  --raw-rollouts "${RAW_ROLLOUTS}" \
  --features "${FEATURES_PATH}" \
  --mathverse-metadata "${MATHVERSE_METADATA}" \
  --output-dir "${MANIFEST_DIR}" \
  --question-count 32 \
  --seed "${SEED}"

score_args=(
  --manifest "${MANIFEST_DIR}/stage_a0_manifest.jsonl"
  --model "${MODEL_PATH}"
  --output-dir "${SCORE_DIR}"
  --project-root "${PROJECT_ROOT}"
  --batch-size "${BATCH_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --seed "${SEED}"
  --local-files-only
)
if [[ "${OVERWRITE}" == "1" ]]; then
  score_args+=(--overwrite)
fi
"${ENV_ROOT}/bin/python" scripts/run_rl03_mcq_audit_vllm.py "${score_args[@]}"

"${ENV_ROOT}/bin/python" scripts/analyze_rl03_mcq_audit.py \
  --scores "${SCORE_DIR}/request_scores.jsonl" \
  --output-dir "${AUDIT_DIR}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --seed "${SEED}"
