#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-formal}
PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814}
VERL_ROOT=${VERL_ROOT:-/data2/hjk/projects/verl_q3_1p7b_base_20260814}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/Qwen3-1.7B-Base}
RUN_ROOT=${RUN_ROOT:-/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814}
CKPT_ROOT=${CKPT_ROOT:-/data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814}
HELDOUT_FILE=${HELDOUT_FILE:-/data2/hjk/data/experiment_2e/q3_1p7b_base_simplerl_strict_v2_seed20260814/simplerl_heldout_256.parquet}

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

case "${MODE}" in
  smoke) STEPS=(100); QUESTION_LIMIT=16 ;;
  formal) STEPS=(0 25 50 75 100 125 150 175 200 225 250); QUESTION_LIMIT=256 ;;
  *) echo "MODE must be smoke or formal" >&2; exit 2 ;;
esac

if [[ "${MODE}" == "smoke" ]]; then
  OUTPUT_ROOT="${RUN_ROOT}/h2_h10_smoke"
else
  OUTPUT_ROOT="${RUN_ROOT}/h2_h10_formal"
fi
mkdir -p "${OUTPUT_ROOT}/work/merged" "${OUTPUT_ROOT}/metrics"

safe_remove_merged() {
  local target=$1
  local root path
  root=$(realpath -m "${OUTPUT_ROOT}/work/merged")
  path=$(realpath -m "${target}")
  case "${path}" in
    "${root}"/*) rm -rf -- "${target}" ;;
    *) echo "refusing unsafe merged-model cleanup: ${path}" >&2; exit 4 ;;
  esac
}

for step in "${STEPS[@]}"; do
  output_dir="${OUTPUT_ROOT}/metrics"
  if [[ -f "${output_dir}/h2_h10_step$(printf '%03d' "${step}").parquet" && -f "${output_dir}/h2_h10_audit_step$(printf '%03d' "${step}").json" ]]; then
    echo "CHECKPOINT_ALREADY_COMPLETE step=${step}"
    continue
  fi
  model="${MODEL_PATH}"
  merged=""
  if (( step > 0 )); then
    merged="${OUTPUT_ROOT}/work/merged/step${step}"
    if [[ ! -f "${merged}/config.json" ]]; then
      if [[ -e "${merged}" ]]; then safe_remove_merged "${merged}"; fi
      cd "${VERL_ROOT}"
      "${ENV_ROOT}/bin/python" -m verl.model_merger merge --backend fsdp \
        --local_dir "${CKPT_ROOT}/global_step_${step}/actor" \
        --target_dir "${merged}" --use_cpu_initialization
    fi
    model="${merged}"
  fi
  cd "${PROJECT_ROOT}"
  "${ENV_ROOT}/bin/python" -m experiment_2e.q3_h2_h10_extract \
    --run-id q3_1p7b_h2_h10_formal \
    --generation-file "${RUN_ROOT}/heldout/generations/${step}.jsonl" \
    --heldout-file "${HELDOUT_FILE}" --model-path "${model}" \
    --tokenizer-path "${MODEL_PATH}" --model-name Qwen3-1.7B-Base \
    --global-step "${step}" --total-steps 250 \
    --output-dir "${output_dir}" --question-limit "${QUESTION_LIMIT}"
  if (( step > 0 )); then safe_remove_merged "${merged}"; fi
done

cd "${PROJECT_ROOT}"
"${ENV_ROOT}/bin/python" -m experiment_2e.analyze_q3_h2_h10 \
  --metrics-dir "${OUTPUT_ROOT}/metrics" \
  --generation-dir "${RUN_ROOT}/heldout/generations" \
  --output-dir "${OUTPUT_ROOT}/analysis" \
  --steps "${STEPS[@]}"

echo "Q3_H2_H10_COMPLETE mode=${MODE} output=${OUTPUT_ROOT}"
