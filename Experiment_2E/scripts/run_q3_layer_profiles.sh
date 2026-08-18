#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-smoke}
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
  smoke)
    STEPS=(0 50 250)
    QUESTION_LIMIT=16
    OUTPUT_ROOT="${RUN_ROOT}/layer_profiles_smoke"
    ;;
  formal)
    STEPS=(0 25 50 75 100 125 150 175 200 225 250)
    QUESTION_LIMIT=256
    OUTPUT_ROOT="${RUN_ROOT}/layer_profiles_formal"
    SMOKE_APPROVAL="${RUN_ROOT}/layer_profiles_smoke/smoke_approval.json"
    "${ENV_ROOT}/bin/python" - "${SMOKE_APPROVAL}" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get("status") == "passed", p
assert p.get("expected_steps") == [0, 50, 250], p
assert p.get("expected_questions") == 16, p
PY
    ;;
  *)
    echo "MODE must be smoke or formal" >&2
    exit 2
    ;;
esac

test -f "${MODEL_PATH}/config.json"
test -f "${HELDOUT_FILE}"
test -d "${VERL_ROOT}/verl/model_merger"
mkdir -p "${OUTPUT_ROOT}/work/merged" "${OUTPUT_ROOT}/metrics" "${OUTPUT_ROOT}/analysis"
STATUS_FILE="${OUTPUT_ROOT}/run_status.json"

write_status() {
  local state=$1
  local exit_code=$2
  "${ENV_ROOT}/bin/python" - "${STATUS_FILE}" "${MODE}" "${state}" "${exit_code}" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(
    json.dumps(
        {
            "mode": sys.argv[2],
            "status": sys.argv[3],
            "exit_code": int(sys.argv[4]),
            "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
temporary.replace(path)
PY
}

finish() {
  local exit_code=$?
  set +e
  if (( exit_code == 0 )); then
    write_status completed 0
  else
    write_status failed "${exit_code}"
  fi
}
trap finish EXIT
write_status running 0

safe_remove_merged() {
  local target=$1
  local merged_root
  local merged_path
  merged_root=$(realpath -m "${OUTPUT_ROOT}/work/merged")
  merged_path=$(realpath -m "${target}")
  case "${merged_path}" in
    "${merged_root}"/*) rm -rf -- "${target}" ;;
    *)
      echo "refusing unsafe merged-model cleanup: ${merged_path}" >&2
      exit 4
      ;;
  esac
}

checkpoint_complete() {
  local step=$1
  "${ENV_ROOT}/bin/python" - "${OUTPUT_ROOT}/metrics" "${step}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
step = int(sys.argv[2])
name = f"step{step:03d}"
required = [
    root / f"layer_profiles_{name}.parquet",
    root / f"vertical_metrics_{name}.parquet",
    root / f"controls_{name}.parquet",
    root / f"model_identity_{name}.json",
    root / f"extraction_audit_{name}.json",
]
if not all(path.is_file() for path in required):
    raise SystemExit(1)
audit = json.loads(required[-1].read_text(encoding="utf-8"))
raise SystemExit(0 if audit.get("passed") else 1)
PY
}

echo "Q3_LAYER_PROFILES_START mode=${MODE} steps=${STEPS[*]} questions=${QUESTION_LIMIT}"
for step in "${STEPS[@]}"; do
  if checkpoint_complete "${step}"; then
    echo "CHECKPOINT_ALREADY_COMPLETE step=${step}"
    continue
  fi
  model="${MODEL_PATH}"
  source_args=()
  base_sample_args=()
  MERGED=""
  if (( step > 0 )); then
    MERGED="${OUTPUT_ROOT}/work/merged/step${step}"
    if [[ ! -f "${MERGED}/config.json" ]]; then
      if [[ -e "${MERGED}" ]]; then
        safe_remove_merged "${MERGED}"
      fi
      cd "${VERL_ROOT}"
      "${ENV_ROOT}/bin/python" -m verl.model_merger merge --backend fsdp \
        --local_dir "${CKPT_ROOT}/global_step_${step}/actor" \
        --target_dir "${MERGED}" --use_cpu_initialization
    fi
    model="${MERGED}"
    source_args=(--source-checkpoint-dir "${CKPT_ROOT}/global_step_${step}/actor")
    base_sample_args=(--base-parameter-sample "${OUTPUT_ROOT}/metrics/model_identity_step000.json")
  fi
  cd "${PROJECT_ROOT}"
  "${ENV_ROOT}/bin/python" -m experiment_2e.q3_layer_extract \
    --run-id "q3_1p7b_native_layer_profiles" \
    --generation-file "${RUN_ROOT}/heldout/generations/${step}.jsonl" \
    --heldout-file "${HELDOUT_FILE}" --model-path "${model}" \
    --tokenizer-path "${MODEL_PATH}" --model-name "Qwen3-1.7B-Base" \
    --global-step "${step}" --total-steps 250 \
    --work-dir "${OUTPUT_ROOT}/work" --output-dir "${OUTPUT_ROOT}/metrics" \
    --calibrator-path "${OUTPUT_ROOT}/metrics/base_calibrators.npz" \
    --question-limit "${QUESTION_LIMIT}" "${source_args[@]}" "${base_sample_args[@]}"
  if (( step > 0 )); then
    safe_remove_merged "${MERGED}"
  fi
  echo "CHECKPOINT_EXTRACTION_COMPLETE step=${step}"
done

expected_steps=$(IFS=,; echo "${STEPS[*]}")
analysis_args=()
if [[ "${MODE}" == smoke ]]; then
  analysis_args=(--approval-output "${OUTPUT_ROOT}/smoke_approval.json")
fi
cd "${PROJECT_ROOT}"
"${ENV_ROOT}/bin/python" -m experiment_2e.q3_layer_analysis \
  --profiles-dir "${OUTPUT_ROOT}/metrics" \
  --generation-dir "${RUN_ROOT}/heldout/generations" \
  --output-dir "${OUTPUT_ROOT}/analysis" \
  --expected-steps "${expected_steps}" --expected-questions "${QUESTION_LIMIT}" \
  "${analysis_args[@]}"

echo "Q3_LAYER_PROFILES_COMPLETE mode=${MODE} output=${OUTPUT_ROOT}"
