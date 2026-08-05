#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/Qwen2.5-7B-Instruct}
MANIFEST=${MANIFEST:?Set MANIFEST to frozen formal_run_manifest.json}
CKPT_DIR=${CKPT_DIR:?Set CKPT_DIR to the audited formal checkpoint root}
ROLLOUT_DIR=${ROLLOUT_DIR:?Set ROLLOUT_DIR to the audited six-checkpoint rollout directory}

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/data2/hjk/cache/huggingface}
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mapfile -t FROZEN < <("${ENV_ROOT}/bin/python" - "${MANIFEST}" <<'PY'
import json
import sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
print(m["run_id"])
print(m["total_training_steps"])
PY
)
RUN_ID=${FROZEN[0]}
TOTAL_STEPS=${FROZEN[1]}
ROOT=${SMOKE_ROOT:-/data2/hjk/results/experiment_2e/${RUN_ID}/hidden_smoke}
WORK=${ROOT}/work
METRICS=${ROOT}/metrics
CALIBRATOR=${ROOT}/base_calibrators.npz
FINAL_ADAPTER=${CKPT_DIR}/global_step_${TOTAL_STEPS}/actor/huggingface
ROLLOUT_AUDIT=${ROLLOUT_AUDIT:-${ROLLOUT_DIR}/rollout_audit.json}

test -f "${MODEL_PATH}/config.json"
test -d "${FINAL_ADAPTER}"
"${ENV_ROOT}/bin/python" - "${ROLLOUT_AUDIT}" <<'PY'
import json
import sys
audit = json.load(open(sys.argv[1], encoding="utf-8"))
assert audit["passed"], audit
assert audit["n_records"] == 12288, audit
PY
mkdir -p "${WORK}" "${METRICS}"

"${ENV_ROOT}/bin/python" -m experiment_2e.hidden_extract \
  --run-id "${RUN_ID}_hidden_smoke" --checkpoint base \
  --rollout-dir "${ROLLOUT_DIR}" --base-model-path "${MODEL_PATH}" \
  --work-dir "${WORK}" --output-dir "${METRICS}" --calibrator-path "${CALIBRATOR}" \
  --question-limit 2 --rollouts-per-question-limit 2 --keep-pooled-cache

"${ENV_ROOT}/bin/python" -m experiment_2e.hidden_extract \
  --run-id "${RUN_ID}_hidden_smoke" --checkpoint final \
  --rollout-dir "${ROLLOUT_DIR}" --base-model-path "${MODEL_PATH}" --adapter-path "${FINAL_ADAPTER}" \
  --work-dir "${WORK}" --output-dir "${METRICS}" --calibrator-path "${CALIBRATOR}" \
  --question-limit 2 --rollouts-per-question-limit 2 --keep-pooled-cache

"${ENV_ROOT}/bin/python" - "${METRICS}" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
for checkpoint in ("base", "final"):
    audit = json.loads((root / f"extraction_audit_{checkpoint}.json").read_text())
    assert audit["n_unique_rollouts"] == 4, audit
    assert audit["h8_not_duplicated_by_representation"], audit
print("HIDDEN_SMOKE_AUDIT_PASSED")
PY
echo "EXPERIMENT_2E_HIDDEN_SMOKE_COMPLETE"
