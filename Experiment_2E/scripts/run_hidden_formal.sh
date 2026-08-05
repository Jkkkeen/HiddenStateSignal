#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/Qwen2.5-7B-Instruct}
MANIFEST=${MANIFEST:?Set MANIFEST}
CKPT_DIR=${CKPT_DIR:?Set CKPT_DIR}
ROLLOUT_DIR=${ROLLOUT_DIR:?Set ROLLOUT_DIR}

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
print(m["save_frequency"])
PY
)
RUN_ID=${FROZEN[0]}
INTERVAL=${FROZEN[1]}
ROOT=${RESULT_ROOT:-/data2/hjk/results/experiment_2e/${RUN_ID}}
WORK=${ROOT}/hidden_work
METRICS=${ROOT}/metrics
CALIBRATOR=${METRICS}/base_calibrators.npz
ROLLOUT_AUDIT=${ROLLOUT_AUDIT:-${ROLLOUT_DIR}/rollout_audit.json}
SMOKE_METRICS=${SMOKE_METRICS:-${ROOT}/hidden_smoke/metrics}
mkdir -p "${WORK}" "${METRICS}"

"${ENV_ROOT}/bin/python" - "${ROLLOUT_AUDIT}" "${SMOKE_METRICS}" <<'PY'
import json
import pathlib
import sys
rollout = json.load(open(sys.argv[1], encoding="utf-8"))
assert rollout["passed"] and rollout["n_records"] == 12288, rollout
root = pathlib.Path(sys.argv[2])
for checkpoint in ("base", "final"):
    audit = json.loads((root / f"extraction_audit_{checkpoint}.json").read_text())
    assert audit["n_unique_rollouts"] == 4, audit
    assert audit["h8_not_duplicated_by_representation"], audit
PY

"${ENV_ROOT}/bin/python" -m experiment_2e.hidden_extract \
  --run-id "${RUN_ID}" --checkpoint base --rollout-dir "${ROLLOUT_DIR}" \
  --base-model-path "${MODEL_PATH}" --work-dir "${WORK}" --output-dir "${METRICS}" \
  --calibrator-path "${CALIBRATOR}"

checkpoints=(20pct 40pct 60pct 80pct final)
for index in 1 2 3 4 5; do
  checkpoint=${checkpoints[$((index - 1))]}
  step=$((INTERVAL * index))
  adapter=${CKPT_DIR}/global_step_${step}/actor/huggingface
  test -d "${adapter}"
  "${ENV_ROOT}/bin/python" -m experiment_2e.hidden_extract \
    --run-id "${RUN_ID}" --checkpoint "${checkpoint}" --rollout-dir "${ROLLOUT_DIR}" \
    --base-model-path "${MODEL_PATH}" --adapter-path "${adapter}" \
    --work-dir "${WORK}" --output-dir "${METRICS}" --calibrator-path "${CALIBRATOR}"
done
echo "EXPERIMENT_2E_HIDDEN_FORMAL_COMPLETE"
