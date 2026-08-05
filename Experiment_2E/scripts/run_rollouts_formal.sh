#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
MODEL_PATH=${MODEL_PATH:-/data2/hjk/models/Qwen2.5-7B-Instruct}
MANIFEST=${MANIFEST:?Set MANIFEST to frozen formal_run_manifest.json}
CKPT_DIR=${CKPT_DIR:?Set CKPT_DIR to the audited formal GRPO checkpoint root}

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/data2/hjk/cache/huggingface}
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mapfile -t FROZEN < <("${ENV_ROOT}/bin/python" - "${MANIFEST}" <<'PY'
import json
import sys

m = json.load(open(sys.argv[1], encoding="utf-8"))
interval = int(m["save_frequency"])
print(m["run_id"])
print(m["eval_file"])
print(m["rollouts_per_eval_question"])
print(m["base_seed"])
print(int(m["total_training_steps"]))
print(" ".join(str(interval * i) for i in range(6)))
PY
)
RUN_ID=${FROZEN[0]}
EVAL_FILE=${FROZEN[1]}
ROLLOUTS_PER_QUESTION=${FROZEN[2]}
SEED=${FROZEN[3]}
TOTAL_STEPS=${FROZEN[4]}
GLOBAL_STEPS=${FROZEN[5]}
SEED_MANIFEST=$(dirname "${MANIFEST}")/rollout_seed_manifest.parquet
OUTPUT_DIR=${OUTPUT_DIR:-/data2/hjk/results/experiment_2e/${RUN_ID}/rollouts}
AUDIT=${AUDIT:-/data2/hjk/results/experiment_2e/${RUN_ID}/rollouts/rollout_audit.json}
TRAINING_AUDIT=${TRAINING_AUDIT:-/data2/hjk/results/experiment_2e/${RUN_ID}/training/formal_training_audit.json}

test "${ROLLOUTS_PER_QUESTION}" = "8"
test -f "${MODEL_PATH}/config.json"
test -f "${EVAL_FILE}"
test -f "${SEED_MANIFEST}"
test -d "${CKPT_DIR}/global_step_${TOTAL_STEPS}/actor/huggingface"
"${ENV_ROOT}/bin/python" - "${TRAINING_AUDIT}" <<'PY'
import json
import sys
audit = json.load(open(sys.argv[1], encoding="utf-8"))
assert audit["passed"], audit
PY
mkdir -p "${OUTPUT_DIR}"

"${ENV_ROOT}/bin/python" -m experiment_2e.rollouts \
  --run-id "${RUN_ID}" \
  --eval-file "${EVAL_FILE}" \
  --base-model-path "${MODEL_PATH}" \
  --checkpoint-root "${CKPT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoints base 20pct 40pct 60pct 80pct final \
  --global-steps ${GLOBAL_STEPS} \
  --rollouts-per-question 8 \
  --max-new-tokens 1536 \
  --temperature 1.0 \
  --top-p 1.0 \
  --seed "${SEED}" \
  --seed-manifest "${SEED_MANIFEST}"

"${ENV_ROOT}/bin/python" -m experiment_2e.rollout_audit \
  --rollout-dir "${OUTPUT_DIR}" \
  --eval-file "${EVAL_FILE}" \
  --rollouts-per-question 8 \
  --seed-manifest "${SEED_MANIFEST}" \
  --output "${AUDIT}"
echo "EXPERIMENT_2E_ROLLOUTS_COMPLETE"
