#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
SWANLAB_ENV_FILE=${SWANLAB_ENV_FILE:-/data2/hjk/secrets/experiment_2e_swanlab.env}
ANALYSIS_DIR=${ANALYSIS_DIR:?Set ANALYSIS_DIR to a completed analysis directory}
FIGURES_DIR=${FIGURES_DIR:?Set FIGURES_DIR to the corresponding figures/primary directory}
CHECKPOINT_MANIFEST=${CHECKPOINT_MANIFEST:?Set CHECKPOINT_MANIFEST to checkpoint_manifest.json}
PROJECT_NAME=${PROJECT_NAME:-experiment_2e_math_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?Set EXPERIMENT_NAME to the formal SwanLab experiment name}
SWANLAB_RUN_ID=${SWANLAB_RUN_ID:?Set SWANLAB_RUN_ID to the resumed training run ID}
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-/data2/hjk/logs/experiment_2e/swanlab/${EXPERIMENT_NAME}}

if [[ ! -r "${SWANLAB_ENV_FILE}" ]]; then
  echo "SwanLab credential file is unreadable: ${SWANLAB_ENV_FILE}" >&2
  exit 1
fi
if [[ "$(stat -c '%a' "${SWANLAB_ENV_FILE}")" != "600" ]]; then
  echo "SwanLab credential file must have mode 600: ${SWANLAB_ENV_FILE}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${SWANLAB_ENV_FILE}"
if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY must be defined in ${SWANLAB_ENV_FILE}" >&2
  exit 1
fi

test -f "${ANALYSIS_DIR}/primary_standardized.parquet"
test -f "${ANALYSIS_DIR}/outcome_effects.csv"
test -f "${CHECKPOINT_MANIFEST}"
mkdir -p "${SWANLAB_LOG_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
"${ENV_ROOT}/bin/python" -m experiment_2e.swanlab_upload \
  --analysis-dir "${ANALYSIS_DIR}" \
  --figures-dir "${FIGURES_DIR}" \
  --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
  --project "${PROJECT_NAME}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --run-id "${SWANLAB_RUN_ID}" \
  --log-dir "${SWANLAB_LOG_DIR}"
