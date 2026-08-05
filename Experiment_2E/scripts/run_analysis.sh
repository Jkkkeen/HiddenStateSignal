#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
METRICS_DIR=${METRICS_DIR:?Set METRICS_DIR to completed scalar metric outputs}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to the formal result directory}
BOOTSTRAP=${BOOTSTRAP:-4000}

export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
"${ENV_ROOT}/bin/python" -m experiment_2e.analyze_discovery \
  --metrics-dir "${METRICS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap "${BOOTSTRAP}"
echo "EXPERIMENT_2E_ANALYSIS_COMPLETE"
