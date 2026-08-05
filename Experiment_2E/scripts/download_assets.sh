#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState/Experiment_2E}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
HF_HOME=${HF_HOME:-/data2/hjk/cache/huggingface}
MODEL_DIR=${MODEL_DIR:-/data2/hjk/models/Qwen2.5-7B-Instruct}
RAW_DATA_DIR=${RAW_DATA_DIR:-/data2/hjk/data/experiment_2e/math_lighteval_raw}
PREPARED_DATA_DIR=${PREPARED_DATA_DIR:-/data2/hjk/data/experiment_2e/math_prepared_seed20260805}
LOG_DIR=${LOG_DIR:-/data2/hjk/logs/experiment_2e}

export HF_ENDPOINT HF_HOME
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="${ENV_ROOT}/bin:${PATH}"

mkdir -p "${MODEL_DIR}" "${RAW_DATA_DIR}" "${PREPARED_DATA_DIR}" "${LOG_DIR}"

python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="${MODEL_DIR}",
    max_workers=8,
)
PY

python - <<PY
from pathlib import Path
from datasets import load_dataset

destination = Path("${RAW_DATA_DIR}")
if not (destination / "dataset_dict.json").exists():
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval")
    dataset.save_to_disk(destination)
else:
    print(f"Raw dataset already present: {destination}")
PY

python -m experiment_2e.data_prep \
  --local-dataset-path "${RAW_DATA_DIR}" \
  --output-dir "${PREPARED_DATA_DIR}" \
  --eval-size 256 \
  --seed 20260805

python -m experiment_2e.reward_audit \
  --data-file "${PREPARED_DATA_DIR}/eval_256.parquet" \
  --output "${PREPARED_DATA_DIR}/reward_audit.json"

python -m experiment_2e.environment_audit \
  --model-path "${MODEL_DIR}" \
  --data-path "${PREPARED_DATA_DIR}/train_level3_5.parquet" \
  --data-path "${PREPARED_DATA_DIR}/eval_256.parquet" \
  --output "${PREPARED_DATA_DIR}/environment_audit.json" \
  --strict

echo "ASSET_PREPARATION_COMPLETE"
