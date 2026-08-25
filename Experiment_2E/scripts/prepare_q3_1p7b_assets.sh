#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/data2/hjk/models/Qwen3-1.7B-Base}
VERL_DIR=${VERL_DIR:-/data2/hjk/projects/verl_q3_1p7b_base_20260814}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
LOG=${LOG:-/data2/hjk/logs/experiment_2e/q3_1p7b_assets_20260814.log}

mkdir -p "$(dirname "${MODEL_DIR}")" "$(dirname "${VERL_DIR}")" "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

if [[ ! -d "${VERL_DIR}/.git" ]]; then
  if [[ -e "${VERL_DIR}" ]]; then
    echo "Refusing to overwrite non-git path: ${VERL_DIR}" >&2
    exit 1
  fi
  git clone https://github.com/sunnykaibai/verl_shx_v2.git "${VERL_DIR}"
fi

git -C "${VERL_DIR}" fetch origin main
git -C "${VERL_DIR}" switch main
git -C "${VERL_DIR}" pull --ff-only origin main

HF_ENDPOINT=https://hf-mirror.com \
  "${ENV_ROOT}/bin/huggingface-cli" download Qwen/Qwen3-1.7B-Base \
  --local-dir "${MODEL_DIR}"

test -f "${MODEL_DIR}/config.json"
"${ENV_ROOT}/bin/python" -m verl.hidden_probe.doctor

echo "ASSETS_COMPLETE"
