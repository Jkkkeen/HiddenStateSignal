#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/data2/hjk/models/Qwen3-1.7B-Base}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
LOG=${LOG:-/data2/hjk/logs/experiment_2e/q3_1p7b_model_download_20260814.log}

mkdir -p "${MODEL_DIR}" "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

HF_ENDPOINT=https://hf-mirror.com \
  "${ENV_ROOT}/bin/huggingface-cli" download Qwen/Qwen3-1.7B-Base \
  --local-dir "${MODEL_DIR}"

test -f "${MODEL_DIR}/config.json"
"${ENV_ROOT}/bin/python" - "${MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
print(json.dumps({
    "model_type": config.get("model_type"),
    "num_hidden_layers": config.get("num_hidden_layers"),
    "hidden_size": config.get("hidden_size"),
}, sort_keys=True))
PY

echo "MODEL_DOWNLOAD_COMPLETE"
