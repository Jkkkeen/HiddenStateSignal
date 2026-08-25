#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data2/hjk/data/experiment_2e/simpleRL_zoo_level3_5_20260814}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "${OUTPUT_ROOT}"

"${ENV_ROOT}/bin/hf" download hkust-nlp/SimpleRL-Zoo-Data \
  --repo-type dataset \
  --include 'simplelr_qwen_level3to5/*.parquet' \
  --local-dir "${OUTPUT_ROOT}"

find "${OUTPUT_ROOT}" -type f -name '*.parquet' -print
echo SIMPLERL_DOWNLOAD_COMPLETE
