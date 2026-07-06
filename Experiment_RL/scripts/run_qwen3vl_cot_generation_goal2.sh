#!/usr/bin/env bash
set -euo pipefail

cd /data2/hjk/projects/AI-HiddenState-ER

mkdir -p logs rl_audit/qwen3vl_cot_generation_goal2 \
  /data2/hjk/cache/vllm /data2/hjk/cache/torch /data2/hjk/tmp

export HF_HOME=/data2/hjk/models/huggingface
export HUGGINGFACE_HUB_CACHE=/data2/hjk/models/huggingface/hub
export TRANSFORMERS_CACHE=/data2/hjk/models/huggingface
export TORCH_HOME=/data2/hjk/cache/torch
export VLLM_CACHE_ROOT=/data2/hjk/cache/vllm
export XDG_CACHE_HOME=/data2/hjk/cache
export TMPDIR=/data2/hjk/tmp
export VLLM_WORKER_MULTIPROC_METHOD=spawn

/data2/hjk/envs/verl_qwen3vl_py311/bin/python scripts/qwen3vl_cot_generation_audit.py \
  --limit 16 \
  --max-tokens 16384 \
  --max-model-len 18432 \
  --gpu-memory-utilization 0.88 \
  --output-dir /data2/hjk/projects/AI-HiddenState-ER/rl_audit/qwen3vl_cot_generation_goal2
status=$?

echo "QWEN3VL_COT_GOAL2_EXIT_CODE=${status}"
df -h / /data2
exit "${status}"
