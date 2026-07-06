#!/usr/bin/env bash
# RL02 C2.1 smoke: C2 zscore reward with vLLM prompt-logprobs option probes.

set -xeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-qwen3vl8b_stage3_c2_1_vllm_probe_smoke18_seed${SEED}}

export PROJECT_ROOT
export SEED
export RUN_NAME
export TOTAL_STEPS=${TOTAL_STEPS:-18}
export ROLLOUT_N=${ROLLOUT_N:-2}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.50}
export OPTION_GAIN_BACKEND=${OPTION_GAIN_BACKEND:-vllm}
export OPTION_GAIN_RESPONSE_FRACS=${OPTION_GAIN_RESPONSE_FRACS:-0.0,0.25,0.50,0.90}
export OPTION_GAIN_REWARD_START_INDEX=${OPTION_GAIN_REWARD_START_INDEX:-1}
export OPTION_GAIN_MAX_RESPONSE_CHARS=${OPTION_GAIN_MAX_RESPONSE_CHARS:-4096}
export LOG=${LOG:-${PROJECT_ROOT}/logs/${RUN_NAME}.log}
export CKPT_DIR=${CKPT_DIR:-/data2/hjk/checkpoints/verl_qwen3vl/${RUN_NAME}}
export VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${PROJECT_ROOT}/rl_audit/${RUN_NAME}/validation_generations}

bash "${PROJECT_ROOT}/scripts/run_verl_qwen3vl_stage3_c2_zscore_pilot.sh"
