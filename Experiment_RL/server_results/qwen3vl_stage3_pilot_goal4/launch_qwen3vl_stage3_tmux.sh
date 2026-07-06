#!/usr/bin/env bash
# Launch one RL02 Stage 3 pilot in tmux from inside H200 bash.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/hjk/projects/AI-HiddenState-ER}
ENV_ROOT=${ENV_ROOT:-/data2/hjk/envs/verl_qwen3vl_py311}
EXPERIMENT=${EXPERIMENT:-a1}
SEED=${SEED:-42}
KEEP_OPEN_SECONDS=${KEEP_OPEN_SECONDS:-3600}

case "${EXPERIMENT}" in
    a1)
        SESSION=${SESSION:-qwen3vl_stage3_a1_seed42}
        RUN_NAME=${RUN_NAME:-qwen3vl8b_stage3_a1_long_answer_pilot_trainonly_prompt4096_seed${SEED}}
        RUN_SCRIPT=scripts/run_verl_qwen3vl_stage3_a1_long_pilot.sh
        EXIT_LABEL=TMUX_A1_EXIT_CODE
        ;;
    c2)
        SESSION=${SESSION:-qwen3vl_stage3_c2_seed42}
        RUN_NAME=${RUN_NAME:-qwen3vl8b_stage3_c2_zscore_pilot_trainonly_prompt4096_seed${SEED}}
        RUN_SCRIPT=scripts/run_verl_qwen3vl_stage3_c2_zscore_pilot.sh
        EXIT_LABEL=TMUX_C2_EXIT_CODE
        ;;
    *)
        echo "Unknown EXPERIMENT=${EXPERIMENT}; expected a1 or c2" >&2
        exit 2
        ;;
esac

cd "${PROJECT_ROOT}"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
"${ENV_ROOT}/bin/ray" stop -f || true

tmux new-session -d -s "${SESSION}" -c "${PROJECT_ROOT}" \
    "RUN_NAME=${RUN_NAME} SEED=${SEED} bash ${RUN_SCRIPT}; code=\$?; echo ${EXIT_LABEL}=\${code}; sleep ${KEEP_OPEN_SECONDS}"

tmux ls | grep "${SESSION}"
echo "SESSION=${SESSION}"
echo "RUN_NAME=${RUN_NAME}"
echo "LOG=${PROJECT_ROOT}/logs/${RUN_NAME}.log"
