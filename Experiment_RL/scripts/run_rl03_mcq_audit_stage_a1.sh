#!/usr/bin/env bash
set -euo pipefail

STAGE_NAME=stage_a1
SELECTION_MODE=all-parseable
RUN_NAME=${RUN_NAME:-rl03_stage_a1_v2_full_seed20260713}

export STAGE_NAME SELECTION_MODE RUN_NAME
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_rl03_mcq_audit_stage_a0.sh"
