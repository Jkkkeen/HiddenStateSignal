#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"
PYTHON="${PYTHON:-/data2/hjk/envs/hs_er/bin/python}"
ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-/data2/hjk/envs/verl_qwen3vl_py311/bin/python}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}"
SESSION="${SESSION:-entropy_band_confirm120}"
RUN_NAME="${RUN_NAME:-entropy_band_confirm120_20260723}"
SEED="${SEED:-20260725}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

RUN_DIR="${ROOT}/experiment0_hidden_dynamics/${RUN_NAME}"
CANDIDATE_DIR="${RUN_DIR}/candidates"
DATA_DIR="${ROOT}/data_long/${RUN_NAME}"
RAW="${DATA_DIR}/rollouts_mt16384.jsonl"
LABELED="${DATA_DIR}/rollouts_mt16384_labeled.jsonl"
MANIFEST_DIR="${RUN_DIR}/manifest"
LOG="${ROOT}/logs/${RUN_NAME}_extension50.log"
MATHVERSE_JSON="${ROOT}/data/mathverse/testmini.json"
PRIOR_THINKING="${ROOT}/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl"
EXISTING_IDS="${CANDIDATE_DIR}/candidate_ids_all800.txt"
COMBINED_IDS="${CANDIDATE_DIR}/candidate_ids_all850.txt"
EXTENSION_IDS="${CANDIDATE_DIR}/candidate_ids_extension50.txt"

freeze_extension() {
  "${PYTHON}" scripts/prepare_entropy_band_confirm120.py extend \
    --mathverse-json "${MATHVERSE_JSON#${ROOT}/}" \
    --exclude-rollouts "${PRIOR_THINKING#${ROOT}/}" \
    --existing-candidate-ids "${EXISTING_IDS#${ROOT}/}" \
    --output-dir "${CANDIDATE_DIR#${ROOT}/}" \
    --extension-count 50 \
    --batch-size 50 \
    --seed "${SEED}"
}

generate_extension() {
  "${ROLLOUT_PYTHON}" scripts/roll_mathverse_vllm.py \
    --data-dir "${ROOT}/data/mathverse" \
    --split testmini \
    --output "${RAW}" \
    --model "${ROLLOUT_MODEL}" \
    --rollouts 8 \
    --limit -1 \
    --temperature 0.7 \
    --top-p 0.95 \
    --max-tokens 16384 \
    --max-model-len 32768 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --question-ids "${EXTENSION_IDS}" \
    --resume
}

label_and_select() {
  mkdir -p "${DATA_DIR}/label_audit" "${MANIFEST_DIR}"
  "${PYTHON}" scripts/inspect_long_rollouts.py \
    --input "${RAW}" \
    --output-dir "${DATA_DIR}/label_audit" \
    --labeled-output "${LABELED}"
  "${PYTHON}" scripts/prepare_entropy_band_confirm120.py select \
    --labeled-rollouts "${LABELED}" \
    --candidate-ids "${COMBINED_IDS}" \
    --output-dir "${MANIFEST_DIR}" \
    --target-questions 120 \
    --min-correct 2 \
    --min-wrong 2
}

validate_extension() {
  "${PYTHON}" - "${RAW}" "${EXTENSION_IDS}" "${MANIFEST_DIR}/eligibility_status.json" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

raw_path, ids_path, status_path = map(Path, sys.argv[1:])
extension_ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
seen = defaultdict(list)
with raw_path.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row.get("question_id", ""))
        if qid in extension_ids:
            seen[qid].append(int(row["rollout_id"]))
bad = {
    qid: sorted(rollout_ids)
    for qid, rollout_ids in seen.items()
    if sorted(rollout_ids) != list(range(8))
}
missing = [qid for qid in extension_ids if qid not in seen]
if missing or bad or len(seen) != 50:
    raise SystemExit(
        f"extension validation failed: missing={missing}, bad={bad}, seen={len(seen)}"
    )
status = json.loads(status_path.read_text(encoding="utf-8"))
print(json.dumps({
    "status": "EXTENSION50_COMPLETE",
    "extension_questions": len(seen),
    "extension_rollouts": sum(len(ids) for ids in seen.values()),
    "eligible_questions_after_extension": status["selected_questions"],
    "cohort_ready": status["ready"],
}))
PY
}

run_probe() {
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="/data2/hjk/models/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false
  mkdir -p "${CANDIDATE_DIR}" "${DATA_DIR}" "${MANIFEST_DIR}" "${ROOT}/logs"

  freeze_extension
  echo "EXTENSION50_GENERATION_START"
  generate_extension
  echo "EXTENSION50_GENERATION_COMPLETE"
  label_and_select
  validate_extension
}

if [[ "${1:-}" == "--run" ]]; then
  run_probe
  exit 0
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "${ROOT}/logs"
printf -v tmux_command \
  "cd %q && env ROOT=%q PYTHON=%q ROLLOUT_PYTHON=%q ROLLOUT_MODEL=%q SESSION=%q RUN_NAME=%q SEED=%q GPU_MEMORY_UTILIZATION=%q bash scripts/launch_entropy_band_confirm120_extension50_tmux.sh --run 2>&1 | tee -a %q" \
  "${ROOT}" "${ROOT}" "${PYTHON}" "${ROLLOUT_PYTHON}" "${ROLLOUT_MODEL}" "${SESSION}" "${RUN_NAME}" "${SEED}" "${GPU_MEMORY_UTILIZATION}" "${LOG}"
tmux new-session -d -s "${SESSION}" "${tmux_command}"
echo "launched tmux=${SESSION} log=${LOG} run_dir=${RUN_DIR}"
