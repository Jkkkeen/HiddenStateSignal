#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"
PYTHON="${PYTHON:-/data2/hjk/envs/hs_er/bin/python}"
ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-/data2/hjk/envs/verl_qwen3vl_py311/bin/python}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}"
SESSION="${SESSION:-entropy_band_confirm120}"
RUN_NAME="${RUN_NAME:-entropy_band_confirm120_20260723}"
SEED="${SEED:-20260725}"
BOOTSTRAP="${BOOTSTRAP:-4000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
RUN_DIR="${ROOT}/experiment0_hidden_dynamics/${RUN_NAME}"
CANDIDATE_DIR="${RUN_DIR}/candidates"
DATA_DIR="${ROOT}/data_long/${RUN_NAME}"
RAW="${DATA_DIR}/rollouts_mt16384.jsonl"
LABELED="${DATA_DIR}/rollouts_mt16384_labeled.jsonl"
MANIFEST_DIR="${RUN_DIR}/manifest"
EXTRACTION_DIR="${RUN_DIR}/extraction"
RESULT_DIR="${RUN_DIR}/results"
LOG="${ROOT}/logs/${RUN_NAME}.log"
MATHVERSE_JSON="${ROOT}/data/mathverse/testmini.json"
PRIOR_THINKING="${ROOT}/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl"

freeze_candidates() {
  mkdir -p "${CANDIDATE_DIR}"
  "${PYTHON}" scripts/prepare_entropy_band_confirm120.py freeze \
    --mathverse-json "${MATHVERSE_JSON}" \
    --exclude-rollouts "${PRIOR_THINKING}" \
    --output-dir "${CANDIDATE_DIR}" \
    --candidate-count 800 \
    --primary-count 600 \
    --reserve-batch 50 \
    --seed "${SEED}"
}

generate_batch() {
  local question_ids="$1"
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
    --question-ids "${question_ids}" \
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
    --candidate-ids "${CANDIDATE_DIR}/candidate_ids_all800.txt" \
    --output-dir "${MANIFEST_DIR}" \
    --target-questions 120 \
    --min-correct 2 \
    --min-wrong 2
}

cohort_ready() {
  "${PYTHON}" - "${MANIFEST_DIR}/eligibility_status.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print("yes" if json.load(handle)["ready"] else "no")
PY
}

extract_features() {
  "${PYTHON}" scripts/extract_entropy_band_qwen3vl.py \
    --input "${MANIFEST_DIR}/manifest_confirm120.jsonl" \
    --output-dir "${EXTRACTION_DIR}" \
    --model Qwen/Qwen3-VL-8B-Thinking \
    --layers 14,15,16,17,18,19 \
    --progress-bins 10 \
    --progress-bin 6 \
    --entropy-chunk-size 512 \
    --min-correct 2 \
    --min-wrong 2 \
    --local-files-only \
    --resume
}

analyze_features() {
  "${PYTHON}" scripts/analyze_entropy_band_confirm120.py \
    --input-dir "${EXTRACTION_DIR}" \
    --output-dir "${RESULT_DIR}" \
    --expected-questions 120 \
    --bootstrap "${BOOTSTRAP}" \
    --seed "${SEED}"
}

validate_outputs() {
  "${PYTHON}" - \
    "${MANIFEST_DIR}/eligibility_status.json" \
    "${MANIFEST_DIR}/manifest_confirm120.jsonl" \
    "${EXTRACTION_DIR}/EXTRACTION_SUMMARY.json" \
    "${RESULT_DIR}/analysis_meta.json" <<'PY'
import json
import sys
from pathlib import Path

status_path, manifest_path, extraction_path, analysis_path = map(Path, sys.argv[1:])
status = json.loads(status_path.read_text(encoding="utf-8"))
if not status["ready"] or status["selected_questions"] != 120:
    raise SystemExit(f"formal cohort is not exactly 120 questions: {status}")
questions = {
    str(json.loads(line)["question_id"])
    for line in manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
if len(questions) != 120:
    raise SystemExit(f"formal manifest has {len(questions)} questions")
extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
if extraction["failed"] != 0 or extraction["available_feature_shards"] != 120:
    raise SystemExit(f"extraction incomplete: {extraction}")
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
if analysis["questions"] != 120 or analysis["alternate_cell_scan"]:
    raise SystemExit(f"analysis validation failed: {analysis}")
print(json.dumps({"status": "PIPELINE_COMPLETE", "outcome": analysis["primary_outcome"]}))
PY
  test -s "${RESULT_DIR}/ENTROPY_BAND_CONFIRM120_RESULTS.md"
  test -s "${RESULT_DIR}/confirmatory_gate_results.csv"
  test -s "${RESULT_DIR}/figures/E03_F1_question_correct_wrong.png"
  test -s "${RESULT_DIR}/figures/E03_F2_question_auc.png"
}

run_pipeline() {
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="/data2/hjk/models/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false
  mkdir -p "${DATA_DIR}" "${MANIFEST_DIR}" "${EXTRACTION_DIR}" "${RESULT_DIR}" "${ROOT}/logs"

  freeze_candidates
  batches=("${CANDIDATE_DIR}/candidate_ids_primary600.txt")
  while IFS= read -r reserve_file; do
    batches+=("${reserve_file}")
  done < <(find "${CANDIDATE_DIR}" -maxdepth 1 -name 'candidate_ids_reserve50_*.txt' -print | sort)

  ready="no"
  for batch in "${batches[@]}"; do
    echo "GENERATION_BATCH_START ${batch}"
    generate_batch "${batch}"
    echo "GENERATION_BATCH_COMPLETE ${batch}"
    label_and_select
    ready="$(cohort_ready)"
    echo "ELIGIBILITY_READY ${ready}"
    if [[ "${ready}" == "yes" ]]; then
      break
    fi
  done
  if [[ "${ready}" != "yes" ]]; then
    echo "Candidate reserve exhausted before reaching 120 eligible questions" >&2
    exit 2
  fi

  extract_features
  analyze_features
  validate_outputs
}

if [[ "${1:-}" == "--run" ]]; then
  run_pipeline
  exit 0
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "${ROOT}/logs"
printf -v tmux_command \
  "cd %q && env ROOT=%q PYTHON=%q ROLLOUT_PYTHON=%q ROLLOUT_MODEL=%q SESSION=%q RUN_NAME=%q SEED=%q BOOTSTRAP=%q GPU_MEMORY_UTILIZATION=%q bash scripts/launch_entropy_band_confirm120_tmux.sh --run 2>&1 | tee %q" \
  "${ROOT}" "${ROOT}" "${PYTHON}" "${ROLLOUT_PYTHON}" "${ROLLOUT_MODEL}" "${SESSION}" "${RUN_NAME}" "${SEED}" "${BOOTSTRAP}" "${GPU_MEMORY_UTILIZATION}" "${LOG}"
tmux new-session -d -s "${SESSION}" "${tmux_command}"
echo "launched tmux=${SESSION} log=${LOG} run_dir=${RUN_DIR}"
