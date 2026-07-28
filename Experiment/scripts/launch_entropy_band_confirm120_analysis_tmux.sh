#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"
PYTHON="${PYTHON:-/data2/hjk/envs/hs_er/bin/python}"
MODEL="${MODEL:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}"
SESSION="${SESSION:-entropy_band_confirm120_analysis}"
RUN_NAME="${RUN_NAME:-entropy_band_confirm120_20260723}"
SEED="${SEED:-20260725}"
BOOTSTRAP="${BOOTSTRAP:-4000}"
MIN_FREE_MEMORY_MIB="${MIN_FREE_MEMORY_MIB:-100000}"

RUN_DIR="${ROOT}/experiment0_hidden_dynamics/${RUN_NAME}"
MANIFEST_DIR="${RUN_DIR}/manifest"
MANIFEST="${MANIFEST_DIR}/manifest_confirm120.jsonl"
STATUS="${MANIFEST_DIR}/eligibility_status.json"
ALL_CANDIDATES="${RUN_DIR}/candidate_extension_20260724/candidate_ids_all1300.txt"
EXTRACTION_DIR="${RUN_DIR}/extraction"
RESULT_DIR="${RUN_DIR}/results"
LOG="${ROOT}/logs/${RUN_NAME}_analysis_20260725.log"

MANIFEST_SHA="ea1dfb084581ca081d847f9792c4d180fd839135449189e94b2a88c429969684"
STATUS_SHA="c78e1db3d3e431e12bc5733285a1b1f669f260adf23c016155ce1f1d9a15b54a"
ALL_CANDIDATES_SHA="f7aad9e97a1ff2142716618d842a94808d321c2912aa033b3b84868e9e3e970c"

verify_sha() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA mismatch for ${path}: expected=${expected} actual=${actual}" >&2
    exit 2
  fi
}

resource_preflight() {
  local free_memory
  free_memory="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  echo "GPU_PREFLIGHT free_memory_mib=${free_memory} required_mib=${MIN_FREE_MEMORY_MIB}"
  if (( free_memory < MIN_FREE_MEMORY_MIB )); then
    echo "insufficient free GPU memory before hidden extraction: ${free_memory} MiB" >&2
    exit 3
  fi
}

verify_formal_manifest() {
  verify_sha "${MANIFEST}" "${MANIFEST_SHA}"
  verify_sha "${STATUS}" "${STATUS_SHA}"
  verify_sha "${ALL_CANDIDATES}" "${ALL_CANDIDATES_SHA}"
  if [[ ! -d "${MODEL}" ]]; then
    echo "local model snapshot is missing: ${MODEL}" >&2
    exit 4
  fi
  "${PYTHON}" - "${MANIFEST}" "${STATUS}" "${ALL_CANDIDATES_SHA}" <<'PY'
import collections
import json
import sys
from pathlib import Path

manifest_path, status_path = map(Path, sys.argv[1:3])
candidate_sha = sys.argv[3]
status = json.loads(status_path.read_text(encoding="utf-8"))
if status.get("ready") is not True or status.get("selected_questions") != 120:
    raise SystemExit(f"formal status is not ready for exactly 120 questions: {status}")
if status.get("entropy_used_for_selection") is not False:
    raise SystemExit("formal cohort was not selected entropy-blind")
if status.get("candidate_ids_sha256") != candidate_sha:
    raise SystemExit("status does not reference the frozen all1300 candidate hash")

rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 930:
    raise SystemExit(f"formal manifest row count changed: {len(rows)}")
keys = [(str(row["question_id"]), int(row["rollout_id"])) for row in rows]
if len(keys) != len(set(keys)):
    raise SystemExit("formal manifest contains duplicate question/rollout keys")
groups = collections.defaultdict(list)
for row in rows:
    groups[str(row["question_id"])].append(bool(row["is_correct"]))
if len(groups) != 120:
    raise SystemExit(f"formal manifest question count changed: {len(groups)}")
bad = {
    qid: {"correct": sum(labels), "wrong": len(labels) - sum(labels)}
    for qid, labels in groups.items()
    if sum(labels) < 2 or len(labels) - sum(labels) < 2
}
if bad:
    raise SystemExit(f"formal manifest lost 2+2 eligibility: {bad}")
print(json.dumps({
    "status": "FORMAL_MANIFEST_VERIFIED",
    "questions": len(groups),
    "rollouts": len(rows),
    "strict_3plus3_questions": status["strict_3plus3_questions"],
    "entropy_used_for_selection": status["entropy_used_for_selection"],
}))
PY
}

extract_features() {
  "${PYTHON}" scripts/extract_entropy_band_qwen3vl.py \
    --input "${MANIFEST}" \
    --output-dir "${EXTRACTION_DIR}" \
    --model "${MODEL}" \
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
  "${PYTHON}" - "${EXTRACTION_DIR}/EXTRACTION_SUMMARY.json" "${RESULT_DIR}/analysis_meta.json" <<'PY'
import json
import sys
from pathlib import Path

extraction_path, analysis_path = map(Path, sys.argv[1:])
extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
if extraction.get("failed") != 0 or extraction.get("available_feature_shards") != 120:
    raise SystemExit(f"extraction incomplete: {extraction}")
if extraction.get("layers") != [14, 15, 16, 17, 18, 19]:
    raise SystemExit(f"unexpected extraction layers: {extraction}")
if extraction.get("progress_bins") != 10 or extraction.get("progress_bin") != 6:
    raise SystemExit(f"unexpected progress specification: {extraction}")
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
if analysis.get("questions") != 120 or analysis.get("alternate_cell_scan") is not False:
    raise SystemExit(f"analysis validation failed: {analysis}")
print(json.dumps({
    "status": "ENTROPY_CONFIRM120_COMPLETE",
    "outcome": analysis["primary_outcome"],
    "questions": analysis["questions"],
}))
PY
  test -s "${RESULT_DIR}/ENTROPY_BAND_CONFIRM120_RESULTS.md"
  test -s "${RESULT_DIR}/confirmatory_gate_results.csv"
  test -s "${RESULT_DIR}/figures/E03_F1_question_correct_wrong.png"
  test -s "${RESULT_DIR}/figures/E03_F2_question_auc.png"
}

check_prelaunch() {
  cd "${ROOT}"
  verify_formal_manifest
  resource_preflight
  echo "ANALYSIS_PRELAUNCH_CHECK_OK"
}

run_pipeline() {
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="/data2/hjk/models/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false
  mkdir -p "${EXTRACTION_DIR}" "${RESULT_DIR}" "${ROOT}/logs"

  verify_formal_manifest
  resource_preflight
  echo "HIDDEN_EXTRACTION_START layers=14-19 progress_bin=6 questions=120 rollouts=930"
  extract_features
  echo "HIDDEN_EXTRACTION_COMPLETE"
  analyze_features
  echo "CONFIRMATORY_ANALYSIS_COMPLETE"
  validate_outputs
}

if [[ "${1:-}" == "--check" ]]; then
  check_prelaunch
  exit 0
fi

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
  "cd %q && env ROOT=%q PYTHON=%q MODEL=%q SESSION=%q RUN_NAME=%q SEED=%q BOOTSTRAP=%q MIN_FREE_MEMORY_MIB=%q bash scripts/launch_entropy_band_confirm120_analysis_tmux.sh --run 2>&1 | tee -a %q" \
  "${ROOT}" "${ROOT}" "${PYTHON}" "${MODEL}" "${SESSION}" "${RUN_NAME}" "${SEED}" \
  "${BOOTSTRAP}" "${MIN_FREE_MEMORY_MIB}" "${LOG}"
tmux new-session -d -s "${SESSION}" "${tmux_command}"
echo "launched tmux=${SESSION} log=${LOG} run_dir=${RUN_DIR}"
