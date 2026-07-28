#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"
PYTHON="${PYTHON:-/data2/hjk/envs/hs_er/bin/python}"
ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-/data2/hjk/envs/verl_qwen3vl_py311/bin/python}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}"
SESSION="${SESSION:-entropy_band_confirm120_batches}"
BASE_RUN_NAME="${BASE_RUN_NAME:-entropy_band_confirm120_20260723}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MIN_FREE_MEMORY_MIB="${MIN_FREE_MEMORY_MIB:-100000}"
START_BATCH="${START_BATCH:-2}"
END_BATCH="${END_BATCH:-10}"

BASE_RUN_DIR="${ROOT}/experiment0_hidden_dynamics/${BASE_RUN_NAME}"
EXTENSION_DIR="${BASE_RUN_DIR}/candidate_extension_20260724"
ALL_CANDIDATES="${EXTENSION_DIR}/candidate_ids_all1300.txt"
ALL_CANDIDATES_SHA="f7aad9e97a1ff2142716618d842a94808d321c2912aa033b3b84868e9e3e970c"
BASE_CANDIDATES="${BASE_RUN_DIR}/candidates/candidate_ids_all800.txt"
BASE_CANDIDATES_SHA="a6be58071e4cba09297c7d2446d0fc632958f5759e7c0c8ca491d02e8da4eb80"
DATA_DIR="${ROOT}/data_long/${BASE_RUN_NAME}"
RAW="${DATA_DIR}/rollouts_mt16384.jsonl"
LABELED="${DATA_DIR}/rollouts_mt16384_labeled.jsonl"
MANIFEST_DIR="${BASE_RUN_DIR}/manifest"
STATUS="${MANIFEST_DIR}/eligibility_status.json"
LOG="${ROOT}/logs/${BASE_RUN_NAME}_remaining_batches_20260724.log"

verify_frozen_candidates() {
  local base_sha all_sha
  base_sha="$(sha256sum "${BASE_CANDIDATES}" | awk '{print $1}')"
  all_sha="$(sha256sum "${ALL_CANDIDATES}" | awk '{print $1}')"
  if [[ "${base_sha}" != "${BASE_CANDIDATES_SHA}" ]]; then
    echo "base candidate SHA mismatch: ${base_sha}" >&2
    exit 2
  fi
  if [[ "${all_sha}" != "${ALL_CANDIDATES_SHA}" ]]; then
    echo "all1300 candidate SHA mismatch: ${all_sha}" >&2
    exit 3
  fi
  if ! cmp -s <(head -n 800 "${ALL_CANDIDATES}") "${BASE_CANDIDATES}"; then
    echo "all1300 candidate prefix does not match frozen all800" >&2
    exit 4
  fi
}

cohort_ready() {
  "${PYTHON}" - "${STATUS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("no")
else:
    print("yes" if json.loads(path.read_text(encoding="utf-8"))["ready"] else "no")
PY
}

resource_preflight() {
  local free_memory
  free_memory="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  echo "GPU_PREFLIGHT free_memory_mib=${free_memory} required_mib=${MIN_FREE_MEMORY_MIB}"
  if (( free_memory < MIN_FREE_MEMORY_MIB )); then
    echo "insufficient free GPU memory before batch: ${free_memory} MiB" >&2
    exit 5
  fi
}

generate_batch() {
  local batch_file="$1"
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
    --question-ids "${batch_file}" \
    --resume
}

label_and_select() {
  "${PYTHON}" scripts/inspect_long_rollouts.py \
    --input "${RAW}" \
    --output-dir "${DATA_DIR}/label_audit" \
    --labeled-output "${LABELED}"
  "${PYTHON}" scripts/prepare_entropy_band_confirm120.py select \
    --labeled-rollouts "${LABELED}" \
    --candidate-ids "${ALL_CANDIDATES}" \
    --output-dir "${MANIFEST_DIR}" \
    --target-questions 120 \
    --min-correct 2 \
    --min-wrong 2
}

validate_batch() {
  local batch_file="$1"
  local batch_index="$2"
  "${PYTHON}" - "${RAW}" "${batch_file}" "${STATUS}" "${batch_index}" <<'PY'
import collections
import json
import sys
from pathlib import Path

raw_path, batch_path, status_path = map(Path, sys.argv[1:4])
batch_index = int(sys.argv[4])
batch = [line.strip() for line in batch_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(batch) != 50 or len(set(batch)) != 50:
    raise SystemExit(f"batch {batch_index} does not contain 50 unique questions")
batch_set = set(batch)
counts = collections.defaultdict(set)
seen = set()
duplicates = []
with raw_path.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row.get("question_id", ""))
        rollout_id = int(row.get("rollout_id", -1))
        key = (qid, rollout_id)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        if qid in batch_set:
            counts[qid].add(rollout_id)
if duplicates:
    raise SystemExit(f"duplicate rollout keys found: {duplicates[:5]}")
incomplete = {qid: sorted(counts[qid]) for qid in batch if counts[qid] != set(range(8))}
if incomplete:
    raise SystemExit(f"batch {batch_index} incomplete: {incomplete}")
status = json.loads(status_path.read_text(encoding="utf-8"))
print(json.dumps({
    "status": "BATCH_COMPLETE",
    "batch_index": batch_index,
    "batch_questions": len(batch),
    "batch_rollouts": sum(len(counts[qid]) for qid in batch),
    "eligible_questions": status["selected_questions"],
    "strict_3plus3_questions": status["strict_3plus3_questions"],
    "formal_ready": status["ready"],
}))
PY
  cp "${STATUS}" "${EXTENSION_DIR}/eligibility_status_after_batch_${batch_index}.json"
}

run_batches() {
  local batch batch_file batch_label ready
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="/data2/hjk/models/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false
  mkdir -p "${DATA_DIR}/label_audit" "${MANIFEST_DIR}" "${ROOT}/logs"

  verify_frozen_candidates
  ready="$(cohort_ready)"
  if [[ "${ready}" == "yes" ]]; then
    echo "COHORT_ALREADY_READY"
    exit 0
  fi

  for ((batch = START_BATCH; batch <= END_BATCH; batch++)); do
    printf -v batch_label '%02d' "${batch}"
    batch_file="${EXTENSION_DIR}/candidate_ids_extension50_${batch_label}.txt"
    if [[ ! -s "${batch_file}" ]]; then
      echo "missing frozen batch file: ${batch_file}" >&2
      exit 6
    fi
    resource_preflight
    echo "GENERATION_BATCH_START batch=${batch} file=${batch_file}"
    generate_batch "${batch_file}"
    echo "GENERATION_BATCH_COMPLETE batch=${batch}"
    label_and_select
    validate_batch "${batch_file}" "${batch}"
    ready="$(cohort_ready)"
    if [[ "${ready}" == "yes" ]]; then
      echo "COHORT_READY_AFTER_BATCH batch=${batch}"
      exit 0
    fi
  done

  echo "FROZEN_CANDIDATE_POOL_EXHAUSTED_WITHOUT_120" >&2
  exit 7
}

check_prelaunch() {
  local batch batch_file batch_label unique_count
  cd "${ROOT}"
  verify_frozen_candidates
  for ((batch = START_BATCH; batch <= END_BATCH; batch++)); do
    printf -v batch_label '%02d' "${batch}"
    batch_file="${EXTENSION_DIR}/candidate_ids_extension50_${batch_label}.txt"
    if [[ ! -s "${batch_file}" ]]; then
      echo "missing frozen batch file: ${batch_file}" >&2
      exit 6
    fi
    unique_count="$(sort -u "${batch_file}" | sed '/^$/d' | wc -l)"
    if [[ "${unique_count}" -ne 50 ]]; then
      echo "batch ${batch} has ${unique_count} unique question IDs" >&2
      exit 8
    fi
  done
  resource_preflight
  echo "PRELAUNCH_CHECK_OK start_batch=${START_BATCH} end_batch=${END_BATCH}"
}

if [[ "${1:-}" == "--run" ]]; then
  run_batches
  exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
  check_prelaunch
  exit 0
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "${ROOT}/logs"
printf -v tmux_command \
  "cd %q && env ROOT=%q PYTHON=%q ROLLOUT_PYTHON=%q ROLLOUT_MODEL=%q SESSION=%q BASE_RUN_NAME=%q GPU_MEMORY_UTILIZATION=%q MIN_FREE_MEMORY_MIB=%q START_BATCH=%q END_BATCH=%q bash scripts/launch_entropy_band_remaining_batches_tmux.sh --run 2>&1 | tee -a %q" \
  "${ROOT}" "${ROOT}" "${PYTHON}" "${ROLLOUT_PYTHON}" "${ROLLOUT_MODEL}" \
  "${SESSION}" "${BASE_RUN_NAME}" "${GPU_MEMORY_UTILIZATION}" "${MIN_FREE_MEMORY_MIB}" \
  "${START_BATCH}" "${END_BATCH}" "${LOG}"
tmux new-session -d -s "${SESSION}" "${tmux_command}"
echo "launched tmux=${SESSION} log=${LOG}"
