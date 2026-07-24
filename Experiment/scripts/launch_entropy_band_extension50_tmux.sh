#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"
PYTHON="${PYTHON:-/data2/hjk/envs/hs_er/bin/python}"
ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-/data2/hjk/envs/verl_qwen3vl_py311/bin/python}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-/data2/hjk/models/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}"
SESSION="${SESSION:-entropy_band_extension50_20260724}"
BASE_RUN_NAME="${BASE_RUN_NAME:-entropy_band_confirm120_20260723}"
SEED="${SEED:-20260725}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
PEER_PID="${PEER_PID:-696866}"
BASE_RUN_DIR="${ROOT}/experiment0_hidden_dynamics/${BASE_RUN_NAME}"
BASE_CANDIDATES="${BASE_RUN_DIR}/candidates/candidate_ids_all800.txt"
BASE_CANDIDATE_SHA="a6be58071e4cba09297c7d2446d0fc632958f5759e7c0c8ca491d02e8da4eb80"
EXTENSION_DIR="${BASE_RUN_DIR}/candidate_extension_20260724"
DATA_DIR="${ROOT}/data_long/${BASE_RUN_NAME}"
RAW="${DATA_DIR}/rollouts_mt16384.jsonl"
LABELED="${DATA_DIR}/rollouts_mt16384_labeled.jsonl"
MANIFEST_DIR="${BASE_RUN_DIR}/manifest"
MATHVERSE_JSON="${ROOT}/data/mathverse/testmini.json"
PRIOR_THINKING="${ROOT}/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl"
LOG="${ROOT}/logs/${BASE_RUN_NAME}_extension50_20260724.log"

run_pipeline() {
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="/data2/hjk/models/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false
  mkdir -p "${EXTENSION_DIR}" "${DATA_DIR}/label_audit" "${MANIFEST_DIR}" "${ROOT}/logs"

  if [[ ! -s "${BASE_CANDIDATES}" ]]; then
    echo "missing frozen base candidates: ${BASE_CANDIDATES}" >&2
    exit 2
  fi
  actual_base_sha="$(sha256sum "${BASE_CANDIDATES}" | awk '{print $1}')"
  if [[ "${actual_base_sha}" != "${BASE_CANDIDATE_SHA}" ]]; then
    echo "base candidate SHA mismatch: ${actual_base_sha}" >&2
    exit 4
  fi
  peer_process="$(ps -p "${PEER_PID}" -o user=,cmd= || true)"
  if [[ -z "${peer_process}" ]]; then
    echo "peer PID is no longer visible before launch: ${PEER_PID}" >&2
    exit 3
  fi
  printf '%s\n' "${peer_process}" >"${EXTENSION_DIR}/peer_process_before.txt"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader >"${EXTENSION_DIR}/gpu_processes_before.txt"
  nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader >"${EXTENSION_DIR}/gpu_before.txt"

  "${PYTHON}" scripts/prepare_entropy_band_confirm120.py extend \
    --mathverse-json "${MATHVERSE_JSON}" \
    --exclude-rollouts "${PRIOR_THINKING}" \
    --existing-candidate-ids "${BASE_CANDIDATES}" \
    --output-dir "${EXTENSION_DIR}" \
    --extension-count 500 \
    --batch-size 50 \
    --seed "${SEED}"

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
    --question-ids "${EXTENSION_DIR}/candidate_ids_extension50_01.txt" \
    --resume

  "${PYTHON}" scripts/inspect_long_rollouts.py \
    --input "${RAW}" \
    --output-dir "${DATA_DIR}/label_audit" \
    --labeled-output "${LABELED}"
  "${PYTHON}" scripts/prepare_entropy_band_confirm120.py select \
    --labeled-rollouts "${LABELED}" \
    --candidate-ids "${EXTENSION_DIR}/candidate_ids_all1300.txt" \
    --output-dir "${MANIFEST_DIR}" \
    --target-questions 120 \
    --min-correct 2 \
    --min-wrong 2

  "${PYTHON}" - "${RAW}" "${EXTENSION_DIR}/candidate_ids_extension50_01.txt" \
    "${MANIFEST_DIR}/eligibility_status.json" "${PEER_PID}" <<'PY'
import collections
import json
import sys
from pathlib import Path

raw_path, batch_path, status_path = map(Path, sys.argv[1:4])
peer_pid = int(sys.argv[4])
batch = {
    line.strip()
    for line in batch_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
counts = collections.defaultdict(set)
seen = set()
duplicates = []
with raw_path.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row.get("question_id", ""))
        rollout_id = int(row.get("rollout_id", -1))
        key = (question_id, rollout_id)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        if question_id in batch:
            counts[question_id].add(rollout_id)
if duplicates:
    raise SystemExit(f"duplicate rollout keys found: {duplicates[:5]}")
incomplete = {
    qid: sorted(counts[qid])
    for qid in batch
    if counts[qid] != set(range(8))
}
if incomplete:
    raise SystemExit(f"extension batch incomplete: {incomplete}")
if not Path(f"/proc/{peer_pid}/stat").exists():
    raise SystemExit(f"peer PID disappeared: {peer_pid}")
status = json.loads(status_path.read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "status": "EXTENSION50_COMPLETE",
            "batch_questions": len(batch),
            "batch_rollouts": sum(len(values) for values in counts.values()),
            "eligible_questions": status["selected_questions"],
            "strict_3plus3_questions": status["strict_3plus3_questions"],
            "formal_ready": status["ready"],
        },
        indent=2,
    )
)
PY

  ps -p "${PEER_PID}" -o user=,pid=,stat=,etime=,cmd= \
    >"${EXTENSION_DIR}/peer_process_after.txt"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader >"${EXTENSION_DIR}/gpu_processes_after.txt"
  echo "EXTENSION50_PIPELINE_COMPLETE ${EXTENSION_DIR}"
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
  "cd %q && env ROOT=%q PYTHON=%q ROLLOUT_PYTHON=%q ROLLOUT_MODEL=%q SESSION=%q BASE_RUN_NAME=%q SEED=%q GPU_MEMORY_UTILIZATION=%q PEER_PID=%q bash scripts/launch_entropy_band_extension50_tmux.sh --run 2>&1 | tee %q" \
  "${ROOT}" "${ROOT}" "${PYTHON}" "${ROLLOUT_PYTHON}" "${ROLLOUT_MODEL}" \
  "${SESSION}" "${BASE_RUN_NAME}" "${SEED}" "${GPU_MEMORY_UTILIZATION}" "${PEER_PID}" "${LOG}"
tmux new-session -d -s "${SESSION}" "${tmux_command}"
echo "launched tmux=${SESSION} log=${LOG}"
