#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"
PYTHON="${PYTHON:-/data2/hjk/envs/hs_er/bin/python}"
RUN_NAME="${RUN_NAME:-experiment04_fixed256_20260726}"
SMOKE_SESSION="${SMOKE_SESSION:-exp04_fixed256_smoke}"
DISCOVERY_SESSION="${DISCOVERY_SESSION:-exp04_fixed256_discovery}"
CONFIRM_SESSION="${CONFIRM_SESSION:-exp04_fixed256_confirm}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
SEED="${SEED:-20260726}"
BOOTSTRAP="${BOOTSTRAP:-4000}"
PERMUTATIONS="${PERMUTATIONS:-4000}"
SMOKE_BOOTSTRAP="${SMOKE_BOOTSTRAP:-200}"
SMOKE_PERMUTATIONS="${SMOKE_PERMUTATIONS:-40}"
GPU_TOTAL_MIB="${GPU_TOTAL_MIB:-143771}"

SOURCE="${ROOT}/data_long/entropy_band_confirm120_20260723/rollouts_mt16384_labeled.jsonl"
RUN_DIR="${ROOT}/${RUN_NAME}"
MANIFEST_DIR="${RUN_DIR}/manifest"
SMOKE_DIR="${RUN_DIR}/smoke"
DISCOVERY_DIR="${RUN_DIR}/discovery"
CONFIRM_DIR="${RUN_DIR}/confirm"
FROZEN_HYPOTHESES="${RUN_DIR}/frozen_hypotheses.json"
SMOKE_LOG="${ROOT}/logs/${RUN_NAME}_smoke.log"
DISCOVERY_LOG="${ROOT}/logs/${RUN_NAME}_discovery.log"
CONFIRM_LOG="${ROOT}/logs/${RUN_NAME}_confirm.log"

prepare_manifests() {
  cd "${ROOT}"
  mkdir -p "${MANIFEST_DIR}" "${ROOT}/logs"
  "${PYTHON}" scripts/prepare_experiment04.py \
    --input "${SOURCE}" \
    --output-dir "${MANIFEST_DIR}" \
    --seed "${SEED}" \
    --expected-rollouts 8
  test -s "${MANIFEST_DIR}/manifest_discovery.jsonl"
  test -s "${MANIFEST_DIR}/manifest_confirm.jsonl"
  test -s "${MANIFEST_DIR}/manifest_smoke.jsonl"
  test -s "${MANIFEST_DIR}/split_manifest.json"
}

run_extract() {
  local manifest="$1"
  local output_dir="$2"
  mkdir -p "${output_dir}"
  "${PYTHON}" scripts/extract_experiment04_qwen3vl.py \
    --input "${manifest}" \
    --output-dir "${output_dir}" \
    --model "${MODEL}" \
    --chunk-size 256 \
    --last-n 25 \
    --local-files-only \
    --resume
}

run_reduce() {
  local manifest="$1"
  local extraction_dir="$2"
  local output_dir="$3"
  mkdir -p "${output_dir}"
  "${PYTHON}" scripts/reduce_experiment04.py \
    --input "${manifest}" \
    --extraction-dir "${extraction_dir}" \
    --output-dir "${output_dir}" \
    --mode discovery \
    --rolling 4
}

run_analysis() {
  local scalar_dir="$1"
  local output_dir="$2"
  local bootstrap="$3"
  local permutations="$4"
  local cluster_top="$5"
  mkdir -p "${output_dir}"
  "${PYTHON}" scripts/analyze_experiment04_discovery.py \
    --input-dir "${scalar_dir}" \
    --output-dir "${output_dir}" \
    --bootstrap "${bootstrap}" \
    --permutations "${permutations}" \
    --progress-bins 20 \
    --cluster-top-families "${cluster_top}" \
    --seed "${SEED}"
}

validate_extraction() {
  local extraction_dir="$1"
  "${PYTHON}" - "${extraction_dir}/progress.json" <<'PY'
import json
import sys
status = json.load(open(sys.argv[1], encoding="utf-8"))
if status["failed"] != 0 or status["completed"] != status["total"]:
    raise SystemExit(f"incomplete extraction: {status}")
PY
}

validate_results() {
  local scalar_dir="$1"
  local result_dir="$2"
  test -s "${scalar_dir}/calibration.npz"
  test -s "${scalar_dir}/horizontal_metrics.parquet"
  test -s "${scalar_dir}/vertical_metrics.parquet"
  test -s "${scalar_dir}/summary_metrics.parquet"
  test -s "${result_dir}/DISCOVERY_REPORT.md"
  test -s "${result_dir}/cell_effects.csv"
  test -s "${result_dir}/candidate_worksheet.csv"
}

validate_smoke_resources() {
  local telemetry="$1"
  local output="$2"
  "${PYTHON}" - "${telemetry}" "${output}" "${GPU_TOTAL_MIB}" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
complete = [row for row in rows if row.get("status") == "complete"]
failed = [row for row in rows if row.get("status") == "failed"]
if failed or not complete:
    raise SystemExit(f"smoke extraction failures: {failed}")
peak = max(float(row["peak_gpu_reserved_mib"]) for row in complete)
total = float(sys.argv[3])
payload = {
    "rollouts": len(complete),
    "peak_gpu_reserved_mib": peak,
    "gpu_total_mib": total,
    "headroom_fraction": 1.0 - peak / total,
    "mean_seconds_per_rollout": sum(float(row["elapsed_seconds"]) for row in complete) / len(complete),
    "max_cache_bytes": max(int(row["cache_bytes"]) for row in complete),
}
if payload["headroom_fraction"] < 0.15:
    raise SystemExit(f"GPU smoke gate failed: {payload}")
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

run_smoke_pipeline() {
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="/data2/hjk/models/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false
  test -s "${MANIFEST_DIR}/manifest_smoke.jsonl"
  mkdir -p "${SMOKE_DIR}/extraction" "${SMOKE_DIR}/scalars" "${SMOKE_DIR}/results"
  run_extract "${MANIFEST_DIR}/manifest_smoke.jsonl" "${SMOKE_DIR}/extraction"
  validate_extraction "${SMOKE_DIR}/extraction"
  validate_smoke_resources \
    "${SMOKE_DIR}/extraction/extraction_telemetry.jsonl" \
    "${SMOKE_DIR}/resource_gate.json"
  run_reduce \
    "${MANIFEST_DIR}/manifest_smoke.jsonl" \
    "${SMOKE_DIR}/extraction" \
    "${SMOKE_DIR}/scalars"
  run_analysis \
    "${SMOKE_DIR}/scalars" \
    "${SMOKE_DIR}/results" \
    "${SMOKE_BOOTSTRAP}" \
    "${SMOKE_PERMUTATIONS}" \
    5
  validate_results "${SMOKE_DIR}/scalars" "${SMOKE_DIR}/results"
  date -Is > "${SMOKE_DIR}/SMOKE_SUCCESS"
  echo "SMOKE_COMPLETE ${SMOKE_DIR}"
}

run_discovery_pipeline() {
  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="/data2/hjk/models/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false
  test -s "${SMOKE_DIR}/SMOKE_SUCCESS"
  test -s "${MANIFEST_DIR}/manifest_discovery.jsonl"
  mkdir -p "${DISCOVERY_DIR}/extraction" "${DISCOVERY_DIR}/scalars" "${DISCOVERY_DIR}/results"
  run_extract "${MANIFEST_DIR}/manifest_discovery.jsonl" "${DISCOVERY_DIR}/extraction"
  validate_extraction "${DISCOVERY_DIR}/extraction"
  run_reduce \
    "${MANIFEST_DIR}/manifest_discovery.jsonl" \
    "${DISCOVERY_DIR}/extraction" \
    "${DISCOVERY_DIR}/scalars"
  run_analysis \
    "${DISCOVERY_DIR}/scalars" \
    "${DISCOVERY_DIR}/results" \
    "${BOOTSTRAP}" \
    "${PERMUTATIONS}" \
    0
  validate_results "${DISCOVERY_DIR}/scalars" "${DISCOVERY_DIR}/results"
  date -Is > "${DISCOVERY_DIR}/DISCOVERY_SUCCESS"
  echo "DISCOVERY_COMPLETE ${DISCOVERY_DIR}"
}

validate_confirm_guard() {
  test -s "${FROZEN_HYPOTHESES}" || {
    echo "confirm blocked: missing ${FROZEN_HYPOTHESES}" >&2
    return 1
  }
  "${PYTHON}" - "${FROZEN_HYPOTHESES}" "${MANIFEST_DIR}/split_manifest.json" <<'PY'
import hashlib
import json
import sys
frozen = json.load(open(sys.argv[1], encoding="utf-8"))
actual = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
if frozen.get("split_manifest_sha256") != actual:
    raise SystemExit("confirm blocked: split manifest SHA256 mismatch")
if not frozen.get("hypotheses") or len(frozen["hypotheses"]) > 3:
    raise SystemExit("confirm blocked: expected 1-3 frozen hypotheses")
PY
}

run_confirm_pipeline() {
  validate_confirm_guard
  echo "confirm remains blocked until the reducer accepts a frozen metric filter" >&2
  return 1
}

start_tmux() {
  local session="$1"
  local internal_mode="$2"
  local log="$3"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    return 1
  fi
  mkdir -p "${ROOT}/logs"
  printf -v tmux_command \
    "cd %q && env ROOT=%q PYTHON=%q RUN_NAME=%q MODEL=%q SEED=%q BOOTSTRAP=%q PERMUTATIONS=%q GPU_TOTAL_MIB=%q bash scripts/launch_experiment04_tmux.sh %q 2>&1 | tee -a %q" \
    "${ROOT}" "${ROOT}" "${PYTHON}" "${RUN_NAME}" "${MODEL}" "${SEED}" "${BOOTSTRAP}" "${PERMUTATIONS}" "${GPU_TOTAL_MIB}" "${internal_mode}" "${log}"
  tmux new-session -d -s "${session}" "${tmux_command}"
  echo "launched tmux=${session} log=${log} run_dir=${RUN_DIR}"
}

case "${1:-}" in
  --prepare)
    prepare_manifests
    ;;
  --run-smoke)
    run_smoke_pipeline
    ;;
  --run-discovery)
    run_discovery_pipeline
    ;;
  --run-confirm)
    run_confirm_pipeline
    ;;
  --smoke)
    test -s "${MANIFEST_DIR}/manifest_smoke.jsonl" || prepare_manifests
    start_tmux "${SMOKE_SESSION}" "--run-smoke" "${SMOKE_LOG}"
    ;;
  --discovery)
    test -s "${SMOKE_DIR}/SMOKE_SUCCESS" || {
      echo "discovery blocked: smoke has not passed (${SMOKE_DIR}/SMOKE_SUCCESS missing)" >&2
      exit 1
    }
    start_tmux "${DISCOVERY_SESSION}" "--run-discovery" "${DISCOVERY_LOG}"
    ;;
  --confirm)
    validate_confirm_guard
    start_tmux "${CONFIRM_SESSION}" "--run-confirm" "${CONFIRM_LOG}"
    ;;
  *)
    echo "usage: $0 --prepare|--smoke|--discovery|--confirm" >&2
    exit 2
    ;;
esac
