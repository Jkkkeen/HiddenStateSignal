# Entropy-Band Candidate Extension First-50 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze MathVerse candidate ranks 801-1300 under the original confirmatory ordering, then safely generate and audit only ranks 801-850 alongside an unrelated idle Jupyter kernel.

**Architecture:** Extend the existing candidate-preparation module with an immutable prefix-validated `extend` command, without changing the frozen entropy feature or selection rule. Add a separate first-50 tmux launcher that appends resumably to the existing raw rollout file at `gpu_memory_utilization=0.65`, rebuilds only derived labels/eligibility outputs, and never invokes hidden extraction. Synchronize the two changed scripts to H200, launch a unique session, and observe it at launch, five minutes, and ten minutes before leaving the single batch running.

**Tech Stack:** Python 3.10+, pytest, Bash, vLLM in `/data2/hjk/envs/verl_qwen3vl_py311`, analysis Python in `/data2/hjk/envs/hs_er`, tmux, NVIDIA H200.

## Global Constraints

- Preserve `Experiment/2dimension03.md`: entropy feature remains L14-L19 mean at progress bin6; no entropy is computed during this plan.
- Existing ranks 1-800 and their SHA-256 `a6be58071e4cba09297c7d2446d0fc632958f5759e7c0c8ca491d02e8da4eb80` are immutable.
- Extension order uses seed `20260725`; freeze ranks 801-1300 before generation and generate ranks 801-850 only.
- Raw generation remains 8 rollouts/question, temperature `0.7`, top-p `0.95`, max tokens `16384`, max model length `32768`.
- vLLM uses `gpu_memory_utilization=0.65` for this concurrent smoke.
- Do not signal, pause, attach to, or modify user `hao.lin` PID `696866` or its Cursor/Jupyter process tree.
- Existing raw JSONL is append-only. The labeled JSONL and eligibility files are derived artifacts and may be atomically rebuilt.
- Do not launch later extension batches, hidden extraction, statistics, or RL in this plan.
- On safety failure, stop only the newly created tmux session and its process tree.

---

### Task 1: Immutable Candidate Extension Command

**Files:**
- Modify: `Experiment/scripts/prepare_entropy_band_confirm120.py`
- Modify: `Experiment/tests/test_prepare_entropy_band_confirm120.py`

**Interfaces:**
- Consumes: MathVerse records, old Thinking-500 question IDs, frozen ranks 1-800, seed `20260725`.
- Produces: `freeze_candidate_extension(...) -> tuple[list[str], list[str]]` and CLI command `extend`.
- Writes: `candidate_ids_all1300.txt`, `candidate_ids_extension500.txt`, ten `candidate_ids_extension50_XX.txt` files, and `candidate_extension_audit.json`.

- [ ] **Step 1: Add failing unit tests for prefix validation and extension ranks**

Append these imports and tests to `Experiment/tests/test_prepare_entropy_band_confirm120.py`:

```python
import pytest

from prepare_entropy_band_confirm120 import (  # noqa: E402
    freeze_candidate_extension,
    freeze_candidates,
    select_formal_cohort,
)


def test_freeze_candidate_extension_preserves_prefix_and_takes_next_ranks() -> None:
    records = [_record(f"q{i:02d}") for i in range(20)]
    existing = freeze_candidates(
        records,
        excluded_question_ids=set(),
        candidate_count=8,
        seed=20260725,
    )

    combined, extension = freeze_candidate_extension(
        records,
        excluded_question_ids=set(),
        existing_candidate_ids=existing,
        extension_count=5,
        seed=20260725,
    )

    assert combined[:8] == existing
    assert extension == combined[8:13]
    assert len(extension) == 5
    assert not set(existing).intersection(extension)


def test_freeze_candidate_extension_rejects_changed_existing_prefix() -> None:
    records = [_record(f"q{i:02d}") for i in range(20)]
    existing = freeze_candidates(
        records,
        excluded_question_ids=set(),
        candidate_count=8,
        seed=20260725,
    )
    existing[0], existing[1] = existing[1], existing[0]

    with pytest.raises(ValueError, match="existing candidate prefix does not match"):
        freeze_candidate_extension(
            records,
            excluded_question_ids=set(),
            existing_candidate_ids=existing,
            extension_count=5,
            seed=20260725,
        )
```

- [ ] **Step 2: Run the focused test to verify failure**

Run: `python -m pytest Experiment/tests/test_prepare_entropy_band_confirm120.py -q`

Expected: collection fails because `freeze_candidate_extension` is not defined.

- [ ] **Step 3: Refactor deterministic ordering and implement prefix validation**

Replace the body-level ordering inside `freeze_candidates` with these functions in `Experiment/scripts/prepare_entropy_band_confirm120.py`:

```python
def ordered_candidate_ids(
    records: Iterable[dict[str, Any]],
    *,
    excluded_question_ids: set[str],
    seed: int,
) -> list[str]:
    unique: set[str] = set()
    for index, record in enumerate(records):
        qid = question_id(record, index)
        answer = str(record.get("answer", "")).strip().upper()
        if qid in excluded_question_ids or answer not in MCQ_ANSWERS or not _has_query(record):
            continue
        unique.add(qid)
    return sorted(
        unique,
        key=lambda qid: (
            hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest(),
            qid,
        ),
    )


def freeze_candidates(
    records: Iterable[dict[str, Any]],
    *,
    excluded_question_ids: set[str],
    candidate_count: int,
    seed: int,
) -> list[str]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    ordered = ordered_candidate_ids(
        records,
        excluded_question_ids=excluded_question_ids,
        seed=seed,
    )
    if len(ordered) < candidate_count:
        raise ValueError(f"only {len(ordered)} unused MCQ candidates; need {candidate_count}")
    return ordered[:candidate_count]


def freeze_candidate_extension(
    records: Iterable[dict[str, Any]],
    *,
    excluded_question_ids: set[str],
    existing_candidate_ids: list[str],
    extension_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not existing_candidate_ids:
        raise ValueError("existing candidate IDs must not be empty")
    if extension_count <= 0:
        raise ValueError("extension_count must be positive")
    ordered = ordered_candidate_ids(
        records,
        excluded_question_ids=excluded_question_ids,
        seed=seed,
    )
    prefix_size = len(existing_candidate_ids)
    required = prefix_size + extension_count
    if len(ordered) < required:
        raise ValueError(f"only {len(ordered)} unused MCQ candidates; need {required}")
    if ordered[:prefix_size] != existing_candidate_ids:
        raise ValueError("existing candidate prefix does not match deterministic order")
    combined = ordered[:required]
    return combined, combined[prefix_size:]
```

- [ ] **Step 4: Run the focused tests to verify the pure function passes**

Run: `python -m pytest Experiment/tests/test_prepare_entropy_band_confirm120.py -q`

Expected: all tests pass.

- [ ] **Step 5: Add failing CLI artifact test**

Append to `Experiment/tests/test_prepare_entropy_band_confirm120.py`:

```python
import json

from prepare_entropy_band_confirm120 import run_extend


def test_run_extend_writes_combined_extension_batches_and_audit(tmp_path: Path) -> None:
    mathverse = tmp_path / "testmini.json"
    excluded = tmp_path / "old.jsonl"
    existing = tmp_path / "candidate_ids_all8.txt"
    output = tmp_path / "extension"
    records = [_record(f"q{i:02d}") for i in range(20)]
    mathverse.write_text(json.dumps(records), encoding="utf-8")
    excluded.write_text("", encoding="utf-8")
    frozen = freeze_candidates(
        records,
        excluded_question_ids=set(),
        candidate_count=8,
        seed=20260725,
    )
    existing.write_text("".join(f"{qid}\n" for qid in frozen), encoding="utf-8")
    args = type("Args", (), {
        "mathverse_json": str(mathverse),
        "exclude_rollouts": str(excluded),
        "existing_candidate_ids": str(existing),
        "output_dir": str(output),
        "extension_count": 6,
        "batch_size": 2,
        "seed": 20260725,
    })()

    run_extend(args)

    assert len((output / "candidate_ids_all14.txt").read_text().splitlines()) == 14
    assert len((output / "candidate_ids_extension6.txt").read_text().splitlines()) == 6
    assert len(list(output.glob("candidate_ids_extension2_*.txt"))) == 3
    audit = json.loads((output / "candidate_extension_audit.json").read_text())
    assert audit["existing_count"] == 8
    assert audit["extension_count"] == 6
    assert audit["entropy_used_for_selection"] is False
```

- [ ] **Step 6: Run the test to verify the CLI artifact code is missing**

Run: `python -m pytest Experiment/tests/test_prepare_entropy_band_confirm120.py::test_run_extend_writes_combined_extension_batches_and_audit -q`

Expected: collection fails because `run_extend` is not defined.

- [ ] **Step 7: Implement immutable writes and the `extend` subcommand**

Add these helpers and command body:

```python
def write_frozen_ids(path: Path, ids: list[str]) -> None:
    content = "".join(f"{qid}\n" for qid in ids)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"frozen candidate file differs: {path}")
        return
    _atomic_text(path, content)


def run_extend(args: argparse.Namespace) -> None:
    data_path = Path(args.mathverse_json)
    exclude_path = Path(args.exclude_rollouts)
    existing_path = Path(args.existing_candidate_ids)
    output_dir = Path(args.output_dir)
    excluded = {
        str(row.get("question_id", ""))
        for row in load_jsonl(exclude_path)
        if str(row.get("question_id", ""))
    }
    existing = read_ids(existing_path)
    combined, extension = freeze_candidate_extension(
        load_json(data_path),
        excluded_question_ids=excluded,
        existing_candidate_ids=existing,
        extension_count=args.extension_count,
        seed=args.seed,
    )
    if args.batch_size <= 0 or args.extension_count % args.batch_size:
        raise ValueError("extension_count must be divisible by batch_size")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"candidate_ids_all{len(combined)}.txt"
    extension_path = output_dir / f"candidate_ids_extension{len(extension)}.txt"
    write_frozen_ids(combined_path, combined)
    write_frozen_ids(extension_path, extension)
    batch_files = []
    for batch_index, start in enumerate(range(0, len(extension), args.batch_size), start=1):
        path = output_dir / (
            f"candidate_ids_extension{args.batch_size}_{batch_index:02d}.txt"
        )
        write_frozen_ids(path, extension[start : start + args.batch_size])
        batch_files.append({"name": path.name, "sha256": sha256_file(path)})
    audit = {
        "mathverse_json": str(data_path),
        "mathverse_sha256": sha256_file(data_path),
        "exclude_rollouts": str(exclude_path),
        "exclude_sha256": sha256_file(exclude_path),
        "existing_candidate_ids": str(existing_path),
        "existing_candidate_sha256": sha256_file(existing_path),
        "existing_count": len(existing),
        "extension_count": len(extension),
        "combined_count": len(combined),
        "extension_rank_start_one_based": len(existing) + 1,
        "extension_rank_end_one_based": len(combined),
        "combined_sha256": sha256_file(combined_path),
        "extension_sha256": sha256_file(extension_path),
        "batch_size": args.batch_size,
        "batch_files": batch_files,
        "seed": args.seed,
        "entropy_used_for_selection": False,
    }
    audit_path = output_dir / "candidate_extension_audit.json"
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    if audit_path.exists() and audit_path.read_text(encoding="utf-8") != audit_text:
        raise ValueError(f"frozen extension audit differs: {audit_path}")
    if not audit_path.exists():
        _atomic_text(audit_path, audit_text)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
```

Add this parser block after the `freeze` parser:

```python
    extend = subparsers.add_parser("extend", help="Freeze a prefix-validated candidate extension.")
    extend.add_argument("--mathverse-json", required=True)
    extend.add_argument("--exclude-rollouts", required=True)
    extend.add_argument("--existing-candidate-ids", required=True)
    extend.add_argument("--output-dir", required=True)
    extend.add_argument("--extension-count", type=int, default=500)
    extend.add_argument("--batch-size", type=int, default=50)
    extend.add_argument("--seed", type=int, default=20260725)
```

Dispatch without changing the existing commands:

```python
def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        run_freeze(args)
    elif args.command == "extend":
        run_extend(args)
    else:
        run_select(args)
```

- [ ] **Step 8: Run all preparation tests and commit**

Run: `python -m pytest Experiment/tests/test_prepare_entropy_band_confirm120.py -q`

Expected: all tests pass.

```bash
git add Experiment/scripts/prepare_entropy_band_confirm120.py Experiment/tests/test_prepare_entropy_band_confirm120.py
git commit -m "feat: freeze entropy candidate extensions"
```

### Task 2: First-50 Concurrent Tmux Launcher

**Files:**
- Create: `Experiment/scripts/launch_entropy_band_extension50_tmux.sh`
- Modify: `Experiment/tests/test_prepare_entropy_band_confirm120.py`

**Interfaces:**
- Consumes: existing confirm120 run directory and `prepare_entropy_band_confirm120.py extend`.
- Produces: a unique tmux session, append-only raw rollouts for ranks 801-850, rebuilt labeled rollouts and partial eligibility audit.
- Must not call `extract_entropy_band_qwen3vl.py` or `analyze_entropy_band_confirm120.py`.

- [ ] **Step 1: Add a failing launcher contract test**

Append to `Experiment/tests/test_prepare_entropy_band_confirm120.py`:

```python
def test_extension50_launcher_is_bounded_and_does_not_extract_entropy() -> None:
    script = (SCRIPTS / "launch_entropy_band_extension50_tmux.sh").read_text(
        encoding="utf-8"
    )
    assert "--extension-count 500" in script
    assert "--batch-size 50" in script
    assert "candidate_ids_extension50_01.txt" in script
    assert "a6be58071e4cba09297c7d2446d0fc632958f5759e7c0c8ca491d02e8da4eb80" in script
    assert 'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"' in script
    assert "--rollouts 8" in script
    assert "--max-tokens 16384" in script
    assert "--resume" in script
    assert "extract_entropy_band_qwen3vl.py" not in script
    assert "analyze_entropy_band_confirm120.py" not in script
```

- [ ] **Step 2: Run the launcher contract test to verify failure**

Run: `python -m pytest Experiment/tests/test_prepare_entropy_band_confirm120.py::test_extension50_launcher_is_bounded_and_does_not_extract_entropy -q`

Expected: fails with `FileNotFoundError` because the launcher does not exist.

- [ ] **Step 3: Create the bounded launcher**

Create `Experiment/scripts/launch_entropy_band_extension50_tmux.sh` with:

```bash
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
  if ! kill -0 "${PEER_PID}" 2>/dev/null; then
    echo "peer PID is no longer alive before launch: ${PEER_PID}" >&2
    exit 3
  fi
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
import os
import sys
from pathlib import Path

raw_path, batch_path, status_path = map(Path, sys.argv[1:4])
peer_pid = int(sys.argv[4])
batch = {line.strip() for line in batch_path.read_text(encoding="utf-8").splitlines() if line.strip()}
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
incomplete = {qid: sorted(counts[qid]) for qid in batch if counts[qid] != set(range(8))}
if incomplete:
    raise SystemExit(f"extension batch incomplete: {incomplete}")
try:
    os.kill(peer_pid, 0)
except OSError as exc:
    raise SystemExit(f"peer PID disappeared: {peer_pid}: {exc}")
status = json.loads(status_path.read_text(encoding="utf-8"))
print(json.dumps({
    "status": "EXTENSION50_COMPLETE",
    "batch_questions": len(batch),
    "batch_rollouts": sum(len(values) for values in counts.values()),
    "eligible_questions": status["selected_questions"],
    "strict_3plus3_questions": status["strict_3plus3_questions"],
    "formal_ready": status["ready"],
}, indent=2))
PY

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
```

- [ ] **Step 4: Run the launcher and regression tests**

Run: `python -m pytest Experiment/tests/test_prepare_entropy_band_confirm120.py Experiment/tests/test_roll_mathverse_vllm_resume.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the bounded launcher**

```bash
git add Experiment/scripts/launch_entropy_band_extension50_tmux.sh Experiment/tests/test_prepare_entropy_band_confirm120.py
git commit -m "exp: add entropy extension first50 launcher"
```

### Task 3: H200 Synchronization And Ten-Minute Observation

**Files:**
- Copy to H200: `Experiment/scripts/prepare_entropy_band_confirm120.py`
- Copy to H200: `Experiment/scripts/launch_entropy_band_extension50_tmux.sh`
- Create on H200: `experiment0_hidden_dynamics/entropy_band_confirm120_20260723/candidate_extension_20260724/`
- Append on H200: `data_long/entropy_band_confirm120_20260723/rollouts_mt16384.jsonl`

**Interfaces:**
- Consumes: tested Task 1-2 scripts and the current H200 process state.
- Produces: tmux session `entropy_band_extension50_20260724` and a monitored first-50 generation job.

- [ ] **Step 1: Run the full local regression before synchronization**

Run: `python -m pytest Experiment/tests/test_prepare_entropy_band_confirm120.py Experiment/tests/test_roll_mathverse_vllm_resume.py Experiment/tests/test_analyze_entropy_band_confirm120.py Experiment/tests/test_entropy_band_metrics.py -q`

Expected: all tests pass.

- [ ] **Step 2: Synchronize only the two required scripts**

```bash
scp Experiment/scripts/prepare_entropy_band_confirm120.py h200:/data2/hjk/projects/AI-HiddenState-ER/scripts/
scp Experiment/scripts/launch_entropy_band_extension50_tmux.sh h200:/data2/hjk/projects/AI-HiddenState-ER/scripts/
```

Expected: both transfers succeed; no data file is copied or replaced.

- [ ] **Step 3: Verify the peer kernel and absence of the new session**

Run: `ssh h200 "ps -o user=,pid=,etime=,cmd= -p 696866"`

Expected: PID `696866`, user `hao.lin`, `ipykernel_launcher`.

Run: `ssh h200 "tmux has-session -t entropy_band_extension50_20260724"`

Expected: exit 1, because the new session does not yet exist.

Run: `ssh h200 "nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader"`

Expected before launch: approximately 18.5 GiB used and 0% utilization. If another new compute process appears, stop before launch and report it.

- [ ] **Step 4: Launch the first-50 job**

Run:

```bash
ssh h200 "cd /data2/hjk/projects/AI-HiddenState-ER && SESSION=entropy_band_extension50_20260724 BASE_RUN_NAME=entropy_band_confirm120_20260723 GPU_MEMORY_UTILIZATION=0.65 PEER_PID=696866 bash scripts/launch_entropy_band_extension50_tmux.sh"
```

Expected: prints `launched tmux=entropy_band_extension50_20260724` and returns immediately.

- [ ] **Step 5: Perform the launch observation**

Run: `ssh h200 "tmux capture-pane -pt entropy_band_extension50_20260724:0 -S -120"`

Expected: prefix validation succeeds, extension audit reports existing 800 + extension 500, and vLLM initialization begins.

Run: `ssh h200 "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"`

Expected: PID `696866` still exists plus the new vLLM process; total used memory remains below 135 GiB.

- [ ] **Step 6: Perform the five-minute observation**

At approximately five minutes after launch, rerun:

```bash
ssh h200 "tmux capture-pane -pt entropy_band_extension50_20260724:0 -S -120"
ssh h200 "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"
ssh h200 "ps -o user=,pid=,stat=,etime=,cmd= -p 696866"
```

Expected: no `CUDA out of memory`, no engine initialization failure, the log advances, memory remains below 135 GiB, and PID `696866` remains alive.

- [ ] **Step 7: Perform the ten-minute observation and duplicate audit**

At approximately ten minutes after launch, run:

```bash
ssh h200 "tmux capture-pane -pt entropy_band_extension50_20260724:0 -S -160"
ssh h200 "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"
ssh h200 "stat -c '%y %s %n' /data2/hjk/projects/AI-HiddenState-ER/data_long/entropy_band_confirm120_20260723/rollouts_mt16384.jsonl"
```

Then run this read-only duplicate check:

```bash
ssh h200 "/data2/hjk/envs/hs_er/bin/python -c \"import json,collections,pathlib; p=pathlib.Path('/data2/hjk/projects/AI-HiddenState-ER/data_long/entropy_band_confirm120_20260723/rollouts_mt16384.jsonl'); keys=[]; [keys.append((str(r.get('question_id')),int(r.get('rollout_id',-1)))) for r in map(json.loads,p.open())]; duplicates=[k for k,v in collections.Counter(keys).items() if v>1]; print({'rows':len(keys),'duplicates':len(duplicates)}); assert not duplicates\""
```

Expected: output file size or mtime advances, duplicate count is zero, no OOM appears, total memory is below 135 GiB, and the peer kernel remains alive.

- [ ] **Step 8: Apply the safety stop only if an invariant fails**

Do not run this step when the observations pass. If vLLM OOMs, initialization repeatedly fails, memory reaches 135 GiB, or PID `696866` disappears, run:

```bash
ssh h200 "tmux kill-session -t entropy_band_extension50_20260724"
ssh h200 "ps -o user=,pid=,stat=,etime=,cmd= -p 696866"
```

Expected: only the new session is stopped; PID `696866` is not signaled by this procedure.

- [ ] **Step 9: Leave the stable first batch running and report status**

When all observations pass, do not launch extension batches 02-10. Report the current completed-question count, elapsed time, memory usage, estimated first-batch completion time, and the exact tmux/log names:

```text
tmux: entropy_band_extension50_20260724
log:  /data2/hjk/projects/AI-HiddenState-ER/logs/entropy_band_confirm120_20260723_extension50_20260724.log
```

### Task 4: First-Batch Completion Audit

**Files:**
- Read on H200: `experiment0_hidden_dynamics/entropy_band_confirm120_20260723/manifest/eligibility_status.json`
- Read on H200: `experiment0_hidden_dynamics/entropy_band_confirm120_20260723/candidate_extension_20260724/candidate_extension_audit.json`
- Read on H200: `data_long/entropy_band_confirm120_20260723/label_audit/long_smoke_summary.json`

**Interfaces:**
- Consumes: completed Task 3 tmux job.
- Produces: a go/no-go report for later extension batches; no automatic next launch.

- [ ] **Step 1: Detect completion without modifying the session**

Run:

```bash
ssh h200 "grep -F 'EXTENSION50_PIPELINE_COMPLETE' /data2/hjk/projects/AI-HiddenState-ER/logs/entropy_band_confirm120_20260723_extension50_20260724.log"
```

Expected after completion: one matching line. If absent and the tmux session still exists, the batch is still running.

- [ ] **Step 2: Read the post-batch eligibility counts**

Run:

```bash
ssh h200 "cat /data2/hjk/projects/AI-HiddenState-ER/experiment0_hidden_dynamics/entropy_band_confirm120_20260723/manifest/eligibility_status.json"
```

Expected: `selected_questions` is at least 84, `entropy_used_for_selection=false`, and the status records whether 120 has been reached.

- [ ] **Step 3: Verify exactly 50 new questions and 400 rollout keys**

Use the same inline audit embedded in the launcher against `candidate_ids_extension50_01.txt`. Expected: 50 questions, each with rollout IDs 0-7, and no duplicate keys in the complete raw JSONL.

- [ ] **Step 4: Report the next gate without launching it**

If `ready=false`, report the remaining deficit and estimate how many additional frozen 50-question batches are needed from the observed cumulative eligibility rate. If `ready=true`, report that hidden extraction may be scheduled next, but do not run it under this plan.
