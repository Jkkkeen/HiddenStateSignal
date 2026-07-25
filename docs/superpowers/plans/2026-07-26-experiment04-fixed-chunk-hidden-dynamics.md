# Experiment 04 Fixed-Chunk Hidden Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the approved 2Dimension-04 fixed-256-token hidden-dynamics discovery on H200 in a resumable tmux session while keeping the held-out confirm questions blind.

**Architecture:** Freeze a question-level discovery/confirm split from the existing labeled long-response rollouts. Replay only discovery rollouts through Qwen3-VL-8B-Thinking, cache bounded fp16 chunk representations, reduce them in two passes into scalar horizontal/vertical metrics, then generate discovery curves and atlases. Confirm extraction is a separate gated command that cannot run until a frozen hypothesis manifest exists.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, NumPy, pandas/pyarrow, SciPy, scikit-learn, Matplotlib, pytest, Bash, tmux, H200.

## Global Constraints

- Source specification: `Experiment/2dimension04.md` at commit `df61d04`.
- Input model: `Qwen/Qwen3-VL-8B-Thinking`.
- Input rollouts: MathVerse testmini, 8 rollouts/question, `max_tokens=16384`.
- Analyze response start through the complete `</think>` segment only.
- Use non-overlapping 256-token chunks; full chunks are primary and terminal partial chunks are sensitivity-only.
- Hidden-state indexes are `0..36`; discovery scans all indexes and three representations: `last`, `last25`, `mean`.
- Eligibility is at least 2 correct and 2 wrong rollouts per question.
- Split unit is question, seed `20260726`; confirm rows must not be forwarded or analyzed before candidate freeze.
- H200 execution is single-GPU, batch size 1, question-resumable, isolated, and tmux-managed.
- Do not modify or stop unrelated H200 sessions or processes.

---

### Task 1: Pure Metric Kernel

**Files:**
- Create: `Experiment/scripts/experiment04_metrics.py`
- Test: `Experiment/tests/test_experiment04_metrics.py`

**Interfaces:**
- Produces: `coordinate_entropy`, `chunk_representations`, `path_metrics`, `horizontal_records`, and `vertical_records`.
- Consumes: NumPy arrays only; no model or filesystem dependency.

- [ ] **Step 1: Write failing synthetic-array tests**

Cover exact chunk boundaries, `last/last25/mean`, raw and demean state angle, step magnitude, turn cosine, turn-angle standard deviation, path/net/straightness/detour, raw/difference entropy, rolling-4 paths, early/late invalidation below eight values, and vertical scale normalization/near-zero exclusion.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `pytest -q Experiment/tests/test_experiment04_metrics.py`

Expected: import failure because `experiment04_metrics.py` does not yet exist.

- [ ] **Step 3: Implement the metric API**

Use float32 for dot products, norms, arccos, and entropy; clip cosine to `[-1, 1]`; return `NaN` for invalid directions. Use:

Required public signatures are `coordinate_entropy(x: np.ndarray, eps: float = 1e-12) -> np.ndarray`, `chunk_representations(hidden: np.ndarray, chunk_size: int = 256) -> dict[str, np.ndarray]`, `path_metrics(points: np.ndarray, rolling: int = 4) -> dict[str, np.ndarray | float]`, `horizontal_records(vectors: np.ndarray, common: np.ndarray | None) -> pd.DataFrame`, and `vertical_records(vectors: np.ndarray, common: np.ndarray | None, scale: np.ndarray) -> pd.DataFrame`. The module constant is `REPRESENTATIONS = ("last", "last25", "mean")`.

- [ ] **Step 4: Run tests**

Run: `pytest -q Experiment/tests/test_experiment04_metrics.py`

Expected: all tests pass.

### Task 2: Freeze Cohort and Blind Split

**Files:**
- Create: `Experiment/scripts/prepare_experiment04.py`
- Test: `Experiment/tests/test_prepare_experiment04.py`

**Interfaces:**
- Consumes: labeled rollout JSONL.
- Produces: `manifest_discovery.jsonl`, `manifest_confirm.jsonl`, `manifest_smoke.jsonl`, `split_manifest.json`, and SHA256 audit fields.

- [ ] **Step 1: Test deterministic eligibility and splitting**

Use synthetic questions with 1+3, 2+2, and 3+3 label counts. Assert only 2+2/3+3 questions are eligible, no question overlaps, ordering is SHA256 of `20260726:<question_id>`, and smoke selects one discovery question with all eight rollouts.

- [ ] **Step 2: Run the test and verify failure**

Run: `pytest -q Experiment/tests/test_prepare_experiment04.py`

- [ ] **Step 3: Implement preparation and audits**

The CLI must be:

```text
python scripts/prepare_experiment04.py \
  --input data_long/entropy_band_confirm120_20260723/rollouts_mt16384_labeled.jsonl \
  --output-dir experiment04_fixed256_20260726/manifest \
  --seed 20260726
```

Write actual question/rollout counts, strict 3+3 counts, source SHA256, ordered question IDs, and disjointness assertions before any hidden forward.

- [ ] **Step 4: Run tests**

Run: `pytest -q Experiment/tests/test_prepare_experiment04.py`

Expected: all tests pass.

### Task 3: Resumable Qwen3-VL Extraction

**Files:**
- Create: `Experiment/scripts/extract_experiment04_qwen3vl.py`
- Test: `Experiment/tests/test_extract_experiment04_qwen3vl.py`

**Interfaces:**
- Consumes: one frozen manifest JSONL and Qwen3-VL model outputs.
- Produces: one fp16 vector cache per rollout, token-first entropy parquet per question, `progress.json`, and resource telemetry.

- [ ] **Step 1: Test reduction without loading a model**

Construct a tuple of 37 synthetic hidden tensors and assert:

```text
full_vectors shape = (n_full_chunks, 3, 37, hidden_dim)
partial_vectors shape = (0 or 1, 3, 37, hidden_dim)
token entropy rows contain raw/time-diff/layer-diff mean, median, P90, top25mean
```

Also test that a completed rollout cache is skipped under `--resume` and a corrupt cache is recomputed.

- [ ] **Step 2: Run the test and verify failure**

Run: `pytest -q Experiment/tests/test_extract_experiment04_qwen3vl.py`

- [ ] **Step 3: Implement extraction**

Reuse the prompt/message and `</think>` token-span helpers from `run_long_path_smoke_qwen3vl.py`. Run with `torch.inference_mode()`, batch size 1, bf16 model weights, and float32 metric reductions. Save each rollout atomically as `cache/<question_stem>/rollout_<id>.npz`; never save full token hidden states.

Required CLI:

```text
python scripts/extract_experiment04_qwen3vl.py \
  --input <frozen_manifest.jsonl> \
  --output-dir <run>/extraction \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --chunk-size 256 --resume --local-files-only
```

Log elapsed seconds, peak allocated/reserved GPU memory, CPU RSS, response tokens, full chunks, partial tokens, NaN/Inf counts, and skip reason per rollout.

- [ ] **Step 4: Run extraction tests**

Run: `pytest -q Experiment/tests/test_extract_experiment04_qwen3vl.py`

Expected: all tests pass without a GPU.

### Task 4: Two-Pass Scalar Reduction

**Files:**
- Create: `Experiment/scripts/reduce_experiment04.py`
- Test: `Experiment/tests/test_reduce_experiment04.py`

**Interfaces:**
- Consumes: discovery rollout vector caches and token-first parquet files.
- Produces: `calibration.npz`, `horizontal_metrics.parquet`, `vertical_metrics.parquet`, `rollout_summary.parquet`, and `reduction_audit.json`.

- [ ] **Step 1: Test discovery calibration**

Assert that common vectors are rollout-equal means, vertical scales are median-of-rollout-medians, thresholds are `max(1e-8, 1e-4*q)`, and confirm mode rejects calibration generated from confirm data.

- [ ] **Step 2: Test scalar schemas and partial rules**

Assert full chunks enter horizontal movement/path metrics, terminal partial chunks enter only raw entropy/vertical sensitivity rows, all 37 states and 36 updates are present, turning means are excluded as candidates, and only `turn_angle_std/late_std` are emitted.

- [ ] **Step 3: Implement pass 1 and pass 2**

Pass 1 computes `mu_disc[3,37,D]`, vertical scale `[3,36]`, and direction thresholds. Pass 2 streams one rollout cache at a time through `experiment04_metrics.py`, attaches question/rollout labels and coverage, and writes parquet atomically.

- [ ] **Step 4: Run tests**

Run: `pytest -q Experiment/tests/test_reduce_experiment04.py`

Expected: all tests pass.

### Task 5: Discovery Statistics and Figures

**Files:**
- Create: `Experiment/scripts/analyze_experiment04_discovery.py`
- Test: `Experiment/tests/test_analyze_experiment04_discovery.py`

**Interfaces:**
- Consumes: scalar parquets and split manifest.
- Produces: question-equal curves, layer×chunk effect/AUC/coverage atlases, cluster-permutation results, discovery report, and candidate worksheet; never reads confirm scalars.

- [ ] **Step 1: Test question-equal aggregation and within-Q AUC**

Use unequal rollout counts to prove questions, not rollouts, receive equal weight. Test deterministic question bootstrap and within-question label permutation.

- [ ] **Step 2: Test contiguous cluster mass**

Use a synthetic layer×chunk grid with one known connected positive cluster and isolated cells. Assert 4-neighbor connectivity, `|t|>=2.0`, maximum-cluster null selection, and fixed 4000-permutation seed behavior.

- [ ] **Step 3: Implement plots and reports**

Generate:

```text
horizontal_absolute_<metric>.png
horizontal_end_aligned_<metric>.png
horizontal_relative_progress_<metric>.png
vertical_profiles.pdf
vertical_profiles.html
atlas_effect_<family>.png
atlas_auc_<family>.png
atlas_coverage_<family>.png
DISCOVERY_REPORT.md
candidate_worksheet.csv
```

Every correct/wrong curve uses question-equal aggregation and question-bootstrap 95% CI. Low-coverage cells below `max(20, ceil(0.5*n_discovery_questions))` are masked from candidate regions.

- [ ] **Step 4: Run tests**

Run: `pytest -q Experiment/tests/test_analyze_experiment04_discovery.py`

Expected: all tests pass.

### Task 6: H200 Launcher and Blind Confirm Guard

**Files:**
- Create: `Experiment/scripts/launch_experiment04_tmux.sh`
- Test: `Experiment/tests/test_launch_experiment04.py`

**Interfaces:**
- Produces: `--prepare`, `--smoke`, `--discovery`, and guarded `--confirm` modes.
- `--confirm` requires `frozen_hypotheses.json` with source split SHA256 and refuses a missing or mismatched file.

- [ ] **Step 1: Test launcher text and confirm guard**

Assert tmux sessions are isolated, existing session names fail, discovery only references `manifest_discovery.jsonl`, and confirm cannot run without a valid frozen manifest.

- [ ] **Step 2: Implement launcher**

Defaults:

```text
ROOT=/data2/hjk/projects/AI-HiddenState-ER
RUN_NAME=experiment04_fixed256_20260726
SMOKE_SESSION=exp04_fixed256_smoke
DISCOVERY_SESSION=exp04_fixed256_discovery
MODEL=Qwen/Qwen3-VL-8B-Thinking
BOOTSTRAP=4000
PERMUTATIONS=4000
```

The discovery tmux command chains extraction, reduction, analysis, and success marker creation with `set -euo pipefail`. It writes a single append-only log and question-level progress files.

- [ ] **Step 3: Run all local tests**

Run:

```text
pytest -q \
  Experiment/tests/test_experiment04_metrics.py \
  Experiment/tests/test_prepare_experiment04.py \
  Experiment/tests/test_extract_experiment04_qwen3vl.py \
  Experiment/tests/test_reduce_experiment04.py \
  Experiment/tests/test_analyze_experiment04_discovery.py \
  Experiment/tests/test_launch_experiment04.py
```

Expected: all tests pass.

### Task 7: Deploy, Smoke, and Start Formal Discovery

**Files:**
- Deploy local Experiment 04 scripts/tests to `/data2/hjk/projects/AI-HiddenState-ER/scripts` and `/data2/hjk/projects/AI-HiddenState-ER/tests`.
- Create remote run directory: `/data2/hjk/projects/AI-HiddenState-ER/experiment04_fixed256_20260726`.

- [ ] **Step 1: Re-audit H200 before launch**

Run `nvidia-smi`, `tmux ls`, process listing, and `df -h /data2`. Do not stop unrelated sessions.

- [ ] **Step 2: Prepare and freeze manifests**

Run launcher `--prepare`; inspect only eligibility/count/hash fields. Confirm no discovery/confirm question overlap and record exact counts.

- [ ] **Step 3: Launch and monitor smoke**

Launch `1 question × 8 rollouts` in `exp04_fixed256_smoke`. Wait for completion, validate resource telemetry, scalar schemas, no OOM, at least 15% GPU headroom, and numerical audits.

- [ ] **Step 4: Start formal discovery tmux**

Only after smoke passes, launch `exp04_fixed256_discovery`. Verify the tmux session exists, the Python extractor is running, GPU memory/utilization are nonzero, log progress advances, and the run points only to the discovery manifest.

- [ ] **Step 5: Record launch handoff**

Report tmux name, log path, run directory, exact discovery/confirm question counts, measured smoke time per rollout, estimated formal discovery duration, and safe attach/detach commands. Do not run confirm.
