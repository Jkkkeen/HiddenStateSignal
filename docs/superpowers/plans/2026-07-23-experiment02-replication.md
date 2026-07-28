# Experiment 02 Replication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute this plan task-by-task with tests and review at each gate.

**Goal:** Run the frozen 96-question Qwen3-VL-8B-Thinking replication and exploratory hidden-state geometry experiment defined in Experiment/2dimension02.md.

**Architecture:** A deterministic manifest builder freezes the new cohort after excluding FormalDiscovery24. A bounded-memory extractor replays existing responses, computes token entropy/activity and span movement/path metrics online, and writes resumable per-question shards. A separate analyzer combines shards, evaluates the two frozen confirmatory cells, produces exploratory entropy/path outputs, and writes figures and metadata.

**Tech Stack:** Python 3, PyTorch, Transformers, NumPy, pandas/Parquet, scikit-learn, matplotlib, tmux, one H200 GPU.

## Global Constraints

- Reuse existing non-truncated responses with a reliable closing think tag; no generation.
- Main cohort: 96 new questions with at least 2 correct and 2 wrong rollouts.
- Exclude all 24 FormalDiscovery24 questions.
- Confirmatory cells remain mean_w128_s64/L24/bin5/median movement and token/L15/bin9/P90 activity.
- Save full progress curves; do not rescan cells for the main verdict.
- H_act-raw, H_act-z, path length, net displacement, straightness, and log-detour remain exploratory.
- Per-question shards and completion markers must support resume.

---

### Task 1: Pure Experiment 02 Metrics

**Files:**
- Create: Experiment/scripts/experiment02_metrics.py
- Create: Experiment/tests/test_experiment02_metrics.py

**Interfaces:**
- Produces coordinate_energy_entropy(values, center_coordinates).
- Produces trajectory_z_entropy(hidden, sigma_scale, clip).
- Produces path_integrals(displacements, progress_bins).

- [ ] Write tests for raw coordinate centering, rogue-dimension trajectory standardization, sigma-floor behavior, and straight/detoured paths.
- [ ] Run python -m pytest Experiment/tests/test_experiment02_metrics.py -q and verify the tests fail before implementation.
- [ ] Implement vectorized NumPy reference functions with finite-value validation and fixed epsilon=1e-12.
- [ ] Re-run the focused tests and verify all pass.

### Task 2: Frozen Cohort Builder

**Files:**
- Create: Experiment/scripts/prepare_experiment02_replication.py
- Create: Experiment/tests/test_prepare_experiment02_replication.py

**Interfaces:**
- Consumes the labeled rollout JSONL and the FormalDiscovery24 manifest.
- Produces manifest_replication96.jsonl, manifest_smoke4.jsonl, question_splits.csv, and ELIGIBILITY_AUDIT.json.

- [ ] Write tests showing excluded question IDs never enter either manifest and every selected question satisfies 2+2.
- [ ] Run the focused tests and verify failure before implementation.
- [ ] Implement deterministic length-stratified selection with seed 20260724, selecting 96 of the 101 eligible new questions.
- [ ] Record the strict 3+3 subset in the audit without excluding 2+2 questions from the main cohort.
- [ ] Run tests and inspect manifest counts.

### Task 3: Resumable Hidden-State Extractor

**Files:**
- Create: Experiment/scripts/extract_experiment02_qwen3vl.py

**Interfaces:**
- Consumes a frozen manifest.
- Produces per-question progress_features, entropy_features, path_features, optional float16 path_vectors, completion markers, status JSONL, and an extraction summary.

- [ ] Reuse the existing Qwen3-VL prompt/full-response encoding and think-boundary helpers.
- [ ] Compute H_act-raw, H_act-z, update entropy, L15 vertical activity, L24 movement, and L24/L36 path metrics in the same model forward.
- [ ] Use sigma_min = 1e-4 times median sigma and clip z-scores to [-8, 8]; do not re-center z-scores per token.
- [ ] Aggregate every metric into the fixed ten relative-progress bins and retain whole-trajectory path rows.
- [ ] Atomically write shards before marking a question complete; validate all required columns when resuming.
- [ ] Run py_compile and a CPU synthetic reducer check.

### Task 4: Confirmatory and Exploratory Analysis

**Files:**
- Create: Experiment/scripts/analyze_experiment02.py

**Interfaces:**
- Consumes extraction shards and the frozen audit.
- Produces combined Parquet/CSV files, bootstrap summaries, OOF length comparisons, plots, and EXPERIMENT02_RESULTS.md.

- [ ] Compute question-equal within-question AUC, sign fraction, and 4,000-question bootstrap CI for the two frozen cells.
- [ ] Report the 96-question cohort and strict 3+3 sensitivity subset separately.
- [ ] Produce full bin 0-9 movement/activity curves and rollout spaghetti plots.
- [ ] Produce raw/z/update entropy layer-by-progress results and whole/block path results.
- [ ] Run py_compile and analyze a synthetic shard directory.

### Task 5: Smoke, Deploy, and Launch

**Files:**
- Create: Experiment/scripts/launch_experiment02_tmux.sh

**Interfaces:**
- Runs prepare, smoke extraction/analysis, then the resumable formal extraction/analysis.
- Uses tmux session two_dim02_replication96 and run directory experiment0_hidden_dynamics/experiment02_replication96_20260723.

- [ ] Sync only the new Experiment 02 scripts and approved plan to H200.
- [ ] Generate the audit and verify 96 new questions, no overlap with FormalDiscovery24, and at least 2+2 labels per question.
- [ ] Run the four-question smoke and verify all expected shards, finite metrics, and completion markers.
- [ ] Check current GPU memory/utilization without terminating other users' processes.
- [ ] Launch the formal pipeline in tmux with resume, capture its log path, and verify at least one rollout completes without error.
- [ ] Report tmux attach/detach commands and runtime estimate based on smoke seconds per rollout.
