# Entropy Band Confirm120 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a preregistered replication of raw activation entropy at L14-L19/bin6 on 120 entirely new eligible MathVerse long-response questions.

**Architecture:** Freeze an entropy-blind ordered acquisition pool, generate Qwen3-VL-8B-Thinking rollout8 responses in resumable batches, select the first 120 questions meeting 2+2 eligibility, replay only those responses, and reduce hidden states online to one frozen score per rollout. A separate analyzer implements the signal and length-increment gates without scanning other layers or bins.

**Tech Stack:** Python 3, PyTorch/Transformers, vLLM, pandas/NumPy, scikit-learn, pytest, Bash/tmux.

## Global Constraints

- Model and sampling are frozen to `Qwen/Qwen3-VL-8B-Thinking`, rollout8, temperature `0.7`, top-p `0.95`, max tokens `16384`.
- Exclude all 500 prior Thinking questions; candidate ordering seed is `20260725`.
- Primary analysis is exactly the raw centered-coordinate entropy mean over layers 14-19 and progress bin 6.
- Primary cohort has exactly 120 questions with at least 2 correct and 2 wrong clean complete rollouts.
- Do not inspect entropy before cohort selection and do not scan alternate layer/bin cells.
- Use 4,000 question bootstraps and question-grouped OOF folds.
- Preserve unrelated dirty-worktree changes and do not commit unless explicitly requested.

---

### Task 1: Freeze Acquisition And Cohort Selection

**Files:**
- Create: `Experiment/scripts/prepare_entropy_band_confirm120.py`
- Test: `Experiment/tests/test_prepare_entropy_band_confirm120.py`

**Interfaces:**
- Consumes MathVerse JSON, prior rollout JSONL, frozen candidate ID files, and labeled new rollout JSONL.
- Produces `candidate_ids_primary600.txt`, four `candidate_ids_reserve50_*.txt`, `candidate_audit.json`, `manifest_confirm120.jsonl`, and `eligibility_status.json`.

- [ ] Write tests showing prior questions and non-A-D answers are excluded, hash ordering is deterministic, and formal selection takes the first 120 eligible questions in frozen order.
- [ ] Run `pytest Experiment/tests/test_prepare_entropy_band_confirm120.py -q` and confirm the tests fail before implementation.
- [ ] Implement candidate freezing, clean-complete 2+2 eligibility, strict 3+3 flags, SHA256 audit fields, and atomic output writes.
- [ ] Run the test again and require all tests to pass.

### Task 2: Extract Only The Frozen Entropy Band

**Files:**
- Create: `Experiment/scripts/entropy_band_metrics.py`
- Create: `Experiment/scripts/extract_entropy_band_qwen3vl.py`
- Test: `Experiment/tests/test_entropy_band_metrics.py`

**Interfaces:**
- Consumes the frozen formal manifest and Qwen3-VL hidden states.
- Produces one parquet shard per question with `raw_entropy_L14` through `raw_entropy_L19`, `raw_entropy_L14_19_bin6`, exact `think_length`, labels, and rollout IDs.

- [ ] Write numerical tests for per-token coordinate centering, normalized entropy, bin6 token indexing, and six-layer arithmetic mean.
- [ ] Run `pytest Experiment/tests/test_entropy_band_metrics.py -q` and confirm failure before implementation.
- [ ] Implement pure NumPy metric functions.
- [ ] Implement the resumable Qwen3-VL replay extractor using existing message/token-boundary helpers; reduce only layers 14-19/bin6 and release each rollout's tensors.
- [ ] Run both new test files and existing Experiment02 metric tests.

### Task 3: Implement The Two Frozen Statistical Gates

**Files:**
- Create: `Experiment/scripts/analyze_entropy_band_confirm120.py`
- Test: `Experiment/tests/test_analyze_entropy_band_confirm120.py`

**Interfaces:**
- Consumes question parquet shards from Task 2.
- Produces question AUCs, OOF scores, gate results, two figures, analysis metadata, and `ENTROPY_BAND_CONFIRM120_RESULTS.md`.

- [ ] Write tests for question-equal pairwise AUC, question bootstrap, no OOF group overlap, and the exact two-gate decision table.
- [ ] Run the test and confirm failure before implementation.
- [ ] Implement Gate 1 as mean within-question AUC with a 4,000-sample question-bootstrap CI.
- [ ] Implement Gate 2 as paired question-bootstrap CI for grouped-OOF `feature+length` minus `length-only` AUC.
- [ ] Add the preregistered strict 3+3 sensitivity output and concise diagnostic plots without alternate cell selection.
- [ ] Run all new tests and compile-check all new scripts.

### Task 4: Orchestrate Smoke And Formal Tmux Run

**Files:**
- Create: `Experiment/scripts/launch_entropy_band_confirm120_tmux.sh`

**Interfaces:**
- Consumes all scripts above and the existing rollout generator/labeler.
- Produces a resumable remote run under `entropy_band_confirm120_20260723` and a single status log.

- [ ] Implement candidate freeze, primary generation, optional reserve top-up, labeling, selection, extraction, analysis, and output validation as explicit pipeline stages.
- [ ] Sync the new files to H200 and run shell/Python syntax checks.
- [ ] Run a one-question generation smoke at vLLM memory utilization `0.75` without touching the other user's process.
- [ ] Run extractor/analyzer smoke on known eligible old rollouts in an isolated smoke directory.
- [ ] Start the formal pipeline in tmux, verify the model loads and the first question produces eight rollout records, then report the tmux name, log path, run path, and conservative `10-18` hour estimate.
