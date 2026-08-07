# Experiment 2E Batch Capacity Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run isolated train-batch 2/4/8 capacity probes from the audited Experiment 2E step-3348 checkpoint and report stable/useful limits without changing the formal run.

**Architecture:** A focused Bash launcher creates a unique scratch root, links the formal checkpoint read-only, starts one-second GPU telemetry, and runs each candidate sequentially with identical frozen GRPO settings. Ray sockets use a separate short, candidate-unique path under `/data2/hjk/cache/ray/e2b` to remain below the Linux AF_UNIX 107-byte limit. A Python summarizer parses completed-step metrics and telemetry into a machine-readable audit and applies the frozen stability/usefulness rules.

**Tech Stack:** Bash, tmux, veRL/Hydra, Ray, vLLM, `nvidia-smi`, Python 3.11, pytest.

## Global Constraints

- Formal checkpoint root `/data2/hjk/checkpoints/experiment_2e/qwen25_7b_math_grpo_formal_seed20260805` is read-only.
- Resume source is `global_step_3348`; formal `latest_checkpointed_iteration.txt` must remain `3348`.
- Candidates are `2`, `4`, and `8`; `rollout.n=8` and all other formal hyperparameters remain unchanged.
- Each candidate completes two warm-up plus five measured steps.
- SwanLab is out of scope and requires a separate user approval.

---

### Task 1: Batch Capacity Runner

**Files:**
- Create: `Experiment_2E/scripts/run_batch_capacity_smoke.sh`
- Test: `Experiment_2E/tests/test_batch_capacity_smoke.py`

**Interfaces:**
- Consumes: `SOURCE_CKPT_ROOT`, `SOURCE_STEP`, `RUN_ROOT`, and the existing formal model/data paths.
- Produces: one candidate log, telemetry CSV, exit-status file, and scratch checkpoint root per batch.

- [ ] **Step 1: Write failing contract tests**

Test that the launcher requires the source checkpoint, refuses a non-empty GPU, creates candidate-isolated paths, uses `TRAIN_BATCH_SIZE` with `ROLLOUT_N=8`, runs exactly seven post-resume steps, and never passes the formal root as `trainer.default_local_dir`.

- [ ] **Step 2: Run the contract tests and verify failure**

Run: `pytest Experiment_2E/tests/test_batch_capacity_smoke.py -q`

Expected: failure because the launcher does not exist.

- [ ] **Step 3: Implement the isolated runner**

The script must execute candidates in order, write `nvidia-smi` samples once per second, abort before a candidate if another compute process owns the GPU, and wait for candidate processes to release the GPU before continuing.

- [ ] **Step 4: Run contract and repository tests**

Run: `pytest Experiment_2E/tests/test_batch_capacity_smoke.py -q`

Expected: all batch-capacity tests pass.

Run: `pytest Experiment_2E/tests -q`

Expected: all Experiment 2E tests pass.

- [ ] **Step 5: Commit the runner**

```bash
git add Experiment_2E/scripts/run_batch_capacity_smoke.sh Experiment_2E/tests/test_batch_capacity_smoke.py
git commit -m "feat: add isolated GRPO batch capacity smoke"
```

### Task 2: Capacity Audit Summarizer

**Files:**
- Create: `Experiment_2E/experiment_2e/batch_capacity.py`
- Create: `Experiment_2E/tests/test_batch_capacity.py`

**Interfaces:**
- Consumes: per-candidate veRL logs, telemetry CSV files, and exit-status files.
- Produces: `batch_capacity_audit.json` with per-batch completed steps, mean measured step time, mean throughput, peak memory, headroom, stability, and stable/useful maxima.

- [ ] **Step 1: Write parser and decision-rule tests**

Cover successful candidates, an OOM candidate, fewer than seven completed steps, the 10% headroom boundary, and a larger stable batch whose throughput regresses.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest Experiment_2E/tests/test_batch_capacity.py -q`

Expected: failure because `experiment_2e.batch_capacity` does not exist.

- [ ] **Step 3: Implement parsing and audit CLI**

Parse `training/global_step`, `timing_s/step`, `perf/throughput`, actor allocated/reserved memory, telemetry memory, and failure signatures. Exclude the first two completed post-resume steps and average the next five.

- [ ] **Step 4: Run focused and repository tests**

Run: `pytest Experiment_2E/tests/test_batch_capacity.py Experiment_2E/tests/test_batch_capacity_smoke.py -q`

Expected: all focused tests pass.

Run: `pytest Experiment_2E/tests -q`

Expected: all Experiment 2E tests pass.

- [ ] **Step 5: Commit the summarizer**

```bash
git add Experiment_2E/experiment_2e/batch_capacity.py Experiment_2E/tests/test_batch_capacity.py
git commit -m "feat: audit GRPO batch capacity smoke"
```

### Task 3: H200 Execution and Recovery Check

**Files:**
- Deploy: `Experiment_2E/scripts/run_batch_capacity_smoke.sh`
- Deploy: `Experiment_2E/experiment_2e/batch_capacity.py`
- Output: `/data2/hjk/results/experiment_2e/batch_capacity_step3348_<timestamp>/`

**Interfaces:**
- Consumes: the audited runner and source checkpoint.
- Produces: a completed tmux sweep and capacity audit while leaving the formal checkpoint pointer unchanged.

- [ ] **Step 1: Upload the two implementation files and validate syntax**

Run remote `bash -n` on the launcher and import the Python module in the formal environment.

- [ ] **Step 2: Launch the sweep in named tmux**

Use session `exp2e_batch_capacity_3348`; save the exact run root before launch.

- [ ] **Step 3: Monitor candidates through completion**

Verify each candidate starts only after the prior GPU process exits. Stop the sweep on an execution failure while retaining its log and telemetry.

- [ ] **Step 4: Build and inspect the audit**

Run the summarizer against the unique run root and inspect all per-batch measurements and failure signatures.

- [ ] **Step 5: Verify formal recovery state**

Confirm no training process is active, GPU memory is released, and formal `latest_checkpointed_iteration.txt` still equals `3348`.
