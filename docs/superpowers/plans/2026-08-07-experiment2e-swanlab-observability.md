# Experiment 2E SwanLab Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the frozen Experiment 2E GRPO run from checkpoint 3348 with
native SwanLab scalar tracking and a post-extraction hidden-state uploader.

**Architecture:** Use veRL's existing `swanlab` backend for scalar metrics and
keep all hidden-state work outside the active training loop. Credentials stay
in a H200-only file outside Git.

**Tech Stack:** veRL tracking, SwanLab SDK, Bash, Python, pytest.

## Global Constraints

- Preserve `train_batch_size=1`, `rollout.n=8`, manifest hash, and checkpoint 3348.
- Do not put the API key in Git, logs, process arguments, or artifacts.
- Do not run hidden-state forwards concurrently with formal GRPO.

### Task 1: Enable Native Scalar Tracking

**Files:**
- Modify: `Experiment_2E/scripts/run_grpo_formal.sh`
- Test: `Experiment_2E/tests/test_run_grpo_formal.py`

- [ ] Add a guarded `ENABLE_SWANLAB=1` branch that sources only the external
  credential file and validates `SWANLAB_API_KEY`.
- [ ] Change the veRL logger list from `console` to `console,swanlab` only in
  that guarded branch.
- [ ] Test the disabled and enabled command construction without exposing a key.

### Task 2: Add Hidden Result Upload

**Files:**
- Create: `Experiment_2E/experiment_2e/swanlab_upload.py`
- Test: `Experiment_2E/tests/test_swanlab_upload.py`

- [ ] Parse the existing analysis tables and image paths after each fixed
  checkpoint extraction.
- [ ] Log scalar correct/wrong/AUROC/coverage values by checkpoint step and
  upload overview, discriminative, and H1-H8/V1-V8 figures.
- [ ] Test with a fake SwanLab client and fixture table; no network in tests.

### Task 3: Provision and Resume

**Files:**
- Create outside Git: `/data2/hjk/secrets/experiment_2e_swanlab.env`
- Modify remote environment only: `/data2/hjk/envs/verl_qwen3vl_py311`

- [ ] Install the SwanLab SDK in the existing veRL environment.
- [ ] Write the credential file with directory mode 700 and file mode 600.
- [ ] Run an isolated login/log/finish smoke, check the run appears, and remove
  only the smoke log directory.
- [ ] Launch the formal tmux from checkpoint 3348 and verify one scalar arrives.
