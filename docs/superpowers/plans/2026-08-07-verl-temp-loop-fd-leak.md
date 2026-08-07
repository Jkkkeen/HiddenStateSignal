# veRL Temporary Event-Loop FD Leak Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deterministically close veRL temporary asyncio loops and safely resume the Experiment 2E formal GRPO run from checkpoint 2232.

**Architecture:** An idempotent utility performs one exact source rewrite in the frozen veRL checkout. A Linux smoke exercises the real `_run_async_in_temp_loop` bridge and checks `/proc/self/fd`; the formal launcher reapplies/verifies the patch before every start.

**Tech Stack:** Python 3.11, asyncio, veRL 0.9.0.dev0, TransferQueue 0.1.8, PyZMQ 27.1.0, Bash, tmux, H200.

## Global Constraints

- Do not upgrade the frozen environment.
- Do not discard checkpoint 2232 or existing logs.
- Do not start formal training until the FD regression smoke passes.
- Do not stage or commit unrelated dirty-worktree files.

---

### Task 1: Idempotent veRL patch

**Files:**
- Create: `Experiment_2E/scripts/patch_verl_temp_loop.py`
- Test: `Experiment_2E/tests/test_patch_verl_temp_loop.py`

- [x] Test first apply, repeat apply, unknown source, backup mismatch, and exact restore.
- [x] Replace the async stop coroutine with thread-safe stop, join, and explicit loop close.
- [x] Record target, backup, and before/after SHA256 in JSON.

### Task 2: Linux FD regression

**Files:**
- Create: `Experiment_2E/scripts/smoke_verl_temp_loop_fd.py`

- [x] Reproduce unpatched FD growth without forced garbage collection.
- [x] Apply the patch to `/data2/hjk/projects/verl`.
- [x] Complete 5000 bridge calls with FD delta zero.

### Task 3: Formal recovery

**Files:**
- Modify: `Experiment_2E/scripts/run_grpo_formal.sh`

- [x] Verify manifest SHA and complete checkpoint 2232.
- [x] Start `exp2e_grpo_formal` with `RLIMIT_NOFILE=65535`.
- [x] Confirm new steps 2233 through 2240 and stable WorkerDict FD count 154.
- [x] Leave the formal training process running in tmux.
