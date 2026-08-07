# TransferQueue ZMQ FD Leak Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the per-request TransferQueue ZMQ context leak and safely resume the Experiment 2E formal GRPO run from checkpoint 2232.

**Architecture:** An idempotent patch utility performs one exact, auditable source rewrite in the frozen Python environment. A separate Linux regression smoke exercises the real decorated request path and checks process FD counts before the formal trainer is resumed.

**Tech Stack:** Python 3.11, TransferQueue 0.1.8, PyZMQ 27.1.0, veRL 0.9.0.dev0, Bash, tmux, H200.

## Global Constraints

- Do not upgrade the frozen training environment.
- Do not discard checkpoint 2232 or existing logs.
- Do not start formal training until the FD regression smoke passes.
- Do not stage or commit unrelated dirty-worktree files.

---

### Task 1: Idempotent TransferQueue patch

**Files:**
- Create: `Experiment_2E/scripts/patch_transferqueue_zmq.py`
- Test: `Experiment_2E/tests/test_patch_transferqueue_zmq.py`

**Interfaces:**
- Consumes: a Python environment containing the `TransferQueue` distribution.
- Produces: `apply_patch(distribution_root: Path) -> PatchResult` and `restore_patch(distribution_root: Path) -> PatchResult`.

- [ ] Write tests using a temporary fake distribution file for first apply, idempotent reapply, unknown-source refusal, and exact restore.
- [ ] Run `python -m pytest Experiment_2E/tests/test_patch_transferqueue_zmq.py -q`; expect the initial implementation test to fail.
- [ ] Implement exact source replacement, SHA256 audit output, backup, and restore.
- [ ] Re-run the test; expect all cases to pass.

### Task 2: Linux FD regression smoke

**Files:**
- Create: `Experiment_2E/scripts/smoke_transferqueue_zmq_fd.py`

**Interfaces:**
- Consumes: the patched installed `transfer_queue.utils.zmq_utils.with_zmq_socket`.
- Produces: JSON containing call count, baseline/final/peak FD counts, and pass/fail status.

- [ ] Start a local ROUTER endpoint and issue 2000 sequential decorated DEALER requests through the real TransferQueue helper.
- [ ] Sample `/proc/self/fd`, allow a small fixed warm-up delta, and fail on monotonic or unbounded growth.
- [ ] Upload the patch utility and smoke to H200, apply the patch, and run the smoke; expect exit code 0 and bounded FDs.

### Task 3: Resume and observe formal training

**Files:**
- Modify only if required: `Experiment_2E/scripts/run_grpo_formal.sh`

**Interfaces:**
- Consumes: checkpoint 2232, the existing frozen run manifest, and the patched environment.
- Produces: a live `exp2e_grpo_formal` tmux session with advancing steps.

- [ ] Confirm no stale formal trainer/Ray process owns GPUs and checkpoint 2232 is complete.
- [ ] Start the existing resume command in tmux with `RLIMIT_NOFILE=65535` inherited by Ray workers.
- [ ] Observe startup, one or more completed steps, GPU utilization, and repeated WorkerDict FD samples.
- [ ] Record the tmux session, PID, current step, FD samples, and revised ETA without waiting for the full run to finish.
