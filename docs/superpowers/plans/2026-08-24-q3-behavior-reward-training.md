# Qwen3-1.7B Behavior Reward Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Train Qwen3-1.7B Base with the query-relative H2/H10 behavior advantage from `Experiment_final/RLreward_plan02.md`, preserving the existing train/eval cohort and checkpoint schedule while uploading only normal RL metrics to SwanLab.

**Architecture:** During each GRPO update, the actor's old-log-prob forward emits compact final-layer behavior scalars from the complete generated response. The trainer computes H2-conditioned focus/exploration win scores within each rollout group, adds a centered `lambda * A_beh` terminal bonus to the normal math reward, and then computes the existing GRPO advantage. No hidden tensors or H2/H10 plots are persisted; checkpoints and normal validation metrics remain unchanged.

**Tech Stack:** Python, NumPy, PyTorch, veRL DAPO/GRPO, Qwen3-1.7B Base, H200, tmux, SwanLab.

## Global Constraints

- Reuse the frozen Qwen3-1.7B Base train and held-out files selected by the existing calibration decision.
- Keep rollout group size at 8 and compute all behavior comparisons within each `uid` group.
- Do not read correctness labels in behavior scoring; correctness remains only in the existing math reward and offline evaluation.
- Use `lambda=0.2` for the canary/formal candidate, with `lambda=0.1` fallback if the behavior term dominates or destabilizes training.
- Compute hidden behavior scalars online for reward construction, but do not save hidden tensors or publish H2/H10-specific figures.
- Save checkpoints at the same cadence as the prior Qwen3-1.7B run and run the existing held-out validation at each test checkpoint.
- Formal training must run in a named H200 tmux session with a persistent log and audit files.

---

### Task 1: Define and test compact behavior reduction

**Files:**
- Create: `Experiment_2E/experiment_2e/online_behavior.py`
- Test: `Experiment_2E/tests/test_online_behavior.py`

**Interfaces:**
- `reduce_behavior_response(token_hidden, endpoints) -> (values, coverage)` returns `[H2, H2_slope, focus_forward, focus_net, orthogonal]` for one complete response using the final layer and `mean_w128_s32` pooled chunks.
- `reduce_behavior_batch(hidden_states, input_ids, response_mask) -> (values, coverage)` returns a dense `(batch, 5)` array for the actor hook.
- `compute_behavior_advantage(values, coverage, group_ids, lambda_) -> (bonus, metrics)` computes query-relative `R_H`, `W_F`, `W_E`, `U`, centers U within group, and returns a terminal response bonus.

- [ ] Write tests for a straight trajectory (`H2=1`, high focus, low orthogonal), a sideward trajectory (low focus, high orthogonal), and a two-group rank calculation.
- [ ] Run `pytest Experiment_2E/tests/test_online_behavior.py -q` and verify the new tests fail before implementation.
- [ ] Implement the scalar formulas from `RLreward_plan02.md`: `Q=F/(F+B)`, `A=(F+B)/L`, `focus=F/L`, `net=(F-B)/L`, `O=D/L`, and the H2 prefix slope.
- [ ] Implement group-relative wins with `R_H` low favoring focus and `R_H` high favoring exploration; skip incomplete groups and return zero bonus for invalid rows.
- [ ] Run the focused tests and the existing Experiment 2E test suite.

### Task 2: Patch the veRL actor/trainer hook

**Files:**
- Create: `Experiment_2E/scripts/patch_q3_behavior_reward.py`
- Modify at runtime on H200: `/data2/hjk/projects/verl_q3_1p7b_base_20260814/verl/workers/engine/fsdp/transformer_impl.py`
- Modify at runtime on H200: `/data2/hjk/projects/verl_q3_1p7b_base_20260814/verl/trainer/ppo/v1/trainer_base.py`

**Interfaces:**
- The actor forward sets `output_hidden_states=True` only when `EXPERIMENT_2E_BEHAVIOR_REWARD=1`, calls `reduce_behavior_batch`, and returns compact behavior values/coverage through the TransferQueue.
- `_compute_old_log_prob` requests and writes the compact fields without retaining hidden tensors.
- `_compute_advantage` reads the compact fields, calls `compute_behavior_advantage`, adds `lambda * bonus` to the terminal reward, and logs `behavior/*` scale/coverage metrics.

- [ ] Add an idempotent, source-hash-audited patch with backups and an explicit marker.
- [ ] Make the patch fail closed if either known veRL anchor is missing or duplicated.
- [ ] Add a dry-run mode and run it against a clean remote veRL tree before mutating the tree.
- [ ] Apply the patch on H200 and write a patch audit containing before/after hashes and backup paths.

### Task 3: Add the behavior-reward Qwen3 runner

**Files:**
- Create: `Experiment_2E/scripts/run_q3_1p7b_base_behavior_grpo.sh`
- Create: `Experiment_2E/scripts/launch_q3_behavior_grpo_tmux.sh`

**Interfaces:**
- Runner consumes the existing calibration decision, train file, held-out file, max response length, seed, and prior batch/rollout settings.
- Runner sets `EXPERIMENT_2E_BEHAVIOR_REWARD=1`, `EXPERIMENT_2E_BEHAVIOR_LAMBDA=0.2`, `TRAINER_LOGGER='["console","swanlab"]'`, and no hidden sidecar.
- Runner writes `resolved_shell_config.json`, `training.done`, `exit_status.txt`, checkpoints, normal validation generations, and a training audit.

- [ ] Reuse the existing Qwen3-1.7B Base formal command line and frozen train/eval paths.
- [ ] Add a short canary mode with 5 training steps and checkpoint/test frequency 5.
- [ ] Refuse to overwrite an existing non-resume behavior run.
- [ ] Refuse to launch if another veRL trainer or GPU compute process is active.
- [ ] Start only through a named tmux session.

### Task 4: Canary validation and formal launch

**Files:**
- Create: `Experiment_2E/server_results/q3_1p7b_behavior_reward_canary_20260824/`
- Create: `Experiment_2E/server_results/q3_1p7b_behavior_reward_formal_20260824/`

- [ ] Run the canary in tmux with `lambda=0.2`.
- [ ] Verify behavior coverage, group count, behavior advantage scale, KL, clip fraction, checkpoint creation, and SwanLab logging.
- [ ] Verify the normal held-out pass@1/pass@4/pass@8 and mean response length are finite.
- [ ] If `std(lambda*A_beh) > 0.5 * std(A_grpo)`, rerun canary with `lambda=0.1`; do not change formulas silently.
- [ ] After canary passes, launch the full formal training in a separate named tmux session.

### Task 5: Checkpoint evaluation and comparison

**Files:**
- Reuse: existing checkpoint rollout/evaluation scripts and the same held-out manifest.
- Create: `Experiment_2E/server_results/q3_1p7b_behavior_reward_formal_20260824/analysis/`

- [ ] Evaluate each saved behavior-reward checkpoint on the same held-out cohort used by the prior normal RL run.
- [ ] Compute pass@1, pass@4, pass@8 and mean@k using the existing evaluation convention.
- [ ] Compare behavior reward versus the prior normal RL run without mixing train and held-out prompts.
- [ ] Record whether the behavior run changes response length, KL, or checkpoint stability.

## Verification Commands

```powershell
pytest Experiment_2E/tests/test_online_behavior.py -q
ssh h200 'tmux ls'
ssh h200 'tail -100 /data2/hjk/logs/experiment_2e/<behavior-run>.log'
```

Success means: the canary and formal run complete in tmux, every expected checkpoint exists, normal SwanLab metrics are present, and held-out pass@k/mean@k are available for comparison.
