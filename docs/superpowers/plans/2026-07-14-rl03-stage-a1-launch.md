# RL03 Stage A1 Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the live A0 poll, pull and audit its artifacts, and conditionally launch the full RL03 Stage A1 frozen-policy scoring run in tmux.

**Architecture:** Keep the poll in the active Codex session. Extend the existing immutable manifest builder with a backwards-compatible all-parseable selection mode, then add A1-specific bash runner and tmux launcher scripts patterned after the tested A0 scripts. Launch fails closed unless A0's engineering smoke gate passes.

**Tech Stack:** PowerShell, OpenSSH, Bash, tmux, Python 3.11, pytest, vLLM, pandas.

---

### Task 1: Generalize the immutable manifest bundle

**Files:**
- Modify: `Experiment_RL/scripts/build_rl03_mcq_audit_manifest.py`
- Modify: `Experiment_RL/tests/test_rl03_mcq_manifest.py`

- [ ] Add tests proving `all_parseable_question_ids` deterministically selects every parseable question and proving configurable `stage_a1_*` filenames do not change A0 defaults.
- [ ] Run `python -m pytest Experiment_RL/tests/test_rl03_mcq_manifest.py -q` and confirm the new tests fail before implementation.
- [ ] Add `--selection-mode {mixed-smoke,all-parseable}` and `--stage-name`, retaining `mixed-smoke`, `32`, and `stage_a0` as defaults.
- [ ] Generate `stage_a1_manifest.jsonl`, `stage_a1_selected_question_ids.txt`, and `stage_a1_manifest_summary.json` only when `--stage-name stage_a1` is explicit.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Add the Stage A1 runner and tmux launcher

**Files:**
- Create: `Experiment_RL/scripts/run_rl03_mcq_audit_stage_a1.sh`
- Create: `Experiment_RL/scripts/launch_rl03_mcq_audit_stage_a1_tmux.sh`
- Create: `Experiment_RL/tests/test_rl03_stage_a1_launcher.py`

- [ ] Add static launcher tests for `/data2` confinement, all-parseable manifest selection, immutable output, busy-GPU refusal, no process cleanup, and idempotent tmux behavior.
- [ ] Run `python -m pytest Experiment_RL/tests/test_rl03_stage_a1_launcher.py -q` and confirm failure because the scripts do not exist.
- [ ] Implement the runner using the A0 environment/resource/run-contract pattern, changing only run identity and full-manifest selection.
- [ ] Implement an idempotent tmux launcher named `rl03_stage_a1_v2_full`.
- [ ] Run focused launcher tests and shell syntax checks.

### Task 3: Regression verification and deployment

**Files:**
- Deploy reviewed Stage A scripts under `/data2/hjk/projects/AI-HiddenState-ER/scripts`

- [ ] Run `python -m pytest Experiment_RL/tests -q` locally.
- [ ] Run `bash -n` for both new shell scripts.
- [ ] Back up only same-name remote files if they already exist.
- [ ] Copy the reviewed builder, A1 runner, and A1 launcher to H200.
- [ ] Compare local and remote SHA-256 values.

### Task 4: Complete A0 handling and conditionally launch A1

**Files:**
- Pull to: `Experiment_RL/server_results/rl03_stage_a0_v2_smoke32_seed20260713/`

- [ ] Wait for `exit_code.txt`, `scores/run_summary.json`, `audit/metric_summary.json`, and `audit/STAGE_A_AUDIT_REPORT.md` using the active ten-minute poll.
- [ ] Pull the complete A0 run and log locally.
- [ ] Verify exit code, completion rate, finite-score rate, score-failure rate, nonzero within-question variance, required artifacts, and error scan.
- [ ] If any engineering predicate fails, write a local failure analysis and stop before A1.
- [ ] If all predicates pass, invoke the A1 tmux launcher once.
- [ ] Verify the A1 tmux session, exact command, log/output paths, manifest size, request count or growing progress, and clean preflight.
