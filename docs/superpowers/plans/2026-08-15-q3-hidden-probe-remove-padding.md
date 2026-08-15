# Q3 Hidden-Probe Remove-Padding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frozen Q3 hidden probe executable with remove-padding enabled, pass the comparable smoke and resume gates, and leave the 250-step formal training actively running in a verified tmux session.

**Architecture:** The existing runner gains one validated environment setting that is used consistently by Hydra and the resolved configuration. A fresh remove-padding baseline and probe pair provide the timing denominator and numerator; the probe run then resumes to step 20 and feeds the existing approval auditor. Formal mode remains gated by that approval JSON and is launched as a detached, explicitly named tmux session.

**Tech Stack:** Bash, Hydra, veRL/DAPO, PyTorch FSDP2, Ray, vLLM, pytest, SwanLab, tmux, NVIDIA H200.

## Global Constraints

- Freeze `USE_REMOVE_PADDING=True` for the new baseline, probe, resume, and formal jobs.
- Keep SDPA attention; do not install or require FlashAttention.
- Do not change hidden metrics, layers, stages, reducers, reward, optimizer, prompt, dataset, sampling, or response cap.
- Use the frozen SimpleRL decision: response cap 4096, 250 formal steps, and 8,000 accepted groups.
- Exclude `baseline_v6` and the interrupted `probe_v1` from approval inputs.
- Every GPU job must run in a separately named detached tmux session.
- Do not start formal training unless the smoke approval status is `passed`.
- Completion requires the formal tmux session to remain alive after a complete first training step, with its PID, GPU use, resolved config, log, and SwanLab initialization verified.

---

### Task 1: Make Remove-Padding A Frozen Runner Setting

**Files:**
- Modify: `Experiment_2E/scripts/run_q3_1p7b_base_grpo.sh`
- Create: `Experiment_2E/tests/test_q3_runner.py`

**Interfaces:**
- Consumes: environment variable `USE_REMOVE_PADDING`, accepted values `True` or `False`.
- Produces: Hydra override `actor_rollout_ref.model.use_remove_padding=<value>` and JSON boolean `remove_padding` in `resolved_shell_config.json`.

- [ ] **Step 1: Write the failing runner tests**

```python
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_q3_1p7b_base_grpo.sh"
)


def test_q3_runner_freezes_remove_padding_true_by_default() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}" in text
    assert '"remove_padding":${USE_REMOVE_PADDING,,}' in text
    assert (
        'actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}"'
        in text
    )


def test_q3_runner_rejects_unknown_remove_padding_values() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'case "${USE_REMOVE_PADDING}" in' in text
    assert "True|False)" in text
    assert "USE_REMOVE_PADDING must be True or False" in text
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path Experiment_2E).Path
pytest -q Experiment_2E/tests/test_q3_runner.py
```

Expected: both tests fail because the runner still hardcodes `False`.

- [ ] **Step 3: Add and validate the setting**

Add after the existing runner defaults:

```bash
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
case "${USE_REMOVE_PADDING}" in
  True|False) ;;
  *) echo "USE_REMOVE_PADDING must be True or False" >&2; exit 2 ;;
esac
```

Change the resolved JSON field to:

```bash
"remove_padding":${USE_REMOVE_PADDING,,}
```

Change the Hydra override to:

```bash
actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path Experiment_2E).Path
pytest -q Experiment_2E/tests/test_q3_runner.py Experiment_2E/tests/test_q3_smoke_audit.py Experiment_2E/tests/test_q3_simplerl_manifests.py
```

Expected: all tests pass.

- [ ] **Step 5: Run the veRL fallback test**

Run:

```powershell
pytest -q C:/Users/LENOVO/AppData/Local/Temp/verl_q3_1p7b_impl_20260814/tests/hidden_probe/test_attention_utils_fallback.py
```

Expected: test passes without importing FlashAttention.

- [ ] **Step 6: Commit the runner change**

```powershell
git add Experiment_2E/scripts/run_q3_1p7b_base_grpo.sh Experiment_2E/tests/test_q3_runner.py
git commit -m "fix: enable q3 remove-padding probe path"
```

### Task 2: Deploy And Run The Comparable Baseline

**Files:**
- Deploy: `Experiment_2E/scripts/run_q3_1p7b_base_grpo.sh`
- Remote runner: `/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/run_q3_1p7b_base_grpo.sh`
- Remote result: `/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814`
- Remote checkpoint: `/data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814`
- Remote log: `/data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814.log`

**Interfaces:**
- Consumes: runner from Task 1 and deployed Transformers padding fallback at veRL commit `4ebf922`.
- Produces: five remove-padding baseline step times for the approval denominator.

- [ ] **Step 1: Deploy and syntax-check the runner**

```powershell
scp Experiment_2E/scripts/run_q3_1p7b_base_grpo.sh h200:/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/run_q3_1p7b_base_grpo.sh
ssh h200 "chmod 755 /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/run_q3_1p7b_base_grpo.sh && bash -n /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/run_q3_1p7b_base_grpo.sh"
```

Expected: exit code 0.

- [ ] **Step 2: Verify the baseline targets are unused and the GPU is free**

```powershell
ssh h200 "set -e; ! tmux has-session -t q3_1p7b_rmpad_baseline_v1_20260815 2>/dev/null; test ! -e /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814; test ! -e /data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814; test -z \"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)\""
```

Expected: no target exists and GPU memory use is 0 MiB.

- [ ] **Step 3: Launch the baseline in tmux**

```powershell
ssh h200 tmux new-session -d -s q3_1p7b_rmpad_baseline_v1_20260815 env MODE=smoke5 RUN_NAME=q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814 USE_REMOVE_PADDING=True HIDDEN_PROBE_RATE=0 HIDDEN_PROBE_INTERVAL=1 bash /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/run_q3_1p7b_base_grpo.sh
```

- [ ] **Step 4: Verify launch configuration**

```powershell
ssh h200 "set -e; tmux has-session -t q3_1p7b_rmpad_baseline_v1_20260815; python3 -c \"import json; p=json.load(open('/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814/resolved_shell_config.json')); assert p['remove_padding'] is True and p['hidden_probe_rate']==0 and p['steps']==5\"; grep -q Q3_1P7B_GRPO_START /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814.log"
```

Expected:

- tmux `q3_1p7b_rmpad_baseline_v1_20260815` is alive;
- resolved config has `remove_padding=true`, `hidden_probe_rate=0`, and `steps=5`;
- the log contains `Q3_1P7B_GRPO_START`;
- no fatal log pattern appears.

- [ ] **Step 5: Monitor to completion**

```powershell
ssh h200 "tmux has-session -t q3_1p7b_rmpad_baseline_v1_20260815 2>/dev/null && echo running || echo finished; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; grep -c 'timing_s/step:' /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814.log"
ssh h200 "set -e; test \"$(cat /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814/exit_status.txt)\" = 0; test -f /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814/training.done; test -d /data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814/global_step_5; test \"$(grep -c 'timing_s/step:' /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814.log)\" -ge 5; ! grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory|Too many open files' /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814.log"
```

Expected:

- exit status 0 and `training.done`;
- checkpoint `global_step_5`;
- five positive `timing_s/step` samples;
- no hidden history is required;
- no traceback, OOM, or file-descriptor error.

### Task 3: Run The Comparable Hidden Probe

**Files:**
- Remote result: `/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814`
- Remote checkpoint: `/data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814`
- Remote log: `/data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814.log`

**Interfaces:**
- Consumes: the same runner and remove-padding configuration as Task 2.
- Produces: step-5 checkpoint, per-step hidden history, dashboard images, and probe timing samples.

- [ ] **Step 1: Verify probe targets are unused**

```powershell
ssh h200 "set -e; ! tmux has-session -t q3_1p7b_rmpad_probe_v1_20260815 2>/dev/null; test ! -e /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814; test ! -e /data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814"
```

Expected: target tmux, result, and checkpoint paths do not exist.

- [ ] **Step 2: Launch the probe in tmux**

```powershell
ssh h200 tmux new-session -d -s q3_1p7b_rmpad_probe_v1_20260815 env MODE=smoke5 RUN_NAME=q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814 USE_REMOVE_PADDING=True HIDDEN_PROBE_RATE=1 HIDDEN_PROBE_INTERVAL=1 bash /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/run_q3_1p7b_base_grpo.sh
```

- [ ] **Step 3: Verify the first hidden record**

```powershell
ssh h200 "set -e; test -s /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/online_hidden/per_step_summary.jsonl; python3 -c \"import json; p=json.load(open('/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/resolved_shell_config.json')); assert p['remove_padding'] is True and p['hidden_probe_rate']==1\"; ! grep -q '\[hidden_probe\] aggregate failed' /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814.log"
```

Expected after step 1:

- `online_hidden/per_step_summary.jsonl` exists and is non-empty;
- the log contains no `[hidden_probe] aggregate failed`;
- resolved config has remove-padding true and probe rate 1.

- [ ] **Step 4: Monitor to step 5**

```powershell
ssh h200 "tmux has-session -t q3_1p7b_rmpad_probe_v1_20260815 2>/dev/null && echo running || echo finished; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; wc -l /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/online_hidden/per_step_summary.jsonl 2>/dev/null || true"
ssh h200 "set -e; test \"$(cat /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/exit_status.txt)\" = 0; test -d /data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/global_step_5; test -s /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/online_hidden/per_step_summary.jsonl; test \"$(find /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/online_hidden/latest_figures -type f -name '*.png' | wc -l)\" -eq 141"
```

Expected:

- exit status 0;
- checkpoint `global_step_5`;
- five positive timing samples;
- non-empty hidden history;
- the asynchronous sidecar converges to 141 PNG files;
- SwanLab initializes without authentication or upload failure.

### Task 4: Resume The Probe To Step 20

**Files:**
- Reuse the probe result, checkpoint, and log paths from Task 3.

**Interfaces:**
- Consumes: the Task 3 `global_step_5` checkpoint.
- Produces: `global_step_20`, a restore marker, held-out hidden history, and final 141-image dashboard state.

- [ ] **Step 1: Confirm the step-5 checkpoint is complete**

```powershell
ssh h200 "set -e; test \"$(cat /data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/latest_checkpointed_iteration.txt)\" = 5; test -d /data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/global_step_5/actor"
```

Expected: `latest_checkpointed_iteration.txt` contains `5`, and actor/optimizer state files are present.

- [ ] **Step 2: Launch resume in a new tmux session**

```powershell
ssh h200 tmux new-session -d -s q3_1p7b_rmpad_probe_resume20_v1_20260815 env MODE=smoke20 RUN_NAME=q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814 USE_REMOVE_PADDING=True HIDDEN_PROBE_RATE=1 HIDDEN_PROBE_INTERVAL=1 bash /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/run_q3_1p7b_base_grpo.sh
```

- [ ] **Step 3: Verify restore before accepting new steps**

```powershell
ssh h200 "grep -q 'Setting global step to 5' /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814.log"
```

Expected: the combined log contains `Setting global step to 5`; training progress starts at 5 rather than 0.

- [ ] **Step 4: Monitor to step 20**

```powershell
ssh h200 "tmux has-session -t q3_1p7b_rmpad_probe_resume20_v1_20260815 2>/dev/null && echo running || echo finished; cat /data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/latest_checkpointed_iteration.txt 2>/dev/null || true; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
ssh h200 "set -e; test \"$(cat /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/exit_status.txt)\" = 0; test -d /data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/global_step_20; test -s /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/online_hidden/per_step_summary.jsonl; test -s /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/heldout/hidden_summary.jsonl; test \"$(find /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/online_hidden/latest_figures -type f -name '*.png' | wc -l)\" -eq 141"
```

Expected:

- exit status 0;
- `global_step_20`;
- non-empty online and held-out hidden JSONL;
- 141 latest PNG files;
- clean SwanLab sidecar exit;
- no traceback, OOM, or file-descriptor error.

### Task 5: Generate And Audit Smoke Approval

**Files:**
- Create remotely: `/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/smoke_approval.json`

**Interfaces:**
- Consumes: baseline log from Task 2 and probe artifacts from Tasks 3-4.
- Produces: the only `SMOKE_APPROVAL` accepted by formal mode.

- [ ] **Step 1: Generate approval**

```bash
PYTHONPATH=/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814 \
/data2/hjk/envs/verl_qwen3vl_py311/bin/python \
  -m experiment_2e.q3_smoke_audit \
  --baseline-log /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_baseline_v1_seed20260814.log \
  --probe-log /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814.log \
  --probe-result /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814 \
  --checkpoint-root /data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814 \
  --output /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/smoke_approval.json
```

- [ ] **Step 2: Verify every approval field**

Expected:

- `status == "passed"`;
- every item in `checks` is true;
- `checkpoint_restore_passed == true`;
- `dashboard_image_count == 141`;
- `hidden_probe_interval` is 1 or 5.

If overhead is between 1.30 and 1.50, do not edit the approval. Run the pre-registered reduced-group smoke first.

- [ ] **Step 3: Freeze artifact hashes**

Record SHA256 for the runner, approval JSON, dataset decision, base common, and veRL source manifest in the formal launch log.

### Task 6: Launch And Verify Formal Training In Tmux

**Files:**
- Remote result: `/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814`
- Remote checkpoint: `/data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814`
- Remote log: `/data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814.log`

**Interfaces:**
- Consumes: passed approval JSON from Task 5.
- Produces: a live, detached 250-step formal tmux job.

- [ ] **Step 1: Perform final preflight**

```powershell
ssh h200 "set -e; test -z \"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)\"; ! tmux has-session -t q3_1p7b_base_grpo_formal_20260815 2>/dev/null; test ! -e /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814; test ! -e /data2/hjk/checkpoints/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814; test \"$(df --output=avail -B1 /data2 | tail -1)\" -ge 500000000000; test \"$(stat -c '%a' /data2/hjk/secrets/experiment_2e_swanlab.env)\" = 600; bash -n /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/run_q3_1p7b_base_grpo.sh; python3 -c \"import json; p=json.load(open('/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/smoke_approval.json')); assert p['status']=='passed'\""
```

Expected:

- H200 has no conflicting compute process;
- target formal tmux/result/checkpoint paths do not exist;
- at least 500 GB remains on `/data2`;
- approval JSON is passed;
- SwanLab secret is readable with mode 600;
- runner passes `bash -n`.

- [ ] **Step 2: Launch the formal job**

```powershell
ssh h200 tmux new-session -d -s q3_1p7b_base_grpo_formal_20260815 env MODE=formal RUN_NAME=q3_1p7b_base_simplerl_grpo_formal_seed20260814 USE_REMOVE_PADDING=True SMOKE_APPROVAL=/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe_v1_seed20260814/smoke_approval.json bash /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/run_q3_1p7b_base_grpo.sh
```

- [ ] **Step 3: Verify tmux and resolved configuration**

```powershell
ssh h200 "set -e; tmux has-session -t q3_1p7b_base_grpo_formal_20260815; tmux list-panes -t q3_1p7b_base_grpo_formal_20260815 -F '#{session_name}|#{pane_pid}|#{pane_current_command}|#{pane_dead}|#{pane_start_command}'; python3 -c \"import json; p=json.load(open('/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814/resolved_shell_config.json')); assert p['mode']=='formal' and p['steps']==250 and p['remove_padding'] is True and p['max_response_length']==4096 and p['hidden_probe_interval'] in (1,5)\""
```

Expected:

- tmux session `q3_1p7b_base_grpo_formal_20260815` is alive and detached;
- pane start command exactly references `MODE=formal` and the passed approval;
- resolved config has `mode=formal`, `steps=250`, `remove_padding=true`, cap 4096, and the approved hidden interval;
- trainer and sidecar use the formal SwanLab run name.

- [ ] **Step 4: Verify live training**

```powershell
ssh h200 "tmux list-panes -t q3_1p7b_base_grpo_formal_20260815 -F '#{pane_pid}|#{pane_start_command}'; ps -eo pid,ppid,etime,cmd | grep -E 'DAPOTaskRunner|VLLM::Worker|q3_1p7b_base_simplerl_grpo_formal' | grep -v grep; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; grep -E 'swanlab|SwanLab|Traceback|CUDA out of memory|Too many open files' /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814.log | tail -30"
```

Expected:

- Ray/DAPO and vLLM worker PIDs are children of the tmux launch;
- `nvidia-smi` shows the formal worker using the H200;
- no fatal log pattern appears;
- SwanLab initialization succeeds.

- [ ] **Step 5: Wait for and verify the first complete formal step**

```powershell
ssh h200 "set -e; tmux has-session -t q3_1p7b_base_grpo_formal_20260815; grep -q 'step:1 -' /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814.log; grep 'step:1 -' /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814.log | tail -1 | grep -q 'grpo_groups/accepted_group_count:32.0'; ! grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory|Too many open files' /data2/hjk/logs/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814.log; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
```

Expected:

- the log contains a complete `step:1` metrics line;
- accepted groups equal 32;
- hidden history contains the formal step when the approved interval includes step 1;
- tmux remains alive after the check;
- GPU and worker PIDs still belong to the formal run.

Do not terminate the formal tmux session after verification.
