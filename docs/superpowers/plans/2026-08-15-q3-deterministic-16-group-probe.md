# Q3 Deterministic 16-Group Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the frozen deterministic 16-group hidden-probe branch, approve it from comparable smoke artifacts, and leave the 250-step Q3 formal run actively executing in a verified detached tmux session.

**Architecture:** A small veRL helper converts accepted trajectory metadata into a stable whole-group boolean mask. The DAPO trainer attaches that mask before no-padding conversion, and the FSDP hidden extractor emits NaN vectors for unselected trajectories while preserving batch alignment. The project runner and auditor carry the approved group limit and interval from smoke evidence into formal mode.

**Tech Stack:** Python, PyTorch, NumPy, Bash, Hydra, pytest, SSH, tmux, Ray, SwanLab.

## Global Constraints

- Keep Qwen3-1.7B-Base, SimpleRL Level 3-5, response cap 4096, rollout `n=8`, accepted batch 32 groups, SDPA, full-parameter GRPO, and all frozen hidden metric settings unchanged.
- Use `USE_REMOVE_PADDING=True` for reduced smoke, approval artifacts, and formal mode.
- Select the lexicographically first 16 distinct frozen `prompt_hash` values after DAPO filtering; select all eight trajectories for each hash.
- Keep all 32 groups in old/reference log-probability, advantages, loss, and optimizer updates; only hidden-probe extraction is masked.
- Every remote GPU job runs in a separately named detached tmux session.
- Formal mode starts only from a `status=passed` approval JSON and remains in tmux after its first complete training step.

---

### Task 1: Add Deterministic Group Selection

**Files:**
- Create: `C:/Users/LENOVO/AppData/Local/Temp/verl_q3_1p7b_impl_20260814/verl/hidden_probe/selection.py`
- Create: `C:/Users/LENOVO/AppData/Local/Temp/verl_q3_1p7b_impl_20260814/tests/hidden_probe/test_selection.py`

**Interface:** `select_group_mask(extra_info: object, group_limit: int = 32) -> np.ndarray[bool]` returns one boolean per trajectory, validates `1 <= group_limit <= 32`, sorts distinct `prompt_hash` strings, and selects every trajectory whose hash is among the first `group_limit` hashes.

- [ ] Write tests for shuffled trajectory order, whole-group selection, deterministic output, default limit 32, and invalid limits.
- [ ] Run `pytest -q tests/hidden_probe/test_selection.py`; confirm the new tests fail before implementation.
- [ ] Implement the helper using a normalized list of dictionaries and `np.asarray(mask, dtype=bool)` without changing metadata.
- [ ] Run the focused test; expect all tests to pass.
- [ ] Commit as `feat: add deterministic q3 hidden group selection`.

### Task 2: Thread The Mask Through veRL Hidden Extraction

**Files:**
- Modify: `C:/Users/LENOVO/AppData/Local/Temp/verl_q3_1p7b_impl_20260814/verl/trainer/ppo/ray_trainer.py`
- Modify: `C:/Users/LENOVO/AppData/Local/Temp/verl_q3_1p7b_impl_20260814/verl/workers/engine/fsdp/transformer_impl.py`
- Modify: `C:/Users/LENOVO/AppData/Local/Temp/verl_q3_1p7b_impl_20260814/verl/hidden_probe/engine_extract.py`
- Create: `C:/Users/LENOVO/AppData/Local/Temp/verl_q3_1p7b_impl_20260814/tests/hidden_probe/test_group_mask_plumbing.py`

**Interface:** `build_hidden_probe(..., selected: torch.Tensor | None = None)` returns exactly one vector per input sample; false entries are all-NaN and never enter dashboard reduction. `_compute_old_log_prob` reads `HIDDEN_PROBE_GROUP_LIMIT` (default `32`), creates `hidden_probe_selected` aligned to `batch.batch`, and attaches it before `left_right_2_no_padding`.

- [ ] Write unit tests proving a 32-row mask remains 32 rows, false rows become NaN, and the trainer source passes `hidden_probe_selected` plus `HIDDEN_PROBE_GROUP_LIMIT`.
- [ ] Run the focused tests and the existing attention fallback test; confirm the new plumbing assertions fail before implementation.
- [ ] Add the mask tensor in `_compute_old_log_prob`, gate capture on the existing probe settings, pass the dynamic micro-batch mask into `build_hidden_probe`, and preserve the nested output alignment.
- [ ] Add the optional `selected` argument to `build_hidden_probe`; short-circuit false rows before trajectory reduction and retain all existing exception-to-NaN behavior.
- [ ] Run `pytest -q tests/hidden_probe/test_group_mask_plumbing.py tests/hidden_probe/test_attention_utils_fallback.py` and the existing hidden-probe unit suite.
- [ ] Commit as `feat: mask q3 hidden probe to deterministic groups`.

### Task 3: Freeze Runner And Timing Auditor

**Files:**
- Modify: `C:/Users/LENOVO/Desktop/A-G实验/AI-HiddenState/Experiment_2E/scripts/run_q3_1p7b_base_grpo.sh`
- Modify: `C:/Users/LENOVO/Desktop/A-G实验/AI-HiddenState/Experiment_2E/experiment_2e/q3_smoke_audit.py`
- Modify: `C:/Users/LENOVO/Desktop/A-G实验/AI-HiddenState/Experiment_2E/tests/test_q3_runner.py`
- Modify: `C:/Users/LENOVO/Desktop/A-G实验/AI-HiddenState/Experiment_2E/tests/test_q3_smoke_audit.py`

**Interface:** Runner accepts `HIDDEN_PROBE_GROUP_LIMIT` in `1..32`, writes it to `resolved_shell_config.json`, and formal mode reads both `hidden_probe_group_limit` and `hidden_probe_interval` from the passed approval JSON. Auditor accepts `--reduced-probe-log` and `--reduced-probe-result`, parses both `timing_s/step:123` and dictionary timing, and for the middle branch requires reduced ratio `<= 1.50`, 16 probed questions per reduced step, five history rows, and 141 figures.

- [ ] Add failing tests for console timing parsing, the 16-group middle branch, reduced history validation, and formal approval group-limit forwarding.
- [ ] Run the focused project tests and confirm the new assertions fail.
- [ ] Implement runner validation/config serialization and approval extraction; implement auditor checks and approval fields `full_probe_overhead_ratio`, `reduced_probe_overhead_ratio`, `hidden_probe_group_limit`, and `hidden_probe_interval`.
- [ ] Run `pytest -q Experiment_2E/tests/test_q3_runner.py Experiment_2E/tests/test_q3_smoke_audit.py Experiment_2E/tests/test_q3_simplerl_manifests.py`.
- [ ] Commit as `fix: freeze q3 reduced probe approval settings`.

### Task 4: Deploy, Run Reduced Smoke, And Generate Approval

**Remote artifacts:**
- veRL root: `/data2/hjk/projects/verl_q3_1p7b_base_20260814`
- project root: `/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814`
- reduced tmux: `q3_1p7b_rmpad_probe16_v1_20260815`
- reduced run: `q3_1p7b_base_simplerl_grpo_smoke_rmpad_probe16_v1_seed20260814`

- [ ] Deploy changed veRL files and the runner; verify `bash -n`, source hashes, and no existing reduced targets.
- [ ] Launch the reduced five-step run detached with `MODE=smoke5 USE_REMOVE_PADDING=True HIDDEN_PROBE_RATE=1 HIDDEN_PROBE_INTERVAL=1 HIDDEN_PROBE_GROUP_LIMIT=16`.
- [ ] Wait for exit 0 and verify `training.done`, `global_step_5`, five online rows, 141 figures, no fatal patterns, and every row reports `probed_questions=16`.
- [ ] Run the auditor against the existing remove-padding baseline, full 20-step probe artifacts, and reduced result; require `status=passed` and inspect the generated approval JSON.
- [ ] Commit the approval JSON only if it is a repository-tracked artifact; otherwise record its remote absolute path in the final handoff.

### Task 5: Launch And Verify Formal tmux

**Remote artifacts:**
- formal tmux: `q3_1p7b_base_grpo_formal_20260815`
- formal run: `q3_1p7b_base_simplerl_grpo_formal_seed20260814`
- formal mode: `MODE=formal`, `USE_REMOVE_PADDING=True`, `SMOKE_APPROVAL=<passed approval JSON>`

- [ ] Verify the GPU is free and the formal result/checkpoint/log targets do not already exist.
- [ ] Launch the formal command with `tmux new-session -d -s q3_1p7b_base_grpo_formal_20260815 ...`.
- [ ] Verify the tmux session remains alive, the resolved config has 250 steps, cap 4096, remove padding true, group limit 16, and interval 1, and the process tree contains the runner, Ray, and Python workers.
- [ ] Wait through the first complete training step; verify `timing_s/step`, accepted group count 32, hidden `probed_questions=16`, SwanLab initialization, and no fatal pattern while the tmux session remains attached.
- [ ] Leave the formal tmux session running and report its session name, run name, log path, approval path, and verified first-step evidence.

## Self-Review Checklist

- Spec coverage: Tasks 1-2 cover deterministic selection and NaN masking; Task 3 covers both timing formats and approval propagation; Tasks 4-5 cover reduced smoke, gate generation, and the required active formal tmux.
- Placeholder scan: no `TODO`, `TBD`, or unspecified implementation step is used; every remote name and verification condition is explicit.
- Type consistency: `select_group_mask` returns a NumPy boolean vector; runner converts it to a PyTorch boolean batch tensor; `build_hidden_probe` consumes that tensor and returns one vector per sample; auditor consumes the resulting JSONL and approval fields.
