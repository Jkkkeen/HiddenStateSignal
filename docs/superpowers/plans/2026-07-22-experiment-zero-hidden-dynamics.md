# Experiment Zero Hidden Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Experiment 0 on fixed Qwen3-VL-8B-Thinking long responses, measuring token-amplitude and span-direction hidden dynamics across all layers before comparing every candidate signal against raw think length.

**Architecture:** A pure NumPy module owns metric definitions, balanced leave-one-rollout-out scoring, prototype shrinkage, and aggregation. A resumable H200 extractor replays one fixed question at a time, reduces token hidden states online to bin-level scalars, temporarily retains only pooled span vectors for that question, computes cross-rollout features, and atomically writes compact per-question shards. A CPU analyzer combines those shards, performs question-level inference and grouped out-of-fold length baselines, then emits the discovery atlas and report.

**Tech Stack:** Python 3.10+, NumPy, pandas, PyArrow, scikit-learn, matplotlib, PyTorch, Transformers 5.9, pytest; H200 CUDA 12.8 environment `/data2/hjk/envs/hs_er`.

## Global Constraints

- Input is `/data2/hjk/projects/AI-HiddenState-ER/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl`; never call vLLM or generate responses.
- Use `Qwen/Qwen3-VL-8B-Thinking`, only clean non-truncated complete `<think>` segments, and primary discovery questions with at least 3 correct and 3 wrong rollouts.
- Smoke sequence is 2-question extraction validation, then 8-question analysis smoke; formal discovery is 24 questions if available, otherwise all available primary discovery questions down to 16.
- Extract embedding output and every transformer hidden-state index; process one rollout at a time and one question at a time.
- Token stream computes horizontal/vertical L2, first differences, centered coordinate-energy entropy, effective dimensions, counts, mean/median/P90, and mean hidden/update norms in 10 progress bins.
- Span stream computes `span-mean` at 64/32, 128/64, and 256/128; `span-last` is an endpoint control. The primary representation is span-mean 128/64.
- Cross-rollout scores use same-question, same-layer, same-progress-bin nearest-`rho` matching, balanced LOO reference subsets, set direction, prototype direction, prototype shrinkage, and geometric-mean movement-length support.
- Prototype primary validity threshold is `kappa_min=0.20`; `0.10` and `0.30` are sensitivity thresholds. Invalid prototypes do not affect set-direction scores.
- Permanent output must not contain full `[tokens,layers,hidden_dim]` hidden states. Only preselected audit questions may retain float16 pooled span vectors.
- Confidence intervals, permutations, folds, and paired comparisons use question as the independent unit.
- Every candidate must be compared on identical eligible rows and outer folds as `feature-only`, `length-only`, and `feature+length`; both paired AUC improvements must have a question-bootstrap 95% CI lower bound above zero to pass the Experiment 0 gate.
- Do not terminate or modify existing H200 tmux sessions. Launch only when `nvidia-smi` reports no compute process, in a new uniquely named tmux session.

---

### Task 1: Pure Hidden-Dynamics Metrics

**Files:**
- Create: `Experiment/scripts/experiment0_hidden_dynamics.py`
- Create: `Experiment/tests/test_experiment0_hidden_dynamics.py`

**Interfaces:**
- Consumes: NumPy arrays shaped `[tokens, layers, hidden_dim]` or pooled span records.
- Produces: `coordinate_energy_entropy`, `aggregate_token_dynamics`, `pool_span_vectors`, `build_span_direction_records`, `prototype_shrinkage`, and `score_cross_rollout_queries`.

- [ ] **Step 1: Write failing tests for centered entropy, span direction, balanced matching, and shrinkage**

```python
def test_coordinate_energy_entropy_distinguishes_concentrated_and_uniform_energy():
    concentrated = np.array([[3.0, -1.0, -1.0, -1.0]])
    spread = np.array([[1.0, -1.0, 1.0, -1.0]])
    h_conc, n_conc = coordinate_energy_entropy(concentrated)
    h_spread, n_spread = coordinate_energy_entropy(spread)
    assert h_conc[0] < h_spread[0]
    assert n_conc[0] < n_spread[0]

def test_prototype_shrinkage_detects_cancellation():
    aligned = np.array([[1.0, 0.0], [1.0, 0.0]])
    cancelling = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert prototype_shrinkage(aligned) == pytest.approx(1.0)
    assert prototype_shrinkage(cancelling) == pytest.approx(0.0)

def test_cross_rollout_uses_one_nearest_progress_reference_per_rollout():
    scored, diagnostics = score_cross_rollout_queries(make_synthetic_span_records())
    row = scored.query("rollout_id == 0 and span_id == 1").iloc[0]
    assert row.reference_rollout_count_pos == 2
    assert row.reference_rollout_count_neg == 2
    assert diagnostics.groupby("reference_rollout_id").size().max() == 1
```

- [ ] **Step 2: Run the tests and confirm the new module is missing**

Run: `python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py -q`

Expected: collection fails with `ModuleNotFoundError: experiment0_hidden_dynamics`.

- [ ] **Step 3: Implement stable pure metric primitives**

```python
def coordinate_energy_entropy(update: np.ndarray, eps: float = 1e-12):
    update = np.asarray(update, dtype=np.float64)
    centered = update - update.mean(axis=-1, keepdims=True)
    energy = centered * centered
    probs = energy / np.maximum(energy.sum(axis=-1, keepdims=True), eps)
    entropy_raw = -(probs * np.log(probs + eps)).sum(axis=-1)
    entropy = entropy_raw / np.log(update.shape[-1])
    return entropy, np.exp(entropy_raw)

def prototype_shrinkage(unit_vectors: np.ndarray, eps: float = 1e-12) -> float:
    vectors = np.asarray(unit_vectors, dtype=np.float64)
    centroid_norm = np.linalg.norm(vectors.mean(axis=0))
    mean_norm = np.linalg.norm(vectors, axis=1).mean()
    return float(centroid_norm / max(mean_norm, eps))
```

`score_cross_rollout_queries` must call the existing `balanced_reference_subsets`, select one nearest-`rho` record from each reference rollout in the query bin, compute each subset separately, average subset scores, and preserve `kappa_pos`, `kappa_neg`, gate status, reference IDs, and subset ID.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the pure metric core**

```bash
git add Experiment/scripts/experiment0_hidden_dynamics.py Experiment/tests/test_experiment0_hidden_dynamics.py
git commit -m "feat: add experiment zero hidden metrics"
```

### Task 2: Resumable All-Layer H200 Extractor

**Files:**
- Create: `Experiment/scripts/extract_experiment0_hidden_dynamics_qwen3vl.py`
- Create: `Experiment/tests/test_extract_experiment0_hidden_dynamics.py`
- Reuse: `Experiment/scripts/prepare_long_success_smoke.py`
- Reuse: `Experiment/scripts/run_long_path_smoke_qwen3vl.py`

**Interfaces:**
- Consumes: a frozen manifest JSONL and Qwen3-VL processor/model outputs.
- Produces per question: `bin_features/question_*.parquet`, `span_features/question_*.parquet`, `prototype_diagnostics/question_*.parquet`, `question_*.complete.json`, and optional `audit_spans/question_*.npz`.

- [ ] **Step 1: Write failing tests for online reduction and atomic completeness**

```python
def test_reduce_hidden_emits_all_layer_progress_rows():
    hidden = tuple(torch.tensor(make_layer(index)) for index in range(4))
    bins, spans = reduce_rollout_hidden(hidden, segment_start=0, segment_end=320,
                                        progress_bins=10, span_specs=((128, 64),))
    assert set(bins.layer) == {0, 1, 2, 3}
    assert set(bins.progress_bin).issubset(set(range(10)))
    assert {"horizontal_norm_mean", "vertical_norm_p90",
            "coordinate_entropy_mean"}.issubset(bins.columns)

def test_question_complete_requires_all_three_parquet_shards(tmp_path):
    marker = tmp_path / "question_q1.complete.json"
    marker.write_text('{"status":"completed"}', encoding="utf-8")
    assert valid_completed_question(tmp_path, "question_q1") is False
```

- [ ] **Step 2: Run the focused extractor tests and verify failure**

Run: `python -m pytest Experiment/tests/test_extract_experiment0_hidden_dynamics.py -q`

Expected: collection fails because the extractor does not exist.

- [ ] **Step 3: Implement one-question bounded-memory extraction**

```python
with torch.inference_mode():
    outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)
bin_frame, span_payload = reduce_rollout_hidden(
    outputs.hidden_states,
    segment_start=full_segment_start,
    segment_end=full_segment_end,
    progress_bins=10,
    span_specs=((64, 32), (128, 64), (256, 128)),
)
del outputs, full_inputs
torch.cuda.empty_cache()
```

The reducer must iterate layer tensors without stacking the full hidden tuple, move only scalar aggregates and float16 span means/last endpoints to CPU, compute cross-rollout records after all rollouts for one question, atomically rename all parquet files, write the completion marker last, and then release that question's span vectors. `--resume` skips only questions whose marker and all required parquet schemas validate.

- [ ] **Step 4: Run extractor and regression tests**

Run: `python -m pytest Experiment/tests/test_extract_experiment0_hidden_dynamics.py Experiment/tests/test_extract_long_success_hidden.py Experiment/tests/test_long_success_trajectory_common.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the extractor**

```bash
git add Experiment/scripts/extract_experiment0_hidden_dynamics_qwen3vl.py Experiment/tests/test_extract_experiment0_hidden_dynamics.py
git commit -m "feat: extract all-layer experiment zero dynamics"
```

### Task 3: Question-Level Effects And Length Baselines

**Files:**
- Create: `Experiment/scripts/analyze_experiment0_hidden_dynamics.py`
- Create: `Experiment/tests/test_analyze_experiment0_hidden_dynamics.py`

**Interfaces:**
- Consumes: per-question parquet shards from Task 2.
- Produces: `long_experiment_0_bin_features.parquet`, `long_experiment_0_question_effects.csv`, `long_experiment_0_predictor_comparisons.csv`, and `long_experiment_0_prototype_diagnostics.parquet`.

- [ ] **Step 1: Write failing tests for Hedges' g, grouped OOF prediction, and paired deltas**

```python
def test_hedges_g_has_correct_sign():
    assert hedges_g(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0])) > 0

def test_length_baseline_uses_identical_rows_and_group_folds():
    result = compare_feature_to_length(make_grouped_predictor_frame(), seed=7, bootstrap=20)
    assert result.n_rows_feature == result.n_rows_length == result.n_rows_combined
    assert result.test_question_overlap == 0
    assert {"feature_only_auc", "length_only_auc", "combined_auc"}.issubset(result.index)

def test_gate_requires_both_paired_ci_lower_bounds_above_zero():
    assert passes_length_gate(delta_feature_low=0.01, delta_combined_low=0.02)
    assert not passes_length_gate(delta_feature_low=-0.01, delta_combined_low=0.02)
```

- [ ] **Step 2: Verify the analyzer tests fail**

Run: `python -m pytest Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q`

Expected: collection fails because the analyzer does not exist.

- [ ] **Step 3: Implement fixed statistical comparisons**

```python
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
)
folds = list(GroupKFold(n_splits=min(5, frame.question_id.nunique())).split(
    frame, frame.is_correct, groups=frame.question_id
))
```

For each `feature x representation x layer x progress_bin`, first intersect non-null rows once, then reuse those rows and folds for feature-only, raw-think-length-only, and combined predictors. Learn scaler and sign only on outer-train questions. Compute per-question pairwise AUC, average questions equally, bootstrap paired question deltas, report signed `g_length` and `abs(g_length)`, and prohibit test-fold AUC flipping.

- [ ] **Step 4: Run analyzer tests**

Run: `python -m pytest Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the statistical analyzer**

```bash
git add Experiment/scripts/analyze_experiment0_hidden_dynamics.py Experiment/tests/test_analyze_experiment0_hidden_dynamics.py
git commit -m "feat: analyze experiment zero discovery signals"
```

### Task 4: Discovery Atlas And Report

**Files:**
- Modify: `Experiment/scripts/analyze_experiment0_hidden_dynamics.py`
- Modify: `Experiment/tests/test_analyze_experiment0_hidden_dynamics.py`

**Interfaces:**
- Consumes: the combined feature/effect/predictor tables from Task 3.
- Produces: E0-F1 through E0-F7 figures, `LONG_EXPERIMENT_0_RESULTS.md`, and `analysis_meta.json`.

- [ ] **Step 1: Add a failing end-to-end artifact test**

```python
def test_analyzer_writes_all_declared_outputs(tmp_path, monkeypatch):
    run_analyzer_on_synthetic_shards(tmp_path, monkeypatch)
    expected = [
        "LONG_EXPERIMENT_0_RESULTS.md",
        "long_experiment_0_bin_features.parquet",
        "long_experiment_0_question_effects.csv",
        "long_experiment_0_predictor_comparisons.csv",
        "long_experiment_0_prototype_diagnostics.parquet",
        "figures/E0_F1_horizontal_length_angle.png",
        "figures/E0_F2_cross_rollout_support.png",
        "figures/E0_F3_vertical_entropy_activity.png",
        "figures/E0_F4_layer_progress_effects.png",
        "figures/E0_F5_angle_snr_reliability.png",
        "figures/E0_F6_length_strata.png",
        "figures/E0_F7_length_baseline.png",
    ]
    assert all((tmp_path / "results" / name).is_file() for name in expected)
```

- [ ] **Step 2: Run the test and confirm missing figures/report**

Run: `python -m pytest Experiment/tests/test_analyze_experiment0_hidden_dynamics.py::test_analyzer_writes_all_declared_outputs -q`

Expected: FAIL listing the absent artifacts.

- [ ] **Step 3: Implement deterministic plotting and report generation**

The report must state cohort exclusions, actual question/rollout/token counts, extraction layers, all span specifications, kappa invalid rates at `0.10/0.20/0.30`, strongest effect with question-bootstrap CI, raw-length baseline, feature/length/combined paired AUCs, gate decisions, and explicit discovery-only language. Each heatmap must annotate sample coverage and use a symmetric effect-size color scale centered at zero.

- [ ] **Step 4: Run all Experiment 0 tests**

Run: `python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py Experiment/tests/test_extract_experiment0_hidden_dynamics.py Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q`

Expected: all tests pass and no matplotlib figures remain open.

- [ ] **Step 5: Commit report generation**

```bash
git add Experiment/scripts/analyze_experiment0_hidden_dynamics.py Experiment/tests/test_analyze_experiment0_hidden_dynamics.py
git commit -m "feat: report experiment zero discovery atlas"
```

### Task 5: Safe Tmux Launcher And Two-Question Extraction Smoke

**Files:**
- Create: `Experiment/scripts/launch_experiment0_hidden_dynamics_tmux.sh`
- Create on H200: `experiment0_hidden_dynamics/extraction_smoke2_20260722/`

**Interfaces:**
- Consumes: Tasks 1-4, fixed manifest, and an idle H200 GPU.
- Produces: a resumable tmux run with a unique log and two validated completed question markers.

- [ ] **Step 1: Write the launcher with strict preflight checks**

```bash
set -euo pipefail
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d')" ]]; then
  echo "GPU has active compute processes; refusing to launch" >&2
  exit 2
fi
```

The launcher must accept `QUESTION_LIMIT`, `RUN_NAME`, `SESSION`, and `AUDIT_QUESTIONS`, build the manifest with seed `20260721`, invoke extraction with `--resume`, invoke analysis only after extraction markers validate, and write `PIPELINE_COMPLETE` only after all E0 artifacts exist.

- [ ] **Step 2: Run all local tests before synchronization**

Run: `python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py Experiment/tests/test_extract_experiment0_hidden_dynamics.py Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q`

Expected: all tests pass.

- [ ] **Step 3: Synchronize only Experiment 0 scripts and inspect H200 without mutation**

Run: `scp Experiment/scripts/{experiment0_hidden_dynamics.py,extract_experiment0_hidden_dynamics_qwen3vl.py,analyze_experiment0_hidden_dynamics.py,launch_experiment0_hidden_dynamics_tmux.sh} h200:/data2/hjk/projects/AI-HiddenState-ER/scripts/`

Run: `ssh h200 "tmux ls; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"`

Expected: scripts exist; no compute process is active; existing tmux sessions remain unchanged.

- [ ] **Step 4: Launch and monitor extraction smoke**

Run: `ssh h200 "cd /data2/hjk/projects/AI-HiddenState-ER && SESSION=two_dim_e0_extract_smoke2 RUN_NAME=extraction_smoke2_20260722 QUESTION_LIMIT=2 bash scripts/launch_experiment0_hidden_dynamics_tmux.sh"`

Run: `ssh h200 "tmux capture-pane -pt two_dim_e0_extract_smoke2:0 -S -100"`

Expected: two completion markers, no CUDA OOM, finite entropy/norm fields, prototype shrinkage in `[0,1]`, and `PIPELINE_COMPLETE`.

- [ ] **Step 5: Commit the launcher after smoke validation**

```bash
git add Experiment/scripts/launch_experiment0_hidden_dynamics_tmux.sh
git commit -m "exp: add experiment zero tmux launcher"
```

### Task 6: Eight-Question Smoke And Formal Discovery Launch

**Files:**
- Create on H200: `experiment0_hidden_dynamics/smoke8_20260722/`
- Create on H200: `experiment0_hidden_dynamics/formal_discovery24_20260722/`
- Copy back: `Experiment/server_results/experiment0_hidden_dynamics_smoke8_20260722/`

**Interfaces:**
- Consumes: validated Task 5 pipeline.
- Produces: a complete 8-question smoke report and a safely running or completed 16-24-question formal discovery tmux job.

- [ ] **Step 1: Launch the eight-question smoke in a new tmux session**

Run: `ssh h200 "cd /data2/hjk/projects/AI-HiddenState-ER && SESSION=two_dim_e0_smoke8 RUN_NAME=smoke8_20260722 QUESTION_LIMIT=8 bash scripts/launch_experiment0_hidden_dynamics_tmux.sh"`

Expected: the launcher starts only if the GPU is idle and creates `two_dim_e0_smoke8`.

- [ ] **Step 2: Validate smoke outputs before formal launch**

Run: `ssh h200 "grep -R 'PIPELINE_COMPLETE' logs/two_dim_e0_smoke8_20260722.log && find experiment0_hidden_dynamics/smoke8_20260722/results -maxdepth 2 -type f -printf '%P %s bytes\n' | sort"`

Expected: E0-F1 through E0-F7, all four declared tables, report, metadata, finite-value audit, and no traceback/OOM.

- [ ] **Step 3: Copy compact smoke results locally**

Run: `scp -r h200:/data2/hjk/projects/AI-HiddenState-ER/experiment0_hidden_dynamics/smoke8_20260722/results Experiment/server_results/experiment0_hidden_dynamics_smoke8_20260722`

Expected: scalar parquet/CSV, report, figures, and metadata are local; temporary per-question span vectors remain on H200.

- [ ] **Step 4: Launch formal discovery with a unique session**

Run: `ssh h200 "cd /data2/hjk/projects/AI-HiddenState-ER && SESSION=two_dim_e0_formal24 RUN_NAME=formal_discovery24_20260722 QUESTION_LIMIT=24 bash scripts/launch_experiment0_hidden_dynamics_tmux.sh"`

Expected: `two_dim_e0_formal24` appears in `tmux ls`; the log records actual selected question count and advances through extraction without touching other sessions.

- [ ] **Step 5: Record operational handoff**

Report the exact tmux name, log path, run directory, observed per-question time, estimated completion window, `tmux capture-pane` monitoring command, and safe detach sequence `Ctrl-b d`.

## Plan Self-Review

- Spec coverage: E0-A/B/C, all layers, token/span granularity, 10 bins, three span resolutions, span-last and token-angle controls, balanced LOO, geometric length support, prototype shrinkage/gates, length-only floor, question-level inference, E0-F1 through E0-F7, bounded storage, smoke/formal gates, and H200 isolation are assigned to concrete tasks.
- Placeholder scan: no `TODO`, `TBD`, deferred algorithm, or unspecified test command remains.
- Type consistency: the extractor emits per-question parquet schemas consumed directly by the analyzer; `question_id`, `rollout_id`, `progress_bin`, `layer`, `representation`, and `feature` keys are stable across every task.
- Scope: this plan implements discovery Experiment 0 only; it does not modify RL, Experiment 1 historical results, or confirmatory Experiment 3.
