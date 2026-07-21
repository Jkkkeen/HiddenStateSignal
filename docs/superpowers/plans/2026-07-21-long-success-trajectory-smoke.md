# Long Success-Trajectory Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Experiment 1's long-response smoke on fixed Qwen3-VL-8B-Thinking rollouts, producing L24/L36 span representations and balanced-LOO four-cell angle/amplitude results.

**Architecture:** A CPU-only manifest stage freezes eligible `3+3` discovery questions without regenerating text. A resumable H200 extractor replays those fixed sequences and writes one compressed span-hidden shard per question. A CPU-only analyzer loads the shards, exhaustively enumerates balanced leave-one-rollout-out reference subsets, and writes span, rollout, question-interaction, coverage, permutation, and report artifacts.

**Tech Stack:** Python 3.10+, NumPy, pandas, PyArrow, scikit-learn, matplotlib, PyTorch, Transformers 5.9, pytest; H200 CUDA 12.8 environment `/data2/hjk/envs/hs_er`.

## Global Constraints

- Input is `/data2/hjk/projects/AI-HiddenState-ER/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl`; never call vLLM or generate new responses.
- Use `Qwen/Qwen3-VL-8B-Thinking`, non-truncated rows with a valid `</think>` boundary, and only the think-token segment.
- Primary questions have at least 3 correct and 3 wrong clean rollouts; smoke uses 32 discovery questions with seed 20260721.
- Extract layers 24 and 36, full 128-token windows, stride 64, span-last primary and span-mean secondary.
- Main scoring uses 10 relative-progress bins, nearest observed path (`K=1`), balanced LOO, and exhaustive equal-size correct/wrong reference subset pairs.
- All confidence intervals and permutations resample at question level; spans are never treated as independent observations.
- Do not modify or terminate existing H200 tmux sessions. Launch in a new `two_dim_e1_smoke32` session only after confirming GPU 0 is idle.

---

### Task 1: Pure Manifest And Balanced-LOO Primitives

**Files:**
- Create: `Experiment/scripts/long_success_trajectory_common.py`
- Create: `Experiment/tests/test_long_success_trajectory_common.py`

**Interfaces:**
- Consumes: labeled rollout dictionaries and NumPy displacement vectors.
- Produces: `eligible_question_summaries`, `split_primary_questions`, `full_span_bounds`, `balanced_reference_subsets`, `cosine_similarity`, and `pairwise_auc`.

- [ ] **Step 1: Write failing tests for eligibility, full spans, fixed reference size, and all subset combinations**

```python
def test_balanced_subsets_for_three_correct_five_wrong_query_wrong():
    labels = {0: True, 1: True, 2: True, 3: False, 4: False, 5: False, 6: False, 7: False}
    subsets = balanced_reference_subsets(labels, query_id=3)
    assert len(subsets) == 18
    assert all(len(pos) == len(neg) == 2 for pos, neg in subsets)
    assert all(3 not in pos and 3 not in neg for pos, neg in subsets)

def test_full_span_bounds_drop_partial_tail():
    assert full_span_bounds(300, window=128, stride=64) == [(0, 128), (64, 192), (128, 256)]
```

- [ ] **Step 2: Run tests and verify they fail because the module does not exist**

Run: `python -m pytest Experiment/tests/test_long_success_trajectory_common.py -q`

Expected: collection fails with `ModuleNotFoundError: long_success_trajectory_common`.

- [ ] **Step 3: Implement deterministic pure helpers**

```python
def balanced_reference_subsets(labels: dict[int, bool], query_id: int):
    pos = [rid for rid, y in labels.items() if rid != query_id and y]
    neg = [rid for rid, y in labels.items() if rid != query_id and not y]
    m = min(sum(labels.values()) - 1, len(labels) - sum(labels.values()) - 1)
    if m < 1:
        return []
    return [(p, n) for p in combinations(pos, m) for n in combinations(neg, m)]

def full_span_bounds(length: int, window: int, stride: int):
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    return [(start, start + window) for start in range(0, length - window + 1, stride)]
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest Experiment/tests/test_long_success_trajectory_common.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the pure core**

```bash
git add Experiment/scripts/long_success_trajectory_common.py Experiment/tests/test_long_success_trajectory_common.py
git commit -m "feat: add long success trajectory primitives"
```

### Task 2: Freeze The Smoke Manifest

**Files:**
- Create: `Experiment/scripts/prepare_long_success_smoke.py`
- Create: `Experiment/tests/test_prepare_long_success_smoke.py`

**Interfaces:**
- Consumes: existing labeled long-response JSONL.
- Produces: `manifest_smoke32.jsonl`, `question_splits.csv`, `ELIGIBILITY_AUDIT.json`, and `ELIGIBILITY_AUDIT.md`.

- [ ] **Step 1: Write failing tests for clean complete-think filtering and deterministic 70/30 question splitting**

```python
def test_build_manifest_keeps_only_primary_discovery_questions(tmp_path):
    rows = make_rows({"q1": (3, 3), "q2": (2, 4), "q3": (4, 4)})
    result = build_manifest(rows, smoke_questions=1, discovery_fraction=0.7, seed=7)
    assert set(result.primary_questions) == {"q1", "q3"}
    assert len(result.smoke_questions) == 1
    assert all(row["question_id"] in result.smoke_questions for row in result.smoke_rows)
```

- [ ] **Step 2: Verify the focused test fails**

Run: `python -m pytest Experiment/tests/test_prepare_long_success_smoke.py -q`

Expected: import fails because `prepare_long_success_smoke.py` does not exist.

- [ ] **Step 3: Implement manifest selection and audit writing**

```python
def is_primary_row(row):
    return (
        not bool(row.get("truncated"))
        and bool(row.get("has_think_close"))
        and row.get("pred_answer") in {"A", "B", "C", "D"}
    )

def question_is_primary(group):
    positives = sum(bool(row["is_correct"]) for row in group)
    negatives = len(group) - positives
    return positives >= 3 and negatives >= 3
```

- [ ] **Step 4: Run focused and existing long-rollout tests**

Run: `python -m pytest Experiment/tests/test_prepare_long_success_smoke.py Experiment/tests/test_inspect_long_rollouts.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit manifest builder**

```bash
git add Experiment/scripts/prepare_long_success_smoke.py Experiment/tests/test_prepare_long_success_smoke.py
git commit -m "feat: prepare long success smoke manifest"
```

### Task 3: Resumable H200 Span-Hidden Extractor

**Files:**
- Create: `Experiment/scripts/extract_long_success_hidden_qwen3vl.py`
- Create: `Experiment/tests/test_extract_long_success_hidden.py`

**Interfaces:**
- Consumes: frozen smoke JSONL and Qwen3-VL processor/model.
- Produces: one `question_*.npz` per completed question plus `extraction_status.jsonl`.

- [ ] **Step 1: Write failing tests for tensor pooling and shard schema**

```python
def test_pool_span_last_and_mean():
    hidden = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    last, mean = pool_span_representations(hidden, [(0, 4), (2, 6)])
    np.testing.assert_allclose(last[:, :, :], hidden[:, [3, 5], :].transpose(1, 0, 2))
    np.testing.assert_allclose(mean[0], hidden[:, 0:4].mean(axis=1))
```

- [ ] **Step 2: Verify the test fails**

Run: `python -m pytest Experiment/tests/test_extract_long_success_hidden.py -q`

Expected: import fails because the extractor does not exist.

- [ ] **Step 3: Implement GPU pooling without saving token hidden states**

```python
def pool_span_representations(hidden, spans):
    last = np.stack([hidden[:, end - 1, :] for start, end in spans], axis=0)
    mean = np.stack([hidden[:, start:end, :].mean(axis=1) for start, end in spans], axis=0)
    return last.astype(np.float16), mean.astype(np.float16)
```

The main loop must group rows by question, replay every fixed response with `model.eval()` and `torch.inference_mode()`, pool only layers 24/36, atomically write a temporary shard then rename it, and skip existing valid shards under `--resume`.

- [ ] **Step 4: Run extractor and regression tests**

Run: `python -m pytest Experiment/tests/test_extract_long_success_hidden.py Experiment/tests/test_long_path_smoke.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit extractor**

```bash
git add Experiment/scripts/extract_long_success_hidden_qwen3vl.py Experiment/tests/test_extract_long_success_hidden.py
git commit -m "feat: extract long span hidden states"
```

### Task 4: Balanced-LOO Analyzer And Report

**Files:**
- Create: `Experiment/scripts/analyze_long_success_trajectory.py`
- Create: `Experiment/tests/test_analyze_long_success_trajectory.py`

**Interfaces:**
- Consumes: question NPZ shards from Task 3.
- Produces: averaged span scores, rollout features, four-cell question interactions, reference coverage, permutation nulls, AUROC tables, plots, and `LONG_EXPERIMENT_1_RESULTS.md`.

- [ ] **Step 1: Write failing synthetic tests for four-cell signs and amplitude weighting**

```python
def test_four_cell_interaction_positive_for_separated_paths():
    cells = {"A_pp": 0.9, "A_pn": 0.2, "A_np": 0.1, "A_nn": 0.8}
    assert interaction_from_cells(cells) == pytest.approx(1.4)

def test_amplitude_suppresses_tiny_noisy_displacement():
    assert amplitude_score(norm=0.01, s_pos=0.9, s_neg=0.1) == pytest.approx(0.008)
```

- [ ] **Step 2: Verify the analyzer test fails**

Run: `python -m pytest Experiment/tests/test_analyze_long_success_trajectory.py -q`

Expected: import fails because the analyzer does not exist.

- [ ] **Step 3: Implement same-bin nearest-reference scoring and question-level inference**

```python
def interaction_from_cells(cells):
    return (cells["A_pp"] - cells["A_pn"]) - (cells["A_np"] - cells["A_nn"])

def amplitude_score(norm, s_pos, s_neg):
    return float(norm * (s_pos - s_neg))
```

The analyzer must average over all exhaustive balanced subset pairs, aggregate spans within rollout-progress bin before equal-bin rollout aggregation, retain selected reference IDs, bootstrap questions, and rerun the complete reference construction under within-question label permutations.

- [ ] **Step 4: Run all Experiment 1 unit tests**

Run: `python -m pytest Experiment/tests/test_long_success_trajectory_common.py Experiment/tests/test_prepare_long_success_smoke.py Experiment/tests/test_extract_long_success_hidden.py Experiment/tests/test_analyze_long_success_trajectory.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit analyzer**

```bash
git add Experiment/scripts/analyze_long_success_trajectory.py Experiment/tests/test_analyze_long_success_trajectory.py
git commit -m "feat: analyze long success trajectory interactions"
```

### Task 5: H200 Smoke Launch And Result Retrieval

**Files:**
- Create: `Experiment/scripts/launch_long_success_smoke_tmux.sh`
- Create on H200: `long_success_trajectory/smoke32_20260721/`
- Copy back: `Experiment/server_results/long_success_trajectory_smoke32_20260721/`

**Interfaces:**
- Consumes: Tasks 1-4 and idle H200 GPU 0.
- Produces: completed smoke extraction and analysis artifacts available locally.

- [ ] **Step 1: Copy only new scripts to H200 and run manifest audit**

Run: `scp Experiment/scripts/{long_success_trajectory_common.py,prepare_long_success_smoke.py,extract_long_success_hidden_qwen3vl.py,analyze_long_success_trajectory.py,launch_long_success_smoke_tmux.sh} h200:/data2/hjk/projects/AI-HiddenState-ER/scripts/`

Expected: all five files exist on H200; no existing script is overwritten.

- [ ] **Step 2: Reconfirm GPU idleness and launch the independent tmux session**

Run: `ssh h200 "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader && bash /data2/hjk/projects/AI-HiddenState-ER/scripts/launch_long_success_smoke_tmux.sh"`

Expected: session `two_dim_e1_smoke32` appears in `tmux ls`, and its log reaches model loading then extraction progress.

- [ ] **Step 3: Monitor without attaching or interrupting other sessions**

Run: `ssh h200 "tmux capture-pane -pt two_dim_e1_smoke32:0 -S -80"`

Expected: progress advances and no CUDA OOM or missing-image errors occur.

- [ ] **Step 4: Validate result artifacts and copy scalar outputs locally**

Run: `scp -r h200:/data2/hjk/projects/AI-HiddenState-ER/long_success_trajectory/smoke32_20260721/results Experiment/server_results/long_success_trajectory_smoke32_20260721`

Expected: report, CSV/parquet tables, figures, and audit files are present; large hidden NPZ shards remain on H200 unless explicitly needed.

- [ ] **Step 5: Commit launch script and copied scalar results**

```bash
git add Experiment/scripts/launch_long_success_smoke_tmux.sh Experiment/server_results/long_success_trajectory_smoke32_20260721
git commit -m "exp: run long success trajectory smoke"
```

## Plan Self-Review

- Spec coverage: fixed long rollouts, complete-think filtering, 3+3 primary cohort, L24/L36 128/64 extraction, balanced LOO, exhaustive subsets, angle/amplitude, four cells, relative progress, coverage, permutation, and question-level inference are assigned to concrete tasks.
- Placeholder scan: every step names concrete files, commands, expected results, and fixed parameters.
- Type consistency: manifest rows feed question NPZ shards; shards feed balanced-LOO analysis; field names are stable across task boundaries.
