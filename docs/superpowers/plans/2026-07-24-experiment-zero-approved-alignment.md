# Approved Experiment Zero Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the existing long-response Experiment 0 pipeline into exact alignment with the approved multi-resolution design in `Experiment/2dimension.md`, then run a non-overwriting H200 smoke and formal discovery refresh.

**Architecture:** Preserve the existing three-module boundary: `experiment0_hidden_dynamics.py` owns pure geometry and reductions, `extract_experiment0_hidden_dynamics_qwen3vl.py` owns one-question-at-a-time all-layer forward/reduction, and `analyze_experiment0_hidden_dynamics.py` owns question-level statistics and reporting. Extend the compact shard schema with an explicit progress resolution, compute matched angle-background controls before pooled vectors are released, and add raw, within-question-centered, and grouped out-of-fold nuisance-adjusted analyses. Existing 2026-07-22 results remain immutable legacy discovery artifacts; v2 writes to new local and H200 directories.

**Tech Stack:** Python 3.10+, NumPy, pandas, PyArrow, scikit-learn, matplotlib, PyTorch, Transformers 5.9, pytest; H200 CUDA environment `/data2/hjk/envs/hs_er`.

## Global Constraints

- Reuse fixed rows from `/data2/hjk/projects/AI-HiddenState-ER/data_long/rollouts_thinking_smoke500_mt16384_labeled.jsonl`; do not generate new rollouts.
- Use `Qwen/Qwen3-VL-8B-Thinking`, clean complete `<think>` segments, and the frozen primary-discovery split with at least 3 correct and 3 wrong rollouts per question.
- Process one rollout and one question at a time. Never persist full `[tokens, layers, hidden_dim]` tensors.
- Main progress resolution is `B=10`; `B=20` is inspection-only and must never replace the main result after looking at effects.
- Main angle representation is `span-mean 128/64`; `span-mean 64/32`, `span-mean 256/128`, `span-last 128/64`, and token turning are controls.
- Main cross-rollout score is balanced-LOO nearest-reference set direction. Prototype direction is valid only when both shrinkage values pass the frozen `kappa_min=0.20`; `0.10/0.30` are sensitivity thresholds.
- Question is the independent unit for confidence intervals, label permutations, folds, reliability summaries, and paired model comparisons.
- Existing results under `Experiment/server_results/experiment0_hidden_dynamics_*_20260722/` are read-only legacy artifacts. Do not overwrite them or retrospectively relabel them as v2.
- A v2 H200 launcher must refuse to start if another compute process is active and must use a new tmux session.

---

### Task 1: Add Explicit 10/20-Bin Resolution To The Compact Schema

**Files:**
- Modify: `Experiment/scripts/experiment0_hidden_dynamics.py`
- Modify: `Experiment/scripts/extract_experiment0_hidden_dynamics_qwen3vl.py`
- Modify: `Experiment/tests/test_experiment0_hidden_dynamics.py`
- Modify: `Experiment/tests/test_extract_experiment0_hidden_dynamics.py`

**Interfaces:**
- CLI changes from `--progress-bins 10` to `--progress-bins 10,20`.
- Every token, span, prototype, and pairwise-geometry row gains `progress_resolution`.
- Primary inferential rows are selected with `progress_resolution == 10`; resolution 20 is inspection-only.

- [ ] **Step 1: Write failing parser and schema tests**

```python
from extract_experiment0_hidden_dynamics_qwen3vl import parse_progress_bins


def test_parse_progress_bins_requires_unique_positive_values() -> None:
    assert parse_progress_bins("10,20") == (10, 20)
    with pytest.raises(ValueError):
        parse_progress_bins("10,10")
    with pytest.raises(ValueError):
        parse_progress_bins("10,0")


def test_reduce_hidden_emits_both_progress_resolutions() -> None:
    reduced = reduce_rollout_hidden(
        _hidden_states(),
        segment_start=0,
        segment_end=12,
        progress_bins=(3, 4),
        span_specs=((4, 2),),
        endpoint_spec=(4, 2),
        primary_spec=(4, 2),
        question_id="q1",
        rollout_id=0,
        is_correct=True,
        think_length=12,
    )
    assert set(reduced.token_features["progress_resolution"]) == {3, 4}
    assert set(reduced.span_horizontal["progress_resolution"]) == {3, 4}
    keys = [
        "question_id", "rollout_id", "representation", "progress_resolution",
        "progress_bin", "layer", "span_id",
    ]
    assert not reduced.span_horizontal.duplicated(keys).any()
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py Experiment/tests/test_extract_experiment0_hidden_dynamics.py -q`

Expected: failure because `parse_progress_bins` and `progress_resolution` do not exist and `reduce_rollout_hidden` accepts one integer.

- [ ] **Step 3: Implement the resolution parser and row identity**

```python
def parse_progress_bins(value: str) -> tuple[int, ...]:
    bins = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not bins or any(item <= 0 for item in bins):
        raise ValueError("progress bins must be positive")
    if len(set(bins)) != len(bins):
        raise ValueError("progress bins must be unique")
    return bins


def progress_bin_ids(length: int, resolution: int) -> np.ndarray:
    if length <= 0 or resolution <= 0:
        raise ValueError("length and resolution must be positive")
    return np.minimum(
        ((np.arange(length) + 1) * resolution) // length,
        resolution - 1,
    ).astype(np.int16)
```

Change `aggregate_token_dynamics`, `build_span_direction_records`, and `reduce_rollout_hidden` to accept `tuple[int, ...]`. Compute hidden updates once per layer, then loop only over the small resolution tuple when aggregating. Add `progress_resolution` to all grouping keys in:

```python
group_columns = [
    "question_id", "representation", "progress_resolution", "layer", "progress_bin"
]
```

The same key must be used by `score_cross_rollout_queries`, `build_pairwise_geometry`, `score_set_direction_from_geometry`, `aggregate_span_features`, `aggregate_cross_features`, and shard validation.

- [ ] **Step 4: Add a cross-resolution isolation test**

```python
def test_cross_rollout_never_matches_across_progress_resolutions() -> None:
    records = pd.concat([
        _synthetic_cross_records().assign(progress_resolution=10),
        _synthetic_cross_records().assign(
            progress_resolution=20,
            displacement=lambda frame: frame["displacement"].map(lambda value: -value),
        ),
    ], ignore_index=True)
    scores, _ = score_cross_rollout_queries(records)
    assert set(scores["progress_resolution"]) == {10, 20}
    assert scores.groupby("progress_resolution")["cross_set_direction"].mean().notna().all()
```

- [ ] **Step 5: Run regression tests and commit**

Run: `python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py Experiment/tests/test_extract_experiment0_hidden_dynamics.py Experiment/tests/test_long_success_trajectory_common.py -q`

Expected: all tests pass.

```bash
git add Experiment/scripts/experiment0_hidden_dynamics.py Experiment/scripts/extract_experiment0_hidden_dynamics_qwen3vl.py Experiment/tests/test_experiment0_hidden_dynamics.py Experiment/tests/test_extract_experiment0_hidden_dynamics.py
git commit -m "feat: add experiment zero progress resolutions"
```

### Task 2: Quantify Angle SNR Instead Of Reporting Reliability Alone

**Files:**
- Modify: `Experiment/scripts/experiment0_hidden_dynamics.py`
- Modify: `Experiment/scripts/extract_experiment0_hidden_dynamics_qwen3vl.py`
- Modify: `Experiment/tests/test_experiment0_hidden_dynamics.py`
- Modify: `Experiment/tests/test_extract_experiment0_hidden_dynamics.py`

**Interfaces:**
- Token rows add `token_turn_background_cos_mean`, `token_turn_contrast_mean`, and `token_turn_snr`.
- Span rows add `span_turn_background_cos_mean`, `span_turn_contrast_mean`, `span_turn_background_sd`, and `span_turn_snr`.
- The control pairs are deterministic non-adjacent displacement pairs, matched in count within the same question, rollout, representation, layer, and progress resolution.

- [ ] **Step 1: Write failing deterministic-background tests**

```python
def test_matched_nonadjacent_pairs_exclude_turn_neighbors() -> None:
    pairs = matched_nonadjacent_pairs(n_vectors=12, seed=9)
    assert len(pairs) == 10
    assert all(abs(left - right) > 1 for left, right in pairs)
    assert pairs == matched_nonadjacent_pairs(n_vectors=12, seed=9)


def test_angle_snr_is_positive_for_smooth_path() -> None:
    vectors = np.stack([
        np.array([1.0, 0.02 * index], dtype=np.float64)
        for index in range(12)
    ])
    result = angle_snr(vectors, seed=4)
    assert result["turn_cos_mean"] > result["background_cos_mean"]
    assert result["turn_contrast_mean"] > 0
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py -q`

Expected: missing `matched_nonadjacent_pairs` and `angle_snr`.

- [ ] **Step 3: Implement deterministic matched controls**

```python
def matched_nonadjacent_pairs(n_vectors: int, seed: int) -> list[tuple[int, int]]:
    if n_vectors < 4:
        return []
    rng = np.random.default_rng(seed)
    pairs = []
    for left in range(1, n_vectors - 1):
        candidates = np.flatnonzero(np.abs(np.arange(n_vectors) - left) > 1)
        right = int(rng.choice(candidates))
        pairs.append((left, right))
    return pairs


def angle_snr(vectors: np.ndarray, seed: int, eps: float = EPS) -> dict[str, float]:
    values = np.asarray(vectors, dtype=np.float64)
    turns = np.asarray([
        _cosine(values[index], values[index - 1])
        for index in range(1, len(values))
    ])
    background = np.asarray([
        _cosine(values[left], values[right])
        for left, right in matched_nonadjacent_pairs(len(values), seed)
    ])
    turns = turns[np.isfinite(turns)]
    background = background[np.isfinite(background)]
    if turns.size == 0 or background.size < 2:
        return {key: float("nan") for key in (
            "turn_cos_mean", "background_cos_mean", "background_cos_sd",
            "turn_contrast_mean", "turn_snr",
        )}
    contrast = float(turns.mean() - background.mean())
    background_sd = float(background.std(ddof=1))
    return {
        "turn_cos_mean": float(turns.mean()),
        "background_cos_mean": float(background.mean()),
        "background_cos_sd": background_sd,
        "turn_contrast_mean": contrast,
        "turn_snr": contrast / max(background_sd, eps),
    }
```

Use a stable seed derived from `question_id`, `rollout_id`, representation, layer, and resolution with SHA-256; never use Python's randomized `hash()`. Compute token controls on GPU before deleting token updates and span controls from pooled displacement arrays before dropping non-primary vectors.

- [ ] **Step 4: Preserve the existing interleaved reliability control**

Keep `span_turn_cos_split_a/b`, but report it precisely as interleaved non-overlapping-grid reliability: each half uses every other 64-token step, so windows within a half do not overlap for the primary 128/64 specification. Do not call token-level angle a primary metric.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py Experiment/tests/test_extract_experiment0_hidden_dynamics.py -q`

Expected: all tests pass, angle SNR is finite when at least four displacement vectors are available, and short spans produce `NaN` without failing extraction.

```bash
git add Experiment/scripts/experiment0_hidden_dynamics.py Experiment/scripts/extract_experiment0_hidden_dynamics_qwen3vl.py Experiment/tests/test_experiment0_hidden_dynamics.py Experiment/tests/test_extract_experiment0_hidden_dynamics.py
git commit -m "feat: add matched angle snr controls"
```

### Task 3: Add Within-Question And Grouped Nuisance-Adjusted Effects

**Files:**
- Modify: `Experiment/scripts/analyze_experiment0_hidden_dynamics.py`
- Modify: `Experiment/tests/test_analyze_experiment0_hidden_dynamics.py`

**Interfaces:**
- Effect table gains `analysis_scale` with values `raw`, `question_centered`, and `nuisance_residual`.
- Predictor table retains the frozen length gate and adds `nuisance_only_auc`, `feature_plus_nuisance_auc`, and paired deltas.
- Main tables default to `progress_resolution=10`; resolution 20 is separately labeled inspection-only.

- [ ] **Step 1: Write failing centering and leakage tests**

```python
def test_question_centering_removes_question_offsets() -> None:
    frame = pd.DataFrame({
        "question_id": ["q1", "q1", "q2", "q2"],
        "feature": [101.0, 99.0, -49.0, -51.0],
    })
    centered = add_question_centered_feature(frame, "feature")
    assert centered.groupby("question_id")["feature_question_centered"].mean().abs().max() < 1e-12


def test_grouped_oof_residual_never_trains_on_test_question() -> None:
    residual, audit = grouped_oof_residual(
        _nuisance_frame(), target="feature", controls=["think_length", "token_count"], seed=7
    )
    assert residual.notna().all()
    assert audit["fold_question_overlap"].max() == 0
```

- [ ] **Step 2: Run the analyzer tests and confirm failure**

Run: `python -m pytest Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q`

Expected: missing centering and residualization helpers.

- [ ] **Step 3: Implement grouped out-of-fold nuisance residuals**

```python
from sklearn.linear_model import Ridge


def grouped_oof_residual(
    frame: pd.DataFrame,
    target: str,
    controls: list[str],
    seed: int,
) -> tuple[pd.Series, pd.DataFrame]:
    view = frame[["question_id", target, *controls]].dropna().copy()
    prediction = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    audit_rows = []
    splitter = GroupKFold(n_splits=min(5, view["question_id"].nunique()))
    for fold_id, (train, test) in enumerate(
        splitter.split(view, groups=view["question_id"])
    ):
        train_q = set(view.iloc[train]["question_id"])
        test_q = set(view.iloc[test]["question_id"])
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(view.iloc[train][controls], view.iloc[train][target])
        prediction.loc[view.iloc[test].index] = model.predict(view.iloc[test][controls])
        audit_rows.append({
            "fold_id": fold_id,
            "fold_question_overlap": len(train_q.intersection(test_q)),
        })
    return frame[target] - prediction, pd.DataFrame(audit_rows)
```

Use control maps that do not regress a feature on itself:

```python
TOKEN_CONTROLS = ["log_think_length", "token_count", "hidden_norm_mean"]
SPAN_CONTROLS = ["log_think_length", "span_count"]
ANGLE_EXTRA_CONTROLS = ["span_movement_norm_mean"]
```

For token angle/entropy features, include horizontal or vertical update magnitude when available. For span angle and cross-direction features, include `span_movement_norm_mean`. For movement-norm and cross-length features, omit movement norm because it is the target or part of the target definition. Record the exact controls in a `control_columns` string in every output row.

- [ ] **Step 4: Extend predictive comparisons without changing the frozen length gate**

For identical eligible rows and identical outer folds, compute:

```text
feature-only
length-only
feature+length
nuisance-only
feature+nuisance
feature+length+nuisance
```

The existing Experiment 0 gate remains:

```python
passes_length_gate = (
    delta_feature_vs_length_low > 0
    and delta_combined_vs_length_low > 0
)
```

Add a stricter descriptive flag, not a replacement gate:

```python
survives_nuisance = delta_feature_nuisance_vs_nuisance_low > 0
```

- [ ] **Step 5: Make effect inference question-balanced**

Calculate one correct-minus-wrong effect per question first, then average question effects and bootstrap questions. Keep pooled Hedges' `g` only as a descriptive magnitude column. This prevents questions with eight usable rollouts or more valid spans from receiving more inferential weight than questions with six.

- [ ] **Step 6: Run analyzer tests and commit**

Run: `python -m pytest Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q`

Expected: all tests pass; every fold has zero question overlap; raw/centered/residual scales are present; the same row IDs and folds are used for all predictor blocks.

```bash
git add Experiment/scripts/analyze_experiment0_hidden_dynamics.py Experiment/tests/test_analyze_experiment0_hidden_dynamics.py
git commit -m "feat: control experiment zero nuisance effects"
```

### Task 4: Replace The Legacy Atlas With The Approved V2 Views

**Files:**
- Modify: `Experiment/scripts/analyze_experiment0_hidden_dynamics.py`
- Modify: `Experiment/tests/test_analyze_experiment0_hidden_dynamics.py`

**Interfaces:**
- Produces v2 E0-F1 through E0-F7 figures without overwriting the 2026-07-22 directory.
- Report explicitly separates main `B=10`, inspection `B=20`, legacy discovery, and post-selection caveats.

- [ ] **Step 1: Add a failing artifact and content test**

```python
def test_v2_report_declares_resolution_and_adjustment_scales(tmp_path, monkeypatch) -> None:
    run_v2_analyzer_on_synthetic_shards(tmp_path, monkeypatch)
    report = (tmp_path / "results" / "LONG_EXPERIMENT_0_V2_RESULTS.md").read_text("utf-8")
    assert "B=10 main" in report
    assert "B=20 inspection-only" in report
    assert "question-centered" in report
    assert "grouped out-of-fold nuisance residual" in report
    assert "legacy 2026-07-22 results were not overwritten" in report
```

- [ ] **Step 2: Implement the seven fixed views**

```text
E0-F1  B10 horizontal norm x span-turn scatter/hexbin, raw and Q-centered panels
E0-F2  B10 cross-length x set-direction, prototype overlay only where kappa gate is valid
E0-F3  B10 vertical entropy x vertical norm/layer-turn, raw and nuisance-residual panels
E0-F4  B10 layer x progress question-balanced effects; B20 is a separate inspection panel
E0-F5  token, span-last, and three span-mean specifications: angle contrast, SNR, reliability, coverage
E0-F6  <2k / 2-4k / 4-8k / 8k+ effects with question counts and usable-span counts
E0-F7  feature, length, combined, nuisance, and feature+nuisance held-out pairwise AUC
```

Scatter plots must show both raw and question-centered values. Heatmaps use a shared symmetric color bound within a figure. Every panel states question coverage, not only row count. `E0-F5` must plot effect/SNR and reliability together; a reliability-only bar chart is insufficient.

- [ ] **Step 3: Fix permutation reporting**

Only `cross_set_direction` and `cross_length_support` at L24/L36 have reconstructed-geometry permutation p-values. Other features should display `NA (not permuted)`, not numeric `nan`. Resolution is part of every permutation key:

```python
key_columns = [
    "feature", "representation", "progress_resolution", "layer", "progress_bin"
]
```

- [ ] **Step 4: Write complete provenance metadata**

`analysis_meta.json` must include model ID, manifest SHA-256, git commit, source JSONL path and SHA-256, question IDs, audit question IDs, all span specifications, progress resolutions, primary resolution/representation, kappa thresholds, bootstrap/permutation counts, exact nuisance control maps, and `new_generation=false`.

- [ ] **Step 5: Run the end-to-end synthetic test and commit**

Run: `python -m pytest Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q`

Expected: all v2 tables, report, metadata, and seven figures exist; no matplotlib figures remain open.

```bash
git add Experiment/scripts/analyze_experiment0_hidden_dynamics.py Experiment/tests/test_analyze_experiment0_hidden_dynamics.py
git commit -m "feat: report approved experiment zero atlas"
```

### Task 5: Add A Non-Overwriting V2 Tmux Launcher

**Files:**
- Create: `Experiment/scripts/launch_experiment0_v2_tmux.sh`
- Modify: `Experiment/tests/test_extract_experiment0_hidden_dynamics.py`

**Interfaces:**
- Accepts `QUESTION_LIMIT`, `RUN_NAME`, `SESSION`, `BOOTSTRAP`, `PERMUTATIONS`, and `AUDIT_QUESTION_COUNT`.
- Runs `--progress-bins 10,20` and writes only beneath `experiment0_hidden_dynamics_v2/${RUN_NAME}`.

- [ ] **Step 1: Add a launcher text test**

```python
def test_v2_launcher_is_non_overwriting_and_multiresolution() -> None:
    script = (ROOT / "scripts" / "launch_experiment0_v2_tmux.sh").read_text("utf-8")
    assert "experiment0_hidden_dynamics_v2" in script
    assert "--progress-bins 10,20" in script
    assert "nvidia-smi --query-compute-apps=pid" in script
    assert "tmux has-session" in script
    assert "PIPELINE_COMPLETE" in script
```

- [ ] **Step 2: Implement the strict preflight and pipeline**

```bash
set -euo pipefail
ROOT="${ROOT:-/data2/hjk/projects/AI-HiddenState-ER}"
RUN_DIR="${ROOT}/experiment0_hidden_dynamics_v2/${RUN_NAME}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
  echo "GPU has active compute processes; refusing to launch" >&2
  exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "run directory already exists: ${RUN_DIR}" >&2
  exit 3
fi
```

The pipeline freezes the manifest first, writes `AUDIT_QUESTIONS.json` before model loading, extracts with `--progress-bins 10,20 --span-specs 64:32,128:64,256:128 --endpoint-spec 128:64 --primary-spec 128:64`, analyzes only after all completion markers validate, and prints `PIPELINE_COMPLETE` only after every v2 artifact exists.

- [ ] **Step 3: Run all local Experiment 0 tests and commit**

Run: `python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py Experiment/tests/test_extract_experiment0_hidden_dynamics.py Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q`

Expected: all tests pass.

```bash
git add Experiment/scripts/launch_experiment0_v2_tmux.sh Experiment/tests/test_extract_experiment0_hidden_dynamics.py
git commit -m "exp: add experiment zero v2 launcher"
```

### Task 6: H200 Smoke, Formal Refresh, And Local Intake

**Files:**
- Create on H200: `experiment0_hidden_dynamics_v2/smoke8_20260724/`
- Create on H200: `experiment0_hidden_dynamics_v2/formal_discovery24_20260724/`
- Copy back: `Experiment/server_results/experiment0_hidden_dynamics_v2_smoke8_20260724/`
- Copy back: `Experiment/server_results/experiment0_hidden_dynamics_v2_formal_discovery24_20260724/`
- Modify after results exist: `Experiment/2dimension.md`

- [ ] **Step 1: Synchronize only the v2 implementation files**

Run:

```bash
scp Experiment/scripts/experiment0_hidden_dynamics.py h200:/data2/hjk/projects/AI-HiddenState-ER/scripts/
scp Experiment/scripts/extract_experiment0_hidden_dynamics_qwen3vl.py h200:/data2/hjk/projects/AI-HiddenState-ER/scripts/
scp Experiment/scripts/analyze_experiment0_hidden_dynamics.py h200:/data2/hjk/projects/AI-HiddenState-ER/scripts/
scp Experiment/scripts/launch_experiment0_v2_tmux.sh h200:/data2/hjk/projects/AI-HiddenState-ER/scripts/
```

Then inspect without mutation:

```bash
ssh h200 "tmux ls; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
```

Expected: existing sessions are unchanged. Launch only when no compute PID is present.

- [ ] **Step 2: Launch and validate the eight-question smoke**

```bash
ssh h200 "cd /data2/hjk/projects/AI-HiddenState-ER && SESSION=two_dim_e0_v2_smoke8 RUN_NAME=smoke8_20260724 QUESTION_LIMIT=8 BOOTSTRAP=200 PERMUTATIONS=20 bash scripts/launch_experiment0_v2_tmux.sh"
ssh h200 "tmux capture-pane -pt two_dim_e0_v2_smoke8:0 -S -120"
```

Expected: 8 completed question markers; both resolutions present; 37 hidden-state indexes; finite norm/entropy fields; finite SNR when coverage permits; no traceback, CUDA OOM, or overwrite of the legacy directories.

- [ ] **Step 3: Copy smoke results locally and inspect every figure**

```bash
scp -r h200:/data2/hjk/projects/AI-HiddenState-ER/experiment0_hidden_dynamics_v2/smoke8_20260724/results Experiment/server_results/experiment0_hidden_dynamics_v2_smoke8_20260724
```

Open E0-F1 through E0-F7 locally and verify titles, labels, coverage text, B10/B20 separation, and no overlapping or clipped annotations.

- [ ] **Step 4: Launch the formal 24-question refresh**

```bash
ssh h200 "cd /data2/hjk/projects/AI-HiddenState-ER && SESSION=two_dim_e0_v2_formal24 RUN_NAME=formal_discovery24_20260724 QUESTION_LIMIT=24 RUN_LABEL=FormalDiscovery24V2 BOOTSTRAP=1000 PERMUTATIONS=100 bash scripts/launch_experiment0_v2_tmux.sh"
```

Expected: the job runs in a new tmux session and leaves all earlier experiments untouched.

- [ ] **Step 5: Validate formal invariants before copying results**

Run:

```bash
ssh h200 "grep -R 'PIPELINE_COMPLETE' logs/two_dim_e0_v2_formal_discovery24_20260724.log"
ssh h200 "/data2/hjk/envs/hs_er/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
root = Path('/data2/hjk/projects/AI-HiddenState-ER/experiment0_hidden_dynamics_v2/formal_discovery24_20260724/results')
frame = pd.read_parquet(root / 'long_experiment_0_bin_features.parquet')
assert set(frame.progress_resolution.unique()) == {10, 20}
assert set(frame.layer.unique()) == set(range(37))
assert frame.question_id.nunique() == 24
print(frame.shape, frame.question_id.nunique(), sorted(frame.progress_resolution.unique()))
PY"
```

Expected: `PIPELINE_COMPLETE`, 24 questions, 37 hidden-state indexes, resolutions 10 and 20.

- [ ] **Step 6: Copy formal results and update the design document with facts only**

```bash
scp -r h200:/data2/hjk/projects/AI-HiddenState-ER/experiment0_hidden_dynamics_v2/formal_discovery24_20260724/results Experiment/server_results/experiment0_hidden_dynamics_v2_formal_discovery24_20260724
```

Update `Experiment/2dimension.md` with the v2 run path, manifest hash, actual cohort counts, completion status, and a link to `LONG_EXPERIMENT_0_V2_RESULTS.md`. Do not promote a selected layer/bin into a confirmatory claim; the run remains discovery-only.

- [ ] **Step 7: Run the final local regression and commit the intake**

Run: `python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py Experiment/tests/test_extract_experiment0_hidden_dynamics.py Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q`

Expected: all tests pass.

```bash
git add Experiment/2dimension.md Experiment/server_results/experiment0_hidden_dynamics_v2_smoke8_20260724 Experiment/server_results/experiment0_hidden_dynamics_v2_formal_discovery24_20260724
git commit -m "exp: complete approved experiment zero refresh"
```

## Final Review Checklist

- [ ] No existing 2026-07-22 result file changed.
- [ ] No new rollout was generated.
- [ ] `progress_resolution` is part of every key and permutation group.
- [ ] B10 is always labeled main and B20 inspection-only.
- [ ] Token amplitude/entropy and span direction remain separate streams.
- [ ] Angle claims include matched-background contrast, SNR, reliability, and coverage.
- [ ] Cross-rollout permutations rebuild balanced reference sets from scalar geometry.
- [ ] Raw, question-centered, and nuisance-residual results are all present.
- [ ] All confidence intervals and paired comparisons resample questions.
- [ ] H200 launch occurs only on an idle GPU and in a new tmux session.
