# Vertical Toolkit Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `vertical-toolkit` branch that lets collaborators inspect heterogeneous Qwen3-8B and MiMo-7B hidden-state artifacts, compute audited layer-resolved vertical metrics offline, and run formal jobs in tmux without depending on the current Qwen3-1.7B GRPO pipeline.

**Architecture:** A pure-Python package separates source adapters, canonical records, response pooling, Base calibration, vertical profile formulas, statistical summaries, plotting, and audit/restart logic. A manifest-driven configuration selects raw-token or pre-pooled input adapters; all downstream code consumes one canonical record interface. Shell wrappers provide smoke and detached tmux formal execution while keeping real data and outputs outside Git.

**Tech Stack:** Python 3.11; NumPy; pandas; PyArrow; SciPy; Matplotlib; PyYAML; pytest; POSIX shell; tmux. The normal workflow is CPU-only and does not load model weights or run model forward.

## Global Constraints

- The handoff branch is named `vertical-toolkit` and starts from `master`.
- The first release must complete the v0.1 stable scope: V1/V3/V4/V6/V7, audits, outcome summaries, AUROC, depth summaries, figures, and tmux runners.
- Raw and pooled hidden-state inputs are both supported; ambiguous axes or response boundaries fail closed.
- `h_0` is embedding output and `h_1..h_L` are decoder outputs; `relative_depth = layer_index / L`.
- MiMo MTP/non-decoder states are excluded from the main decoder profile and reported separately when auditable.
- Base calibrators are label-blind and fitted separately by model family and representation; MiMo Base calibrators are reused for MiMo SFT.
- Question-equal aggregation is the default; unsupported AUROC or paired analyses report a reason instead of fabricating values.
- The package never commits raw hidden states, model weights, pooled caches, real-data outputs, private absolute paths, or tmux logs.
- Formal execution is launched in an explicitly named detached tmux session and is blocked until smoke approval exists.
- Every persistent write is atomic and every formal output has a machine-readable audit.

---

## File Map

The implementation creates the following focused units:

```text
vertical_readme.md
vertical/
  __init__.py
  schema.py
  config.py
  inspect.py
  pooling.py
  calibrate.py
  profiles.py
  depth.py
  statistics.py
  plotting.py
  audit.py
  cli.py
  adapters/
    __init__.py
    base.py
    npz.py
    numpy_directory.py
    custom_template.py
configs/
  qwen3_8b.example.yaml
  mimo_7b.example.yaml
scripts/
  inspect_vertical_data.sh
  run_vertical_smoke.sh
  run_vertical_formal.sh
  launch_vertical_formal_tmux.sh
examples/
  synthetic_hidden/
tests/
  test_scaffold.py
  test_schema.py
  test_config.py
  test_adapters.py
  test_inspect.py
  test_pooling.py
  test_calibrate.py
  test_profiles.py
  test_depth.py
  test_statistics.py
  test_plotting.py
  test_audit.py
  test_cli.py
  test_vertical_e2e.py
requirements-vertical.txt
```

The existing `Experiment_2E/` implementation is a reference for formulas and
audits only. New `vertical/` modules must not import from `Experiment_2E`.

## Implementation Tasks

### Task 1: Create the package scaffold and dependency contract

**Files:**
- Create: `vertical/__init__.py`
- Create: `vertical/py.typed`
- Create: `requirements-vertical.txt`
- Create: `configs/qwen3_8b.example.yaml`
- Create: `configs/mimo_7b.example.yaml`
- Create: `tests/conftest.py`
- Create: `tests/test_scaffold.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces an importable `vertical` package from repository root with no
  dependency on `Experiment_2E`.
- Config examples expose `input_mode`, `adapter`, `source`, `output_root`,
  `model_family`, `condition`, `decoder_layers`, `hidden_dimension`,
  `representations`, and `question_equal`.

- [ ] **Step 1: Write the import and dependency tests.**

```python
def test_vertical_package_imports_without_experiment_2e():
    import vertical

    assert vertical.__version__


def test_config_examples_are_present():
    from pathlib import Path

    assert Path("configs/qwen3_8b.example.yaml").is_file()
    assert Path("configs/mimo_7b.example.yaml").is_file()
```

- [ ] **Step 2: Run the focused tests and confirm the scaffold is absent.**

Run: `python -m pytest tests/test_scaffold.py -q`

Expected: FAIL because the package and config loader do not exist yet.

- [ ] **Step 3: Add the minimal package, requirements, config examples, and ignores.**

`requirements-vertical.txt` must pin the minimum tested major versions:

```text
numpy>=1.26,<3
pandas>=2.2,<3
pyarrow>=16,<20
scipy>=1.12,<2
matplotlib>=3.8,<4
pyyaml>=6,<7
pytest>=8,<9
```

The package sets `__version__ = "0.1.0"`. The examples use relative source
paths such as `./data/qwen3_8b` and contain no server-specific paths.

- [ ] **Step 4: Run the focused tests and verify the ignore policy.**

Run: `python -m pytest tests/test_scaffold.py -q`

Expected: PASS. Also run `git check-ignore -v runs/example/raw_hidden.npz`
and verify that the rule matches the repository ignore file.

- [ ] **Step 5: Commit the scaffold.**

```bash
git add vertical requirements-vertical.txt configs tests/conftest.py tests/test_scaffold.py .gitignore
git commit -m "feat: scaffold vertical toolkit"
```

### Task 2: Define canonical metadata and payload schemas

**Files:**
- Create: `vertical/schema.py`
- Create: `vertical/config.py`
- Create: `tests/test_schema.py`
- Create: `tests/test_config.py`

**Interfaces:**
- `HiddenPayload(kind: Literal["raw", "pooled"], values: np.ndarray, representation: str | None, endpoints: np.ndarray | None, progress: np.ndarray | None, response_start: int | None, response_stop: int | None, layer_kind: np.ndarray | None)`.
- `RecordMetadata(record_id: str, model_family: str, model_name: str, condition: str, question_id: str, rollout_id: str, is_correct: bool | None, response_token_count: int, num_decoder_layers: int, hidden_state_count: int, hidden_dimension: int, source_path: str, checkpoint: str | None, global_step: int | None, training_progress: float | None, extras: Mapping[str, Any])`.
- `VerticalRecord(metadata: RecordMetadata, payloads: Mapping[str, HiddenPayload])`.
- `VerticalConfig.from_yaml(path: Path) -> VerticalConfig` and `VerticalConfig.validate() -> None`.

- [ ] **Step 1: Write schema validation tests.**

```python
def test_raw_payload_requires_token_axis_and_matching_metadata():
    payload = HiddenPayload(kind="raw", values=np.zeros((4, 5, 8), dtype=np.float32))
    metadata = make_metadata(hidden_state_count=5, hidden_dimension=8)
    record = VerticalRecord(metadata=metadata, payloads={"raw": payload})
    record.validate()


def test_invalid_layer_count_fails_closed():
    payload = HiddenPayload(kind="raw", values=np.zeros((4, 4, 8), dtype=np.float32))
    record = VerticalRecord(metadata=make_metadata(hidden_state_count=5, hidden_dimension=8), payloads={"raw": payload})
    with pytest.raises(ValueError, match="hidden_state_count"):
        record.validate()
```

- [ ] **Step 2: Run the tests to verify the interfaces are missing.**

Run: `python -m pytest tests/test_schema.py -q`

Expected: FAIL with missing schema symbols.

- [ ] **Step 3: Implement immutable metadata, payload, record, and YAML config validation.**

Validation must reject non-3D payloads, non-finite values when strict mode is
enabled, inconsistent layer/dimension metadata, empty identifiers, negative
response lengths, unknown input modes, and duplicate representation names.

- [ ] **Step 4: Run the focused tests.**

Run: `python -m pytest tests/test_schema.py tests/test_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit canonical schemas.**

```bash
git add vertical/schema.py vertical/config.py tests/test_schema.py tests/test_config.py
git commit -m "feat: define vertical record schemas"
```

### Task 3: Implement manifest-driven adapters

**Files:**
- Create: `vertical/adapters/__init__.py`
- Create: `vertical/adapters/base.py`
- Create: `vertical/adapters/npz.py`
- Create: `vertical/adapters/numpy_directory.py`
- Create: `vertical/adapters/custom_template.py`
- Create: `tests/test_adapters.py`

**Interfaces:**
- `AdapterInspection(adapter: str, input_mode: str, files: list[str], sample_shapes: list[tuple[int, ...]], dtypes: list[str], keys: list[str], warnings: list[str], errors: list[str])`.
- `class VerticalAdapter(Protocol): inspect(source: Path, config: VerticalConfig, sample_limit: int) -> AdapterInspection; iter_records(source: Path, config: VerticalConfig) -> Iterator[VerticalRecord]`.
- `get_adapter(name: str) -> VerticalAdapter`.
- NPZ adapter reads one record per `.npz` with configurable `tensor_key` and optional `metadata_json_key`.
- Numpy-directory adapter reads a manifest (`.jsonl` or `.parquet`) whose tensor path column points to `.npy` or `.npz` files.
- Custom-template adapter is a manifest-driven adapter with explicit tensor key, axis order, metadata column, and response-boundary mapping; it must reject missing mappings rather than guess.

- [ ] **Step 1: Add fixtures for one raw NPZ, one pooled NPZ, and one manifest-directory record.**

The fixtures use arrays with shape `(6, 5, 8)`, decoder layer count `4`, and
metadata fields `question_id`, `rollout_id`, `condition`, `is_correct`, and
`response_token_count`.

- [ ] **Step 2: Write adapter tests.**

```python
def test_npz_adapter_normalizes_raw_axes(tmp_path):
    records = list(get_adapter("npz").iter_records(tmp_path, config))
    assert records[0].payloads["raw"].values.shape == (6, 5, 8)


def test_custom_adapter_rejects_unknown_axis_order(tmp_path):
    with pytest.raises(ValueError, match="axis_order"):
        list(get_adapter("custom_template").iter_records(tmp_path, bad_config))
```

- [ ] **Step 3: Run adapter tests to confirm they fail before implementation.**

Run: `python -m pytest tests/test_adapters.py -q`

Expected: FAIL with missing adapter registry or readers.

- [ ] **Step 4: Implement the three adapters and registry.**

Every adapter must preserve `source_path`, validate metadata against the
canonical schema, normalize declared axes to `[token_or_endpoint, layer, dim]`,
and stream records instead of loading the entire source directory.

- [ ] **Step 5: Run adapter tests and commit.**

Run: `python -m pytest tests/test_adapters.py -q`

Expected: PASS.

```bash
git add vertical/adapters tests/test_adapters.py examples/synthetic_hidden
git commit -m "feat: add manifest-driven hidden-state adapters"
```

### Task 4: Build the preflight inspector

**Files:**
- Create: `vertical/inspect.py`
- Create: `tests/test_inspect.py`

**Interfaces:**
- `inspect_source(source: Path, config: VerticalConfig, sample_limit: int = 16) -> AdapterInspection`.
- `write_input_audit(report: AdapterInspection, path: Path) -> None`.
- `inspect_source` must report file count, keys, shape/dtype samples, finite
  rates, layer count, hidden dimension, metadata coverage, duplicate IDs,
  response-boundary status, raw/pooled mode, and MTP candidates.

- [ ] **Step 1: Write tests for successful inspection and fail-closed ambiguity.**

```python
def test_inspector_reports_shape_and_metadata_coverage(raw_fixture, tmp_path):
    report = inspect_source(raw_fixture, config, sample_limit=4)
    assert report.errors == []
    assert report.sample_shapes == [(6, 5, 8)]
    assert report.metadata_coverage["question_id"] == 1.0


def test_inspector_rejects_missing_response_boundary(ambiguous_fixture):
    report = inspect_source(ambiguous_fixture, config, sample_limit=4)
    assert any("response" in error for error in report.errors)
```

- [ ] **Step 2: Run inspector tests and confirm missing implementation.**

Run: `python -m pytest tests/test_inspect.py -q`

Expected: FAIL with missing inspector functions.

- [ ] **Step 3: Implement bounded sampling and JSON audit serialization.**

The inspector must never materialize all tensors. It exits with a nonzero CLI
status when `report.errors` is nonempty and includes a deterministic SHA256 of
the sampled manifest paths.

- [ ] **Step 4: Run tests and commit the inspector.**

Run: `python -m pytest tests/test_inspect.py -q`

Expected: PASS.

```bash
git add vertical/inspect.py tests/test_inspect.py
git commit -m "feat: audit vertical hidden-state inputs"
```

### Task 5: Implement response pooling and B1-B4 stages

**Files:**
- Create: `vertical/pooling.py`
- Create: `tests/test_pooling.py`

**Interfaces:**
- `trajectory_endpoints(response_length: int, window: int = 128, stride: int = 32) -> np.ndarray`.
- `pool_response_states(response_states: np.ndarray, endpoints: np.ndarray, window: int = 128) -> dict[str, np.ndarray]`.
- `progress_from_endpoints(endpoints: np.ndarray, response_length: int) -> np.ndarray`.
- `stage_mask(progress: np.ndarray, stage: int) -> np.ndarray` with B1 `[0,.25)`, B2 `[.25,.5)`, B3 `[.5,.75)`, B4 `[.75,1]`.

- [ ] **Step 1: Write endpoint, pooling, and stage tests.**

```python
def test_pooling_matches_trailing_window_definition():
    tokens = np.arange(10 * 2 * 3, dtype=np.float32).reshape(10, 2, 3)
    endpoints = np.array([2, 5, 10])
    pooled = pool_response_states(tokens, endpoints, window=4)
    assert np.allclose(pooled["last_s32"][1], tokens[4])
    assert np.allclose(pooled["mean_w128_s32"][1], tokens[1:5].mean(axis=0))


def test_final_stage_includes_progress_one():
    assert stage_mask(np.array([0.25, 0.50, 0.75, 1.0]), 3).tolist() == [False, False, False, True]
```

- [ ] **Step 2: Run the tests and verify missing pooling symbols.**

Run: `python -m pytest tests/test_pooling.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement endpoint generation with no empty windows, trailing-window pooling, progress, and stage masks.**

All outputs preserve layer and hidden dimensions, use float32 for computation,
and reject endpoints outside `[1, response_length]`.

- [ ] **Step 4: Run focused tests and commit.**

Run: `python -m pytest tests/test_pooling.py -q`

Expected: PASS.

```bash
git add vertical/pooling.py tests/test_pooling.py
git commit -m "feat: pool response hidden states by progress stage"
```

### Task 6: Port the stable local vertical metrics

**Files:**
- Create: `vertical/profiles.py`
- Create: `tests/test_profiles.py`

**Interfaces:**
- `local_profile_arrays(layers: np.ndarray, base_common: np.ndarray) -> dict[str, np.ndarray]`.
- `reduce_stage_profiles(trajectory: np.ndarray, progress: np.ndarray, representation: str, metadata: Mapping[str, Any], base_common: np.ndarray) -> pd.DataFrame`.
- `PROFILE_COLUMNS = ("v1_raw_update_norm", "v1_relative_update_norm", "v3_demean_state_angle", "v4_layer_update_turning_angle", "v6_raw_activation_entropy", "v7_layer_difference_entropy")`.

- [ ] **Step 1: Write formula and layer-semantics tests.**

```python
def test_local_profile_structural_nan_positions():
    layers = np.array([[1., 0.], [1., 1.], [2., 1.], [2., 3.]])
    result = local_profile_arrays(layers, np.zeros_like(layers))
    assert np.isnan(result["v1_raw_update_norm"][0])
    assert np.isnan(result["v4_layer_update_turning_angle"][:2]).all()
    assert np.allclose(result["v1_raw_update_norm"][1:], [1., 1., 2.])


def test_stage_reduction_preserves_relative_depth_and_coverage():
    frame = reduce_stage_profiles(trajectory, progress, "last_s32", metadata, common)
    assert frame["relative_depth"].min() == 0.0
    assert frame["profile_chunk_count"].min() >= 0
```

- [ ] **Step 2: Run profile tests to verify missing implementation.**

Run: `python -m pytest tests/test_profiles.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement the six local arrays and stage-median reducer.**

Use adjacent updates `h_l - h_(l-1)`, destination-layer alignment, epsilon and
norm floors, `NaN` for structural undefined positions, and explicit per-metric
finite coverage counts. The formulas must not import `Experiment_2E`.

- [ ] **Step 4: Run focused tests and commit.**

Run: `python -m pytest tests/test_profiles.py -q`

Expected: PASS.

```bash
git add vertical/profiles.py tests/test_profiles.py
git commit -m "feat: compute stable layer-local vertical profiles"
```

### Task 7: Add Base calibrators and standardized comparisons

**Files:**
- Create: `vertical/calibrate.py`
- Create: `tests/test_calibrate.py`

**Interfaces:**
- `fit_base_calibrator(records: Iterable[VerticalRecord], representation: str) -> BaseCalibrator`.
- `save_calibrator(calibrator: BaseCalibrator, path: Path) -> None`.
- `load_calibrator(path: Path, representation: str) -> BaseCalibrator`.
- `BaseCalibrator.common`, `.coordinate_mean`, `.coordinate_sigma`, `.model_family`, `.representation`, `.input_sha256`.
- `standardize_layers(layers: np.ndarray, calibrator: BaseCalibrator, sigma_floor: float = 1e-4, z_clip: float = 8.0) -> np.ndarray`.

- [ ] **Step 1: Write tests proving label blindness and family isolation.**

```python
def test_base_calibrator_does_not_change_when_labels_are_permuted(base_records):
    first = fit_base_calibrator(base_records, "last_s32")
    permuted = permute_labels(base_records)
    second = fit_base_calibrator(permuted, "last_s32")
    assert np.allclose(first.common, second.common)
    assert np.allclose(first.coordinate_sigma, second.coordinate_sigma)


def test_mimo_calibrator_rejects_qwen_shape():
    with pytest.raises(ValueError, match="model_family"):
        standardize_layers(qwen_layers, mimo_calibrator)
```

- [ ] **Step 2: Run calibration tests and verify missing symbols.**

Run: `python -m pytest tests/test_calibrate.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement rollout-equal common fitting, chunk-equal coordinate statistics, atomic serialization, and family/representation checks.**

Fitting must use Base records only and reject an SFT-only calibration cohort.
The MiMo Base calibrator is reusable for MiMo SFT, while Qwen and MiMo
calibrators cannot be mixed.

- [ ] **Step 4: Run tests and commit.**

Run: `python -m pytest tests/test_calibrate.py -q`

Expected: PASS.

```bash
git add vertical/calibrate.py tests/test_calibrate.py
git commit -m "feat: add label-blind Base calibration"
```

### Task 8: Materialize profile partitions and output audits

**Files:**
- Create: `vertical/audit.py`
- Create: `tests/test_audit.py`
- Modify: `vertical/profiles.py`

**Interfaces:**
- `write_profile_partition(records: Iterable[VerticalRecord], calibrators: Mapping[str, BaseCalibrator], output_path: Path) -> ProfileAudit`.
- `ProfileAudit.passed`, `.expected_rows`, `.actual_rows`, `.finite_coverage`, `.structural_nan_checks`, `.input_hashes`.
- `atomic_parquet(frame: pd.DataFrame, path: Path) -> None`.
- `write_json_atomic(value: Mapping[str, Any], path: Path) -> None`.
- `load_completed_partition(path: Path, audit_path: Path) -> pd.DataFrame`.

- [ ] **Step 1: Write partition and atomic-write tests.**

```python
def test_profile_partition_has_one_row_per_record_rep_stage_layer(records, tmp_path):
    audit = write_profile_partition(records, calibrators, tmp_path / "profiles.parquet")
    frame = pd.read_parquet(tmp_path / "profiles.parquet")
    assert audit.passed
    assert frame.groupby(["record_id", "representation", "stage", "layer_index"]).size().max() == 1


def test_partial_parquet_is_not_accepted(tmp_path):
    with pytest.raises(ValueError, match="audit"):
        load_completed_partition(tmp_path / "partial.parquet", tmp_path / "missing_audit.json")
```

- [ ] **Step 2: Run audit tests and verify missing implementation.**

Run: `python -m pytest tests/test_audit.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement streaming reduction, atomic Parquet/JSON writes, row-count gates, structural-NaN gates, and input/config hashes.**

The partition writer must never retain all hidden states in memory. It may
buffer bounded profile rows before writing a temporary Parquet file and
atomically replacing the final path.

- [ ] **Step 4: Run focused tests and commit.**

Run: `python -m pytest tests/test_audit.py tests/test_profiles.py -q`

Expected: PASS.

```bash
git add vertical/audit.py vertical/profiles.py tests/test_audit.py
git commit -m "feat: write audited vertical profile partitions"
```

### Task 9: Implement outcome summaries, AUROC, and depth summaries

**Files:**
- Create: `vertical/statistics.py`
- Create: `vertical/depth.py`
- Create: `tests/test_statistics.py`
- Create: `tests/test_depth.py`

**Interfaces:**
- `summarize_policy_profiles(frame: pd.DataFrame, metric: str) -> pd.DataFrame`.
- `summarize_outcome_profiles(frame: pd.DataFrame, metric: str) -> pd.DataFrame`.
- `within_question_auc(frame: pd.DataFrame, score_column: str) -> pd.DataFrame`.
- `summarize_layer_auc(frame: pd.DataFrame, metric: str) -> pd.DataFrame`.
- `summarize_depth_change(policy: pd.DataFrame, metric: str, base_condition: str = "base") -> pd.DataFrame`.
- `analyze_condition_delta(policy: pd.DataFrame, base_condition: str, target_condition: str) -> pd.DataFrame`.

- [ ] **Step 1: Write tests for question-equal weighting and unsupported analyses.**

```python
def test_policy_summary_weights_questions_equally(profile_frame):
    result = summarize_policy_profiles(profile_frame, "v1_raw_update_norm")
    assert result.loc[result["layer_index"] == 1, "n_questions"].iat[0] == 2
    assert result.loc[result["layer_index"] == 1, "profile_mean"].iat[0] == pytest.approx(2.0)


def test_auc_reports_no_mixed_questions_without_fabricating_values(single_outcome_frame):
    result = summarize_layer_auc(single_outcome_frame, "v1_raw_update_norm")
    assert result["n_mixed_questions"].eq(0).all()
    assert result["question_equal_auc"].isna().all()
```

- [ ] **Step 2: Run tests and verify missing analysis functions.**

Run: `python -m pytest tests/test_statistics.py tests/test_depth.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement question-first aggregation, within-question AUROC, Base-vs-condition depth mass, and paired alignment checks.**

The depth summary computes absolute profile change relative to the Base
condition, then peak depth, center, spread, early/middle/late mass, and
transition depth. All summaries retain counts and unsupported reasons.

- [ ] **Step 4: Run focused tests and commit.**

Run: `python -m pytest tests/test_statistics.py tests/test_depth.py -q`

Expected: PASS.

```bash
git add vertical/statistics.py vertical/depth.py tests/test_statistics.py tests/test_depth.py
git commit -m "feat: summarize vertical outcomes and depth"
```

### Task 10: Add v0.1 figures and analysis artifacts

**Files:**
- Create: `vertical/plotting.py`
- Create: `tests/test_plotting.py`

**Interfaces:**
- `plot_policy_heatmap(frame: pd.DataFrame, metric: str, path: Path) -> None`.
- `plot_selected_profiles(frame: pd.DataFrame, metric: str, selected_conditions: Sequence[str], path: Path) -> None`.
- `plot_correct_minus_wrong(frame: pd.DataFrame, metric: str, path: Path) -> None`.
- `plot_auc_heatmap(frame: pd.DataFrame, metric: str, path: Path) -> None`.
- `render_v01_report(profiles_path: Path, output_dir: Path, metrics: Sequence[str]) -> AnalysisAudit`.

- [ ] **Step 1: Write tests for figure count, nonblank files, and NaN masking.**

```python
def test_v01_report_writes_nonblank_figures(profile_path, tmp_path):
    audit = render_v01_report(profile_path, tmp_path, ["v1_raw_update_norm"])
    assert audit.passed
    assert all(path.stat().st_size > 10_000 for path in tmp_path.glob("figures/*.png"))
```

- [ ] **Step 2: Run plotting tests and confirm missing implementation.**

Run: `python -m pytest tests/test_plotting.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement four figure families and analysis audit.**

Plots use relative depth on the y-axis or x-axis consistently, mask
structural NaNs rather than converting them to zero, and include condition,
representation, stage, and metric in filenames. The report writes policy,
outcome, AUROC, and depth Parquet files before figures.

- [ ] **Step 4: Run tests and commit v0.1 plotting.**

Run: `python -m pytest tests/test_plotting.py -q`

Expected: PASS.

```bash
git add vertical/plotting.py tests/test_plotting.py
git commit -m "feat: render audited vertical profile figures"
```

### Task 11: Add configuration CLI and tmux runners

**Files:**
- Create: `vertical/cli.py`
- Create: `scripts/inspect_vertical_data.sh`
- Create: `scripts/run_vertical_smoke.sh`
- Create: `scripts/run_vertical_formal.sh`
- Create: `scripts/launch_vertical_formal_tmux.sh`
- Create: `tests/test_cli.py`
- Modify: `.gitignore`

**Interfaces:**
- `python -m vertical.cli inspect --config CONFIG --output AUDIT_JSON`.
- `python -m vertical.cli smoke --config CONFIG --output-root RUN_ROOT`.
- `python -m vertical.cli formal --config CONFIG --output-root RUN_ROOT --smoke-approval APPROVAL_JSON`.
- `launch_vertical_formal_tmux.sh` creates a detached session named
  `vertical_formal_<run_id>` and refuses an existing target session or missing
  smoke approval.
- `render_tmux_command(session_name: str, runner: Path) -> str` returns the
  fully quoted detached-session command used by the launcher.

- [ ] **Step 1: Write CLI and launcher guard tests.**

```python
def test_formal_cli_requires_smoke_approval(tmp_path):
    result = run_cli(["formal", "--config", str(config), "--output-root", str(tmp_path)])
    assert result.returncode != 0
    assert "smoke approval" in result.stderr.lower()


def test_tmux_launcher_uses_named_detached_session():
    command = render_tmux_command("vertical_formal_mimo", Path("scripts/run_vertical_formal.sh"))
    assert "tmux new-session -d -s vertical_formal_mimo" in command
```

- [ ] **Step 2: Run CLI tests and verify the command surface is absent.**

Run: `python -m pytest tests/test_cli.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement CLI dispatch and POSIX wrappers.**

The shell scripts resolve the repository root, enable `set -euo pipefail`,
write a persistent log, and pass all paths through the YAML config or explicit
arguments. The formal runner checks `smoke_approval.json`, writes transitions
to `run_status.json`, and uses atomic outputs. It does not invoke a model or
GPU process.

- [ ] **Step 4: Run CLI, shell syntax, and launcher tests.**

Run: `python -m pytest tests/test_cli.py -q`

Expected: PASS.

Run: `bash -n scripts/inspect_vertical_data.sh scripts/run_vertical_smoke.sh scripts/run_vertical_formal.sh scripts/launch_vertical_formal_tmux.sh`

Expected: exit code 0.

- [ ] **Step 5: Commit the runnable v0.1 pipeline.**

```bash
git add vertical/cli.py scripts tests/test_cli.py .gitignore
git commit -m "feat: add vertical smoke and tmux runners"
```

### Task 12: Write the collaborator tutorial and synthetic quickstart

**Files:**
- Create: `vertical_readme.md`
- Create: `examples/synthetic_hidden/README.md`
- Create: `examples/synthetic_hidden/make_fixture.py`
- Create: `tests/test_readme.py`
- Modify: `configs/qwen3_8b.example.yaml`
- Modify: `configs/mimo_7b.example.yaml`

**Interfaces:**
- The README must contain copyable commands for install, inspect, smoke,
  formal tmux launch, monitoring, resume, and safe artifact sharing.
- The synthetic fixture command must create raw NPZ and pooled NPZ examples
  that pass the same inspector and v0.1 smoke path.

- [ ] **Step 1: Write a documentation test that checks required tutorial sections.**

```python
def test_readme_contains_required_runbook_sections():
    text = Path("vertical_readme.md").read_text(encoding="utf-8")
    for heading in ("Preflight", "Smoke", "tmux", "Base calibration", "Troubleshooting", "Do not push"):
        assert heading.lower() in text.lower()
```

- [ ] **Step 2: Run the documentation test and confirm the tutorial is absent.**

Run: `python -m pytest tests/test_readme.py -q`

Expected: FAIL because `vertical_readme.md` does not exist yet.

- [ ] **Step 3: Write the tutorial and synthetic fixture generator.**

The tutorial must explain raw versus pooled inputs, response-only semantics,
embedding/decoder indexing, MTP exclusion, Base/SFT alignment, supported
claims, output paths, and the exact tmux commands. It must never instruct a
collaborator to commit raw tensors or private absolute paths.

- [ ] **Step 4: Run the documentation test and synthetic quickstart.**

Run: `python -m pytest tests/test_readme.py -q`

Expected: PASS.

Run: `python examples/synthetic_hidden/make_fixture.py --output examples/synthetic_hidden/generated`

Expected: the fixture directory contains both raw and pooled files and no
single fixture file exceeds 1 MiB.

- [ ] **Step 5: Commit the v0.1 handoff documentation.**

```bash
git add vertical_readme.md examples/synthetic_hidden configs
git commit -m "docs: add vertical toolkit collaborator runbook"
```

### Task 13: Add the full v0.1 end-to-end smoke test and release gate

**Files:**
- Create: `tests/test_vertical_e2e.py`
- Modify: `vertical/audit.py`
- Modify: `vertical/cli.py`

**Interfaces:**
- `run_v01_pipeline(config_path: Path, output_root: Path) -> Mapping[str, Any]`.
- `finalize_run(audits: Sequence[Path], output_root: Path) -> Mapping[str, Any]`.

- [ ] **Step 1: Write the end-to-end test against the synthetic fixture.**

```python
def test_v01_pipeline_produces_audited_profiles_figures_and_status(tmp_path):
    result = run_v01_pipeline(synthetic_config, tmp_path / "run")
    assert result["passed"] is True
    assert (tmp_path / "run/audit/final_audit.json").is_file()
    assert list((tmp_path / "run/figures").glob("*.png"))
    assert not list((tmp_path / "run").glob("**/*hidden*.npz"))
```

- [ ] **Step 2: Run the test and verify the complete pipeline is not wired.**

Run: `python -m pytest tests/test_vertical_e2e.py -q`

Expected: FAIL.

- [ ] **Step 3: Wire inspect, adapter stream, pooling, calibration, profile writing, summaries, plots, and final audit.**

The pipeline must preserve one run manifest, use only the synthetic or
configured source, and mark status `completed` only after every audit gate
passes.

- [ ] **Step 4: Run the full v0.1 test suite and shell checks.**

Run: `python -m pytest tests -q`

Expected: PASS.

Run: `bash -n scripts/*.sh`

Expected: exit code 0.

- [ ] **Step 5: Commit the v0.1 release gate.**

```bash
git add vertical tests/test_vertical_e2e.py
git commit -m "test: gate vertical toolkit v0.1 end to end"
```

### Task 14: Add v0.2 metric-complete vertical companions

**Files:**
- Modify: `vertical/profiles.py`
- Modify: `vertical/statistics.py`
- Modify: `vertical/depth.py`
- Modify: `vertical/plotting.py`
- Create: `tests/test_extended_vertical_metrics.py`

**Interfaces:**
- `v2_raw_state_angle(layers: np.ndarray) -> np.ndarray`.
- `rolling_vertical_geometry(layers: np.ndarray, window_updates: int = 4) -> pd.DataFrame`.
- `cumulative_vertical_geometry(layers: np.ndarray) -> pd.DataFrame`.
- `vertical_effective_rank(layers: np.ndarray, centered: bool) -> float`.
- `vertical_effective_degree(layers: np.ndarray, degree: int = 3) -> Mapping[str, float]`.

- [ ] **Step 1: Write tests for V2 and four-update rolling alignment.**

```python
def test_v2_aligns_adjacent_state_angles_to_destination_layer(layers):
    values = v2_raw_state_angle(layers)
    assert np.isnan(values[0])
    assert len(values) == layers.shape[0]


def test_rolling_window_never_connects_nonadjacent_updates(layers):
    result = rolling_vertical_geometry(layers, window_updates=4)
    assert result["window_updates"].eq(4).all()
    assert result["end_layer"].is_monotonic_increasing
```

- [ ] **Step 2: Run the extended tests and verify missing functions.**

Run: `python -m pytest tests/test_extended_vertical_metrics.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement V2, rolling/cumulative V5, V8, and V9 with coverage and invalid-window records.**

Rolling uses exactly four consecutive layer updates as the primary window.
Cumulative geometry starts at `h_0` and ends at the current layer. Effective
rank uses singular-value entropy with the existing relative tolerance; degree
fits use normalized depth coordinates and return fit coverage, linear fraction,
high-order fraction, RMSE, and condition number.

- [ ] **Step 4: Add figures and output columns for the extended metrics.**

Extended outputs are additive to v0.1 files and carry a metric registry status
of `experimental` until the full smoke and regression suite passes.

- [ ] **Step 5: Run tests and commit v0.2.**

Run: `python -m pytest tests -q`

Expected: PASS.

```bash
git add vertical tests/test_extended_vertical_metrics.py
git commit -m "feat: add extended vertical geometry metrics"
```

### Task 15: Add v0.3 inferential analysis

**Files:**
- Modify: `vertical/statistics.py`
- Modify: `vertical/depth.py`
- Modify: `vertical/plotting.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- `question_bootstrap(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float], n_boot: int, seed: int) -> pd.DataFrame`.
- `cluster_permutation(values: np.ndarray, labels: np.ndarray, n_permutations: int, seed: int) -> pd.DataFrame`.
- `fit_nuisance_controlled_effects(frame: pd.DataFrame, score: str, controls: Sequence[str]) -> pd.DataFrame`.
- `paired_condition_delta(frame: pd.DataFrame, base_condition: str, target_condition: str, keys: Sequence[str]) -> pd.DataFrame`.

- [ ] **Step 1: Write deterministic tests for bootstrap, cluster runs, and paired alignment.**

```python
def test_question_bootstrap_is_reproducible(profile_frame):
    first = question_bootstrap(profile_frame, mean_score, n_boot=100, seed=7)
    second = question_bootstrap(profile_frame, mean_score, n_boot=100, seed=7)
    assert first.equals(second)


def test_paired_delta_rejects_duplicate_condition_keys(duplicated_pair_frame):
    with pytest.raises(ValueError, match="one-to-one"):
        paired_condition_delta(duplicated_pair_frame, "base", "sft", ["question_id", "rollout_id"])
```

- [ ] **Step 2: Run inference tests and verify missing implementations.**

Run: `python -m pytest tests/test_inference.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement question-resampling confidence intervals and reproducible cluster permutation.**

Resampling units are question IDs, not individual rollout rows. Cluster runs
are contiguous relative-depth positions with a test statistic threshold and a
max-cluster correction. Every result records `n_questions`, `n_permutations`,
seed, and correction method.

- [ ] **Step 4: Implement nuisance-controlled and paired Base/SFT summaries.**

Controls are used only when present and finite. Paired comparisons require
one-to-one keys and emit an unavailable audit reason otherwise. Cross-model
results use relative depth and within-family Base standardization; no raw vector
subtraction is introduced.

- [ ] **Step 5: Run the full suite and commit v0.3.**

Run: `python -m pytest tests -q`

Expected: PASS.

```bash
git add vertical tests/test_inference.py
git commit -m "feat: add inferential vertical analyses"
```

### Task 16: Final documentation, security, and release validation

**Files:**
- Modify: `vertical_readme.md`
- Modify: `requirements-vertical.txt`
- Modify: `.gitignore`
- Create: `tests/test_release_gate.py`

**Interfaces:**
- `run_release_gate(repo_root: Path) -> Mapping[str, Any]`.
- The release gate reports tests, shell syntax, fixture size, tracked-file
  safety, README command coverage, and metric registry statuses.

- [ ] **Step 1: Write release-gate tests.**

```python
def test_release_gate_rejects_tracked_real_tensor(tmp_path):
    (tmp_path / "runs/raw_hidden.npz").parent.mkdir(parents=True)
    (tmp_path / "runs/raw_hidden.npz").write_bytes(b"tensor")
    result = run_release_gate(tmp_path)
    assert result["tracked_data_gate"] is False
```

- [ ] **Step 2: Run release tests and verify the gate is incomplete.**

Run: `python -m pytest tests/test_release_gate.py -q`

Expected: FAIL.

- [ ] **Step 3: Update the README with v0.2/v0.3 commands, metric status labels, and returned-artifact examples.**

Document the exact commands:

```bash
bash scripts/inspect_vertical_data.sh --config configs/mimo_7b.example.yaml
bash scripts/run_vertical_smoke.sh --config configs/mimo_7b.example.yaml
bash scripts/launch_vertical_formal_tmux.sh --config configs/mimo_7b.example.yaml
tmux ls
tmux attach -t vertical_formal_<run_id>
```

The README must distinguish Base-only descriptive analysis, MiMo Base/SFT
condition analysis, and claims that require multiple checkpoints.

- [ ] **Step 4: Run the complete release gate.**

Run: `python -m pytest tests -q`

Expected: PASS.

Run: `bash -n scripts/*.sh`

Expected: exit code 0.

Run: `git diff --check` and `git status --short --ignored`

Expected: no whitespace errors; only intended source/docs are tracked and
real-data paths are ignored.

- [ ] **Step 5: Commit the release documentation and gate.**

```bash
git add vertical_readme.md requirements-vertical.txt .gitignore tests/test_release_gate.py
git commit -m "chore: validate vertical toolkit release"
```

## Plan Self-Review

The plan covers every spec section:

- branch and collaboration boundary: Tasks 1, 11, 12, and 16;
- raw/pooled adapters and input audit: Tasks 2-4;
- frozen pooling, stages, layer semantics, MTP handling: Tasks 5-6;
- Base calibrator rules: Task 7;
- v0.1 metrics and output audits: Tasks 6-10;
- tmux smoke/formal workflow: Tasks 11 and 13;
- v0.2 metrics: Task 14;
- v0.3 inference and paired analyses: Task 15;
- Git/data safety and collaborator tutorial: Tasks 12 and 16;
- testing and acceptance gates: Tasks 1-4, 8-13, and 16.

The plan contains no unresolved implementation gaps. Every public
function referenced by a later task is defined in an earlier task or the same
task, and every task ends with a focused test command and a commit boundary.
