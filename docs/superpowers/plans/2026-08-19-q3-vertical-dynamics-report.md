# Qwen3-1.7B Vertical Dynamics Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify one self-contained Chinese HTML presentation that summarizes the completed Qwen3-1.7B GRPO ability, vertical dynamics, local AUROC, horizontal context, and stopping-candidate evidence.

**Architecture:** A single deterministic Python builder reads the completed H200 result root, validates all formal gates, derives compact question-equal summaries, renders summary figures with Matplotlib, embeds selected source figures and generated figures as base64, and atomically writes one offline HTML file. Focused tests load the builder as a module and validate formulas, audit behavior, mappings, and HTML structure before the builder is synchronized to H200.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, PyArrow, Matplotlib, pytest, HTML/CSS, tmux/SSH, Playwright.

## Global Constraints

- Output is a Chinese group-meeting/advisor report, not a dashboard or all-figure gallery.
- Final delivery is one offline HTML with no external font, script, style, image, or network dependency.
- Do not modify training, extraction, metric, checkpoint, or formal result inputs.
- Candidate vertical panel remains V1/V3/V4/V4C/V8; do not select a primary metric.
- Use all 11 checkpoints `0,25,50,75,100,125,150,175,200,225,250`, 256 fixed questions, and 8 rollouts per question.
- Behavior intervals use 4,000 paired question-bootstrap draws with seed `20260818`.
- H2/H5 online and small held-out probes must be labeled separately from the 256-question formal vertical cohort.
- Do not claim causal mediation, a validated universal stopping rule, or justification for H2/H5 reward shaping.
- Cards have radius at most 7 px; sections are unframed; no gradients, decorative blobs, nested cards, or viewport-scaled fonts.
- Builder output writes atomically and fails closed on missing/failed audits, checkpoints, metrics, or source figures.

---

### Task 1: Implement audited calculations

**Files:**

- Create: `Experiment_2E/scripts/build_q3_vertical_dynamics_report.py`
- Create: `Experiment_2E/tests/test_build_q3_vertical_dynamics_report.py`

**Interfaces:**

- Consumes completed result paths and scalar/formal profile artifacts.
- Produces `validate_result_root(root: Path) -> tuple[dict, dict]`.
- Produces `pass_at_k_from_count(correct: int, total: int, k: int) -> float`.
- Produces `paired_bootstrap(values: np.ndarray, *, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.
- Produces `load_behavior(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]`.
- Produces `load_vertical_dynamics(root: Path) -> pd.DataFrame`.
- Produces `load_auc_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]`.
- Produces `load_horizontal_summary(root: Path) -> pd.DataFrame`.

- [x] **Step 1: Write formula and audit tests.**

```python
def test_pass_at_k_uses_roll8_combinatorics():
    assert report.pass_at_k_from_count(0, 8, 4) == 0.0
    assert report.pass_at_k_from_count(8, 8, 4) == 1.0
    assert report.pass_at_k_from_count(1, 8, 1) == 0.125
    assert report.pass_at_k_from_count(1, 8, 4) == 0.5


def test_validate_result_root_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_status"):
        report.validate_result_root(tmp_path)


def test_paired_bootstrap_is_deterministic():
    values = np.arange(24, dtype=float).reshape(3, 8)
    first = report.paired_bootstrap(values, draws=50, seed=20260818)
    second = report.paired_bootstrap(values, draws=50, seed=20260818)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
```

- [x] **Step 2: Run focused tests and confirm failure.**

Run: `python -m pytest Experiment_2E/tests/test_build_q3_vertical_dynamics_report.py -q`

Expected: FAIL because the builder and functions do not exist.

- [x] **Step 3: Implement constants, result-root validation, behavior formulas, and deterministic bootstrap.**

```python
EXPECTED_STEPS = (0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250)
QUESTIONS = 256
ROLLOUTS = 8
BOOTSTRAP_DRAWS = 4000
SEED = 20260818

def pass_at_k_from_count(correct: int, total: int, k: int) -> float:
    if not 0 <= correct <= total or not 1 <= k <= total:
        raise ValueError("invalid correct/total/k")
    failures = total - correct
    missed = math.comb(failures, k) / math.comb(total, k) if failures >= k else 0.0
    return float(1.0 - missed)
```

`validate_result_root` must read `run_status.json` and `analysis_audit.json`, require completed/passed status, exact steps, exact question counts, and all listed Parquet/JSONL/source-figure inputs. Error messages include the missing path or failed key.

- [x] **Step 4: Implement question-equal behavior and vertical loaders.**

`load_behavior` groups each generation JSONL in manifest order into 256 groups of 8, computes `correct_count`, pass@1/4/8, checkpoint means, paired CIs, adjacent differences, final-minus-base, step50-to-final, and step175-to-final.

`load_vertical_dynamics` reads local profile means from `policy_profiles.parquet`, maps V4C from `weighted_layer_update_turning_angle`, maps V8 raw/centered variants, and returns one row per metric/representation/checkpoint with:

```text
metric_family metric_variant representation global_step level_mean
cumulative_relative_l1 adjacent_relative_l1 update_cosine
```

Different stage/layer profile spaces remain separate; local metrics use `stage x layer`, V4C/V8 use `stage` only.

- [x] **Step 5: Implement AUROC and horizontal loaders.**

`load_auc_summary` computes checkpoint median/P90/max separability and a late-step cell table for step175-250. `load_horizontal_summary` reads only final-layer H2/H5 B4 online values, aggregates 25-step blocks, and records the online cohort label and mixed-question coverage.

- [x] **Step 6: Run focused tests.**

Run: `python -m pytest Experiment_2E/tests/test_build_q3_vertical_dynamics_report.py -q`

Expected: PASS for formulas, deterministic bootstrap, fail-closed audit, question-equal aggregation, V4C mapping, V8 mapping, and temporal ordering.

- [x] **Step 7: Commit the audited calculation layer.**

```bash
git add Experiment_2E/scripts/build_q3_vertical_dynamics_report.py Experiment_2E/tests/test_build_q3_vertical_dynamics_report.py
git commit -m "feat: derive Q3 vertical report evidence"
```

### Task 2: Render figures and the self-contained report

**Files:**

- Modify: `Experiment_2E/scripts/build_q3_vertical_dynamics_report.py`
- Modify: `Experiment_2E/tests/test_build_q3_vertical_dynamics_report.py`

**Interfaces:**

- Consumes audited summary DataFrames from Task 1.
- Produces `render_summary_figures(data: dict[str, pd.DataFrame]) -> dict[str, bytes]`.
- Produces `encode_png(data: bytes) -> str`.
- Produces `render_html(data: dict[str, Any], images: dict[str, bytes]) -> str`.
- Produces `build_report(result_root: Path, output: Path) -> dict[str, Any]`.

- [x] **Step 1: Write figure and HTML contract tests.**

```python
def test_encode_png_returns_embeddable_data_uri():
    encoded = report.encode_png(b"\x89PNG\r\n\x1a\nfixture")
    assert encoded.startswith("data:image/png;base64,")


def test_render_html_contains_required_sections(fake_report_data, fake_pngs):
    html = report.render_html(fake_report_data, fake_pngs)
    for section in (
        "overview", "ability", "vertical", "depth", "auroc",
        "horizontal", "stopping", "audit",
    ):
        assert f'id="{section}"' in html
    assert "data:image/png;base64," in html
    assert "39.65%" in html
    assert "63.33%" in html
    assert "step175–200" in html
    assert "H2" in html and "reward shaping" in html
```

- [x] **Step 2: Run the focused HTML tests and confirm failure.**

Run: `python -m pytest Experiment_2E/tests/test_build_q3_vertical_dynamics_report.py -q`

Expected: FAIL on missing renderer functions.

- [x] **Step 3: Render six deterministic summary figures.**

Generate Matplotlib PNG bytes with fixed dimensions and no transparent
background:

```text
ability_passk.png
ability_delta_pass1.png
vertical_cumulative.png
vertical_adjacent.png
auc_distribution.png
horizontal_h2_h5.png
```

Use charcoal, green, blue, amber, vermilion, and violet as distinct semantic
colors. Never overlay raw V1, angle metrics, and ER on one raw-unit axis; the
combined vertical plots use dimensionless relative changes only.

- [x] **Step 4: Embed four layer-location and two AUROC source figures.**

Require and base64-embed these exact formal assets:

```text
policy_heatmap__v1_raw_update_norm__mean_w128_s32__B4.png
policy_heatmap__v3_demean_state_angle__mean_w128_s32__B4.png
policy_heatmap__v4_layer_update_turning_angle__mean_w128_s32__B2.png
selected_profiles__v3_demean_state_angle__mean_w128_s32__B4.png
auc_heatmap__v3_demean_state_angle__mean_w128_s32__B4.png
auc_heatmap__v1_relative_update_norm__mean_w128_s32__B4.png
```

- [x] **Step 5: Implement the eight-section Chinese HTML.**

The renderer includes compact sticky navigation, result-stat blocks, phase
timeline, responsive two-column figure grids, accessible captions/alt text,
scrollable checkpoint tables, the stopping matrix, limitations, and inline
provenance. CSS contains no gradients, external URLs, nested cards, or
viewport-dependent font sizes.

- [x] **Step 6: Implement atomic build orchestration and CLI.**

```text
python Experiment_2E/scripts/build_q3_vertical_dynamics_report.py \
  --result-root /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814 \
  --output /data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814/Q3_VERTICAL_DYNAMICS_REPORT.html
```

The builder writes a temporary sibling, checks file size, required sections,
embedded-image count, and image payload sizes, then atomically renames it. It
prints a JSON build audit with output path, size, image count, checkpoint count,
source audit status, builder version, and seed.

- [x] **Step 7: Run focused and neighboring tests.**

Run:

```text
python -m pytest Experiment_2E/tests/test_build_q3_vertical_dynamics_report.py Experiment_2E/tests/test_q3_layer_analysis.py Experiment_2E/tests/test_grpo3b_behavior_eval.py -q
```

Expected: all tests PASS.

- [x] **Step 8: Commit rendering and report assembly.**

```bash
git add Experiment_2E/scripts/build_q3_vertical_dynamics_report.py Experiment_2E/tests/test_build_q3_vertical_dynamics_report.py
git commit -m "feat: build Q3 vertical dynamics HTML report"
```

### Task 3: Build on H200 and download the artifact

**Files:**

- Read on H200: completed formal result root.
- Create on H200: `Q3_VERTICAL_DYNAMICS_REPORT.html` under the formal result root.
- Create locally: `Experiment_2E/server_results/q3_1p7b_vertical_dynamics_formal_20260818/Q3_VERTICAL_DYNAMICS_REPORT.html`.

- [x] **Step 1: Synchronize only the tested builder to H200.**

Run:

```text
scp Experiment_2E/scripts/build_q3_vertical_dynamics_report.py \
  h200:/data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/
```

Expected: transfer succeeds without touching training or metric files.

- [x] **Step 2: Run syntax and focused calculation checks on H200.**

Run:

```text
ssh h200 "/data2/hjk/envs/verl_qwen3vl_py311/bin/python -m py_compile \
  /data2/hjk/projects/Experiment_2E_q3_1p7b_20260814/scripts/build_q3_vertical_dynamics_report.py"
```

Expected: exit code 0.

- [x] **Step 3: Build the report in a detached tmux session.**

Use session `q3_vertical_report_build`. The launcher command must refuse an
existing session, start detached, write a dedicated log, and leave all formal
inputs read-only.

- [x] **Step 4: Monitor until tmux exits naturally and inspect the build audit.**

Expected: output HTML exists, is larger than 1 MB, has at least 12 embedded PNG
data URIs, and the log ends with a passing JSON build audit.

- [x] **Step 5: Download the HTML to its local final path.**

Run:

```text
scp h200:/data2/hjk/results/experiment_2e/q3_1p7b_base_simplerl_grpo_formal_seed20260814/Q3_VERTICAL_DYNAMICS_REPORT.html \
  Experiment_2E/server_results/q3_1p7b_vertical_dynamics_formal_20260818/
```

Expected: local SHA256 matches the H200 SHA256.

### Task 4: Visual and content verification

**Files:**

- Verify: `Experiment_2E/server_results/q3_1p7b_vertical_dynamics_formal_20260818/Q3_VERTICAL_DYNAMICS_REPORT.html`
- Create temporary screenshots outside the committed result directory.

- [x] **Step 1: Run static content checks.**

Verify required section IDs, no external `http://` or `https://` references,
all data URIs decode, at least 12 images exist, no image payload is below 10 KB,
and the report includes the audited headline values and limitations.

- [x] **Step 2: Start a local static server for browser verification.**

Run on a free localhost port and keep it active until all Playwright checks are
complete.

- [x] **Step 3: Verify desktop rendering at 1440x1000.**

Capture full-page and first-viewport screenshots. Check nonblank pixels,
navigation links, image dimensions, `scrollWidth == clientWidth`, and zero
console/page errors.

- [x] **Step 4: Verify mobile rendering at 390x844.**

Repeat overflow, overlap, image, navigation, and console checks. Confirm long
metric names wrap and tables scroll inside their containers without expanding
the document.

- [x] **Step 5: Correct any visual defects and rebuild/reverify.**

Changes remain scoped to the builder/test and final HTML. Do not alter source
metrics or hand-edit only the generated HTML.

- [x] **Step 6: Run final tests and record artifact facts.**

Report local path, byte size, SHA256, embedded image count, tested viewports,
test counts, and any remaining scientific limitations.

- [x] **Step 7: Commit the verified final artifact.**

```bash
git add Experiment_2E/server_results/q3_1p7b_vertical_dynamics_formal_20260818/Q3_VERTICAL_DYNAMICS_REPORT.html
git commit -m "docs: add Q3 vertical dynamics formal report"
```
