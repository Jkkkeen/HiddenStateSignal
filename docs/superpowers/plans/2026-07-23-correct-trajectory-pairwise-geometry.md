# Correct-Trajectory Pairwise Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an offline analysis that tests whether same-question correct-correct hidden movements are closer than wrong-wrong and correct-wrong movements in angle, relative vector distance, and amplitude.

**Architecture:** Put deterministic geometry reduction and question-level inference in a focused pure-Python module, then expose a separate CLI for schema validation, artifact writing, figures, and the Markdown report. Reduce the 195,280 directed span matches to one observation per undirected rollout pair before bootstrap or permutation so span density never becomes the independent unit.

**Tech Stack:** Python 3, NumPy, pandas, matplotlib, pytest, parquet/pyarrow.

## Global Constraints

- Input is frozen to `long_experiment_0_pairwise_geometry.parquet` from FormalDiscovery24.
- Analyze only `mean_w128_s64` and hidden-state indexes 24 and 36.
- Use existing 10 progress bins and existing nearest-progress reference matches.
- Exclude self-pairs and rows whose query/reference norm is non-finite or at most `1e-12`; report exclusion coverage.
- Treat question as the independent unit; never bootstrap raw pair rows.
- Do not connect H200, generate responses, run model forward passes, or modify existing Experiment 0 artifacts.
- Use seed `20260723`, 4,000 question bootstraps, and 1,000 within-question label permutations.

---

### Task 1: Geometry Metrics And Undirected Pair Reduction

**Files:**
- Create: `Experiment/scripts/correct_trajectory_pairwise_geometry.py`
- Create: `Experiment/tests/test_correct_trajectory_pairwise_geometry.py`

**Interfaces:**
- Consumes: a pandas `DataFrame` with the frozen pairwise-geometry schema.
- Produces: `compute_pair_metrics(frame, eps)`, `aggregate_directed_pairs(frame)`, `symmetrize_pairs(frame)`, `pair_type_means(frame, whole_trajectory)`.

- [ ] **Step 1: Write failing metric and exclusion tests**

```python
def test_compute_pair_metrics_recovers_known_geometry() -> None:
    frame = pd.DataFrame({
        "displacement_norm": [1.0, 1.0, 1.0],
        "reference_norm": [1.0, 1.0, 1.0],
        "cosine_similarity": [1.0, 0.0, -1.0],
    })
    result, exclusions = compute_pair_metrics(frame)
    np.testing.assert_allclose(result["angle_rad"], [0.0, np.pi / 2, np.pi])
    np.testing.assert_allclose(result["delta_vec"], [0.0, np.sqrt(2.0), 2.0])
    np.testing.assert_allclose(result["delta_rel"], [0.0, np.sqrt(2.0) / 2.0, 1.0])
    np.testing.assert_allclose(result["delta_amp"], 0.0)
    assert exclusions["excluded_rows"] == 0


def test_compute_pair_metrics_excludes_undefined_directions() -> None:
    frame = pd.DataFrame({
        "displacement_norm": [1.0, 0.0, np.nan],
        "reference_norm": [1.0, 1.0, 1.0],
        "cosine_similarity": [1.0, 0.0, 0.0],
    })
    result, exclusions = compute_pair_metrics(frame)
    assert len(result) == 1
    assert exclusions == {"input_rows": 3, "excluded_rows": 2, "valid_rows": 1}
```

- [ ] **Step 2: Run the metric tests and verify failure**

Run: `python -m pytest Experiment/tests/test_correct_trajectory_pairwise_geometry.py -v`

Expected: collection fails because `correct_trajectory_pairwise_geometry` does not exist.

- [ ] **Step 3: Implement schema constants and exact recovered metrics**

```python
EPS = 1e-12
METRICS = ("angle_rad", "angle_deg", "delta_vec", "delta_rel", "delta_amp")


def compute_pair_metrics(frame: pd.DataFrame, eps: float = EPS) -> tuple[pd.DataFrame, dict[str, int]]:
    result = frame.copy()
    left = result["displacement_norm"].to_numpy(dtype=np.float64)
    right = result["reference_norm"].to_numpy(dtype=np.float64)
    cosine = result["cosine_similarity"].to_numpy(dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right) & np.isfinite(cosine) & (left > eps) & (right > eps)
    result = result.loc[valid].copy()
    left, right, cosine = left[valid], right[valid], np.clip(cosine[valid], -1.0, 1.0)
    result["angle_rad"] = np.arccos(cosine)
    result["angle_deg"] = np.degrees(result["angle_rad"])
    radicand = np.maximum(left * left + right * right - 2.0 * left * right * cosine, 0.0)
    result["delta_vec"] = np.sqrt(radicand)
    result["delta_rel"] = result["delta_vec"].to_numpy() / (left + right + eps)
    result["delta_amp"] = np.abs(np.log(left + eps) - np.log(right + eps))
    return result, {
        "input_rows": int(len(frame)),
        "excluded_rows": int((~valid).sum()),
        "valid_rows": int(valid.sum()),
    }
```

- [ ] **Step 4: Write failing reduction and weighting tests**

```python
def test_directed_matches_are_symmetrized_once() -> None:
    raw = synthetic_geometry_two_directions()
    metrics, _ = compute_pair_metrics(raw)
    directed = aggregate_directed_pairs(metrics)
    pairs = symmetrize_pairs(directed)
    assert len(directed) == 2
    assert len(pairs) == 1
    assert pairs.iloc[0]["pair_type"] == "++"
    assert pairs.iloc[0]["angle_rad"] == pytest.approx(directed["angle_rad"].mean())


def test_question_pair_type_means_do_not_weight_pair_count() -> None:
    pairs = synthetic_unequal_pair_counts()
    means = pair_type_means(pairs, whole_trajectory=False)
    assert means.groupby(["question_id", "layer", "progress_bin", "pair_type"]).size().eq(1).all()
```

- [ ] **Step 5: Implement directed aggregation, canonicalization, and pair types**

```python
def aggregate_directed_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["question_id", "layer", "progress_bin", "rollout_id", "reference_rollout_id",
            "is_correct", "reference_is_correct"]
    return frame.groupby(keys, as_index=False, observed=True)[list(METRICS)].median()


def symmetrize_pairs(directed: pd.DataFrame) -> pd.DataFrame:
    view = directed.copy()
    view["pair_lo"] = view[["rollout_id", "reference_rollout_id"]].min(axis=1).astype(int)
    view["pair_hi"] = view[["rollout_id", "reference_rollout_id"]].max(axis=1).astype(int)
    view["pair_type"] = np.where(
        view["is_correct"] & view["reference_is_correct"], "++",
        np.where(~view["is_correct"] & ~view["reference_is_correct"], "--", "+-"),
    )
    keys = ["question_id", "layer", "progress_bin", "pair_lo", "pair_hi", "pair_type"]
    return view.groupby(keys, as_index=False, observed=True)[list(METRICS)].mean()


def pair_type_means(pairs: pd.DataFrame, whole_trajectory: bool) -> pd.DataFrame:
    view = pairs.copy()
    if whole_trajectory:
        pair_keys = ["question_id", "layer", "pair_lo", "pair_hi", "pair_type"]
        view = view.groupby(pair_keys, as_index=False, observed=True)[list(METRICS)].mean()
        group_keys = ["question_id", "layer", "pair_type"]
    else:
        group_keys = ["question_id", "layer", "progress_bin", "pair_type"]
    return view.groupby(group_keys, as_index=False, observed=True)[list(METRICS)].mean()
```

- [ ] **Step 6: Run Task 1 tests**

Run: `python -m pytest Experiment/tests/test_correct_trajectory_pairwise_geometry.py -v`

Expected: all geometry and reduction tests pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add Experiment/scripts/correct_trajectory_pairwise_geometry.py Experiment/tests/test_correct_trajectory_pairwise_geometry.py
git commit -m "feat: reduce correct trajectory pair geometry"
```

### Task 2: Question-Level Contrasts, Bootstrap, And Label Permutation

**Files:**
- Modify: `Experiment/scripts/correct_trajectory_pairwise_geometry.py`
- Modify: `Experiment/tests/test_correct_trajectory_pairwise_geometry.py`

**Interfaces:**
- Consumes: undirected rollout-pair rows and question-level pair-type means from Task 1.
- Produces: `question_contrasts(means)`, `directed_four_cell_interactions(directed)`, `summarize_contrasts(contrasts, bootstrap, seed)`, `permutation_inference(pairs, labels, permutations, seed)`.

- [ ] **Step 1: Write failing contrast tests**

```python
def test_question_contrasts_have_expected_negative_sign() -> None:
    means = pd.DataFrame({
        "question_id": ["q1"] * 3,
        "layer": [24] * 3,
        "progress_bin": [5] * 3,
        "pair_type": ["++", "--", "+-"],
        "angle_rad": [0.2, 0.8, 1.0],
    })
    result = question_contrasts(means, metrics=("angle_rad",))
    values = dict(zip(result["contrast"], result["value"]))
    assert values["pp_minus_mm"] == pytest.approx(-0.6)
    assert values["pp_minus_pm"] == pytest.approx(-0.8)


def test_missing_pair_type_only_drops_required_contrast() -> None:
    means = synthetic_pair_type_means_without_mm()
    result = question_contrasts(means, metrics=("angle_rad",))
    assert set(result["contrast"]) == {"pp_minus_pm"}
```

- [ ] **Step 2: Implement long-form paired contrasts**

```python
def question_contrasts(means: pd.DataFrame, metrics: tuple[str, ...] = PRIMARY_METRICS) -> pd.DataFrame:
    id_cols = [column for column in ("question_id", "layer", "progress_bin") if column in means]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        pivot = means.pivot_table(index=id_cols, columns="pair_type", values=metric, aggfunc="first")
        for name, right in (("pp_minus_mm", "--"), ("pp_minus_pm", "+-")):
            if "++" not in pivot or right not in pivot:
                continue
            values = (pivot["++"] - pivot[right]).dropna()
            for key, value in values.items():
                key = key if isinstance(key, tuple) else (key,)
                rows.append({**dict(zip(id_cols, key)), "metric": metric, "contrast": name, "value": float(value)})
    return pd.DataFrame(rows)


def directed_four_cell_interactions(directed: pd.DataFrame) -> pd.DataFrame:
    view = directed.copy()
    view["cell"] = np.select(
        [view["is_correct"] & view["reference_is_correct"],
         view["is_correct"] & ~view["reference_is_correct"],
         ~view["is_correct"] & view["reference_is_correct"]],
        ["++", "+-", "-+"],
        default="--",
    )
    keys = ["question_id", "layer", "progress_bin", "cell"]
    cells = view.groupby(keys, as_index=False, observed=True)[list(PRIMARY_METRICS)].mean()
    rows = []
    for metric in PRIMARY_METRICS:
        pivot = cells.pivot_table(index=keys[:-1], columns="cell", values=metric, aggfunc="first")
        valid = pivot.dropna(subset=["++", "+-", "-+", "--"])
        interaction = (valid["++"] - valid["+-"]) - (valid["-+"] - valid["--"])
        for key, value in interaction.items():
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(keys[:-1], key)), "metric": metric, "interaction": float(value)})
    return pd.DataFrame(rows)
```

- [ ] **Step 3: Write failing deterministic bootstrap and permutation tests**

```python
def test_bootstrap_is_question_level_and_deterministic() -> None:
    contrasts = synthetic_question_contrasts()
    first = summarize_contrasts(contrasts, bootstrap=100, seed=7)
    second = summarize_contrasts(contrasts, bootstrap=100, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["n_questions"].eq(contrasts["question_id"].nunique()).all()


def test_permutation_preserves_label_counts_and_is_deterministic() -> None:
    pairs, labels = synthetic_labeled_pairs()
    first = permutation_inference(pairs, labels, permutations=20, seed=9)
    second = permutation_inference(pairs, labels, permutations=20, seed=9)
    pd.testing.assert_frame_equal(first.nulls, second.nulls)
    assert first.label_count_checks.all()
```

- [ ] **Step 4: Implement question bootstrap, sign consistency, and within-question relabeling**

```python
def summarize_contrasts(contrasts: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_cols = [column for column in ("metric", "contrast", "layer", "progress_bin") if column in contrasts]
    rows = []
    for keys, group in contrasts.groupby(group_cols, sort=True, observed=True):
        values = group.set_index("question_id")["value"].dropna()
        samples = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(bootstrap)])
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(group_cols, keys)), "mean_contrast": float(values.mean()),
                     "ci_low": float(np.quantile(samples, 0.025)),
                     "ci_high": float(np.quantile(samples, 0.975)),
                     "negative_sign_fraction": float((values < 0).mean()),
                     "n_questions": int(len(values))})
    return pd.DataFrame(rows)
```

Implement permutation with these concrete operations:

```python
@dataclass(frozen=True)
class PermutationResult:
    nulls: pd.DataFrame
    label_count_checks: np.ndarray


def permute_label_map(labels: pd.DataFrame, rng: np.random.Generator) -> tuple[dict[tuple[str, int], bool], bool]:
    mapping: dict[tuple[str, int], bool] = {}
    checks = []
    for question_id, group in labels.groupby("question_id", sort=True):
        before = group["is_correct"].to_numpy(dtype=bool)
        after = rng.permutation(before)
        checks.append(int(before.sum()) == int(after.sum()))
        mapping.update({(str(question_id), int(rid)): bool(label)
                        for rid, label in zip(group["rollout_id"], after)})
    return mapping, bool(all(checks))


def adjusted_max_stat_pvalues(observed: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    result = observed.copy()
    result["max_stat_p"] = np.nan
    families = ["metric", "contrast"]
    for keys, group in result.groupby(families, sort=True):
        null_family = nulls[(nulls["metric"] == keys[0]) & (nulls["contrast"] == keys[1])]
        maxima = null_family.groupby("permutation_id")["mean_contrast"].apply(lambda x: x.abs().max())
        for index in group.index:
            observed_abs = abs(float(result.at[index, "mean_contrast"]))
            result.at[index, "max_stat_p"] = (1 + int((maxima >= observed_abs).sum())) / (1 + len(maxima))
    return result
```

`permutation_inference` uses `permute_label_map`, assigns both endpoint labels to every undirected pair, rebuilds `pair_type`, recomputes progress and whole-trajectory question contrasts, and returns all null grid estimates plus one boolean label-count check per permutation.

- [ ] **Step 5: Run Task 2 tests**

Run: `python -m pytest Experiment/tests/test_correct_trajectory_pairwise_geometry.py -v`

Expected: all metric, reduction, bootstrap, and permutation tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add Experiment/scripts/correct_trajectory_pairwise_geometry.py Experiment/tests/test_correct_trajectory_pairwise_geometry.py
git commit -m "feat: test correct trajectory cohesion"
```

### Task 3: CLI, Figures, Report, And Formal Local Run

**Files:**
- Create: `Experiment/scripts/analyze_correct_trajectory_pairwise_geometry.py`
- Create: `Experiment/tests/test_analyze_correct_trajectory_pairwise_geometry.py`
- Generate: `Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722/correct_trajectory_pairwise_geometry/*`

**Interfaces:**
- Consumes: frozen FormalDiscovery24 parquet and Task 1/2 functions.
- Produces: validated parquet/CSV artifacts, four figures, `analysis_meta.json`, and `CORRECT_TRAJECTORY_PAIRWISE_GEOMETRY_REPORT.md`.

- [ ] **Step 1: Write failing CLI integration test**

Create a synthetic 3-question, 3-correct/3-wrong geometry parquet with layers 24/36 and bins 0/1. Call `main()` with `--bootstrap 20 --permutations 10`, then assert these artifacts exist and are non-empty:

```python
EXPECTED = [
    "directed_pair_metrics.parquet",
    "undirected_rollout_pairs.parquet",
    "question_pair_type_means.csv",
    "question_contrasts.csv",
    "directed_four_cell_interactions.csv",
    "contrast_summary.csv",
    "permutation_null.parquet",
    "analysis_meta.json",
    "CORRECT_TRAJECTORY_PAIRWISE_GEOMETRY_REPORT.md",
    "figures/G1_direction_angle_progress.png",
    "figures/G2_relative_vector_progress.png",
    "figures/G3_amplitude_progress.png",
    "figures/G4_whole_trajectory_contrasts.png",
]
```

- [ ] **Step 2: Run CLI test and verify failure**

Run: `python -m pytest Experiment/tests/test_analyze_correct_trajectory_pairwise_geometry.py -v`

Expected: collection fails because the CLI module does not exist.

- [ ] **Step 3: Implement argument parsing and frozen-input validation**

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()
```

Use this validator before metric recovery:

```python
def validate_frozen_input(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if set(frame["representation"].unique()) != {"mean_w128_s64"}:
        raise ValueError("expected only mean_w128_s64")
    if set(frame["layer"].unique()) != {24, 36}:
        raise ValueError("expected hidden-state indexes 24 and 36")
    if (frame["rollout_id"] == frame["reference_rollout_id"]).any():
        raise ValueError("self-pairs are not allowed")
```

Record question, row, layer, progress-bin, and exclusion coverage before analysis.

- [ ] **Step 4: Implement artifact writing and figures**

Write progress curves from `question_pair_type_means.csv` after averaging over questions. Use fixed colors `{"++": "#157a6e", "--": "#b33f40", "+-": "#4c6085"}`, a zero reference line for contrast plots, angle in degrees for display, and two fixed L24/L36 facets. Draw every progress bin with its question-bootstrap 95% interval; do not select or hide bins based on effect size.

- [ ] **Step 5: Implement the report and metadata**

The report must include frozen cohort, exclusion counts, whole-trajectory contrasts, all adjusted progress-bin results, coverage, and an interpretation split into direction, amplitude, and combined geometry. It must explicitly state that negative `++ minus control` means correct trajectories are closer and that this is discovery-only.

- [ ] **Step 6: Run focused and existing regression tests**

Run:

```powershell
python -m pytest Experiment/tests/test_correct_trajectory_pairwise_geometry.py Experiment/tests/test_analyze_correct_trajectory_pairwise_geometry.py -v
python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Run the formal local analysis**

```powershell
python Experiment/scripts/analyze_correct_trajectory_pairwise_geometry.py `
  --input Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722/long_experiment_0_pairwise_geometry.parquet `
  --output-dir Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722/correct_trajectory_pairwise_geometry `
  --bootstrap 4000 `
  --permutations 1000 `
  --seed 20260723
```

Expected: completion message reports 24 questions, layers 24/36, 10 progress bins, zero missing required artifacts, and no H200 access.

- [ ] **Step 8: Inspect generated figures and numeric consistency**

Check every PNG with `view_image`; verify labels are legible, both layers render, confidence intervals do not overlap labels, and no empty panel is silently presented. Recompute one observed contrast manually from `question_pair_type_means.csv` and confirm it matches `contrast_summary.csv`.

- [ ] **Step 9: Commit implementation and report artifacts**

```powershell
git add Experiment/scripts/correct_trajectory_pairwise_geometry.py `
        Experiment/scripts/analyze_correct_trajectory_pairwise_geometry.py `
        Experiment/tests/test_correct_trajectory_pairwise_geometry.py `
        Experiment/tests/test_analyze_correct_trajectory_pairwise_geometry.py
git commit -m "exp: analyze correct trajectory pair similarity"
```

Do not add unrelated dirty-worktree files or bulk generated parquet artifacts to the commit.
