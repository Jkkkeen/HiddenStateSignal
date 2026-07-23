# Rollout-Progress Spaghetti Curves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate an offline E0-F8 figure containing all eligible rollout progress curves plus question-weighted correct/wrong means and bootstrap intervals for four frozen hidden-state signals.

**Architecture:** Add one focused plotting CLI that validates and filters the saved combined Experiment 0 feature parquet, computes question-equal summaries, then writes the figure, summary CSV, and report. Keep summary computation in pure functions so unequal rollout counts, missing bins, and deterministic bootstrap behavior can be tested without invoking matplotlib.

**Tech Stack:** Python 3, pandas, NumPy, matplotlib with the `Agg` backend, pytest, parquet/pyarrow.

## Global Constraints

- Read only `long_experiment_0_bin_features.parquet` from FormalDiscovery24.
- Do not connect to H200, generate responses, or run hidden-state forward passes.
- Freeze the four signal specifications from the approved design: movement L24, turn L24, vertical P90 L15, and coordinate entropy L17.
- Draw raw values over the existing progress bins 0-9 without smoothing, interpolation, or normalization.
- Give every question equal weight in displayed means and 4,000-resample confidence intervals.
- Use seed `20260723` and preserve missing-bin gaps.
- Do not alter E0-F1 through E0-F7.

---

### Task 1: Frozen Signal Selection And Question-Weighted Summaries

**Files:**
- Create: `Experiment/scripts/plot_experiment0_rollout_progress.py`
- Create: `Experiment/tests/test_plot_experiment0_rollout_progress.py`

**Interfaces:**
- Consumes: a combined Experiment 0 pandas `DataFrame` with rollout identifiers, labels, representation, layer, progress bin, and the four frozen feature columns.
- Produces: `SignalSpec`, `select_signal(frame, spec)`, and `question_weighted_summary(selected, spec, bootstrap, seed)`.

- [ ] **Step 1: Write failing filtering and equal-question-weight tests**

```python
from plot_experiment0_rollout_progress import (
    FROZEN_SIGNALS,
    question_weighted_summary,
    select_signal,
)


def test_select_signal_freezes_representation_layer_and_feature() -> None:
    frame = synthetic_combined_features()
    selected = select_signal(frame, FROZEN_SIGNALS[0])
    assert selected["representation"].eq("mean_w128_s64").all()
    assert selected["layer"].eq(24).all()
    assert selected["value"].notna().all()
    assert set(selected["progress_bin"]) == {0, 1}


def test_summary_weights_questions_not_rollout_count() -> None:
    selected = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q1", "q2", "q1", "q2"],
            "rollout_id": [0, 1, 2, 0, 3, 1],
            "is_correct": [True, True, True, True, False, False],
            "progress_bin": [0, 0, 0, 0, 0, 0],
            "value": [0.0, 0.0, 0.0, 10.0, 2.0, 4.0],
        }
    )
    summary = question_weighted_summary(
        selected,
        FROZEN_SIGNALS[0],
        bootstrap=100,
        seed=7,
    )
    correct = summary[summary["is_correct"]].iloc[0]
    assert correct["mean"] == pytest.approx(5.0)
    assert correct["n_questions"] == 2
    assert correct["n_rollouts"] == 4
```

- [ ] **Step 2: Run the focused tests and verify collection failure**

Run:

`python -m pytest Experiment/tests/test_plot_experiment0_rollout_progress.py -v`

Expected: collection fails with `ModuleNotFoundError: plot_experiment0_rollout_progress`.

- [ ] **Step 3: Implement frozen specifications and exact filtering**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalSpec:
    feature: str
    representation: str
    layer: int
    panel_title: str
    y_label: str


FROZEN_SIGNALS = (
    SignalSpec(
        "span_movement_norm_median",
        "mean_w128_s64",
        24,
        "Horizontal movement amplitude",
        "Median span movement norm",
    ),
    SignalSpec(
        "span_turn_cos_median",
        "mean_w128_s64",
        24,
        "Horizontal movement turning",
        "Median span turning cosine",
    ),
    SignalSpec(
        "vertical_norm_p90",
        "token",
        15,
        "Vertical activity burst",
        "P90 vertical update norm",
    ),
    SignalSpec(
        "coordinate_entropy_mean",
        "token",
        17,
        "Vertical coordinate entropy",
        "Mean normalized coordinate entropy",
    ),
)


def select_signal(frame: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    required = {
        "question_id",
        "rollout_id",
        "is_correct",
        "representation",
        "layer",
        "progress_bin",
        spec.feature,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    selected = frame[
        (frame["representation"] == spec.representation)
        & (frame["layer"] == spec.layer)
    ][
        [
            "question_id",
            "rollout_id",
            "is_correct",
            "representation",
            "layer",
            "progress_bin",
            spec.feature,
        ]
    ].copy()
    selected = selected.rename(columns={spec.feature: "value"})
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    selected = selected[np.isfinite(selected["value"])].copy()
    if selected.empty:
        raise ValueError(
            f"no finite rows for {spec.feature} | {spec.representation} | L{spec.layer}"
        )
    return selected.sort_values(
        ["question_id", "rollout_id", "progress_bin"], kind="stable"
    ).reset_index(drop=True)
```

- [ ] **Step 4: Implement question-equal aggregation and deterministic bootstrap**

```python
def question_weighted_summary(
    selected: pd.DataFrame,
    spec: SignalSpec,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    question_means = (
        selected.groupby(
            ["question_id", "is_correct", "progress_bin"],
            as_index=False,
            observed=True,
        )["value"]
        .mean()
    )
    rollout_coverage = (
        selected.groupby(["is_correct", "progress_bin"], observed=True)[
            ["question_id", "rollout_id"]
        ]
        .size()
        .rename("n_rollouts")
    )
    rows = []
    for offset, ((label, progress_bin), group) in enumerate(
        question_means.groupby(["is_correct", "progress_bin"], sort=True)
    ):
        values = group["value"].to_numpy(dtype=np.float64)
        rng = np.random.default_rng(seed + offset)
        samples = rng.choice(
            values,
            size=(bootstrap, len(values)),
            replace=True,
        ).mean(axis=1)
        rows.append(
            {
                "feature": spec.feature,
                "representation": spec.representation,
                "layer": spec.layer,
                "is_correct": bool(label),
                "progress_bin": int(progress_bin),
                "mean": float(values.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "n_questions": int(group["question_id"].nunique()),
                "n_rollouts": int(rollout_coverage.loc[(label, progress_bin)]),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Add deterministic and missing-bin tests**

```python
def test_summary_is_deterministic_and_does_not_create_missing_bins() -> None:
    selected = synthetic_selected_with_missing_bin()
    first = question_weighted_summary(selected, FROZEN_SIGNALS[0], 200, 19)
    second = question_weighted_summary(selected, FROZEN_SIGNALS[0], 200, 19)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["progress_bin"]) == {0, 2}
    assert 1 not in set(first["progress_bin"])
```

- [ ] **Step 6: Run Task 1 tests**

Run:

`python -m pytest Experiment/tests/test_plot_experiment0_rollout_progress.py -v`

Expected: all signal-selection, equal-question, deterministic-bootstrap, and missing-bin tests pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add Experiment/scripts/plot_experiment0_rollout_progress.py `
        Experiment/tests/test_plot_experiment0_rollout_progress.py
git commit -m "feat: summarize rollout progress signals"
```

### Task 2: E0-F8 Figure, Report, And Formal Offline Run

**Files:**
- Modify: `Experiment/scripts/plot_experiment0_rollout_progress.py`
- Modify: `Experiment/tests/test_plot_experiment0_rollout_progress.py`
- Modify after validation: `Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722/LONG_EXPERIMENT_0_RESULTS.md`
- Generate: `Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722/figures/E0_F8_rollout_progress_spaghetti.png`
- Generate: `Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722/long_experiment_0_spaghetti_summary.csv`
- Generate: `Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722/E0_F8_ROLLOUT_PROGRESS_REPORT.md`

**Interfaces:**
- Consumes: the selected frames and summaries produced by Task 1.
- Produces: `plot_spaghetti_panels(...)`, `write_report(...)`, `append_parent_report(...)`, and CLI `main()`.

- [ ] **Step 1: Write a failing CLI artifact test**

```python
def test_cli_writes_png_csv_report_and_idempotent_parent_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "features.parquet"
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    synthetic_all_four_signals().to_parquet(input_path, index=False)
    parent = output_dir / "LONG_EXPERIMENT_0_RESULTS.md"
    parent.write_text("# Existing report\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_experiment0_rollout_progress.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap",
            "20",
            "--seed",
            "11",
        ],
    )
    main()
    main()
    assert (output_dir / "figures/E0_F8_rollout_progress_spaghetti.png").stat().st_size > 0
    assert (output_dir / "long_experiment_0_spaghetti_summary.csv").stat().st_size > 0
    assert (output_dir / "E0_F8_ROLLOUT_PROGRESS_REPORT.md").stat().st_size > 0
    text = parent.read_text(encoding="utf-8")
    assert text.count("## Rollout-Progress Spaghetti Addendum") == 1
```

- [ ] **Step 2: Run the CLI test and verify it fails**

Run:

`python -m pytest Experiment/tests/test_plot_experiment0_rollout_progress.py::test_cli_writes_png_csv_report_and_idempotent_parent_link -v`

Expected: FAIL because `main` and artifact writers do not exist.

- [ ] **Step 3: Implement CLI parsing and matplotlib plotting**

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def plot_spaghetti_panels(
    selected_by_signal: list[tuple[SignalSpec, pd.DataFrame]],
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    colors = {True: "#157a6e", False: "#b33f40"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for axis, (spec, selected) in zip(axes.ravel(), selected_by_signal):
        for (_, _), rollout in selected.groupby(
            ["question_id", "rollout_id"], sort=False
        ):
            rollout = rollout.sort_values("progress_bin")
            if rollout["progress_bin"].nunique() < 2:
                continue
            label = bool(rollout.iloc[0]["is_correct"])
            axis.plot(
                rollout["progress_bin"],
                rollout["value"],
                color=colors[label],
                alpha=0.07,
                linewidth=0.65,
                zorder=1,
            )
        signal_summary = summary[summary["feature"] == spec.feature]
        for label in (True, False):
            curve = signal_summary[signal_summary["is_correct"] == label].sort_values(
                "progress_bin"
            )
            x = curve["progress_bin"].to_numpy(dtype=float)
            axis.fill_between(
                x,
                curve["ci_low"],
                curve["ci_high"],
                color=colors[label],
                alpha=0.18,
                zorder=2,
            )
            axis.plot(
                x,
                curve["mean"],
                color=colors[label],
                linewidth=2.6,
                marker="o",
                markersize=4,
                zorder=3,
            )
        axis.set_title(
            f"{spec.panel_title}\n{spec.feature} | {spec.representation} | L{spec.layer}"
        )
        axis.set_ylabel(spec.y_label)
        axis.set_xticks(range(10))
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Relative-progress bin")
    fig.suptitle("Qwen3-VL-8B-Thinking rollout-level hidden dynamics", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
```

Add a figure-level legend above the panels with `Line2D` handles for faint correct/wrong rollout
lines, thick correct/wrong means, and a `Patch` for the question-bootstrap interval:

```python
handles = [
    Line2D([0], [0], color=colors[True], alpha=0.20, linewidth=1, label="Correct rollout"),
    Line2D([0], [0], color=colors[False], alpha=0.20, linewidth=1, label="Wrong rollout"),
    Line2D([0], [0], color=colors[True], linewidth=2.6, label="Correct mean"),
    Line2D([0], [0], color=colors[False], linewidth=2.6, label="Wrong mean"),
    Patch(facecolor="#777777", alpha=0.18, label="95% question-bootstrap CI"),
]
fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.955))
```

Place it below the suptitle and above the panel titles so it does not cover data.

- [ ] **Step 4: Implement report writing and idempotent parent-report append**

```python
ADDENDUM_MARKER = "## Rollout-Progress Spaghetti Addendum"


def append_parent_report(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if ADDENDUM_MARKER in text:
        return
    addition = (
        "\n## Rollout-Progress Spaghetti Addendum\n\n"
        "- `figures/E0_F8_rollout_progress_spaghetti.png`\n"
        "- `long_experiment_0_spaghetti_summary.csv`\n"
        "- `E0_F8_ROLLOUT_PROGRESS_REPORT.md`\n"
    )
    path.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")
```

`write_report` must state the frozen four signals, 24-question/191-rollout cohort, raw-line and
question-bootstrap weighting distinction, missing-bin handling, and for every panel list the correct
and wrong means at bins 0 and 9 plus the bin with the largest absolute displayed mean gap. It must
explicitly call these descriptive curves and avoid significance claims from visual separation.

- [ ] **Step 5: Implement `main()` and write all artifacts before appending**

```python
def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    frame = pd.read_parquet(args.input)
    selected_by_signal = []
    summaries = []
    for index, spec in enumerate(FROZEN_SIGNALS):
        selected = select_signal(frame, spec)
        selected_by_signal.append((spec, selected))
        summaries.append(
            question_weighted_summary(
                selected,
                spec,
                bootstrap=args.bootstrap,
                seed=args.seed + index * 100,
            )
        )
    summary = pd.concat(summaries, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "figures/E0_F8_rollout_progress_spaghetti.png"
    summary_path = args.output_dir / "long_experiment_0_spaghetti_summary.csv"
    report_path = args.output_dir / "E0_F8_ROLLOUT_PROGRESS_REPORT.md"
    plot_spaghetti_panels(selected_by_signal, summary, figure_path)
    summary.to_csv(summary_path, index=False)
    write_report(report_path, frame, selected_by_signal, summary, args)
    for path in (figure_path, summary_path, report_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty artifact: {path}")
    append_parent_report(args.output_dir / "LONG_EXPERIMENT_0_RESULTS.md")
    print(
        f"Completed E0-F8: {frame['question_id'].nunique()} questions, "
        f"{frame[['question_id', 'rollout_id']].drop_duplicates().shape[0]} rollouts, "
        "H200 accessed: False"
    )
```

- [ ] **Step 6: Run focused and regression tests**

Run:

```powershell
python -m pytest Experiment/tests/test_plot_experiment0_rollout_progress.py -v
python -m pytest Experiment/tests/test_experiment0_hidden_dynamics.py `
                 Experiment/tests/test_analyze_experiment0_hidden_dynamics.py -q
```

Expected: all new tests pass and the existing 14 Experiment 0 tests pass.

- [ ] **Step 7: Run the formal offline plotting command**

```powershell
python Experiment/scripts/plot_experiment0_rollout_progress.py `
  --input Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722/long_experiment_0_bin_features.parquet `
  --output-dir Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722 `
  --bootstrap 4000 `
  --seed 20260723
```

Expected: completion output reports 24 questions, 191 rollouts, all four signals, and no H200 access.

- [ ] **Step 8: Inspect image and numeric consistency**

Open `E0_F8_rollout_progress_spaghetti.png` with `view_image`. Verify all four panels are nonblank,
individual curves remain visible, means and bands are distinguishable, titles/legend do not overlap,
and bins 0-9 are shown. Independently recompute one label/bin question-equal mean from the parquet
and assert it matches `long_experiment_0_spaghetti_summary.csv` to floating-point precision.

- [ ] **Step 9: Commit implementation without bulk generated data**

```powershell
git add Experiment/scripts/plot_experiment0_rollout_progress.py `
        Experiment/tests/test_plot_experiment0_rollout_progress.py
git commit -m "exp: plot rollout progress trajectories"
```

Do not add the 43 MB source parquet or other unrelated dirty-worktree files. The small PNG, CSV,
Markdown report, and parent-report addendum remain local analysis artifacts unless explicitly requested
for version control.
