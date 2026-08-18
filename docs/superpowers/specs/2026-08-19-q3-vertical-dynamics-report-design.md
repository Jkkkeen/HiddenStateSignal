# Qwen3-1.7B Vertical Dynamics Presentation Report Design

**Date:** 2026-08-19

**Status:** Approved direction; implementation pending written-spec review

## Purpose

Build a self-contained Chinese HTML report for a group meeting or advisor
presentation. The report must explain the completed Qwen3-1.7B GRPO result as
one auditable evidence chain:

1. RL produced a real held-out ability gain;
2. V1/V3/V4/V8 show large early vertical reorganization and a late plateau;
3. the reorganization remains concentrated in stable deep-layer bands;
4. vertical correctness AUROC is weak globally but has stable local signals;
5. step175-200 is a stopping candidate, not a validated stopping rule;
6. the current run does not robustly establish H2 AUROC 0.67 or justify H2
   reward shaping.

The first screen must communicate the result without requiring the reader to
understand the extraction pipeline. Detailed definitions, provenance, and
limitations belong later in the report.

## Deliverables

Create:

```text
Experiment_2E/scripts/build_q3_vertical_dynamics_report.py
Experiment_2E/tests/test_build_q3_vertical_dynamics_report.py
Experiment_2E/server_results/q3_1p7b_vertical_dynamics_formal_20260818/
  Q3_VERTICAL_DYNAMICS_REPORT.html
```

The HTML is the presentation artifact. It must contain all CSS, JavaScript,
tables, and images inline, open without a server, and have no external network
dependency. The builder is the reproducibility artifact; it is not exposed as
part of the presentation.

## Source Contract

The builder consumes the completed H200 run root:

```text
/data2/hjk/results/experiment_2e/
q3_1p7b_base_simplerl_grpo_formal_seed20260814
```

Required inputs:

```text
heldout/generations/{0,25,...,250}.jsonl
heldout/hidden_summary.jsonl
online_hidden/per_step_summary.jsonl
layer_profiles_formal/run_status.json
layer_profiles_formal/analysis/analysis_audit.json
layer_profiles_formal/analysis/policy_profiles.parquet
layer_profiles_formal/analysis/layer_auc.parquet
layer_profiles_formal/analysis/depth_summaries.parquet
layer_profiles_formal/metrics/vertical_metrics_step{000..250}.parquet
layer_profiles_formal/analysis/figures/*.png
```

The builder must fail closed unless:

- `run_status.status == "completed"` and `exit_code == 0`;
- `analysis_audit.passed == true`;
- all 11 expected checkpoint steps exist;
- every checkpoint contains exactly 256 questions and 8 rollouts per question;
- all required metric variants and selected source figures exist;
- all generated figures are nonblank.

No raw response text, chain-of-thought content, model weights, hidden tensors,
or server credentials enter the HTML.

## Statistical Contract

### Behavior

For a question with `c` correct responses among 8 rollouts:

```text
pass@k = 1 - C(8-c, k) / C(8, k), k in {1,4,8}
```

Checkpoint values are question means. Confidence intervals use 4,000 paired
question-bootstrap draws with seed `20260818`. The report shows:

- pass@1/4/8 levels by checkpoint;
- final-minus-base estimates and 95% confidence intervals;
- adjacent pass@1 differences;
- step50-to-final and step175-to-final paired differences.

### Vertical Dynamics

The candidate panel is V1/V3/V4/V4C/V8, with no primary selection:

| Family | Variant used in summary |
|---|---|
| V1 | raw and relative layer-update norm |
| V3 | demeaned adjacent-state angle |
| V4 | layer-update turning angle |
| V4C | amplitude-weighted turning companion |
| V8 | layer-update effective rank, with centered sensitivity |

For each metric and representation, build a question-equal checkpoint profile.
Report cumulative relative L1 change from base and adjacent-checkpoint relative
L1 change. Preserve `mean_w128_s32` and `last_s32`; the main summary chart uses
`mean_w128_s32`, while the text and appendix show whether `last_s32` agrees.

V4C must be read from the existing V4 family metric
`weighted_layer_update_turning_angle` and labeled V4C in the report. V8 centered
must remain a sensitivity variant, not a separate family.

### AUROC

Layer-resolved correctness results use the completed 256-question formal
profile analysis. Report both raw direction and separability
`max(AUROC, 1-AUROC)`. The report must state that a reverse-direction AUROC is
not a new fitted classifier.

H2/H5 are handled separately:

- online values come from training-time 16-question probes and may describe
  trends but not formal held-out generalization;
- checkpoint held-out H2/H5 have only about 5-11 mixed questions per cell and
  are labeled underpowered;
- no H2/H5 value is presented as a 256-question formal estimate;
- no reward-shaping recommendation is made.

## Report Structure

### 1. Executive Result

The opening band contains:

- model/run identity;
- `pass@1 39.65% -> 63.33%`;
- final-minus-base `+23.68 pp` and its CI;
- `11 checkpoints`, `22,528 rollouts`, `192/192 figures passed`;
- one literal conclusion: early learning, later consolidation, step175-200
  stopping candidate.

It must not use a marketing hero, decorative gradients, or oversized type.

### 2. Ability Evidence

Show:

1. pass@1/4/8 checkpoint curves with bootstrap bands;
2. adjacent `Delta pass@1` with zero line and interval CIs;
3. a compact table of all checkpoints.

Annotate the three phases:

- step0-50: primary ability gain;
- step50-175: smaller cumulative improvement;
- step175-250: no detectable additional gain.

### 3. Vertical Reorganization

Show:

1. cumulative change from base for V1/V3/V4/V4C/V8;
2. adjacent-checkpoint change speed;
3. V8 raw and centered sensitivity trajectories;
4. a metric-role table explaining magnitude, direction, turning, weighted
   turning, and effective rank.

The section conclusion must distinguish level, change speed, and correctness
classification.

### 4. Where Reorganization Occurs

Embed the most informative completed formal figures:

- V1 raw policy heatmap, `mean_w128_s32`, B4;
- V3 policy heatmap, `mean_w128_s32`, B4;
- V4 policy heatmap, `mean_w128_s32`, B2;
- V3 selected profiles, `mean_w128_s32`, B4.

Add a compact depth summary showing:

- V1 raw peak L28;
- V3 peak L28;
- V4 mean-window peak L24 and last-window late-layer behavior;
- no shallow-to-deep peak migration.

### 5. Correctness Signal

Show:

1. AUROC separability distribution by checkpoint;
2. V3 B4 layer-AUROC heatmap;
3. V1 relative B4 layer-AUROC heatmap;
4. a table for the stable late cells.

The section must report:

- step250 median separability about 0.529;
- step250 P90 about 0.576;
- V3 B4 L28 late mean about 0.650;
- reverse-direction V1 relative B4 L28 late separability about 0.648.

### 6. Horizontal Context

Show a compact two-panel H2/H5 online trend chart for the final layer. The
visual separates correct and wrong means and never overlays online values with
formal 256-question vertical estimates as if they shared a cohort.

The text must state that the current run's long-run H2 AUROC is around 0.58 in
its best sufficiently covered online cells, while H5 primarily shows a level
trend and inverse-direction separation. Individual checkpoint spikes based on
5-11 mixed questions are not highlighted as discoveries.

### 7. Stopping Interpretation

Use a four-state matrix:

| Vertical change | Ability gain | Interpretation |
|---|---|---|
| high | positive | effective learning candidate |
| high | flat/negative | drift or overtraining candidate |
| low | positive | readout/policy refinement candidate |
| low | flat | stopping candidate |

Place step0-50, step50-175, and step175-250 into the matrix. State explicitly
that the rollout split-noise baseline from `vertical_dynamic_plan.md` is still
required before validating a stopping rule.

### 8. Audit and Limitations

Include:

- exact checkpoint list;
- question and rollout counts;
- extraction and analysis gate status;
- metric definitions and representation/stage semantics;
- source hashes from `analysis_audit.json`;
- single-seed limitation;
- time-trend confounding in checkpoint correlations;
- absent rollout sampling-noise correction;
- H2/H5 cohort and power limitations.

## Visual Design

- Chinese scientific-report layout, optimized for a 16:9 presentation screen
  but fully usable on mobile.
- Quiet white/light-gray paper with charcoal text; green for verified ability
  gain, blue for geometry, amber for stopping candidates, and vermilion for
  limitations. No one-hue palette and no decorative gradients or blobs.
- Cards use a maximum 7 px radius and only frame individual statistics or
  figures. Sections remain unframed full-width bands; no nested cards.
- Headings inside compact panels remain compact. Font sizes do not scale with
  viewport width and letter spacing is zero except for small uppercase labels.
- Tables scroll horizontally on narrow screens. Figures use explicit aspect
  ratios so loading cannot shift the layout.
- Every image has meaningful Chinese alt text and a caption that states the
  cohort, representation, stage, and interpretation boundary.
- A sticky section navigation supports live presentation without adding an
  application shell or landing page.
- Existing source PNGs and generated Matplotlib PNGs are base64 embedded. No
  external fonts, icon libraries, scripts, or stylesheets are required.

## Builder Boundaries

`build_q3_vertical_dynamics_report.py` owns four responsibilities:

1. validate the completed result root;
2. derive compact behavior/vertical/AUROC/horizontal summaries;
3. render deterministic Matplotlib summary figures;
4. assemble and atomically write the self-contained HTML.

Pure calculation helpers must be independently testable. HTML composition must
consume already-derived tables and image data; it must not hide metric formulas
inside template string manipulation.

## Error Handling

- Missing/failed audits stop the build with a precise path and failed gate.
- Missing metric cells or checkpoint steps stop the build; they are not silently
  interpolated.
- Unsupported AUROC cells render as unavailable only when the source explicitly
  records inadequate mixed-question coverage.
- A figure below 10 KB is treated as blank and fails the build.
- The final HTML is written to a temporary sibling and atomically renamed.
- The builder records input hashes, build timestamp, builder version, and all
  fixed seeds in an inline provenance block.

## Verification

Automated tests cover:

- exact pass@1/4/8 combinatorics;
- paired bootstrap determinism;
- question-equal aggregation;
- V4C and V8 variant mapping;
- expected checkpoint/profile differences;
- fail-closed audit behavior;
- required section IDs, captions, embedded images, and provenance.

Final visual verification uses a local static server and Playwright at desktop
`1440x1000` and mobile `390x844`. Acceptance requires:

- no horizontal document overflow;
- no overlapping navigation, text, tables, or figures;
- all embedded images decode and have nonblank rendered pixels;
- all section links work;
- no browser-console errors;
- the first viewport shows the run identity, ability result, and stopping
  candidate while leaving a visible hint of the next section.

## Non-Goals

- Do not include all 192 source figures.
- Do not build a filterable dashboard.
- Do not modify training, extraction, metrics, checkpoints, or result files.
- Do not select a vertical primary metric.
- Do not claim causal mediation or a validated universal stopping rule.
- Do not recommend rewarding H2/H5 from this single run.
