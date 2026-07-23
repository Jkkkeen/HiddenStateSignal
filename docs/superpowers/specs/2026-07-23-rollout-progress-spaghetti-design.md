# Rollout-Progress Spaghetti Curves Design

## Objective

Visualize how four frozen hidden-state signals evolve over relative reasoning progress in every
FormalDiscovery24 rollout. The figure must retain all individual rollout trajectories while also
showing question-weighted correct/wrong trends and uncertainty. The analysis is offline and must not
connect to H200 or rerun hidden-state extraction.

## Frozen Input And Signals

Input:

`Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722/long_experiment_0_bin_features.parquet`

The cohort is the existing 24 questions and 191 Qwen3-VL-8B-Thinking long-response rollouts. Use the
saved 10 relative-progress bins without smoothing or redefining progress.

The four panels are fixed before plotting:

| Panel | Feature | Representation | Hidden-state index |
|---|---|---|---:|
| Horizontal movement amplitude | `span_movement_norm_median` | `mean_w128_s64` | 24 |
| Horizontal movement turning | `span_turn_cos_median` | `mean_w128_s64` | 24 |
| Vertical activity burst | `vertical_norm_p90` | `token` | 15 |
| Vertical coordinate entropy | `coordinate_entropy_mean` | `token` | 17 |

No panel, layer, representation, progress bin, or sign may be changed after viewing this figure.

## Individual Curves

For each panel, draw one line for every eligible `(question_id, rollout_id)` across progress bins
0-9. Use the saved feature value directly, with no interpolation or normalization. A rollout with a
missing bin has a gap; it is not imputed. A rollout with fewer than two finite bins is excluded from
the line layer and counted in the coverage artifact.

Correct rollouts use translucent green lines and wrong rollouts use translucent red lines. Lines are
thin enough that dense regions remain visible. Individual trajectories are background observations,
not independent inferential units.

## Mean Curves And Confidence Intervals

Every question receives equal weight even when its correct/wrong rollout counts differ. For each
`feature x label x progress_bin`:

1. Average eligible rollout values within each `question_id x label` cell.
2. Average those question-level means to obtain the displayed correct or wrong mean.
3. Bootstrap eligible questions with replacement 4,000 times using seed `20260723`.
4. Use the 2.5th and 97.5th bootstrap percentiles as the 95% confidence band.

The overlaid mean curves are opaque and substantially thicker than the individual lines. Confidence
bands use the matching label color. Report the number of eligible questions and rollouts at every bin.

## Figure And Artifacts

Create one high-resolution 2x2 figure:

`figures/E0_F8_rollout_progress_spaghetti.png`

All panels share the x-axis definition (`relative-progress bin`, 0-9) but keep independent y-scales
because the metrics have different units. Titles name the feature, representation, and layer. The
legend distinguishes individual rollouts, correct mean, wrong mean, and 95% question-bootstrap CI
without covering data.

Write:

- `long_experiment_0_spaghetti_summary.csv`: label/bin means, confidence limits, and coverage.
- `E0_F8_ROLLOUT_PROGRESS_REPORT.md`: frozen cohort, formulas, coverage, and a descriptive reading
  of the generated curves without significance claims based on visual overlap.

Append the E0-F8 artifact names to the existing Experiment 0 report only after all outputs validate.

## Implementation Boundary

Add a focused plotting CLI that consumes the saved combined feature parquet and writes only the new
E0-F8 artifacts. Do not modify extraction, regenerate rollouts, run model forward passes, or alter
E0-F1 through E0-F7.

## Validation

Automated tests must verify:

- exact signal/representation/layer filtering;
- question-equal means despite unequal rollout counts;
- deterministic question bootstrap under the fixed seed;
- missing-bin coverage and no interpolation;
- creation of a non-empty PNG, CSV, and Markdown report.

Visually inspect the final PNG for all four panels, readable legends, non-overlapping labels, visible
individual curves, complete progress ticks, and nonblank confidence bands.
