# Correct-Trajectory Pairwise Geometry Design

## Objective

Test whether successful rollouts for the same question have more similar hidden-state movements than failed rollouts at matched reasoning progress. The analysis separates direction, amplitude, and their combined vector difference instead of collapsing them into the existing `cross_length_support` score.

This is a discovery-only offline analysis. It uses existing Experiment 0 pairwise geometry and does not run generation, model forward passes, RL updates, or H200 jobs.

## Frozen Input

- File: `Experiment/server_results/experiment0_hidden_dynamics_formal_discovery24_20260722/long_experiment_0_pairwise_geometry.parquet`
- Cohort: 24 questions from FormalDiscovery24
- Representation: `mean_w128_s64`
- Hidden-state indexes: 24 and 36
- Progress alignment: existing 10 relative-progress bins and nearest-progress reference span within each bin
- Required columns: query/reference rollout IDs and correctness labels, layer, progress bin, query/reference movement norm, and cosine similarity

Self-pairs are excluded. No representation, layer, or progress resolution is selected after observing the new results.

## Pairwise Metrics

For a query movement `D_i`, reference movement `D_j`, norms `R_i` and `R_j`, and saved cosine `c_ij`, calculate:

1. Directional angle in radians and degrees:

   `theta_ij = arccos(clip(c_ij, -1, 1))`

   Smaller values mean more similar movement directions.

2. Exact vector difference recovered without the original vectors:

   `delta_vec_ij = sqrt(max(R_i^2 + R_j^2 - 2 R_i R_j c_ij, 0))`

3. Scale-normalized vector difference:

   `delta_rel_ij = delta_vec_ij / (R_i + R_j + epsilon)`

   This is the primary combined direction-and-amplitude comparison because raw vector distance is sensitive to layer scale.

4. Pure amplitude difference:

   `delta_amp_ij = abs(log(R_i + epsilon) - log(R_j + epsilon))`

Raw `delta_vec` remains a diagnostic. `theta`, `delta_rel`, and `delta_amp` separate direction, combined geometry, and magnitude.

Rows with a non-finite norm or either norm less than or equal to `1e-12` are excluded from all primary metrics because their movement direction is undefined. The exclusion count and fraction are reported by layer and progress bin. The square-root radicand for `delta_vec` is clipped at zero only to absorb floating-point roundoff.

## Pair Construction And Weighting

Each saved row is a directed query-reference match. The analysis first aggregates rows within:

`question x layer x progress_bin x query_rollout x reference_rollout`

using the median of each metric. It then canonicalizes the two rollout IDs and averages the two available directions (`i -> j` and `j -> i`) to create one undirected rollout-pair observation. This prevents dense spans and asymmetric nearest-progress matches from receiving extra weight.

Undirected pairs are assigned to:

- `++`: correct-correct
- `--`: wrong-wrong
- `+-`: correct-wrong

Within each question, layer, and progress bin, each pair type is averaged separately. Every eligible question then contributes one value per pair type, regardless of how many correct or wrong rollout pairs it contains.

Missing pair types are omitted only from contrasts that require them. Coverage counts must be reported for every layer, progress bin, and contrast.

## Primary Contrasts

For distance metrics, smaller means more similar. The primary paired within-question contrasts are:

- Correct cohesion versus wrong cohesion: `mean(++) - mean(--)`
- Correct cohesion versus cross-label separation: `mean(++) - mean(+-)`

Negative values support the hypothesis that correct trajectories are more similar. The result is considered structurally persuasive only if correct-correct pairs are closer than both wrong-wrong and correct-wrong pairs; `++ < --` alone is insufficient.

As a diagnostic, retain the directed four cells `A++`, `A+-`, `A-+`, and `A--` before symmetrization and report the interaction:

`I = (A++ - A+-) - (A-+ - A--)`

## Progress And Whole-Trajectory Views

Two views are required:

1. Progress curves: pair-type means for all 10 bins, separately for layers 24 and 36.
2. Whole-trajectory profile: first average each undirected rollout pair over its available progress bins with equal bin weight, then compare `++`, `--`, and `+-` within each question.

Equal bin weighting prevents longer answers or bins containing more spans from dominating the whole-trajectory score.

## Statistical Inference

- Independent unit: question
- Point estimate: mean of the within-question paired contrasts
- Uncertainty: question bootstrap 95% confidence interval
- Stability: fraction of eligible questions with the hypothesized negative sign
- Null test: shuffle rollout correctness labels within each question while preserving the observed correct/wrong counts, rebuild pair types, and recompute all contrasts
- Multiple-grid control: report max-statistic permutation-adjusted p-values across `2 layers x 10 progress bins` for each metric family

Raw pair rows are never treated as independent samples.

## Outputs

- Pair-level table after directed aggregation and symmetrization
- Question-level pair-type means and paired contrasts
- Progress-bin and whole-trajectory summary table with bootstrap intervals, sign consistency, coverage, and permutation p-values
- Direction-angle progress curves for `++`, `--`, and `+-`, faceted by layer
- Scale-normalized vector-difference progress curves, faceted by layer
- Pure amplitude-difference progress curves, faceted by layer
- Per-question whole-trajectory contrast plots
- Markdown report stating whether any apparent similarity is directional, amplitude-driven, or present only in the combined metric

## Interpretation Rules

- `++` closer than both controls in angle but not amplitude: successful trajectories share movement direction rather than step size.
- `++` closer only in amplitude: successful trajectories have similar update scale, not a common direction.
- `++` closer only in `delta_rel`: direction and amplitude interact, but neither marginal metric is sufficient alone.
- A null result does not prove successful trajectories lack structure; it only rejects a single progress-matched cohesion pattern at L24/L36 under `mean_w128_s64`.

## Validation

Automated tests must cover:

- angle and vector-distance recovery on identical, orthogonal, and opposite synthetic vectors
- numerical clipping and zero-norm handling
- self-pair removal
- directed-to-undirected pair canonicalization
- equal question weighting despite unequal pair counts
- correctness-label permutation preserving per-question label counts
- deterministic bootstrap/permutation output under a fixed seed

The implementation must verify the frozen input schema and fail with an explicit message if required columns or expected layer/representation values are missing.
