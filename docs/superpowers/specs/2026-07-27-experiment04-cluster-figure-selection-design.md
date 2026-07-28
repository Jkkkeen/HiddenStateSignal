# Experiment04 Cluster-Validated Figure Selection Design

## Goal

Make the Experiment04 discovery report select primary figures from corrected cluster-permutation results instead of uncorrected maximum single-cell statistics, while treating numerically duplicated token-entropy families as one statistical hypothesis.

## Statistical Unit

- Geometry and pooled-vector metrics use `(direction, representation, feature)` as the family key.
- Features prefixed with `token_` do not depend on direction or chunk representation in the current scalar tables. They use one canonical family key `(token, feature)` and are tested once using `horizontal / mean` rows.
- Canonicalization happens before family ranking and permutation seeding. Duplicate rows must not be tested with different seeds and then minimized over p-values.

## Primary Figures

- Run cluster permutation for every unique family.
- Save exact cluster cells as `[[layer, progress_bin], ...]` in `cluster_permutation.csv`.
- Primary clusters require `p_value <= 0.05`.
- Select at most one primary cluster per unique family: lowest p-value, then largest cluster mass.
- Rank primary figures by p-value ascending, mass descending, then stable family key.
- Select the curve anchor only from cells in the selected cluster, using maximum absolute cell t-statistic.
- Atlas figures show the full family t-statistic grid and outline the selected significant cluster.
- Curves show the selected layer across all progress bins and shade the cluster's progress-bin span.
- If no cluster is significant, generate no `Fxx` primary figures and state that explicitly.

## Exploratory Appendix

- Preserve the old maximum-absolute-t ranking as an explicitly uncorrected appendix.
- Write appendix figures under `figures/exploratory_single_cell/` with `Axx` prefixes.
- Report titles and captions must say `Uncorrected exploratory single-cell ranking`.

## Report

- `DISCOVERY_REPORT.md/.html` contains separate sections for cluster-significant primary findings and the exploratory single-cell appendix.
- Each primary entry reports direction, representation, feature, cluster p-value, mass, sign, exact cell count, anchor layer/bin, anchor t-statistic, AUC, and question support.
- The report records the number of raw families, unique tested families, and duplicated token families removed.
- Confirm remains blind and is not read or forwarded by this change.

## Verification

- Synthetic tests must prove that an isolated higher-|t| non-significant cell cannot displace a lower-|t| significant cluster.
- Synthetic tests must prove that duplicated token features produce one canonical family.
- The curve anchor must belong to the selected cluster.
- Existing permutation determinism and question-equal aggregation tests must continue to pass.
- Reanalysis uses the existing discovery scalar parquet files only; no hidden-state forward or reduction is rerun.
