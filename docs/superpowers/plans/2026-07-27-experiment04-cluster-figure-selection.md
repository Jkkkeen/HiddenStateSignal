# Experiment04 Cluster-Validated Figure Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Experiment04's maximum-single-cell primary figure selection with canonical, cluster-significant figure selection and move the former figures into an explicitly exploratory appendix.

**Architecture:** Add pure helpers for canonical family construction, significant-cluster selection, and in-cluster anchor selection. The main analysis loop will test canonical families once, retain exact cells, generate corrected primary figures, and separately generate the existing single-cell appendix. Existing scalar parquet inputs and inferential kernels remain unchanged.

**Tech Stack:** Python, NumPy, pandas, Matplotlib, pytest, H200 tmux.

## Global Constraints

- Confirm data remains blind and must not be read or forwarded.
- Do not rerun hidden-state extraction or scalar reduction.
- Token entropy is tested once per feature using canonical `horizontal / mean` rows.
- Primary figures require cluster `p_value <= 0.05`.
- Old maximum-|t| figures are appendix-only.

---

### Task 1: Canonical Families and Cluster Selection

**Files:**
- Modify: `Experiment/scripts/analyze_experiment04_discovery.py`
- Test: `Experiment/tests/test_analyze_experiment04_discovery.py`

**Interfaces:**
- Produces: `canonical_family_table(candidates) -> pd.DataFrame`
- Produces: `select_primary_clusters(clusters, top_figures) -> pd.DataFrame`
- Produces: `select_cluster_anchor(effects, cluster_cells) -> pd.Series`

- [ ] Add a test with six duplicated `token_raw_entropy_mean` families and assert one canonical `horizontal / mean` family.
- [ ] Add a test where an isolated cell has larger `|t|` than a significant connected cluster and assert the cluster is selected.
- [ ] Add a test that the selected anchor coordinate is contained in the cluster cell list.
- [ ] Run `pytest -q Experiment/tests/test_analyze_experiment04_discovery.py` and verify the new tests fail.
- [ ] Implement the three pure helpers with stable sorting and explicit validation.
- [ ] Rerun the focused test and verify it passes.

### Task 2: Corrected Figures and Report

**Files:**
- Modify: `Experiment/scripts/analyze_experiment04_discovery.py`
- Test: `Experiment/tests/test_analyze_experiment04_discovery.py`

**Interfaces:**
- Primary output: `figures/Fxx_<family>_atlas.png` and `_curve.png`
- Appendix output: `figures/exploratory_single_cell/Axx_<family>_atlas.png` and `_curve.png`
- Audit output: `cluster_permutation.csv` with a `cells` JSON column

- [ ] Add an integration-style synthetic test for corrected primary and appendix paths.
- [ ] Store exact `[layer, progress_bin]` cells for each cluster.
- [ ] Outline the selected cluster on the atlas and shade its progress span on the curve.
- [ ] Generate primary figures only from `p<=0.05` clusters, at most one cluster per unique family.
- [ ] Generate the old max-|t| ranking only under `exploratory_single_cell/`.
- [ ] Split Markdown and HTML reports into corrected primary and uncorrected appendix sections.
- [ ] Run all Experiment04 analysis tests.

### Task 3: Reanalysis and Verification

**Files:**
- Deploy: `Experiment/scripts/analyze_experiment04_discovery.py`
- Read: existing H200 discovery scalar parquet files
- Replace: discovery result report/CSV/figures only

- [ ] Run the complete local Experiment04 test set.
- [ ] Check `git diff --check` for the modified script and tests.
- [ ] Sync the analyzer to H200.
- [ ] Launch analysis-only tmux using 4,000 bootstrap and 4,000 permutations.
- [ ] Verify token entropy family count is deduplicated and confirm data is untouched.
- [ ] Verify every `Fxx` figure maps to a cluster with `p<=0.05` and every appendix figure is under the exploratory directory.
- [ ] Pull the corrected results into the local Experiment04 server-results directory and visually inspect the primary atlas/curve pairs.
