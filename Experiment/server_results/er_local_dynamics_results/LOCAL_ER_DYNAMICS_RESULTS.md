# Local ER Dynamics Results

ERV/ERA are computed from existing local ER parquet files; no model forward is used.

## Data

- Chunk rows: 68764
- Rollout-layer rows: 17872
- Questions: 1117
- Rollouts: 8936
- Base metric: `er_centered`

## Top Rollout-Level Dynamics Features

| layer | feature | questions | AUROC(+feature) | 95% CI | AUROC(-feature) |
|---:|---|---:|---:|---:|---:|
| 36 | erv_adj_late_min | 1057 | 0.6925 | [0.6763, 0.7106] | 0.3075 |
| 36 | erv_adj_all_min | 1057 | 0.6925 | [0.6763, 0.7106] | 0.3075 |
| 24 | erv_hist_late_min | 1057 | 0.6903 | [0.6725, 0.7081] | 0.3097 |
| 24 | erv_hist_all_min | 1057 | 0.6903 | [0.6725, 0.7081] | 0.3097 |
| 36 | erv_hist_late_min | 1057 | 0.6878 | [0.6703, 0.7062] | 0.3122 |
| 36 | erv_hist_all_min | 1057 | 0.6878 | [0.6703, 0.7062] | 0.3122 |
| 24 | erv_adj_all_min | 1057 | 0.6803 | [0.6627, 0.6984] | 0.3197 |
| 24 | erv_adj_late_min | 1057 | 0.6803 | [0.6627, 0.6984] | 0.3197 |
| 36 | era_adj_late_min | 904 | 0.6767 | [0.6568, 0.6948] | 0.3233 |
| 36 | era_adj_all_min | 904 | 0.6767 | [0.6568, 0.6948] | 0.3233 |
| 36 | era_hist_all_min | 904 | 0.6758 | [0.6557, 0.6940] | 0.3242 |
| 36 | era_hist_late_min | 904 | 0.6758 | [0.6557, 0.6940] | 0.3242 |
| 24 | era_adj_all_min | 904 | 0.6571 | [0.6361, 0.6763] | 0.3429 |
| 24 | era_adj_late_min | 904 | 0.6571 | [0.6361, 0.6763] | 0.3429 |
| 24 | era_hist_all_min | 904 | 0.6563 | [0.6347, 0.6766] | 0.3437 |
| 24 | era_hist_late_min | 904 | 0.6563 | [0.6347, 0.6766] | 0.3437 |
| 24 | era_hist_all_abs_mean | 904 | 0.5271 | [0.5068, 0.5490] | 0.4729 |
| 24 | era_hist_late_abs_mean | 904 | 0.5271 | [0.5068, 0.5490] | 0.4729 |
| 36 | erv_hist_all_abs_mean | 1057 | 0.5269 | [0.5103, 0.5452] | 0.4731 |
| 24 | erv_adj_all_abs_mean | 1057 | 0.5255 | [0.5084, 0.5425] | 0.4745 |
| 24 | era_adj_late_abs_mean | 904 | 0.5198 | [0.4999, 0.5404] | 0.4802 |
| 24 | era_adj_all_abs_mean | 904 | 0.5198 | [0.4999, 0.5404] | 0.4802 |
| 36 | erv_adj_all_abs_mean | 1057 | 0.5186 | [0.5013, 0.5365] | 0.4814 |
| 24 | erv_hist_all_abs_mean | 1057 | 0.5171 | [0.4995, 0.5349] | 0.4829 |
| 24 | erv_hist_late_mean | 1057 | 0.5073 | [0.4903, 0.5249] | 0.4927 |
| 36 | erv_hist_late_abs_mean | 1057 | 0.5056 | [0.4887, 0.5232] | 0.4944 |
| 36 | era_hist_all_abs_mean | 904 | 0.5018 | [0.4815, 0.5219] | 0.4982 |
| 36 | era_hist_late_abs_mean | 904 | 0.5018 | [0.4815, 0.5219] | 0.4982 |
| 24 | erv_adj_late_abs_mean | 1057 | 0.5014 | [0.4839, 0.5197] | 0.4986 |
| 36 | erv_adj_late_abs_mean | 1057 | 0.4963 | [0.4793, 0.5147] | 0.5037 |

## Figures

- `er_local_dynamics_results/figures/ERD_erv_hist_curve.png`
- `er_local_dynamics_results/figures/ERD_era_hist_curve.png`
- `er_local_dynamics_results/figures/ERD_erv_adj_curve.png`
- `er_local_dynamics_results/figures/ERD_era_adj_curve.png`
