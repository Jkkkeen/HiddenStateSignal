# Path Length vs ERV Combo Results

This analysis checks whether Method 3 path-length and Local ER dynamics are redundant or complementary.

## Data

- Joined rollout rows: 8167
- Questions: 1094
- Rollouts: 8167

## Correlations

| x | y | scope | rho | n |
|---|---|---|---:|---:|
| path_score | erv36_adj_late_min | global | 0.6582 | 8167 |
| path_score | erv36_adj_late_min | within_question_mean | 0.3921 | 1070 |
| path_score | erv36_hist_late_min | global | 0.6935 | 8167 |
| path_score | erv36_hist_late_min | within_question_mean | 0.4167 | 1070 |
| path_score | erv24_hist_late_min | global | 0.6983 | 8167 |
| path_score | erv24_hist_late_min | within_question_mean | 0.4042 | 1070 |

## AUROC

| score | questions | mean AUROC | median | 95% CI | opposite mean |
|---|---:|---:|---:|---:|---:|
| combo_path_erv36_hist | 1057 | 0.7153 | 0.8000 | [0.6970, 0.7339] | 0.2847 |
| combo_path_erv36_adj | 1057 | 0.7152 | 0.8000 | [0.6971, 0.7331] | 0.2848 |
| combo_path_erv24_hist | 1057 | 0.7134 | 0.8000 | [0.6951, 0.7305] | 0.2866 |
| path_score | 1057 | 0.7109 | 0.8125 | [0.6917, 0.7285] | 0.2891 |
| erv36_adj_late_min | 1057 | 0.6925 | 0.7500 | [0.6763, 0.7106] | 0.3075 |
| erv24_hist_late_min | 1057 | 0.6903 | 0.7500 | [0.6725, 0.7081] | 0.3097 |
| erv36_hist_late_min | 1057 | 0.6878 | 0.7500 | [0.6703, 0.7062] | 0.3122 |

## Figure

- `combo_results/figures/path_score_vs_erv36_late_min.png`
