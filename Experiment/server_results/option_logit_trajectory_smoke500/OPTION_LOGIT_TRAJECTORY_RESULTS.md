# Option Logit Trajectory Results

This experiment stores A/B/C/D logits at fixed reasoning prefixes and evaluates derived margin trajectories.

## Data

- Probe rows: 65586
- Rollout feature rows: 3858
- Questions: 499
- Rollouts: 3858
- Skipped rollouts: 0
- Top option counts: {'A': 65516, 'B': 60, 'C': 8, 'D': 2}

## Primary AUROC

| feature | direction | questions | AUROC(+feature) | 95% CI | AUROC(-feature) | 95% CI | best |
|---|---|---:|---:|---:|---:|---:|---:|
| think_relative_final_margin_max | pos | 206 | 0.7796 | [0.7402, 0.8216] | 0.2204 | [0.1764, 0.2627] | 0.7796 |
| think_early_to_final_gain_max | pos | 206 | 0.7737 | [0.7262, 0.8170] | 0.2263 | [0.1830, 0.2700] | 0.7737 |
| think_relative_late_margin_max | pos | 206 | 0.7458 | [0.6999, 0.7876] | 0.2542 | [0.2134, 0.3001] | 0.7458 |
| think_early_to_late_gain_max | pos | 206 | 0.7299 | [0.6834, 0.7731] | 0.2701 | [0.2283, 0.3169] | 0.7299 |
| think_late_margin_drop_max | neg | 206 | 0.2701 | [0.2269, 0.3166] | 0.7299 | [0.6831, 0.7717] | 0.7299 |
| think_entropy_delta | neg | 206 | 0.4935 | [0.4475, 0.5444] | 0.5065 | [0.4561, 0.5535] | 0.5065 |
| think_final_margin_mean |  | 206 | 0.7890 | [0.7467, 0.8307] | 0.2110 | [0.1683, 0.2527] | 0.7890 |
| think_relative_final_margin_mean |  | 206 | 0.7890 | [0.7467, 0.8307] | 0.2110 | [0.1683, 0.2527] | 0.7890 |
| think_final_margin_max |  | 206 | 0.7796 | [0.7402, 0.8216] | 0.2204 | [0.1764, 0.2627] | 0.7796 |
| think_early_to_final_gain_mean |  | 206 | 0.7763 | [0.7296, 0.8181] | 0.2237 | [0.1836, 0.2684] | 0.7763 |
| think_mean_step_gain_mean |  | 206 | 0.7763 | [0.7296, 0.8181] | 0.2237 | [0.1836, 0.2684] | 0.7763 |
| think_late_margin_mean_mean |  | 206 | 0.7738 | [0.7244, 0.8128] | 0.2262 | [0.1843, 0.2703] | 0.7738 |
| think_relative_late_margin_mean |  | 206 | 0.7738 | [0.7244, 0.8128] | 0.2262 | [0.1843, 0.2703] | 0.7738 |
| think_mean_step_gain_max |  | 206 | 0.7737 | [0.7262, 0.8170] | 0.2263 | [0.1830, 0.2700] | 0.7737 |
| think_max_margin_mean |  | 206 | 0.7613 | [0.7194, 0.8031] | 0.2387 | [0.1959, 0.2815] | 0.7613 |
| think_early_to_late_gain_mean |  | 206 | 0.7501 | [0.7025, 0.7938] | 0.2499 | [0.2111, 0.2998] | 0.7501 |
| think_late_margin_drop_mean |  | 206 | 0.2499 | [0.2062, 0.2975] | 0.7501 | [0.7002, 0.7889] | 0.7501 |
| think_late_margin_mean_max |  | 206 | 0.7458 | [0.6999, 0.7876] | 0.2542 | [0.2134, 0.3001] | 0.7458 |
| think_max_margin_max |  | 206 | 0.7166 | [0.6737, 0.7637] | 0.2834 | [0.2435, 0.3268] | 0.7166 |
| think_min_margin_max |  | 206 | 0.7013 | [0.6623, 0.7424] | 0.2987 | [0.2605, 0.3392] | 0.7013 |
| think_margin_std_mean |  | 206 | 0.6893 | [0.6441, 0.7361] | 0.3107 | [0.2701, 0.3578] | 0.6893 |
| think_positive_gain_rate_mean |  | 206 | 0.6769 | [0.6382, 0.7126] | 0.3231 | [0.2861, 0.3611] | 0.6769 |
| think_min_margin_mean |  | 206 | 0.6650 | [0.6209, 0.7099] | 0.3350 | [0.2878, 0.3781] | 0.6650 |
| think_positive_gain_rate_max |  | 206 | 0.6516 | [0.6188, 0.6880] | 0.3484 | [0.3148, 0.3860] | 0.6516 |

## Figures

- `figures/O1_option_logit_top_auc.png`
- `figures/O2_correct_margin_max_curve.png`
- `figures/O3_correct_margin_mean_curve.png`

## Interpretation Notes

- `top_option` is diagnostic only because raw label logits can have strong option priors.
- Main signals are continuous correct-option margin gain/drop over thinking.
- `prompt_only_margin` controls for questions where the prompt already makes the answer easy.
