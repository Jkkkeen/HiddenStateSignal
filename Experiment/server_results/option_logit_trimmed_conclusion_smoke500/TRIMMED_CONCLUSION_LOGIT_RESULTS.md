# Trimmed-Conclusion Option Logit Results

This experiment removes the final conclusion sentence from thinking and probes A/B/C/D logits once.

## Data

- Probe rows: 3858
- Feature rows: 3858
- Questions: 499
- Rollouts: 3858
- Skipped rollouts: 0
- Trim status: {'trigger': 3797, 'last_sentence': 61}
- Top option counts: {'A': 3755, 'B': 100, 'C': 3}

## Primary AUROC

| feature | direction | questions | AUROC(+feature) | 95% CI | AUROC(-feature) | 95% CI | best |
|---|---|---:|---:|---:|---:|---:|---:|
| trimmed_margin_mean | pos | 206 | 0.7967 | [0.7580, 0.8360] | 0.2033 | [0.1645, 0.2439] | 0.7967 |
| trimmed_relative_margin_mean | pos | 206 | 0.7967 | [0.7580, 0.8360] | 0.2033 | [0.1645, 0.2439] | 0.7967 |
| trimmed_margin_max | pos | 206 | 0.7892 | [0.7486, 0.8280] | 0.2108 | [0.1749, 0.2524] | 0.7892 |
| trimmed_relative_margin_max | pos | 206 | 0.7892 | [0.7486, 0.8280] | 0.2108 | [0.1749, 0.2524] | 0.7892 |
| original_minus_trimmed_final_margin_max | neg | 206 | 0.5500 | [0.5080, 0.5913] | 0.4500 | [0.4079, 0.4892] | 0.5500 |
| original_minus_trimmed_final_margin_mean | neg | 206 | 0.5442 | [0.5033, 0.5875] | 0.4558 | [0.4130, 0.4997] | 0.5442 |
| think_final_margin_mean |  | 206 | 0.7890 | [0.7467, 0.8307] | 0.2110 | [0.1683, 0.2527] | 0.7890 |
| think_relative_final_margin_mean |  | 206 | 0.7890 | [0.7467, 0.8307] | 0.2110 | [0.1683, 0.2527] | 0.7890 |
| think_final_margin_max |  | 206 | 0.7796 | [0.7402, 0.8216] | 0.2204 | [0.1764, 0.2627] | 0.7796 |
| think_relative_final_margin_max |  | 206 | 0.7796 | [0.7402, 0.8216] | 0.2204 | [0.1764, 0.2627] | 0.7796 |
| trimmed_entropy |  | 206 | 0.5264 | [0.4779, 0.5768] | 0.4736 | [0.4245, 0.5236] | 0.5264 |
| trimmed_is_top_correct |  | 206 | 0.5026 | [0.4967, 0.5075] | 0.4974 | [0.4918, 0.5038] | 0.5026 |
| prompt_only_margin_max |  | 206 | 0.5000 | [0.5000, 0.5000] | 0.5000 | [0.5000, 0.5000] | 0.5000 |
| prompt_only_margin_mean |  | 206 | 0.5000 | [0.5000, 0.5000] | 0.5000 | [0.5000, 0.5000] | 0.5000 |

## Figures

- `figures/T1_trimmed_top_auc.png`
- `figures/T2_original_vs_trimmed_margin.png`

## Interpretation Notes

- If trimmed margins remain strong, the option-logit signal is not only final-answer leakage.
- If original-minus-trimmed is the strongest feature, the previous result depended heavily on final conclusion text.
