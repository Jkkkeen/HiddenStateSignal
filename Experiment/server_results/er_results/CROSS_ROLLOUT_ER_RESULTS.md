# Cross-rollout ER Results

Current version uses chunk-mean hidden states from `h_chunk`.

## Data

- Rows: 9002
- Questions: 1117
- Layers: 24, 36
- Metric plotted: `er_group_centered`

## Spearman Correlations

| layer | phase | target | rho | p-value | n |
|---:|---|---|---:|---:|---:|
| 24 | early | question_accuracy | -0.1309 | 1.13e-05 | 1117 |
| 24 | early | answer_entropy_norm | 0.1322 | 9.32e-06 | 1117 |
| 24 | early | majority_confidence | -0.1397 | 2.79e-06 | 1117 |
| 24 | mid | question_accuracy | -0.3294 | 3.51e-26 | 978 |
| 24 | mid | answer_entropy_norm | 0.2656 | 2.93e-17 | 978 |
| 24 | mid | majority_confidence | -0.2748 | 2.12e-18 | 978 |
| 24 | late | question_accuracy | -0.4737 | 1.05e-51 | 903 |
| 24 | late | answer_entropy_norm | 0.3597 | 5.63e-29 | 903 |
| 24 | late | majority_confidence | -0.3373 | 1.81e-25 | 903 |
| 36 | early | question_accuracy | -0.0912 | 0.00228 | 1117 |
| 36 | early | answer_entropy_norm | 0.0996 | 0.000853 | 1117 |
| 36 | early | majority_confidence | -0.1191 | 6.59e-05 | 1117 |
| 36 | mid | question_accuracy | -0.3256 | 1.39e-25 | 978 |
| 36 | mid | answer_entropy_norm | 0.2485 | 3.1e-15 | 978 |
| 36 | mid | majority_confidence | -0.2635 | 5.41e-17 | 978 |
| 36 | late | question_accuracy | -0.4636 | 2.53e-49 | 903 |
| 36 | late | answer_entropy_norm | 0.3560 | 2.29e-28 | 903 |
| 36 | late | majority_confidence | -0.3350 | 4e-25 | 903 |

## Figures

- `er_results/figures/B2_er_group_by_accuracy_layer24.png`
- `er_results/figures/B1_er_group_by_accuracy_layer36.png`
- `er_results/figures/B3_er_vs_question_accuracy.png`
- `er_results/figures/B4_er_vs_answer_entropy.png`
- `er_results/figures/B5_er_group_heatmap_layer36.png`

## Interpretation Notes

- Higher centered ER means same-question rollouts are more dispersed at that chunk position.
- This is a question-level uncertainty/consensus signal, not a single-rollout reward.
- A last-token version is pending new forward output with `h_last`.
