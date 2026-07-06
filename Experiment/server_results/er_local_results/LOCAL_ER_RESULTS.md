# Local ER Results

Experiment A computes token-level chunk ER online and saves scalar metrics only.

## Data

- Rows: 68764
- Rollouts: 8936
- Questions: 1117
- Layers: 24, 36
- Metric: `er_centered`

## Top Chunk-Level AUROC

| layer | chunk | questions | AUROC(+ER) | 95% CI | AUROC(-ER) | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0 | 1117 | 0.5216 | [0.5054, 0.5379] | 0.4784 | [0.4623, 0.4950] |
| 24 | 0 | 1117 | 0.5138 | [0.4966, 0.5299] | 0.4862 | [0.4714, 0.5035] |
| 36 | 4 | 665 | 0.4971 | [0.4927, 0.5010] | 0.5029 | [0.4991, 0.5070] |
| 24 | 4 | 665 | 0.4971 | [0.4927, 0.5010] | 0.5029 | [0.4991, 0.5070] |
| 24 | 1 | 1057 | 0.4374 | [0.4198, 0.4551] | 0.5626 | [0.5465, 0.5792] |
| 36 | 1 | 1057 | 0.4191 | [0.4014, 0.4360] | 0.5809 | [0.5641, 0.5975] |
| 24 | 2 | 904 | 0.4132 | [0.3929, 0.4330] | 0.5868 | [0.5682, 0.6069] |
| 24 | 3 | 792 | 0.4106 | [0.3891, 0.4339] | 0.5894 | [0.5672, 0.6134] |
| 36 | 2 | 904 | 0.3844 | [0.3639, 0.4052] | 0.6156 | [0.5962, 0.6358] |
| 36 | 3 | 792 | 0.3828 | [0.3612, 0.4053] | 0.6172 | [0.5955, 0.6388] |

## Figures

- `er_local_results/figures/A2_correct_vs_incorrect_er_curve_layer24.png`
- `er_local_results/figures/A1_correct_vs_incorrect_er_curve_layer36.png`
- `er_local_results/figures/A3_relative_position_er_curve.png`
- `er_local_results/figures/A4_within_question_auroc_by_chunk.png`
- `er_local_results/figures/A5_er_distribution_by_correctness.png`
