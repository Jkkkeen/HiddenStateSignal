# Semantic-Step Relative Analysis

This analysis compares correct vs wrong rollouts within the same question across relative answer progress bins.

- Segment: `answer`
- Progress axis: `relative answer progress`

## Figures

- `figures/G1_relative_state_angle_curve.png`
- `figures/G2_relative_turn_angle_curve.png`
- `figures/G3_correct_wrong_gap_by_relative_step.png`
- `figures/G4_within_question_auc_by_relative_step.png`
- `figures/G5_margin_vs_angle_relative_heatmap.png`
- `figures/G6_turn_to_correct_gap_by_relative_step.png`
- `figures/G7_turn_to_correct_auc_by_relative_step.png`
- `figures/G8_state_margin_curve.png`
- `figures/G9_step_correct_gain_curve.png`

## Top Relative-Bin AUROC

| layer | rel bin | questions | AUROC(+margin) | AUROC(-margin) | best |
|---:|---:|---:|---:|---:|---:|
| 24 | 8 | 65 | 0.6038 | 0.3962 | 0.6038 |
| 24 | 4 | 58 | 0.4017 | 0.5983 | 0.5983 |
| 36 | 2 | 111 | 0.4409 | 0.5591 | 0.5591 |
| 36 | 4 | 58 | 0.5425 | 0.4575 | 0.5425 |
| 36 | 0 | 206 | 0.4589 | 0.5411 | 0.5411 |
| 36 | 9 | 206 | 0.4708 | 0.5292 | 0.5292 |
| 36 | 5 | 159 | 0.4716 | 0.5284 | 0.5284 |
| 36 | 3 | 146 | 0.5184 | 0.4816 | 0.5184 |
| 24 | 7 | 100 | 0.5170 | 0.4830 | 0.5170 |
| 24 | 9 | 206 | 0.4833 | 0.5167 | 0.5167 |
