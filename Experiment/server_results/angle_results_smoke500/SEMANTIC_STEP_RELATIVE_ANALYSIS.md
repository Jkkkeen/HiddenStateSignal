# Semantic-Step Relative Analysis

This analysis compares correct vs wrong rollouts within the same question across relative thinking progress bins.

- Segment: `thinking`
- Progress axis: `relative thinking progress`

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
| 24 | 9 | 206 | 0.5466 | 0.4534 | 0.5466 |
| 36 | 7 | 206 | 0.4564 | 0.5436 | 0.5436 |
| 24 | 6 | 206 | 0.5393 | 0.4607 | 0.5393 |
| 36 | 1 | 206 | 0.4707 | 0.5293 | 0.5293 |
| 36 | 9 | 206 | 0.4744 | 0.5256 | 0.5256 |
| 36 | 6 | 206 | 0.4776 | 0.5224 | 0.5224 |
| 36 | 8 | 206 | 0.4783 | 0.5217 | 0.5217 |
| 24 | 2 | 206 | 0.4786 | 0.5214 | 0.5214 |
| 36 | 2 | 206 | 0.4826 | 0.5174 | 0.5174 |
| 24 | 8 | 206 | 0.5173 | 0.4827 | 0.5173 |
