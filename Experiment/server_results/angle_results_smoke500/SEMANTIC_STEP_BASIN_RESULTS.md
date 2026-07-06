# Semantic-Step Correct-Answer Basin Results

This experiment forwards existing rollouts and computes answer-basin alignment on semantic steps only.

## Data

- Step-layer rows: 1394036
- Rollout-layer rows: 7716
- Rollouts: 3858
- Questions: 499
- Skipped rollouts: 0
- Mean semantic steps: 180.67
- Mean rethink steps: 58.23

## Primary AUROC

| layer | feature | questions | AUROC(+feature) | 95% CI | AUROC(-feature) | 95% CI | best |
|---:|---|---:|---:|---:|---:|---:|---:|
| 24 | early_to_late_margin_gain | 206 | 0.5321 | [0.4877, 0.5691] | 0.4679 | [0.4266, 0.5072] | 0.5321 |
| 24 | productive_turn_p90 | 206 | 0.4696 | [0.4335, 0.5083] | 0.5304 | [0.4903, 0.5664] | 0.5304 |
| 36 | late_turn_to_correct | 206 | 0.5295 | [0.4908, 0.5668] | 0.4705 | [0.4335, 0.5106] | 0.5295 |
| 36 | mean_step_correct_gain | 206 | 0.4724 | [0.4331, 0.5106] | 0.5276 | [0.4861, 0.5731] | 0.5276 |
| 24 | late_margin_mean | 206 | 0.5261 | [0.4819, 0.5644] | 0.4739 | [0.4338, 0.5173] | 0.5261 |
| 24 | mean_step_correct_gain | 206 | 0.5220 | [0.4794, 0.5632] | 0.4780 | [0.4397, 0.5206] | 0.5220 |
| 36 | productive_turn_p90 | 206 | 0.4783 | [0.4421, 0.5116] | 0.5217 | [0.4868, 0.5598] | 0.5217 |
| 36 | late_margin_mean | 206 | 0.4836 | [0.4417, 0.5214] | 0.5164 | [0.4797, 0.5523] | 0.5164 |
| 24 | late_turn_to_correct | 206 | 0.4880 | [0.4472, 0.5272] | 0.5120 | [0.4739, 0.5532] | 0.5120 |
| 36 | early_to_late_margin_gain | 206 | 0.4949 | [0.4530, 0.5359] | 0.5051 | [0.4692, 0.5397] | 0.5051 |

## Interpretation

- `late_margin_mean`: late semantic states are closer to correct option than wrong options.
- `mean_step_correct_gain`: semantic steps increase correct-answer margin on average.
- `late_turn_to_correct`: late step displacements point toward the correct option direction.
- `productive_turn_p90`: large turns that also increase correct-answer margin.

## Files

- `semantic_step_basin_steps.parquet`
- `semantic_step_basin_rollout_features.parquet`
- `semantic_step_basin_eval.csv`
- `semantic_step_basin_skipped.jsonl`
