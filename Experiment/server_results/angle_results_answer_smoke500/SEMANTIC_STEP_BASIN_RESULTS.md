# Semantic-Step Correct-Answer Basin Results

This experiment forwards existing rollouts and computes answer-basin alignment on semantic steps only.

## Data

- Segment: answer
- Step mode: auto
- Step-layer rows: 47662
- Rollout-layer rows: 7716
- Rollouts: 3858
- Questions: 499
- Skipped rollouts: 0
- Mean semantic steps: 6.18
- Mean rethink steps: 0.22

## Primary AUROC

| layer | feature | questions | AUROC(+feature) | 95% CI | AUROC(-feature) | 95% CI | best |
|---:|---|---:|---:|---:|---:|---:|---:|
| 36 | mean_step_correct_gain | 206 | 0.4280 | [0.3888, 0.4686] | 0.5720 | [0.5336, 0.6102] | 0.5720 |
| 24 | mean_step_correct_gain | 206 | 0.4442 | [0.4068, 0.4811] | 0.5558 | [0.5184, 0.5933] | 0.5558 |
| 24 | productive_turn_p90 | 206 | 0.4516 | [0.4153, 0.4895] | 0.5484 | [0.5100, 0.5819] | 0.5484 |
| 36 | late_margin_mean | 206 | 0.4545 | [0.4125, 0.4910] | 0.5455 | [0.5078, 0.5859] | 0.5455 |
| 24 | late_turn_to_correct | 206 | 0.4638 | [0.4307, 0.5009] | 0.5362 | [0.5003, 0.5709] | 0.5362 |
| 24 | early_to_late_margin_gain | 206 | 0.4778 | [0.4367, 0.5183] | 0.5222 | [0.4873, 0.5557] | 0.5222 |
| 24 | late_margin_mean | 206 | 0.4796 | [0.4421, 0.5213] | 0.5204 | [0.4811, 0.5566] | 0.5204 |
| 36 | early_to_late_margin_gain | 206 | 0.5201 | [0.4798, 0.5567] | 0.4799 | [0.4394, 0.5139] | 0.5201 |
| 36 | productive_turn_p90 | 206 | 0.5046 | [0.4654, 0.5410] | 0.4954 | [0.4565, 0.5334] | 0.5046 |
| 36 | late_turn_to_correct | 206 | 0.4968 | [0.4600, 0.5364] | 0.5032 | [0.4618, 0.5400] | 0.5032 |

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
