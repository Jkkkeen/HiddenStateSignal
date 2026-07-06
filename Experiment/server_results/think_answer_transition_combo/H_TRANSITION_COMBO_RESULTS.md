# Think-to-Answer Transition Combo Results

No model forward is used. This analysis joins existing think-stage and answer-stage basin parquet files.

## Data

- Joined rollout-layer rows: 7716
- Rollouts: 3858
- Questions: 499
- Layers: 24, 36

## Top Features

| layer | feature | questions | AUROC(+feature) | 95% CI | AUROC(-feature) | 95% CI | best |
|---:|---|---:|---:|---:|---:|---:|---:|
| 24 | think_final_margin | 206 | 0.6211 | [0.5811, 0.6619] | 0.3789 | [0.3396, 0.4192] | 0.6211 |
| 24 | transition_final_drop | 206 | 0.4001 | [0.3629, 0.4399] | 0.5999 | [0.5588, 0.6393] | 0.5999 |
| 24 | think_step_count | 206 | 0.4239 | [0.3846, 0.4621] | 0.5761 | [0.5397, 0.6156] | 0.5761 |
| 36 | think_step_count | 206 | 0.4239 | [0.3846, 0.4621] | 0.5761 | [0.5397, 0.6156] | 0.5761 |
| 36 | combo_think_answer_low_margin | 206 | 0.5759 | [0.5355, 0.6147] | 0.4241 | [0.3836, 0.4637] | 0.5759 |
| 36 | z_answer_lock_score | 206 | 0.5720 | [0.5325, 0.6104] | 0.4280 | [0.3893, 0.4664] | 0.5720 |
| 36 | answer_mean_gain | 206 | 0.4280 | [0.3896, 0.4675] | 0.5720 | [0.5336, 0.6107] | 0.5720 |
| 36 | answer_lock_score | 206 | 0.5720 | [0.5325, 0.6104] | 0.4280 | [0.3893, 0.4664] | 0.5720 |
| 24 | z_answer_stability_score | 206 | 0.4285 | [0.3896, 0.4671] | 0.5715 | [0.5322, 0.6079] | 0.5715 |
| 24 | answer_stability_score | 206 | 0.4285 | [0.3896, 0.4671] | 0.5715 | [0.5322, 0.6079] | 0.5715 |
| 24 | answer_step_count | 206 | 0.4366 | [0.4007, 0.4728] | 0.5634 | [0.5259, 0.5999] | 0.5634 |
| 36 | answer_step_count | 206 | 0.4366 | [0.4007, 0.4728] | 0.5634 | [0.5259, 0.5999] | 0.5634 |
| 36 | z_answer_stability_score | 206 | 0.4436 | [0.4033, 0.4849] | 0.5564 | [0.5171, 0.5949] | 0.5564 |
| 36 | answer_stability_score | 206 | 0.4436 | [0.4033, 0.4849] | 0.5564 | [0.5171, 0.5949] | 0.5564 |
| 24 | answer_mean_gain | 206 | 0.4442 | [0.4073, 0.4844] | 0.5558 | [0.5193, 0.5949] | 0.5558 |
| 24 | answer_lock_score | 206 | 0.5558 | [0.5156, 0.5927] | 0.4442 | [0.4051, 0.4807] | 0.5558 |
| 24 | z_answer_lock_score | 206 | 0.5558 | [0.5156, 0.5927] | 0.4442 | [0.4051, 0.4807] | 0.5558 |
| 24 | answer_productive_turn_p90 | 206 | 0.4516 | [0.4136, 0.4902] | 0.5484 | [0.5088, 0.5860] | 0.5484 |
| 36 | answer_gain_abs_mean | 206 | 0.5472 | [0.5104, 0.5831] | 0.4528 | [0.4155, 0.4880] | 0.5472 |
| 36 | answer_low_gain_abs_score | 206 | 0.4528 | [0.4169, 0.4896] | 0.5472 | [0.5120, 0.5845] | 0.5472 |
| 36 | answer_gain_std | 206 | 0.5466 | [0.5074, 0.5842] | 0.4534 | [0.4139, 0.4922] | 0.5466 |
| 24 | think_late_gain_mean | 206 | 0.5463 | [0.5096, 0.5857] | 0.4537 | [0.4176, 0.4915] | 0.5463 |
| 36 | answer_low_late_margin_score | 206 | 0.5455 | [0.5070, 0.5870] | 0.4545 | [0.4140, 0.4923] | 0.5455 |
| 36 | answer_late_margin_mean | 206 | 0.4545 | [0.4130, 0.4930] | 0.5455 | [0.5077, 0.5860] | 0.5455 |
| 36 | answer_early_gain_mean | 206 | 0.4555 | [0.4174, 0.4961] | 0.5445 | [0.5065, 0.5836] | 0.5445 |
| 24 | combo_answer_lock_stability | 206 | 0.4577 | [0.4191, 0.4983] | 0.5423 | [0.5023, 0.5801] | 0.5423 |
| 36 | z_answer_low_early_margin_score | 206 | 0.5373 | [0.4984, 0.5757] | 0.4627 | [0.4241, 0.5033] | 0.5373 |
| 36 | answer_low_early_margin_score | 206 | 0.5373 | [0.4984, 0.5757] | 0.4627 | [0.4241, 0.5033] | 0.5373 |
| 36 | answer_early_margin_mean | 206 | 0.4627 | [0.4243, 0.5016] | 0.5373 | [0.4967, 0.5759] | 0.5373 |
| 24 | answer_late_turn_mean | 206 | 0.4638 | [0.4303, 0.5012] | 0.5362 | [0.5000, 0.5701] | 0.5362 |

## Figures

- `figures/H1_transition_combo_top_auc.png`
- `figures/H2_think_late_vs_answer_early_margin.png`
- `figures/H3_think_final_margin_by_correctness.png`

## Notes

- `transition_margin_drop = answer_early_margin_mean - think_late_margin_mean`.
- `answer_lock_score = -answer_mean_gain`, so larger means less answer-stage gain.
- `answer_stability_score = -abs(answer_mean_gain)`, so larger means less answer-stage drift.
