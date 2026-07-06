# Path Dynamics Results

Experiment E decomposes Method 3 path length into adjacent cross-chunk displacement dynamics.

## Data

- Chunk-step rows: 50892
- Rollout-layer rows: 16334
- Questions: 1094
- Rollouts: 8167
- Layers: 24, 36
- Source representation: chunk-mean hidden states from `chunklayer/h_gpu*.npz`

## Primary Metrics

| layer | feature | direction | questions | committed AUROC | AUROC(+feature) | AUROC(-feature) |
|---:|---|---|---:|---:|---:|---:|
| 36 | d_late_mean | neg | 1057 | 0.7211 | 0.2789 [0.2608, 0.2973] | 0.7211 [0.7027, 0.7392] |
| 36 | d_ratio_late_early | neg | 1057 | 0.7167 | 0.2833 [0.2655, 0.3009] | 0.7167 [0.6991, 0.7345] |
| 24 | d_ratio_late_early | neg | 1057 | 0.7156 | 0.2844 [0.2666, 0.3018] | 0.7156 [0.6982, 0.7334] |
| 36 | d_hist_late_min | pos | 904 | 0.2884 | 0.2884 [0.2695, 0.3082] | 0.7116 [0.6918, 0.7305] |
| 24 | d_late_mean | neg | 1057 | 0.7051 | 0.2949 [0.2771, 0.3128] | 0.7051 [0.6872, 0.7229] |
| 36 | pv_late_max | neg | 904 | 0.7021 | 0.2979 [0.2787, 0.3176] | 0.7021 [0.6824, 0.7213] |
| 24 | d_hist_late_min | pos | 904 | 0.3029 | 0.3029 [0.2815, 0.3240] | 0.6971 [0.6760, 0.7185] |
| 24 | pv_late_max | neg | 904 | 0.6965 | 0.3035 [0.2843, 0.3237] | 0.6965 [0.6763, 0.7157] |

## Top Rollout-Level Features

| layer | feature | primary | questions | best AUROC | AUROC(+feature) | AUROC(-feature) |
|---:|---|---:|---:|---:|---:|---:|
| 36 | d_late_mean | True | 1057 | 0.7211 | 0.2789 | 0.7211 |
| 36 | d_late_max | False | 1057 | 0.7211 | 0.2789 | 0.7211 |
| 36 | d_std | False | 1057 | 0.7199 | 0.2801 | 0.7199 |
| 36 | d_max | False | 1057 | 0.7191 | 0.2809 | 0.7191 |
| 36 | d_ratio_late_early | True | 1057 | 0.7167 | 0.2833 | 0.7167 |
| 24 | d_ratio_late_early | True | 1057 | 0.7156 | 0.2844 | 0.7156 |
| 24 | d_std | False | 1057 | 0.7127 | 0.2873 | 0.7127 |
| 24 | d_cv | False | 1057 | 0.7126 | 0.2874 | 0.7126 |
| 36 | d_hist_late_min | True | 904 | 0.7116 | 0.2884 | 0.7116 |
| 36 | d_hist_late_max | False | 904 | 0.7116 | 0.2884 | 0.7116 |
| 36 | d_hist_late_mean | False | 904 | 0.7116 | 0.2884 | 0.7116 |
| 36 | path_length | False | 1057 | 0.7109 | 0.2891 | 0.7109 |
| 36 | d_cv | False | 1057 | 0.7078 | 0.2922 | 0.7078 |
| 24 | d_late_max | False | 1057 | 0.7051 | 0.2949 | 0.7051 |
| 24 | d_late_mean | True | 1057 | 0.7051 | 0.2949 | 0.7051 |
| 24 | d_max | False | 1057 | 0.7042 | 0.2958 | 0.7042 |
| 36 | pv_late_max | True | 904 | 0.7021 | 0.2979 | 0.7021 |
| 36 | pv_late_min | False | 904 | 0.7021 | 0.2979 | 0.7021 |
| 36 | pv_late_mean | False | 904 | 0.7021 | 0.2979 | 0.7021 |
| 24 | path_length | False | 1057 | 0.7008 | 0.2992 | 0.7008 |
| 36 | pv_abs_late_mean | False | 904 | 0.6986 | 0.3014 | 0.6986 |
| 24 | d_hist_late_min | True | 904 | 0.6971 | 0.3029 | 0.6971 |
| 24 | d_hist_late_mean | False | 904 | 0.6971 | 0.3029 | 0.6971 |
| 24 | d_hist_late_max | False | 904 | 0.6971 | 0.3029 | 0.6971 |
| 24 | pv_abs_late_mean | False | 904 | 0.6966 | 0.3034 | 0.6966 |
| 24 | pv_late_max | True | 904 | 0.6965 | 0.3035 | 0.6965 |
| 24 | pv_late_min | False | 904 | 0.6965 | 0.3035 | 0.6965 |
| 24 | pv_late_mean | False | 904 | 0.6965 | 0.3035 | 0.6965 |
| 36 | d_mean | False | 1057 | 0.6403 | 0.3597 | 0.6403 |
| 24 | d_mean | False | 1057 | 0.6086 | 0.3914 | 0.6086 |

## Chunk-Level AUROC Using -d

| layer | step | questions | AUROC(-d) | 95% CI |
|---:|---:|---:|---:|---:|
| 24 | 0 | 1057 | 0.4127 | [0.3940, 0.4304] |
| 24 | 1 | 904 | 0.4031 | [0.3820, 0.4245] |
| 24 | 2 | 792 | 0.4153 | [0.3926, 0.4366] |
| 24 | 3 | 665 | 0.5075 | [0.4816, 0.5330] |
| 36 | 0 | 1057 | 0.4717 | [0.4542, 0.4892] |
| 36 | 1 | 904 | 0.4155 | [0.3961, 0.4376] |
| 36 | 2 | 792 | 0.4350 | [0.4109, 0.4593] |
| 36 | 3 | 665 | 0.5495 | [0.5247, 0.5736] |

## Figures

- `path_dynamics_results/figures/E1_d_curve.png`
- `path_dynamics_results/figures/E2_pv_curve.png`
- `path_dynamics_results/figures/E3_auroc_by_chunk.png`
- `path_dynamics_results/figures/E4_path_dynamics_vs_erv.png`

## Interpretation Notes

- Smaller late displacement means the rollout is moving less in hidden space near the end.
- `pv` is the change in displacement, so positive late `pv` indicates late acceleration.
- This experiment tests whether temporal patterns add information beyond total path length.
- No new model forward is required.
