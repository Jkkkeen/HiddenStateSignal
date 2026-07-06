# Hidden-State Trajectory Results

本文汇总目前在 MathVerse + Qwen3-VL-8B-Instruct 上完成的主要实验结果，并标明每个结论对应使用的代码、结果文件和生成图片。

## 1. Experimental Setup

### Data

- Dataset: MathVerse `testmini`
- Raw rollout file: `data/rollouts_mathverse_full_cot.jsonl`
- Raw rollouts: 31,520 = 3,940 questions x 8 rollouts
- MCQ labeled subset: `data/rollouts_mathverse_full_cot_labeled_mcq.jsonl`
- Mixed-correctness subset: `data/rollouts_mathverse_full_cot_labeled_mcq_mixed.jsonl`
- Mixed questions: 1,117
- Mixed rollouts: 8,936

### Model And Hidden States

- Rollout model: `Qwen/Qwen3-VL-8B-Instruct`
- Chunk size: 256 tokens
- Main layers: layer 24 and layer 36
- Chunk hidden files:
  - `chunklayer/h_gpu0.npz` ... `chunklayer/h_gpu7.npz`
  - Total rollouts: 8,936
  - Total chunks: 34,382
  - Skipped rollouts: 0
  - Stored representation: chunk-mean hidden states
  - Shape per hidden chunk: `[37, 4096]`

Important note: the existing `chunklayer/h_gpu*.npz` files store chunk-mean hidden states, not last-token hidden states. Any last-token analysis needs a new forward pass that saves `h_last`.

## 2. Main Findings

The strongest current result is:

> Correct rollouts have more stable hidden-state semantic trajectories. This appears as shorter cross-chunk path length, smoother ER dynamics, and modestly higher angular consistency in middle/late chunks. Across rollouts, higher same-question semantic dispersion predicts lower question accuracy.

| analysis | level | best signal | result | interpretation |
|---|---|---:|---:|---|
| Chunk trajectory Method 3 | single rollout | `M3_L36_path_length` | AUROC 0.7109, CI [0.6923, 0.7284] | Correct rollouts move through hidden space with shorter, more stable trajectories. |
| Local ER dynamics | single rollout | `L36 erv_adj_late_min` | AUROC 0.6925, CI [0.6763, 0.7106] | ER dynamics are stronger than raw ER; semantic breadth changes are informative. |
| Local ER raw | single rollout | `L36 chunk 3 -ER` | AUROC 0.6172 | Later lower ER is more predictive than raw high ER. |
| Cross-rollout ER | question level | `L24 late ER vs accuracy` | Spearman rho -0.4737, p = 1.05e-51 | Hard questions keep rollouts semantically dispersed. |
| Angular dynamics | single rollout | `L36 chunk 3 cos_mean` | AUROC 0.5798 | Correct rollouts show modestly stronger local directional stability. |
| Path + ERV combo | single rollout | `path + erv36_hist` | AUROC 0.7153 | ERV adds small lift over path length, so the two are related but not identical. |

## 3. Chunk Trajectory Method 0/1/3

This experiment evaluates rollout-level hidden trajectories using the chunk-mean hidden states in `chunklayer/h_gpu*.npz`.

### Main Results

| metric | verdict | mean AUROC | median | 95% CI | questions |
|---|---:|---:|---:|---:|---:|
| `M3_L36_path_length` | PASS | 0.7109 | 0.8125 | [0.6923, 0.7284] | 1057 |
| `M3_L24_path_length` | PASS | 0.7008 | 0.8000 | [0.6813, 0.7192] | 1057 |
| `M1_tat_ang` | PASS | 0.6773 | 0.7500 | [0.6594, 0.6946] | 1117 |
| `M3_L24_curvature_ratio` | PASS | 0.6054 | 0.6667 | [0.5869, 0.6232] | 1057 |
| `M3_L36_curvature_ratio` | PASS | 0.5956 | 0.6000 | [0.5771, 0.6125] | 1057 |
| `M0_Z_late_std` | PASS | 0.5913 | 0.5714 | [0.5755, 0.6071] | 1117 |

### Interpretation

Method 3 path length is the strongest single-rollout signal so far. The result supports the hypothesis that correct reasoning traces do not wander as much in hidden-state space. Instead, they show shorter and more coherent semantic trajectories across chunks.

Some direction or magnitude metrics are below 0.5, which usually means the opposite sign is informative or that the metric is not aligned with correctness. Therefore the primary claim should focus on `path_length`, not all trajectory metrics.

### Code And Outputs

| item | path |
|---|---|
| reducer script | `scripts/reducer_chunk_trajectory.py` |
| evaluation script | `scripts/eval_all.py` |
| input hidden files | `chunklayer/h_gpu*.npz` |
| score files | `scores/*.npz` |
| result file | `results/RESULTS.md` |
| local copied result | `RESULTS.md` |

## 4. ER Experiment B: Cross-Rollout ER

Experiment B computes same-question cross-rollout ER. For each question, chunk position, and layer, the 8 rollout chunk-mean hidden vectors are stacked into an `N x 4096` matrix, and ER is computed over this group matrix.

This is not a single-rollout reward. It is a question-level uncertainty or consensus metric.

### Main Results

| layer | phase | target | Spearman rho | p-value | n |
|---:|---|---|---:|---:|---:|
| 24 | early | question accuracy | -0.1309 | 1.13e-05 | 1117 |
| 24 | mid | question accuracy | -0.3294 | 3.51e-26 | 978 |
| 24 | late | question accuracy | -0.4737 | 1.05e-51 | 903 |
| 36 | early | question accuracy | -0.0912 | 0.00228 | 1117 |
| 36 | mid | question accuracy | -0.3256 | 1.39e-25 | 978 |
| 36 | late | question accuracy | -0.4636 | 2.53e-49 | 903 |
| 24 | late | answer entropy | 0.3597 | 5.63e-29 | 903 |
| 36 | late | answer entropy | 0.3560 | 2.29e-28 | 903 |

### Interpretation

Higher cross-rollout ER means that different rollouts for the same question are more dispersed in hidden-state space. The negative correlation with question accuracy shows that lower-accuracy questions maintain stronger semantic disagreement, especially in the late phase.

This supports the question-level story:

- Easy or high-consensus questions tend to converge across rollouts.
- Hard or ambiguous questions keep rollouts semantically spread out.
- Cross-rollout ER helps explain when majority vote or centrality-based selection may be reliable.

### Code, Outputs, And Figures

| item | path |
|---|---|
| ER computation script | `scripts/run_cross_rollout_er.py` |
| plotting script | `scripts/plot_cross_rollout_er.py` |
| input hidden files | `chunklayer/h_gpu*.npz` |
| output parquet | `er_results/cross_rollout_er.parquet` |
| question summary | `er_results/question_summary.parquet` |
| report | `er_results/CROSS_ROLLOUT_ER_RESULTS.md` |
| figure B1 | `er_results/figures/B1_er_group_by_accuracy_layer36.png` |
| figure B2 | `er_results/figures/B2_er_group_by_accuracy_layer24.png` |
| figure B3 | `er_results/figures/B3_er_vs_question_accuracy.png` |
| figure B4 | `er_results/figures/B4_er_vs_answer_entropy.png` |
| figure B5 | `er_results/figures/B5_er_group_heatmap_layer36.png` |

## 5. ER Experiment A: Local ER

Experiment A computes token-level ER inside each rollout chunk. Unlike Experiment B, this is a single-rollout metric and can be used as a candidate process-level risk feature.

### Raw Local ER Results

| layer | chunk | questions | AUROC(+ER) | 95% CI | AUROC(-ER) |
|---:|---:|---:|---:|---:|---:|
| 36 | 0 | 1117 | 0.5216 | [0.5054, 0.5379] | 0.4784 |
| 24 | 0 | 1117 | 0.5138 | [0.4966, 0.5299] | 0.4862 |
| 24 | 1 | 1057 | 0.4374 | [0.4198, 0.4551] | 0.5626 |
| 36 | 1 | 1057 | 0.4191 | [0.4014, 0.4360] | 0.5809 |
| 24 | 2 | 904 | 0.4132 | [0.3929, 0.4330] | 0.5868 |
| 36 | 2 | 904 | 0.3844 | [0.3639, 0.4052] | 0.6156 |
| 24 | 3 | 792 | 0.4106 | [0.3891, 0.4339] | 0.5894 |
| 36 | 3 | 792 | 0.3828 | [0.3612, 0.4053] | 0.6172 |

### Interpretation

Raw local ER is position-dependent. In the first chunk, higher ER gives a weak positive signal, possibly reflecting useful early exploration. In middle and late chunks, lower ER is more predictive of correctness, suggesting that correct rollouts tend to semantically concentrate after exploration.

Raw ER alone is not the strongest signal. The stronger result comes from ER dynamics.

### Code, Outputs, And Figures

| item | path |
|---|---|
| local ER script | `scripts/run_local_er_qwen3vl.py` |
| plotting script | `scripts/plot_local_er.py` |
| input rollouts | `data/rollouts_mathverse_full_cot_labeled_mcq_mixed.jsonl` |
| output parquet | `er_local_results/local_er_gpu*.parquet` |
| report | `er_local_results/LOCAL_ER_RESULTS.md` |
| figure A1 | `er_local_results/figures/A1_correct_vs_incorrect_er_curve_layer36.png` |
| figure A2 | `er_local_results/figures/A2_correct_vs_incorrect_er_curve_layer24.png` |
| figure A3 | `er_local_results/figures/A3_relative_position_er_curve.png` |
| figure A4 | `er_local_results/figures/A4_within_question_auroc_by_chunk.png` |
| figure A5 | `er_local_results/figures/A5_er_distribution_by_correctness.png` |

## 6. Local ER Dynamics: ERV And ERA

ERV and ERA are computed from the existing local ER parquet files, without another model forward pass. These features summarize how local ER changes over time.

### Main Results

| layer | feature | questions | AUROC(+feature) | 95% CI | AUROC(-feature) |
|---:|---|---:|---:|---:|---:|
| 36 | `erv_adj_late_min` | 1057 | 0.6925 | [0.6763, 0.7106] | 0.3075 |
| 36 | `erv_adj_all_min` | 1057 | 0.6925 | [0.6763, 0.7106] | 0.3075 |
| 24 | `erv_hist_late_min` | 1057 | 0.6903 | [0.6725, 0.7081] | 0.3097 |
| 36 | `erv_hist_late_min` | 1057 | 0.6878 | [0.6703, 0.7062] | 0.3122 |
| 24 | `erv_adj_late_min` | 1057 | 0.6803 | [0.6627, 0.6984] | 0.3197 |
| 36 | `era_adj_late_min` | 904 | 0.6767 | [0.6568, 0.6948] | 0.3233 |
| 36 | `era_hist_late_min` | 904 | 0.6758 | [0.6557, 0.6940] | 0.3242 |
| 24 | `era_adj_late_min` | 904 | 0.6571 | [0.6361, 0.6763] | 0.3429 |

### Interpretation

The dynamics of ER are much more informative than raw ER values. This suggests that the important signal is not simply whether a chunk has high or low semantic breadth, but whether the rollout is stabilizing or destabilizing across the reasoning process.

This is one of the best supporting results for a future semantic risk score `u_k`.

### Code, Outputs, And Figures

| item | path |
|---|---|
| analysis script | `scripts/analyze_local_er_dynamics.py` |
| input parquet | `er_local_results/local_er_gpu*.parquet` |
| report | `er_local_dynamics_results/LOCAL_ER_DYNAMICS_RESULTS.md` |
| figure ERV historical | `er_local_dynamics_results/figures/ERD_erv_hist_curve.png` |
| figure ERA historical | `er_local_dynamics_results/figures/ERD_era_hist_curve.png` |
| figure ERV adjacent | `er_local_dynamics_results/figures/ERD_erv_adj_curve.png` |
| figure ERA adjacent | `er_local_dynamics_results/figures/ERD_era_adj_curve.png` |

## 7. Experiment D: Local Chunk Angular Dynamics

Experiment D measures local directional stability inside each chunk. It uses micro-window hidden-state displacements and computes angular or cosine-based statistics.

Precommitted primary metrics:

- `cos_mean`
- `cos_p10`
- `spike_rate_90`
- `AV_cos_mean`

Secondary exploratory metrics include `LAD_*`, `cos_std`, `cos_min`, `spike_rate_105`, `AE_norm_B6`, and angular acceleration features.

### Main Results

| layer | chunk | feature | AUROC(+feature) | interpretation |
|---:|---:|---|---:|---|
| 36 | 3 | `cos_mean` | 0.5798 | Correct rollouts have stronger late local directional consistency. |
| 36 | 2 | `cos_mean` | 0.5635 | Middle-late directional stability is also informative. |
| 24 | 3 | `cos_p10` | 0.5636 | Extreme low-consistency events help distinguish wrong rollouts. |
| 24 | 3 | `AV_cos_mean` | 0.5514 | Angular dynamics provide modest additional signal. |

### Selection Bias Check

Later chunks have sample selection bias because short responses do not reach them. For example, questions without later chunks tend to be easier and shorter.

The selection-bias report shows that the main chunk 2/3 angular AUROCs are stable under common-question restriction:

| layer | chunk | feature | native AUROC | common-question AUROC |
|---:|---:|---|---:|---:|
| 24 | 3 | `cos_min` | 0.5853 | 0.5853 |
| 36 | 3 | `cos_mean` | 0.5798 | 0.5798 |
| 24 | 3 | `cos_p10` | 0.5636 | 0.5636 |
| 36 | 2 | `cos_mean` | 0.5635 | 0.5634 |
| 24 | 2 | `cos_min` | 0.5563 | 0.5563 |

Therefore, later-chunk results should not be compared naively with chunk 0, but the mid/late within-subset angular signal itself appears stable.

### Interpretation

Angular dynamics provide a modest but consistent complementary signal. They are weaker than path length and ERV/ERA, but they support the broader story that correct reasoning trajectories are directionally more stable in middle and late reasoning phases.

### Code, Outputs, And Figures

| item | path |
|---|---|
| angular extraction script | `scripts/run_local_angular_qwen3vl.py` |
| plotting script | `scripts/plot_local_angular.py` |
| selection-bias script | `scripts/analyze_angular_selection_bias.py` |
| tests | `tests/test_local_angular_metrics.py` |
| angular parquet | `angular_results/local_angular_gpu*.parquet` |
| angular dynamics parquet | `angular_results/local_angular_with_dynamics.parquet` |
| main report | `angular_results/LOCAL_ANGULAR_RESULTS.md` |
| selection-bias report | `angular_results/selection_bias/ANGULAR_SELECTION_BIAS.md` |
| figure D cos mean | `angular_results/figures/D_cos_mean_curve.png` |
| figure D cos p10 | `angular_results/figures/D_cos_p10_curve.png` |
| figure D spike rate | `angular_results/figures/D_spike_rate_90_curve.png` |
| figure D angular velocity | `angular_results/figures/D_av_cos_mean_curve.png` |
| figure D AUROC | `angular_results/figures/D_primary_auc_by_chunk.png` |
| figure D4 ER x angular | `angular_results/figures/D4_er_x_cos_mean_scatter.png` |

## 8. Path Length vs ERV Combo

This analysis checks whether the strongest chunk trajectory signal, `M3_L36_path_length`, is redundant with local ER dynamics.

### Correlations

| x | y | scope | Spearman rho | n |
|---|---|---|---:|---:|
| `path_score` | `erv36_adj_late_min` | global | 0.6582 | 8167 |
| `path_score` | `erv36_adj_late_min` | within-question mean | 0.3921 | 1070 |
| `path_score` | `erv36_hist_late_min` | global | 0.6935 | 8167 |
| `path_score` | `erv36_hist_late_min` | within-question mean | 0.4167 | 1070 |
| `path_score` | `erv24_hist_late_min` | global | 0.6983 | 8167 |
| `path_score` | `erv24_hist_late_min` | within-question mean | 0.4042 | 1070 |

### AUROC

| score | questions | mean AUROC | median | 95% CI |
|---|---:|---:|---:|---:|
| `combo_path_erv36_hist` | 1057 | 0.7153 | 0.8000 | [0.6970, 0.7339] |
| `combo_path_erv36_adj` | 1057 | 0.7152 | 0.8000 | [0.6971, 0.7331] |
| `combo_path_erv24_hist` | 1057 | 0.7134 | 0.8000 | [0.6951, 0.7305] |
| `path_score` | 1057 | 0.7109 | 0.8125 | [0.6917, 0.7285] |
| `erv36_adj_late_min` | 1057 | 0.6925 | 0.7500 | [0.6763, 0.7106] |

### Interpretation

Path length and ERV are strongly related globally, but their within-question correlation is only moderate. Combining them gives a small AUROC lift from 0.7109 to 0.7153.

This means they are not fully redundant, but the incremental gain is small. For a clean primary claim, `M3_L36_path_length` should remain the main metric, and ERV/ERA should be presented as mechanism-level supporting evidence.

### Code, Outputs, And Figures

| item | path |
|---|---|
| analysis script | `scripts/analyze_path_erv_combo.py` |
| input scores | `scores/M3_L36_path_length.npz` |
| input ER dynamics | `er_local_dynamics_results/*.parquet` or derived files |
| report | `combo_results/PATH_ERV_COMBO_RESULTS.md` |
| figure | `combo_results/figures/path_score_vs_erv36_late_min.png` |

## 9. Overall Interpretation

The current experiments support a consistent hidden-state trajectory story:

1. Correct rollouts tend to have shorter and more stable chunk-level semantic trajectories.
2. Raw local ER is useful but position-dependent: early high ER may reflect exploration, while late lower ER may reflect convergence.
3. ERV/ERA are stronger than raw ER because they measure semantic dynamics rather than static breadth.
4. Cross-rollout ER captures question-level uncertainty and consensus, not single-rollout correctness.
5. Angular dynamics provide complementary but weaker evidence for local directional stability.
6. Combining path length with ERV gives only a small lift, so the best current single metric is still `M3_L36_path_length`.

## 10. Caveats And Next Steps

### Caveats

- The current hidden cache uses chunk-mean hidden states. Last-token versions are not done.
- Later chunks have selection bias because shorter responses do not reach them.
- Cross-rollout ER should not be used directly as a single-rollout reward because it depends on other rollouts from the same question.
- Raw ER should not be interpreted as simply "higher is better" or "lower is better"; the sign depends on reasoning phase.
- Some metrics were exploratory. For formal reporting, focus on precommitted or clearly motivated primary metrics.

### Recommended Next Steps

1. Add a length-control analysis for `M3_L36_path_length`, ERV, and angular metrics.
2. Run a last-token hidden extraction version that saves `h_last`.
3. Re-run Experiment B with both chunk-mean and last-token hidden states.
4. Build a compact semantic risk score using path length, ERV/ERA, and selected angular features.
5. Keep `M3_L36_path_length` as the main claim and ERV/ERA as the mechanistic support.

## 11. Quick File Index

| experiment | scripts | report | figures |
|---|---|---|---|
| Rollout generation | `scripts/roll_mathverse_vllm.py` | none | none |
| Rollout labeling | `scripts/label_mathverse_rollouts.py` if present on server | none | none |
| Chunk hidden extraction | `scripts/extract_chunk_hidden_qwen3vl.py` | none | none |
| Chunk trajectory M0/M1/M3 | `scripts/reducer_chunk_trajectory.py`, `scripts/eval_all.py` | `results/RESULTS.md` | none |
| ER B cross-rollout | `scripts/run_cross_rollout_er.py`, `scripts/plot_cross_rollout_er.py` | `er_results/CROSS_ROLLOUT_ER_RESULTS.md` | `er_results/figures/B*.png` |
| ER A local raw | `scripts/run_local_er_qwen3vl.py`, `scripts/plot_local_er.py` | `er_local_results/LOCAL_ER_RESULTS.md` | `er_local_results/figures/A*.png` |
| ERV/ERA dynamics | `scripts/analyze_local_er_dynamics.py` | `er_local_dynamics_results/LOCAL_ER_DYNAMICS_RESULTS.md` | `er_local_dynamics_results/figures/ERD_*.png` |
| Angular dynamics D | `scripts/run_local_angular_qwen3vl.py`, `scripts/plot_local_angular.py` | `angular_results/LOCAL_ANGULAR_RESULTS.md` | `angular_results/figures/D*.png` |
| Angular selection bias | `scripts/analyze_angular_selection_bias.py` | `angular_results/selection_bias/ANGULAR_SELECTION_BIAS.md` | none |
| Path + ERV combo | `scripts/analyze_path_erv_combo.py` | `combo_results/PATH_ERV_COMBO_RESULTS.md` | `combo_results/figures/*.png` |

