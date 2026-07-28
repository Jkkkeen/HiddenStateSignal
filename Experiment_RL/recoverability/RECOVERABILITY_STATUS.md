# Recoverability Goals Status

Date: 2026-07-11

## Goal Status

| Goal | Status | Result |
|---|---|---|
| Goal 1: freeze protocol and build manifest | completed | 16 questions, 64 incorrect prefixes, 4 prefixes per question |
| Goal 2: K=4 revision smoke | completed | 256 generations, strict mean recovery 0.3125 |
| Goal 3: within-question recoverability analysis | completed, gate failed | primary AUC 0.4333, CI [0.2111, 0.6668] |
| Goal 4: untouched holdout and controls | not started by stop gate | discovery signal did not justify new data collection |
| Goal 5: good-error failure/case audit | completed | recovery is dominated by question/error type, not option margin |
| Goal 6: recoverability-aware GRPO | not started by stop gate | no validated reward signal |

## Error-Type Discovery Branch

After the primary stop gate, a low-cost branch was implemented on the same 64 prefixes. It does not use additional H200 generation.

Created:

```text
scripts/build_error_type_annotation.py
scripts/analyze_error_type_predictors.py
recoverability/error_type_discovery_v1/error_type_annotation_blinded.csv
recoverability/error_type_discovery_v1/ERROR_TYPE_ANNOTATION_CODEBOOK.md
recoverability/error_type_discovery_v1/analysis/
```

The blinded sheet excludes recovery outcomes, revision generations, option-logit features, signal ranks, and the precomputed commitment flag.

Exploratory text-proxy results:

| predictor | within-question AUC | 95% CI |
|---|---:|---:|
| explicit wrong commitment, negative direction | 0.5315 | [0.4444, 0.6296] |
| self-correction marker density | 0.4630 | [0.2296, 0.6963] |
| visual-reference density | 0.3963 | [0.1481, 0.6444] |
| option-mapping marker density | 0.5870 | [0.4556, 0.7222] |

None passes the discovery gate. The earlier anchored/non-anchored mean difference is mainly a between-question observation; explicit commitment has little within-question ranking value in this sample.

## Key Findings

```text
strict mean recovery rate = 0.3125
prefixes with any recovery = 34 / 64
non-truncated format failure rate = 3.62%
budget exhausted without explicit answer = 57 / 256
between-question share of recovery variance = 0.87
```

Primary result:

```text
trimmed_margin_mean within-question recovery AUC = 0.4333
95% CI = [0.2111, 0.6668]
mean within-question corr = 0.0401
high/low recovery ratio = 1.19
```

The strongest exploratory feature was negative late margin drop max:

```text
AUC = 0.6074
95% CI = [0.4037, 0.7926]
```

It did not pass the frozen gate.

## Interpretation

The forced-answer option margin predicts final rollout correctness in the original smoke500 analysis, but it does not predict whether an already-wrong reasoning trace can be repaired by additional prompted reasoning.

Recoverable examples in the case audit are mainly:

```text
visual values reread correctly during revision
correct computation remapped to the correct option
```

Persistent errors are often committed geometric or semantic interpretations. Explicitly restating the original wrong answer is associated with lower recovery:

```text
anchored errors mean recovery = 0.2031
non-anchored errors mean recovery = 0.3490
```

This is observational and requires a dedicated intervention before being treated as causal.

## Candidate-Scoring Follow-Up

The existing 64 prefixes were rescored without generating new revisions. The follow-up compares forced A/B/C/D token logits with length-normalized full-option-content log probabilities. Vision-only option text comes from the matching `testmini.json` `question_for_eval` record; real candidate sets range from three to five options.

Primary margin comparison:

| feature | within-question AUC | 95% CI |
|---|---:|---:|
| lower letter mean-margin | 0.5667 | [0.3407, 0.8037] |
| lower content mean-margin | 0.5648 | [0.3278, 0.7741] |
| lower calibrated content mean-margin | 0.5648 | [0.3296, 0.7852] |

The paired AUC difference between content and letter mean-margin is `-0.0019`, with 95% CI `[-0.3056, 0.2352]`. Full option content therefore does not improve the low-margin recoverability hypothesis in this sample.

The strongest exploratory results are uncertainty features:

| feature | within-question AUC | 95% CI |
|---|---:|---:|
| smaller letter top-two gap | 0.6963 | [0.4556, 0.9074] |
| higher calibrated content entropy | 0.6889 | [0.4519, 0.8889] |
| smaller calibrated content top-two gap | 0.6741 | [0.4444, 0.8889] |

These estimates remain inconclusive because only 9 questions have within-question recovery variation and every lower confidence bound is below 0.50. They support an uncertainty-focused follow-up, not recoverability-aware RL.

Artifacts:

```text
recoverability/recoverability_candidate_scoring_v1/
recoverability/recoverability_candidate_scoring_v1/analysis/RECOVERABILITY_CANDIDATE_SCORING.md
```

## H200 Changes

All project artifacts were placed on `/data2`:

```text
/data2/hjk/projects/AI-HiddenState-ER/scripts/build_recoverability_manifest.py
/data2/hjk/projects/AI-HiddenState-ER/scripts/run_recoverability_revision_vllm.py
/data2/hjk/projects/AI-HiddenState-ER/scripts/run_recoverability_revision_smoke.sh
/data2/hjk/projects/AI-HiddenState-ER/scripts/launch_recoverability_revision_tmux.sh
/data2/hjk/projects/AI-HiddenState-ER/scripts/rescore_recoverability_generations.py
/data2/hjk/projects/AI-HiddenState-ER/scripts/analyze_recoverability.py
/data2/hjk/projects/AI-HiddenState-ER/scripts/recoverability_candidate_scoring.py
/data2/hjk/projects/AI-HiddenState-ER/scripts/score_recoverability_candidates_vllm.py
/data2/hjk/projects/AI-HiddenState-ER/scripts/analyze_recoverability_candidate_scores.py
/data2/hjk/projects/AI-HiddenState-ER/scripts/run_recoverability_candidate_scoring.sh
/data2/hjk/projects/AI-HiddenState-ER/scripts/launch_recoverability_candidate_scoring_tmux.sh
/data2/hjk/projects/AI-HiddenState-ER/rl_recoverability/
/data2/hjk/projects/AI-HiddenState-ER/logs/recoverability_*.log
```

Runtime caches were redirected to:

```text
/data2/hjk/cache/vllm
/data2/hjk/cache/torchinductor
/data2/hjk/cache/triton
/data2/hjk/cache/huggingface
```

The first preflight briefly created a 15 MB vLLM compile-cache directory under `/home/hjk/.cache/vllm/torch_compile_cache/1274cd14aa`. Its timestamp and path were verified, then that exact directory was removed. Subsequent runs used `/data2`; no new vLLM files under `/home/hjk/.cache/vllm` were found after the cache redirect.

The formal smoke exited with code 0 and released the H200 GPU.

## Recommended Next Branch

Do not continue with recoverability-aware RL using `trimmed_margin_mean`, full-content mean-margin, or commitment gap.

The engineering and text-proxy part of the error-type branch is complete. The next scientifically justified actions are to fill the blinded 64-row semantic annotation sheet and to pre-register an uncertainty-focused confirmation using top-two gap/entropy. Do not launch recoverability-aware RL unless one of those predictors passes a question-clustered confirmation gate.
