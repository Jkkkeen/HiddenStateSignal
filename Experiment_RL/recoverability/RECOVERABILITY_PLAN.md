# Recoverability Study: Frozen Protocol v1

## Scientific Question

Can an option-logit trajectory distinguish recoverable errors from collapsed errors?

We define recoverability operationally. Given an incorrect rollout, remove its final conclusion and ask the same model to independently review the retained reasoning and produce a corrected answer:

```text
recovery_rate = correct revision continuations / K
any_recovery  = 1[recovery_rate > 0]
```

An incorrect rollout is a good error only when it has high empirical recovery rate. A high option margin alone does not define a good error.

## Data Roles

The existing `smoke500` MathVerse questions were used to discover and compare option-logit features. They are therefore discovery data, not an untouched holdout.

Version 1 uses these questions only for an engineering and discovery smoke:

```text
split_role = discovery_smoke
questions = 16
incorrect rollouts per question = 4
total prefixes = 64
```

Within each eligible question, four incorrect rollouts are selected at evenly spaced ranks of `trimmed_margin_mean`. Questions are sampled deterministically with seed `20260711` from questions containing at least four incorrect rollouts.

A valid confirmatory result requires newly sampled questions that were not used to choose features, directions, probe positions, thresholds, or prompts.

## Frozen Prefix Definition

For each incorrect rollout:

1. Extract the thinking segment from `response`.
2. Apply the existing `trim_final_conclusion` rule.
3. Retain the resulting text as `revision_prefix`.
4. Flag explicit answer claims that remain in the retained prefix.

An explicit claim for the gold option is excluded from the primary manifest to avoid making recovery trivial. An explicit claim for the rollout's original wrong option is retained: it is part of the erroneous reasoning state, not ground-truth leakage. We record `original_prediction_claim_flag` and report recovery separately for anchored and non-anchored errors.

## Frozen Revision Prompt

The model receives the original image and MCQ prompt, followed by the retained reasoning prefix and this instruction:

```text
The previous reasoning stopped before its final conclusion. Independently review it against the question and image. Correct any mistake you find. Continue the reasoning briefly, then give exactly one final option letter in the format "Final answer: X", where X is A, B, C, or D.
```

The instruction does not reveal whether the original rollout was correct or which option it selected.

## Generation Configuration

```text
model = Qwen3-VL-8B-Thinking base checkpoint
K = 4 independent revision samples per prefix
temperature = 0.8
top_p = 0.95
top_k = -1
max_new_tokens = 4096
max_model_len = 20480
seed = 20260711
```

The first smoke runs on the H200 using only `/data2` for code, data, cache, logs, and outputs.

## Frozen Predictor Set

Primary predictors:

```text
trimmed_margin_mean
trimmed_margin_max
trimmed_relative_margin_mean
trimmed_relative_margin_max
exact_c2_gain_max
exact_c2_gain_mean
```

Secondary controls:

```text
prompt_only_margin_mean
prompt_only_margin_max
think_final_margin_mean
think_final_margin_max
think_late_margin_drop_mean
think_late_margin_drop_max
trimmed_entropy
think_entropy_delta
response length
```

`exact_c2_gain` must use the online C2 definition:

```text
probe fractions = [0.0, 0.25, 0.50, 0.75, 0.90]
reward transitions start after prompt-only
gain = mean(clip(diff(margins[1:]), -0.5, 0.5))
```

If the existing trajectory grid cannot reproduce these exact fractions, exact C2 features must be recomputed before confirmatory analysis. A nearby trajectory feature is not a substitute.

## Primary Analyses

Among incorrect rollouts only:

```text
within-question corr(signal, recovery_rate)
within-question pairwise AUC(signal, recovery_rate)
AUC(signal, any_recovery)
high-vs-low signal recovery-rate ratio
```

The pairwise recovery AUC compares all within-question rollout pairs with unequal recovery rates. Ties in the predictor receive half credit.

Models to compare:

```text
prompt-only
trimmed level
exact C2 gain
trimmed level + exact C2 gain
entropy and response-length controls
```

The incremental value of gain is tested after controlling for trimmed level. Feature directions are fixed before looking at recovery outcomes.

## Smoke Gates

The generation pipeline passes Goal 2 when:

```text
non-truncated answer-format failure rate < 5%
recovery_rate is not almost entirely 0 or 1
no H200 CPU/GPU OOM
manual audit confirms the original image and prefix are present
manual audit finds no gold-answer leakage in the revision instruction or manifest
```

The discovery signal is promising when:

```text
within-question recovery AUC > 0.65
question-bootstrap CI lower bound > 0.50
high-signal errors recover at least 1.5x as often as low-signal errors
direction is consistent across answer labels and question strata
```

These thresholds decide whether to collect a new untouched holdout. They do not by themselves establish the paper claim.

## Confirmatory Controls

Before claiming generalization:

```text
new-question untouched holdout
shuffled recovery labels -> AUC near 0.5
option-order permutation
answer-label stratification
leakage-free subset
question difficulty and response-length controls
```

No reward-aware RL experiment starts until the recoverability signal survives the new-question holdout.

## Decision Path

```text
Goal 1: freeze protocol and manifest
Goal 2: generate K revisions and score recovery
Goal 3: test recoverability prediction on discovery data
Goal 4: collect and evaluate untouched questions with controls
Goal 5: audit good-error mechanisms and build a taxonomy
Goal 6: compare answer-only, naive gain, and recoverability-aware RL
```
