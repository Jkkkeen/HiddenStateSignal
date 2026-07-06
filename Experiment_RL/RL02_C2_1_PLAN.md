# RL02 C2.1 Reward-Probe Architecture Fix Plan

> Scope: fix Experiment C2 engineering so Qwen3-VL option-logit gain can run without a second persistent Qwen3-VL model in the reward worker.

## Root Cause

Goal 4 showed that C2 is blocked by resource architecture, not by reward math:

- A1 answer-only Qwen3-VL Stage 3 finished the 50-step pilot.
- C2 failed around step 15 with vLLM wake-up CUDA OOM.
- The C2 log showed the reward worker loaded a full probe model on CUDA:
  `loaded probe model ... device=cuda`.
- After that, vLLM failed to wake weights/KV cache with CUDA allocator OOM.
- CPU probe avoided GPU OOM but one step took about 664 seconds, so it is not viable.

## Reward Definition To Preserve

C2.1 must keep the same scientific reward:

```text
margin_j = logit(correct_option | prompt + response_prefix_j + probe_suffix)
           - max_wrong logit(wrong_option | prompt + response_prefix_j + probe_suffix)

R_gain_raw = mean_j clip(margin_j - margin_{j-1}, -c, c)
B_gain = clip(zscore_group(R_gain_raw), -2, 2)
R_total = R_answer + lambda * B_gain
```

Main probe points remain late reasoning points. The 0.0 fraction can be logged as a diagnostic anchor, but the reward gain should mainly use 0.25/0.50/0.90 transitions.

## Chosen Architecture

### Preferred path: vLLM prompt-logprobs probe

Use the already-running rollout vLLM server for option probes instead of loading `Qwen3VLForConditionalGeneration.from_pretrained()` inside the reward worker.

For each rollout and each probe fraction:

1. Build `prompt + response_prefix + probe_suffix + option_token`.
2. Request vLLM with `prompt_logprobs=0` and `max_tokens=1`.
3. Read the logprob of the appended option token from `TokenOutput.extra_fields["prompt_logprobs"]`.
4. Repeat for A/B/C/D.
5. Compute margin and clipped gain.
6. Write `option_logit_gain_raw` and per-probe diagnostics into `extra_fields.reward_extra_info`.
7. Let the existing C2 zscore manager compute group zscore and final reward.

This keeps one model owner for rollout/probe and removes the second persistent reward-worker GPU model.

### Fallback path: actor forward probe

If vLLM prompt-logprobs cannot expose the needed option-token logprobs, add an actor-side probe method using the already-loaded FSDP actor model. This is more invasive because it needs TensorDict construction for multimodal probe prompts, so it is fallback only.

### Diagnostic-only path: CPU probe

Keep CPU probe only for correctness debugging. It is too slow for online RL and should not be used for Stage 3 or Stage 4.

## Implementation Tasks

1. Add pure helper tests locally:
   - option-token logprob extraction from vLLM `prompt_logprobs`;
   - option margin from A/B/C/D logprobs;
   - clipped late gain;
   - zscore manager compatibility when `option_logit_gain_raw` is filled later.

2. Add a new C2.1 helper module under `Experiment_RL/scripts`:
   - no model loading;
   - no CUDA allocation;
   - pure construction/parsing functions for vLLM probe.

3. Patch H200 VERL minimally:
   - backup files before edit;
   - allow vLLM `generate()` to preserve integer `prompt_logprobs` and integer `logprobs`;
   - add trainer-side C2.1 probe pass after reward score and before zscore postprocess;
   - write raw probe metrics into `extra_fields.reward_extra_info`.

4. Add a C2.1 smoke runner:
   - `ROLLOUT_N=2`;
   - `TOTAL_STEPS=18` minimum, because old C2 failed around step 15;
   - `MAX_RESPONSE_LENGTH=16384`;
   - no reward-worker probe model;
   - log probe timing and failure counts.

5. If smoke passes, run matched Stage 3 C2.1 50-step pilot:
   - same data/model/settings as A1-long;
   - no Stage 4 until this completes.

## Pass Criteria

C2.1 smoke passes only if all are true:

- no log line saying `loaded probe model ... device=cuda`;
- GPU memory returns to a stable state after Ray stops;
- no vLLM wake-up OOM through at least step 18;
- `option_probe_failed` is near zero;
- `c2_frac_groups_with_nonzero_b_gain_std` is not always zero;
- timing overhead is recorded and not obviously explosive.

If vLLM probe overhead is too high:

- first reduce reward probe points to `0.25,0.50,0.90`;
- if still too slow, pause and implement the actor forward fallback rather than running Stage 3.

## H200 Safety Rules

- All generated files stay under `/data2`.
- No Docker or system package changes.
- Do not run concurrent long jobs on the single H200 GPU.
- Before editing H200 files, create timestamped `.bak_c2_1_YYYYMMDD_HHMMSS` backups.
- Final report must list every H200 file changed and every new result/log path.
