# Goal 4 Stage 3 Pilot Report

## Status

Goal 4 reached a clear stopping point: A1-long answer-only completed, but C2-long-zscore did not complete a matched 50-step pilot. Stage 4 should not start yet.

C2 is not numerically dead: early C2 steps show nonzero group-level B-gain ranking. The blocker is engineering/resource architecture: the current reward worker loads a second Qwen3-VL probe model, which conflicts with VERL/vLLM GPU memory during rollout wake-up.

## Runs

| run | setting | status | log |
|---|---|---|---|
| A1 | answer-only, max_response=16384, prompt=4096, n=2, steps=50 | completed, exit 0 | `qwen3vl8b_stage3_a1_long_answer_pilot_trainonly_prompt4096_seed42.log` |
| C2-gpu050 | answer + zscored option gain, GPU probe, rollout_gpu_memory_utilization=0.50 | failed after step 15 | `qwen3vl8b_stage3_c2_zscore_pilot_trainonly_prompt4096_seed42.log` |
| C2-gpu040 | same C2, rollout_gpu_memory_utilization=0.40 | failed after step 15 | `qwen3vl8b_stage3_c2_zscore_pilot_trainonly_prompt4096_gpu040_seed42.log` |
| C2-cpu-probe-smoke3 | same C2 reward, `OPTION_GAIN_DEVICE=cpu`, steps=3 | manually stopped after step 1 because too slow | `qwen3vl8b_stage3_c2_zscore_pilot_trainonly_prompt4096_cpu_probe_smoke3_seed42.log` |

## A1 Result

A1 completed all 50 training steps.

Key parsed metrics:

| metric | value |
|---|---:|
| steps | 50 |
| mean training score | 0.82 |
| final training score | 1.0 |
| mean response length | 4399 tokens |
| max step mean response length | 14065.5 tokens |
| mean step time | 78.86 s |
| mean generation time | 42.61 s |

This confirms the Qwen3-VL long-CoT training path can run answer-only under the Stage 3 settings.

## C2 Early Signal

For both GPU-probe C2 attempts, the first 15 steps were consistent and showed process-reward signal before crashing.

Key parsed metrics from C2-gpu050/gpu040 first 15 steps:

| metric | value |
|---|---:|
| last completed step | 15 |
| mean training score | 0.8667 |
| mean response length | 3592.77 tokens |
| mean `c2_frac_groups_with_nonzero_b_gain_std` | 0.2667 |
| mean `c2_frac_all_same_answer_groups_with_b_gain_ranking` | 0.2333 |
| max `c2_frac_groups_with_nonzero_b_gain_std` | 0.5 |
| max `c2_frac_all_same_answer_groups_with_b_gain_ranking` | 0.5 |

Interpretation: `B_gain` sometimes changes within-group ranking, including all-correct/all-wrong groups where answer reward alone is constant. So the C2 reward is not all zero and has the kind of GRPO-relevant signal RL02 wanted to check.

## Blocker

Both GPU-probe C2 runs failed at the same point with vLLM allocator OOM during wake-up:

```text
RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:62
Invocation of wake_up method failed
```

Lowering `actor_rollout_ref.rollout.gpu_memory_utilization` from 0.50 to 0.40 did not solve it. A1 completed under the same model/data/training settings, so the added GPU-resident probe model is the likely cause.

CPU-probe avoids the GPU-resident probe model but is not viable as a formal pilot path: step 1 took 664.4 s and CPU memory reached about 64 GiB. A 50-step C2 pilot would be roughly 9+ hours, before considering additional long-response variance.

## Decision

Do not run Stage 4. Do not treat this as a scientific failure of option-logit gain. Treat it as an implementation/resource blocker for the current online-probe design.

Recommended next goal: C2.1 reward-probe architecture fix. The best direction is to avoid a second full Qwen3-VL model on GPU inside the reward worker, for example:

1. Compute option logits from an already-loaded model path in VERL/actor/ref if accessible.
2. Implement full-sequence multi-position logits in one forward rather than repeated probe calls.
3. Try a much smaller/quantized probe only as a diagnostic, clearly marked as no longer exactly the same reward as Qwen3-VL self-probe.
4. Reduce probe frequency or compute C2 bonus offline/periodically for analysis before putting it in the online reward loop.

## Local Artifacts

This directory contains pulled logs, runner scripts, reward scripts, and `goal4_log_summary.json`:

```text
server_results/qwen3vl_stage3_pilot_goal4/
```

## H200 Changes Made In This Goal

Added or updated earlier under `/data2/hjk/projects/AI-HiddenState-ER/scripts/`:

```text
run_verl_qwen3vl_stage3_a1_long_pilot.sh
run_verl_qwen3vl_stage3_c2_zscore_pilot.sh
launch_qwen3vl_stage3_tmux.sh
```

Generated/used data:

```text
/data2/hjk/projects/AI-HiddenState-ER/rl_data/mathverse_qwen3vl_stage3_pilot200_128/
```

Generated logs:

```text
/data2/hjk/projects/AI-HiddenState-ER/logs/qwen3vl8b_stage3_a1_long_answer_pilot_trainonly_prompt4096_seed42.log
/data2/hjk/projects/AI-HiddenState-ER/logs/qwen3vl8b_stage3_c2_zscore_pilot_trainonly_prompt4096_seed42.log
/data2/hjk/projects/AI-HiddenState-ER/logs/qwen3vl8b_stage3_c2_zscore_pilot_trainonly_prompt4096_gpu040_seed42.log
/data2/hjk/projects/AI-HiddenState-ER/logs/qwen3vl8b_stage3_c2_zscore_pilot_trainonly_prompt4096_cpu_probe_smoke3_seed42.log
```

No Docker/system install was performed. Runtime caches/checkpoints/temp remained under `/data2`. At the end of the run, Ray was stopped and GPU memory was released.
