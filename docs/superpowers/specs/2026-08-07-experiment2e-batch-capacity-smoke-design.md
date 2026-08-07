# Experiment 2E Batch Capacity Smoke Design

## Goal

Measure the largest useful `data.train_batch_size` for the current single-H200
Qwen2.5-7B-Instruct GRPO configuration without modifying the formal run or its
checkpoint history.

## Frozen Starting Point

- Source checkpoint: formal `global_step_3348`.
- The source checkpoint directory is read-only for the smoke.
- Model, dataset, seed, `rollout.n=8`, prompt/response limits, LoRA settings,
  optimizer settings, and dynamic token limits remain unchanged.
- Candidate train batch sizes are tested in order: `2`, `4`, `8`.

## Isolation

Each candidate uses a separate scratch checkpoint root, Ray temp directory,
log, telemetry CSV, and tmux execution context. The scratch root references the
audited step-3348 checkpoint only for loading and must not create or overwrite
files in the formal checkpoint root. Smoke outputs are excluded from the six
formal checkpoints and all hidden-state analyses.

## Measurement

For each candidate:

1. Resume from step 3348.
2. Complete at least two warm-up steps and five measured steps.
3. Record step time, throughput, token count, response length, actor allocated
   and reserved peak memory, one-second `nvidia-smi` memory/utilization samples,
   and any OOM, vLLM preemption, Ray failure, or token-splitting warning.
4. Stop the candidate cleanly and release all GPU processes before starting the
   next candidate.

## Decision Rule

`maximum stable batch` is the largest candidate that completes the measured
steps without an execution failure and retains at least 10% H200 memory
headroom at the observed peak. `maximum useful batch` is the largest stable
candidate whose rollout throughput is not worse than the preceding candidate.
Both values are reported; the formal run remains `train_batch_size=1`.

## Recovery

After the smoke, the formal launcher must still report
`latest_checkpointed_iteration=3348`. SwanLab integration is a separate change
requiring user approval. Once approved, the formal run resumes from 3348 with
`train_batch_size=1` and `rollout.n=8`.
