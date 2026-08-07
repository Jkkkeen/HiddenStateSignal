# Experiment 2E SwanLab Observability Design

## Goal

Expose the resumed Qwen2.5-7B MATH GRPO run in SwanLab without changing its
frozen training semantics: `train_batch_size=1`, `rollout.n=8`, the frozen
manifest, or the checkpoint at `global_step_3348`.

## Architecture

veRL's built-in `swanlab` tracking backend logs the scalar metrics it already
emits on every optimizer step. The formal runner loads the API key from a
mode-600 H200-only environment file outside the repository and enables the
backend alongside `console`.

All hidden-state families are computed only through the existing fixed
checkpoint pipeline: frozen roll8 evaluation cohort, response-only forward,
and scalar reduction. An uploader sends its complete H1-H8/V1-V8, B1-B4
tables and generated images to the same SwanLab experiment after extraction.
This avoids a second all-layer forward inside the active RL loop.

## Observable Data

- Every training step: reward/correctness, advantage, actor loss, KL,
  policy entropy, gradient norm, learning rate, clip fractions, response and
  prompt lengths, truncation, token throughput, timing, and actor memory.
- Every frozen checkpoint: all horizontal and vertical metric families for
  both mean/last representations, correct-vs-wrong means, gaps, AUROC,
  coverage, B1-B4 response stages, overview heatmap, and discriminative plot.

## Security

- The API key is stored only in `/data2/hjk/secrets/experiment_2e_swanlab.env`
  with mode `600` and directory mode `700`.
- The key is never committed, printed, placed in a tmux command line, or
  copied into an artifact.
- The runner fails closed when SwanLab is enabled but the credential file is
  unreadable or does not define `SWANLAB_API_KEY`.

## Validation

1. Install and import the SwanLab package in the veRL Python environment.
2. Start a no-training SwanLab smoke that creates a private test run and logs
   a scalar, then finish it successfully.
3. Verify the resumed formal run is still at step 3348 before launch, uses
   `console` and `swanlab`, and keeps batch one/roll8.
4. Confirm a fresh SwanLab run receives a normal veRL scalar before treating
   the dashboard as live.
