# Qwen3-1.7B Layer-Resolved Hidden Profile Design

## Goal

Use the completed, behaviorally effective Qwen3-1.7B-Base GRPO run to locate
where vertical hidden-state dynamics change across decoder depth. The first
analysis uses each checkpoint's own saved held-out responses, so it measures
the combined effect of policy-weight drift and the resulting text drift. A
fixed-response, representation-only sensitivity analysis is explicitly out of
scope for this first run.

This work does not retrain the model and does not regenerate responses. It
performs one teacher-forced hidden-state forward over the already saved
responses because the formal run retained only reduced online hidden metrics,
not full layer-resolved tensors.

## Frozen Inputs

The experiment uses the existing formal artifacts on H200:

- base model: `/data2/hjk/models/Qwen3-1.7B-Base`;
- FSDP checkpoints: steps 25 through 250 at intervals of 25 under the formal
  checkpoint root;
- generation files: steps 0 through 250 at intervals of 25 under
  `heldout/generations`;
- held-out manifest: the frozen SimpleRL `simplerl_heldout_256.parquet` file.

The audited cohort contains 11 checkpoints, 256 questions per checkpoint, and
8 rollout slots per question, for 22,528 responses. Generation-file order is
an exact `256 x 8` match to the held-out manifest. Every saved response has
already passed an encode/decode round-trip audit, and the longest combined
prompt and response is 4,529 tokens.

The adapter assigns the manifest's stable `question_id` by question position
and assigns rollout slots 0 through 7 within each question. It rejects a file
unless all eight prompts in every group are identical and equal the rendered
manifest prompt. Correctness comes from `acc`; format correctness and reward
are retained as companion metadata.

## Reused Components

The implementation reuses rather than redefines:

- Hugging Face Qwen3 `output_hidden_states` for the model forward;
- veRL `verl.model_merger` for FSDP-to-Hugging-Face checkpoint conversion;
- `trajectory_endpoints` and the frozen 128-token window, 32-token stride;
- `mean_w128_s32` and `last_s32` representations;
- B1-B4 response-progress stages;
- the existing base-common and coordinate calibrator semantics;
- existing V1-V9 aggregate metric functions as a compatibility baseline.

The new code is limited to the Q3 generation adapter, memory-safe pooled
forward, layer-local profile reduction, audits, analysis, and tmux runners.

## Checkpoint Processing

Step 0 loads the base model directly. Each trained checkpoint is converted
with the veRL FSDP merger into a temporary Hugging Face directory. Checkpoints
are processed serially, and only the current temporary merged model is kept.
The temporary model is removed only after that checkpoint's extraction output
and audit have been atomically finalized.

Extraction is restartable at rollout and checkpoint boundaries. Each completed
rollout has a deterministic cache name derived from its rollout ID. An
interrupted run reuses valid pooled caches and never treats a partial output as
a completed checkpoint.

## Teacher-Forced Forward And Pooling

Prompt and response text are tokenized separately without added special
tokens, then concatenated. The extractor records the exact response start and
stop positions and rejects empty responses or a token round-trip mismatch.

The model forward requests hidden states but does not retain full token hidden
tensors on CPU or disk. For every hidden-state layer, response-only states are
pooled on GPU with cumulative sums:

- `mean_w128_s32`: mean of the trailing 128 response tokens at each endpoint;
- `last_s32`: the final response token at each endpoint.

Only the pooled `chunk x hidden-state-layer x hidden-dimension` matrices are
transferred to CPU and written temporarily as float16. Token log probability,
policy entropy, and final-layer hidden norm are reduced to stage controls and
stored as scalars. The implementation limits Qwen3 logits to response
prediction positions so prompt logits are not materialized unnecessarily.

No permanent `response x token x layer x hidden_dim` artifact is allowed.

## Base Calibration

V3 uses a label-blind base common component separately for each representation.
Step-0 pooled caches are retained temporarily, and the existing calibrator
definition is applied:

- common component: rollout-equal after each rollout's chunk mean;
- coordinate mean and sigma: chunk-equal over all base rollouts.

After the calibrator is finalized, step 0 is reduced with that calibrator just
like every trained checkpoint. Temporary pooled caches may then be deleted.
Correctness labels must not affect calibrator fitting.

## Layer-Local Output Schema

Each checkpoint writes one wide Parquet profile partition. A row identifies:

`model, checkpoint, global_step, training_progress, question_id, rollout_id,
rollout_slot, is_correct, stage, representation, layer_index,
relative_depth`.

It also retains response length, trajectory-point count, policy entropy,
token-log-probability mean, hidden-norm mean, difficulty, format correctness,
and reward where available.

Qwen3-1.7B-Base has 28 decoder blocks and therefore 29 hidden-state positions
including position 0, the embedding state. `relative_depth` is
`layer_index / 28`. For update metrics, `layer_index=l` names the destination
state of update `h_l - h_(l-1)`:

- V1 raw and relative update norm exist at layers 1 through 28;
- V3 demeaned adjacent-state angle exists at layers 1 through 28;
- V4 turning between updates ending at `l-1` and `l` exists at layers 2
  through 28;
- V6 raw activation entropy exists at layers 0 through 28;
- V7 centered coordinate entropy of `h_l - h_(l-1)` exists at layers 1
  through 28.

Within each rollout, representation, stage, and layer, values are the median
over valid response chunks in that stage. Structurally undefined locations are
NaN with explicit coverage fields; they are never written as zero.

The existing all-layer V1-V9 long-form aggregate output is also written for
compatibility. This first implementation does not add V2 or the rolling and
cumulative V5/V8/V9 companions.

## Analysis Outputs

The first analysis produces, for V1, V3, V4, V6, and V7:

- layer-by-checkpoint policy-dynamics heatmaps;
- base, early, best, and final layer profiles;
- correct-minus-wrong layer profiles;
- layer-wise within-question AUROC heatmaps;
- checkpoint curves for peak depth, depth center, depth spread, and
  early/middle/late mass.

The behaviorally best checkpoint is selected only from the already frozen
held-out accuracy table. The analysis does not select checkpoints based on a
hidden metric. Question-equal aggregation is used before cross-question
summaries so questions with more valid rollouts do not receive extra weight.

Bootstrap confidence intervals, cluster permutation, nuisance-controlled
models, V2, rolling/cumulative V5/V8/V9, and the fixed-response sensitivity arm
remain follow-up work after the first profile and heatmap audit.

## Smoke And Formal Gates

The smoke cohort is frozen to steps 0, 50, and 250, with the first 16 manifest
questions and all 8 rollout slots. It runs in its own tmux session and must
verify:

- 128 unique rollouts at each checkpoint;
- exact prompt, question, slot, label, and response-boundary mapping;
- 29 hidden-state positions and 28 decoder blocks;
- float16 pooled cache and no full token-hidden files;
- expected structural coverage for all five local profile families;
- agreement between local-profile aggregation and existing V1/V3/V4/V6/V7
  aggregate functions on audited examples;
- checkpoint model identity, input hashes, elapsed time, peak GPU memory, and
  a measured formal ETA;
- nonblank smoke heatmaps with all three checkpoint columns.

Any mismatch blocks formal launch. After the smoke audit passes, the full 11
checkpoint extraction is launched in a detached, explicitly named tmux
session with a persistent log and status JSON. The launcher refuses to start
if the target session already exists, a veRL trainer or hidden extractor is
active, the smoke approval is absent, or the H200 has an unrelated compute
process.

## Failure Handling

Generation/manifest disagreement, tokenizer round-trip failure, missing
checkpoint shards, failed model merging, unexpected layer count, non-finite
profile coverage, or an incomplete Parquet write causes a nonzero exit and a
checkpoint-specific failure record. Existing valid caches and source artifacts
are retained for diagnosis.

The pipeline never silently substitutes the base model for a trained
checkpoint. Before extraction, the checkpoint audit records the requested
global step, SHA256 manifests for the source model shards and merged
safetensors, and a deterministic parameter sample. Every trained checkpoint's
sample must differ from the base sample. Cleanup is restricted to the
experiment's explicitly resolved temporary merge and pooled-cache directories.

## Verification

Unit tests cover generation adaptation, rollout IDs, prompt mismatch failure,
token boundaries, GPU/NumPy pooling equivalence, local profile formulas, layer
index semantics, stage medians, calibrator label blindness, and audit gates.
Runner tests cover serial checkpoint merging, guarded temporary cleanup,
restart behavior, smoke approval, and detached tmux launch.

Remote verification consists of the three-checkpoint smoke audit, inspection
of representative pooled matrices and Parquet rows, heatmap rendering, GPU and
disk telemetry, and the final tmux session/process/log check for the formal
run.
