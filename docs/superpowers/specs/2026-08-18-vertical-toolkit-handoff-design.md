# Vertical Toolkit Handoff Design

## Goal

Create a standalone `vertical-toolkit` branch for collaborators who already
have saved hidden states from Qwen3-8B-Base and MiMo-7B-Base/SFT. The branch
must let them inspect heterogeneous artifacts, compute layer-resolved vertical
metrics offline, run a smoke test, and launch the formal calculation in tmux
without understanding the current Qwen3-1.7B GRPO/checkpoint pipeline.

The primary deliverable is a reproducible tutorial plus a model-agnostic CPU
analysis toolkit. Collaborators run the toolkit on their own server and return
small audit and summary artifacts. Raw hidden states, model weights, pooled
caches, and large formal outputs are never committed to Git.

## Branch Strategy

The handoff branch is named `vertical-toolkit` and starts from `master`.

It does not start from `codex/rl03-stage-a` because that branch contains many
unrelated experiments and the current Qwen3-1.7B layer-profile pipeline is
tightly coupled to Experiment 2E paths, checkpoint conversion, saved
generation files, and H200 runtime assumptions.

The existing Qwen3-1.7B formal pipeline remains on its current experimental
branch. It is a validated reference implementation, not a runtime dependency
of the handoff toolkit. Useful formulas and tests may be ported into the new
toolkit, but Qwen3-1.7B checkpoint merging, model forward, generation adapters,
and server-specific paths are out of scope.

## Users And Collaboration Model

The first users are collaborators with shell access to another compute server
and locally stored hidden-state artifacts. They should be able to follow
`vertical_readme.md` from environment setup through final audit.

The expected collaboration flow is:

1. We push documentation, code, tests, example configurations, and a tiny
   synthetic fixture to `vertical-toolkit`.
2. A collaborator clones the repository and checks out that branch.
3. The collaborator runs the read-only inspector on Qwen3 and MiMo data.
4. The collaborator configures the appropriate adapter and runs a smoke test.
5. After the smoke gates pass, the collaborator launches the formal job in a
   detached tmux session.
6. The collaborator returns only small audit JSON files, summary tables,
   selected figures, and an optional adapter patch.

Collaborators are not required to push experimental outputs. If an adapter
must change, it may be contributed through a small feature branch or pull
request. Large data and results remain on the compute server.

## Repository Layout

The branch adds a self-contained top-level toolkit:

```text
vertical_readme.md
vertical/
  __init__.py
  schema.py
  inspect.py
  pooling.py
  calibrate.py
  profiles.py
  depth.py
  statistics.py
  plotting.py
  audit.py
  adapters/
    __init__.py
    base.py
    npz.py
    numpy_directory.py
    custom_template.py
configs/
  qwen3_8b.example.yaml
  mimo_7b.example.yaml
scripts/
  inspect_vertical_data.sh
  run_vertical_smoke.sh
  run_vertical_formal.sh
  launch_vertical_formal_tmux.sh
examples/
  synthetic_hidden/
tests/
requirements-vertical.txt
```

The exact module names may be refined during the implementation plan, but the
ownership boundaries are frozen:

- adapters only read source artifacts and emit the canonical record stream;
- pooling only converts response-token states into frozen representations;
- calibration only fits and loads label-blind Base references;
- profiles contain per-layer metric formulas;
- statistics and plotting consume tabular profile outputs, not raw tensors;
- audit owns gates and machine-readable run status.

## Tutorial Structure

`vertical_readme.md` is the collaborator-facing entry point. It contains:

1. the scientific question and the limits of supported claims;
2. environment creation and dependency installation;
3. data inventory commands that do not load an entire dataset;
4. the required metadata and hidden-state axis semantics;
5. inspector usage and interpretation of its report;
6. adapter selection and configuration examples;
7. response-only pooling, representations, and B1-B4 stage definitions;
8. Base calibration rules;
9. formulas and layer indexing for every supported metric;
10. smoke execution, gates, and representative output checks;
11. formal tmux launch, monitoring, restart, and failure diagnosis;
12. output directory and table schemas;
13. safe result-sharing instructions;
14. interpretation warnings for Qwen3/MiMo and Base/SFT comparisons.

Commands in the tutorial use configuration files and relative project paths.
No example depends on `/data2/hjk`, a local Windows path, or a specific user
name. Server-specific paths live only in untracked local configuration files.

## Input Modes

The saved artifact formats are currently heterogeneous or unknown, so the
toolkit supports two canonical input modes.

### Raw response-token states

A record supplies a tensor with logical shape:

```text
[response_token, hidden_state_position, hidden_dimension]
```

The adapter may transpose a source-specific physical layout, but it must
declare the source axes and produce this canonical order. If prompt or padding
tokens are present, the adapter must also provide an exact response slice or
mask. The core never guesses response boundaries.

### Pooled response trajectories

A record supplies one or both tensors with logical shape:

```text
[trajectory_endpoint, hidden_state_position, hidden_dimension]
```

The record must include endpoint token indices or normalized response progress
and must identify the representation. The toolkit verifies rather than assumes
that the representation matches the frozen pooling specification.

## Read-Only Preflight Inspector

Every new data source must pass `inspect` before smoke or formal execution.
The inspector samples a bounded number of files and reports:

- file format, file count, and size distribution;
- available keys and metadata fields;
- tensor shape, dtype, finite-value rate, and sampled ranges;
- declared and plausible axis orders;
- hidden-state-position count and hidden dimension;
- response boundary or progress coverage;
- prompt/padding inclusion where metadata makes this auditable;
- question, rollout, model, condition, and label identifiers;
- duplicate identifiers and missing records;
- possible MTP or non-decoder states;
- whether Base and SFT records are aligned by question and response;
- whether raw or pooled mode is usable.

The inspector writes `input_audit.json` and exits nonzero when a required fact
cannot be established. It may suggest adapter configuration, but it never
silently chooses an ambiguous axis order or layer interpretation.

## Canonical Record Contract

Each adapter emits records with the following required metadata:

```text
record_id
model_family
model_name
condition
question_id
rollout_id
is_correct
response_token_count
num_decoder_layers
hidden_state_count
hidden_dimension
source_path
```

Each record also provides either raw response-token states or pooled
trajectories. Optional metadata includes checkpoint, training progress,
rollout slot, reward, format correctness, difficulty, response hash, token
log-probability, policy entropy, and hidden norm.

`condition` distinguishes at least `base` and `sft`. `checkpoint` is optional
because Qwen3-8B currently has only Base data, while MiMo has Base/SFT data.
Layer counts and dimensions are read from audited artifacts or configuration;
they are never hard-coded to Qwen3-1.7B values.

## Layer Semantics

The canonical main profile contains the embedding output followed by decoder
block outputs:

```text
h_0, h_1, ..., h_L
```

where `h_0` is the embedding-state position and `L` is the number of decoder
blocks. `relative_depth = layer_index / L`.

For update metrics, `layer_index=l` names the destination state of the update
`h_l - h_(l-1)`. Structurally undefined values remain NaN with explicit
coverage fields; they are never written as zero.

MiMo MTP or other predictive-head states are excluded from the decoder main
profile. If their semantics can be audited, they are retained in a separate
sensitivity output with an explicit component label. They do not alter the
decoder denominator used for relative depth.

## Frozen Pooling And Stages

Raw response-token states are reduced using two representations:

- `mean_w128_s32`: mean of the trailing window of at most 128 response tokens
  at endpoints separated by 32 response tokens;
- `last_s32`: the final response-token state at the same endpoints.

Progress is endpoint index divided by response-token count. Response progress
is divided into four stages:

```text
B1: [0.00, 0.25)
B2: [0.25, 0.50)
B3: [0.50, 0.75)
B4: [0.75, 1.00]
```

Within a rollout, representation, stage, and layer, chunk-level metric values
are reduced by their median. Coverage counts are preserved for every metric.

## Base Calibration

V3 and standardized comparisons use label-blind Base calibrators fitted
separately for each model family and representation.

- Qwen3 Base fits the Qwen3 calibrator.
- MiMo Base fits the MiMo calibrator.
- The same MiMo Base calibrator is applied to both MiMo Base and MiMo SFT.
- A Qwen3 calibrator is never applied to MiMo, and vice versa.
- Correctness labels never affect calibrator fitting.

The common component uses rollout-equal weighting after each rollout's chunk
mean. Coordinate mean and sigma use chunk-equal weighting across the Base
calibration cohort. Calibrator inputs, weighting rules, shapes, and hashes are
recorded in a calibration audit.

## Metric Delivery Stages

The branch ultimately targets the complete layer-resolved plan, but delivery
is staged so data and layer semantics are validated before expensive or
inferential extensions are added.

### v0.1 stable: end-to-end handoff

The first runnable release contains:

- V1 raw and relative layer-update norm;
- V3 demeaned adjacent-state angle;
- V4 adjacent layer-update turning angle;
- V6 raw activation coordinate-energy entropy;
- V7 layer-difference coordinate-energy entropy;
- policy profiles, correct-minus-wrong profiles, and within-question AUROC;
- peak depth, depth center, depth spread, and early/middle/late mass;
- standard heatmaps and selected profile curves;
- input, extraction, analysis, and final audits;
- smoke and detached tmux formal runners.

### v0.2 metrics-complete

The second release adds:

- V2 raw adjacent-state angle;
- rolling and cumulative V5 path length, net displacement, straightness, and
  log-detour;
- rolling and cumulative V8 layer-state and layer-update effective rank;
- rolling and cumulative V9 vertical effective degree;
- transition depth and metric-specific depth summaries.

Rolling companions use four consecutive layer updates as the primary window.
Cumulative companions start at the embedding state and end at the current
decoder depth.

### v0.3 inference-complete

The third release adds:

- question bootstrap confidence intervals;
- layer-wise cluster permutation for continuous significant depth bands;
- nuisance-controlled analyses for response length, trajectory-point count,
  policy entropy when available, and correctness where appropriate;
- paired MiMo Base/SFT analysis when record alignment permits it;
- within-family Base-standardized cross-model summaries;
- ability-coupling summaries only when enough independent model states or
  checkpoints exist.

The metric registry labels each metric and analysis as `stable` or
`experimental`. The tutorial does not describe an experimental component as
validated before its tests and smoke gates pass.

## Statistical Units And Supported Claims

Question-equal aggregation is the default so questions with more valid
rollouts do not receive extra weight. Correct-versus-wrong AUROC is computed
within questions and therefore requires mixed-outcome questions with enough
rollouts. The audit reports when this analysis is unsupported.

Qwen3-8B Base alone supports descriptive layer profiles and outcome separation
but not training-dynamics claims. MiMo Base/SFT supports a condition comparison.
If the two conditions used different generated responses, the comparison is
explicitly labeled as combined weight and text drift. A representation-only
claim requires the same prompt and response to have been processed by both
conditions.

Raw hidden vectors and raw norms are not compared directly across model
families. Cross-model summaries use relative depth, within-family Base
standardization, normalized entropy where hidden dimensions differ, and effect
direction. Coordinate-level subtraction across Qwen3 and MiMo is prohibited.

Ability coupling is descriptive when only Qwen3 Base and MiMo Base/SFT are
available. Inferential correlation requires a larger set of independent
checkpoints or model states and aligned held-out performance measurements.

## Output Contract

Each run writes under an explicitly configured output root:

```text
runs/<run_id>/
  manifest.json
  run_status.json
  audit/
    input_audit.json
    pooling_audit.json
    calibration_audit.json
    profile_audit.json
    analysis_audit.json
    final_audit.json
  calibrators/
  profiles/
    vertical_profiles_<model>_<condition>.parquet
  analysis/
    policy_profiles.parquet
    outcome_profiles.parquet
    layer_auc.parquet
    depth_summaries.parquet
  figures/
  logs/
```

Profile rows preserve model, condition, question, rollout, representation,
stage, layer index, relative depth, metric values, structural NaNs, and
coverage counts. Every formal output includes an input manifest, configuration
hash, code revision, start/end timestamps, completion status, and relevant
artifact hashes.

Writes that determine completion are atomic. A partial Parquet or JSON file is
never treated as a completed partition. Restart logic reuses only outputs that
pass their audit and identity checks.

## Smoke And Formal Workflow

The smoke run uses a small but structurally representative cohort from every
configured model and condition. Its gates include:

- adapter and schema validation;
- expected hidden-state positions and relative depths;
- response-only or audited response-boundary handling;
- finite-value and structural-NaN coverage;
- local-profile formula checks on representative records;
- correct Base calibrator family and representation;
- mixed-outcome availability reporting;
- nonblank representative figures;
- bounded peak memory and measured formal runtime projection;
- successful restart from an interrupted smoke partition.

Formal launch is blocked until the smoke approval is present. The launcher
starts a detached, explicitly named tmux session with a persistent log and
`run_status.json`. The tutorial shows commands to list the session, attach,
capture recent output, inspect resource use, and resume after failure.

The formal run is CPU-only unless a source adapter requires an explicitly
documented tensor conversion. No model forward or weight loading is part of the
toolkit's normal workflow.

## Git And Data Safety

The branch tracks only source code, tests, documentation, example
configurations, and a tiny synthetic fixture. `.gitignore` excludes at least:

- raw hidden-state artifacts;
- model weights and checkpoints;
- pooled caches and calibrators generated from real data;
- formal Parquet/CSV tables and figure directories;
- logs, tmux captures, and local run-status files;
- local configuration files containing absolute server paths.

Before any push, a size and tracked-file audit verifies that no large tensor,
weight, cache, or real-data artifact is staged. Returned configurations must
replace private absolute paths with placeholders.

Small collaboration artifacts may be shared separately after review:

- audit JSON files;
- redacted run manifests;
- summary tables;
- selected figures;
- adapter patches and tests.

## Error Handling

The pipeline fails closed on ambiguous axes, missing response boundaries,
unexpected layer counts, mixed hidden dimensions within a model condition,
non-finite calibrators, incorrect Base/SFT calibrator usage, duplicate record
IDs, incomplete partitions, or failed output audits.

Unsupported analyses are recorded as unavailable with a reason rather than
fabricated from insufficient data. Examples include within-question AUROC with
no mixed-outcome questions, paired Base/SFT analysis without aligned record
IDs, and ability coupling with too few independent model states.

## Testing Strategy

Tests use synthetic arrays and small temporary files. They cover:

- adapter axis normalization and ambiguity failures;
- raw and pooled canonical record validation;
- response-boundary and progress handling;
- pooling endpoint and stage semantics;
- layer indexing and structural NaNs;
- V1/V3/V4/V6/V7 formulas and aggregate compatibility;
- Base calibrator label blindness and family isolation;
- MTP exclusion from main relative depth;
- question-equal outcome summaries and AUROC coverage;
- depth summaries and cross-dimension entropy normalization;
- atomic writes, restart behavior, audit gates, and tmux command construction;
- Git fixture size and exclusion rules.

Later releases add direct tests for V2, rolling/cumulative V5/V8/V9,
bootstrap, cluster permutation, and nuisance-controlled analyses before those
components are labeled stable.

## Acceptance Criteria

The handoff is complete when a collaborator can, using only
`vertical_readme.md` and the branch contents:

1. inspect an unknown Qwen3 or MiMo hidden-state directory without loading it
   fully;
2. produce a machine-readable input audit and select/configure an adapter;
3. pass a representative smoke run;
4. launch the formal offline job in a detached tmux session;
5. resume safely after an interruption;
6. obtain audited profiles, summaries, and nonblank figures;
7. distinguish supported scientific conclusions from unsupported ones;
8. return only small, reviewed artifacts without exposing raw hidden states,
   model weights, or private server paths.

