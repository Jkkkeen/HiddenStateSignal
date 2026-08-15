# Q3 Hidden-Probe Remove-Padding Compatibility Design

## Context

The Qwen3-1.7B-Base GRPO runner currently freezes
`actor_rollout_ref.model.use_remove_padding=False` because the original veRL
padding utilities imported `flash_attn`. The deployed veRL branch now provides
a tested Transformers fallback for those utilities, so FlashAttention is no
longer required.

The online hidden probe is implemented only in the no-padding FSDP forward path.
It explicitly requires `use_remove_padding=True`. As a result, setting
`HIDDEN_PROBE_RATE=1` with the current runner silently executes an ordinary
training step and produces no hidden history.

## Decision

Make remove-padding an explicit runner setting and freeze it to `True` for the
Q3 experiment:

- add `USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}` to the runner;
- pass that value to
  `actor_rollout_ref.model.use_remove_padding`;
- record the resolved value in `resolved_shell_config.json`;
- reject values other than `True` or `False`.

No hidden metric, layer, stage, reducer, reward, optimizer, prompt, dataset, or
response-cap definition changes.

## Comparable Smoke Runs

The existing `baseline_v6` remains valid as a system smoke but cannot be used
as the timing denominator because it used remove-padding `False`.

Run two new isolated five-step jobs with identical training configuration:

1. remove-padding `True`, `HIDDEN_PROBE_RATE=0`;
2. remove-padding `True`, `HIDDEN_PROBE_RATE=1`, interval 1.

Both jobs use the frozen SimpleRL smoke and validation manifests, response cap
4096, SDPA attention, one H200, and separate run/checkpoint names. Both jobs run
inside explicitly named tmux sessions.

The probe job must produce a non-empty per-step hidden history and exactly 141
latest dashboard PNG files. Its last five step times are compared with the new
baseline:

- overhead at most 30%: freeze interval 1;
- overhead above 50%: freeze interval 5;
- overhead between 30% and 50%: run the pre-registered reduced-group smoke
  before resolving the interval.

The completed full-probe smoke measured an overhead ratio of approximately
`1.49`, so the reduced-group branch is required.

## Deterministic 16-Group Probe

Add an explicit `HIDDEN_PROBE_GROUP_LIMIT` setting. Its default is 32, while
the reduced smoke and formal run freeze it to 16.

Selection happens after DAPO group filtering has accepted the 32 mixed prompt
groups and before the old-policy log-probability forward:

1. read each accepted trajectory's frozen `extra_info.prompt_hash`;
2. sort the 32 distinct hashes lexicographically;
3. select all eight trajectories belonging to the first 16 hashes;
4. attach a per-trajectory boolean mask to the old-log-probability batch;
5. return NaN hidden vectors for unselected trajectories so the existing
   question-equal aggregator excludes them.

All 32 accepted groups still participate in old/reference log-probability,
advantage, loss, and optimizer updates. The setting changes only which groups
enter hidden reduction and aggregation.

The mask must survive dynamic-batch reordering and restoration. A worker
micro-batch with no selected trajectory may skip `output_hidden_states`;
otherwise the engine computes dashboard vectors only for selected
trajectories. The online summary must record 16 probed questions and the actual
number of valid probed rollouts.

Run a new five-step reduced probe in its own tmux session with:

- `USE_REMOVE_PADDING=True`;
- `HIDDEN_PROBE_RATE=1`;
- `HIDDEN_PROBE_GROUP_LIMIT=16`;
- `HIDDEN_PROBE_INTERVAL=1`.

The reduced probe must exit cleanly, write five online history rows, retain 141
latest PNG files, and report 16 probed questions per step. If it fails or its
median overhead still exceeds 50%, formal training remains blocked.

## Timing Audit Compatibility

The smoke auditor must parse both supported veRL timing formats:

- dictionary telemetry such as `'timing_s/step': 10.0`;
- console telemetry such as `timing_s/step:10.0`.

For the middle branch, approval records both the full-probe and reduced-probe
ratios, freezes `hidden_probe_group_limit=16`, and freezes
`hidden_probe_interval=1`. Formal mode reads both values from the passed
approval JSON instead of accepting launch-time overrides.

## Resume And Formal Gate

After the five-step probe succeeds, resume its checkpoint to step 20 with the
same remove-padding and hidden settings. The smoke approval remains failed until
all existing checks pass:

- clean exit and no fatal log pattern;
- checkpoints at steps 5 and 20;
- verified restoration from step 5;
- non-empty online and held-out hidden histories;
- exactly 141 latest dashboard images;
- resolved hidden-probe interval.

Formal mode continues to read the frozen dataset decision, so SimpleRL selects
250 steps and 8,000 accepted groups. It also requires the passed smoke approval
JSON before a detached formal tmux session can start.

## Failure Handling

If remove-padding `True` still reaches an unavailable FlashAttention path or
fails tensor round-trip checks, stop before training and fix the compatibility
layer. Do not disable hidden capture, silently switch the timing denominator, or
add a padded hidden-extraction implementation during this experiment without a
new design review.

Interrupted or invalid smoke artifacts are retained for audit but are excluded
from approval inputs.

## Verification

- runner tests assert the default, Hydra argument, and resolved config all use
  the same remove-padding value;
- runner tests assert formal mode reads the approved group limit and interval;
- attention utility fallback tests remain green;
- Q3 selection and smoke-audit tests cover dictionary and console timing plus
  the 16-group middle branch;
- hidden-probe tests prove hash selection is deterministic, keeps whole groups,
  and masks unselected trajectories without changing batch size;
- remote resolved configs and logs prove both timing jobs used remove-padding
  `True`;
- the final approval JSON is generated from the comparable baseline, full
  probe, reduced probe, and completed 20-step resume artifacts.
