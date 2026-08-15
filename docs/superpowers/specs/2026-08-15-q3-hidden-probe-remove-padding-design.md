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
- attention utility fallback tests remain green;
- Q3 selection and smoke-audit tests remain green;
- remote resolved configs and logs prove both timing jobs used remove-padding
  `True`;
- the final approval JSON is generated only from the new comparable pair.
