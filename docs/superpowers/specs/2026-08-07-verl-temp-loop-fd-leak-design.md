# veRL Temporary Event-Loop FD Leak Design

## Problem

The Experiment 2E GRPO run failed twice with `ZMQError: Too many open files`. Raising `RLIMIT_NOFILE` from 1024 to 65535 delayed, but did not remove, the failure. The traceback ends while TransferQueue creates a ZMQ context, but the exhausted descriptors originate one layer above it: `verl.utils.transferqueue_utils._run_async_in_temp_loop` creates a new asyncio event loop for every synchronous/async bridge call, stops its thread, and never closes the loop.

A 5000-call concurrent TransferQueue-only smoke held exactly 15 FDs, disproving the initial per-request ZMQ-context hypothesis. A targeted veRL bridge smoke grew from 17 to 38 FDs in 300 calls without forced garbage collection. The loop owns epoll and wakeup-pipe descriptors and must release them deterministically rather than relying on cyclic GC.

## Approved Design

Patch only `verl/utils/transferqueue_utils.py` in the frozen H200 checkout. In `_run_async_in_temp_loop`, schedule `tmp_event_loop.stop` thread-safely, join the loop thread, then call `tmp_event_loop.close()`. Do not alter TransferQueue, PyZMQ, model, optimizer, data, seeds, checkpoints, or formal training hyperparameters.

The patch is applied by an idempotent repository script. It records before/after SHA256 values, keeps one byte-identical backup, refuses unknown source shapes, supports exact restore, and is invoked as a preflight by the formal run script.

## Verification

1. Unit tests cover apply, idempotent reapply, unknown-source refusal, backup mismatch, and exact restore.
2. The real veRL bridge completes 5000 calls without forced GC and without FD growth.
3. The formal run resumes from checkpoint 2232 and completes multiple new steps.
4. WorkerDict inherits `RLIMIT_NOFILE=65535`; its FD count remains bounded across live training steps.

## Failure Handling

If the synthetic smoke fails, restore the original veRL file and do not start training. If live WorkerDict FDs still grow materially per step, stop only this formal run and preserve its checkpoints/logs. Raising the limit to 1048576 is not an accepted root fix.
