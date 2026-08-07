# TransferQueue ZMQ FD Leak Design

## Problem

The formal Experiment 2E GRPO run failed twice with `ZMQError: Too many open files`. Raising `RLIMIT_NOFILE` from 1024 to 65535 only delayed the second failure by about 1061 steps. TransferQueue 0.1.8 creates and terminates a new `zmq.asyncio.Context` for every decorated request; 0.1.9 and upstream `main` retain the same implementation.

## Approved Design

Patch only `transfer_queue/utils/zmq_utils.py` in the frozen H200 environment. Each request will reuse `self.zmq_context` when the owner already has one, otherwise it will reuse the process-wide `zmq.asyncio.Context.instance()`. The request still receives a fresh DEALER socket, and that socket is closed with `linger=0` after the awaited request completes. A request must never terminate a shared context.

The patch is applied by an idempotent repository script. It records the target path and before/after SHA256 values, keeps one byte-identical backup, refuses unknown source shapes, and supports an explicit restore command. It does not upgrade TransferQueue, veRL, PyZMQ, or any training dependency.

## Verification

1. A synthetic request loop must complete at least 2000 request/response cycles without monotonic `/proc/self/fd` growth.
2. The formal run resumes from checkpoint 2232 in its existing tmux session name.
3. During a short live observation window, training steps advance, GPU utilization is nonzero, process `RLIMIT_NOFILE` is 65535, and WorkerDict FD counts remain bounded rather than increasing every step.

## Failure Handling

If the synthetic test fails, restore the original TransferQueue file and do not start training. If live FD counts still grow, stop only this formal run, preserve its checkpoints/logs, and restore the original package file. Increasing the limit to 1048576 is not an accepted root fix.
