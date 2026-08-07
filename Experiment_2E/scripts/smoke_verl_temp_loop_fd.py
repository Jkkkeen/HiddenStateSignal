#!/usr/bin/env python3
"""Measure FD growth in veRL's synchronous-to-async temporary-loop bridge."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
from pathlib import Path

from verl.utils.transferqueue_utils import _run_async_in_temp_loop


def fd_count() -> int:
    return len(os.listdir(Path("/proc/self/fd")))


async def round_trip(value: int) -> int:
    await asyncio.sleep(0)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--warmup-calls", type=int, default=10)
    parser.add_argument("--max-fd-delta", type=int, default=4)
    parser.add_argument("--expect-leak", action="store_true")
    parser.add_argument("--collect-after", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    for index in range(args.warmup_calls):
        assert _run_async_in_temp_loop(round_trip, index) == index
    gc.collect()
    baseline = fd_count()

    for index in range(args.calls):
        assert _run_async_in_temp_loop(round_trip, index) == index
    if args.collect_after:
        gc.collect()
    final = fd_count()
    delta = final - baseline
    leaked = delta > args.max_fd_delta
    payload = {
        "calls": args.calls,
        "warmup_calls": args.warmup_calls,
        "baseline_fds": baseline,
        "final_fds": final,
        "fd_delta": delta,
        "max_fd_delta": args.max_fd_delta,
        "collect_after": args.collect_after,
        "leaked": leaked,
        "passed": leaked if args.expect_leak else not leaked,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
