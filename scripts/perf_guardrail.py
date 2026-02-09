#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wordle_rl import ActionMaskMode, WordleEnv


def measure_steps_per_sec(steps: int, *, mode: ActionMaskMode) -> float:
    env = WordleEnv(action_mask_mode=mode)
    env.reset(seed=0)
    start = time.perf_counter()
    for _ in range(steps):
        action = env.sample_valid_action()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=0)
    elapsed = time.perf_counter() - start
    return steps / elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Performance guardrail for Wordle RL env.")
    parser.add_argument("--steps", type=int, default=300, help="How many single-env steps to measure.")
    parser.add_argument(
        "--min-auto-mask-sps",
        type=float,
        default=120.0,
        help="Fail if auto-mask throughput is below this value.",
    )
    parser.add_argument(
        "--min-consistent-mask-sps",
        type=float,
        default=35.0,
        help="Fail if consistent-mask throughput is below this value.",
    )
    args = parser.parse_args()

    auto_sps = measure_steps_per_sec(args.steps, mode=ActionMaskMode.AUTO)
    consistent_sps = measure_steps_per_sec(args.steps, mode=ActionMaskMode.CONSISTENT)

    print(f"auto_mask_steps_per_sec={auto_sps:.2f}")
    print(f"consistent_mask_steps_per_sec={consistent_sps:.2f}")

    if auto_sps < args.min_auto_mask_sps:
        print(
            f"FAIL: auto mask throughput {auto_sps:.2f} < threshold {args.min_auto_mask_sps:.2f}",
            file=sys.stderr,
        )
        return 1

    if consistent_sps < args.min_consistent_mask_sps:
        print(
            (
                "FAIL: consistent mask throughput "
                f"{consistent_sps:.2f} < threshold {args.min_consistent_mask_sps:.2f}"
            ),
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
