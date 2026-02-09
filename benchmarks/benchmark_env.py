#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wordle_rl import ActionMaskMode, WordleEnv, WordleVectorEnv


def run_single_env_benchmark(steps: int, *, consistent_mask: bool) -> Dict[str, float]:
    mode = ActionMaskMode.CONSISTENT if consistent_mask else ActionMaskMode.AUTO
    env = WordleEnv(action_mask_mode=mode)
    env.reset(seed=0)

    start = time.perf_counter()
    for _ in range(steps):
        action = env.sample_valid_action()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=0)
    elapsed = time.perf_counter() - start

    return {
        "steps": float(steps),
        "seconds": elapsed,
        "steps_per_sec": steps / elapsed,
    }


def run_vector_env_benchmark(steps: int, num_envs: int) -> Dict[str, float]:
    vec = WordleVectorEnv(num_envs=num_envs)
    vec.reset(seed=0)
    total_steps = steps * num_envs

    start = time.perf_counter()
    for _ in range(steps):
        actions = [env.sample_valid_action() for env in vec.envs]
        vec.step(actions)
    elapsed = time.perf_counter() - start
    vec.close()

    return {
        "steps": float(total_steps),
        "seconds": elapsed,
        "steps_per_sec": total_steps / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Wordle RL environment benchmarks.")
    parser.add_argument("--single-steps", type=int, default=500, help="Single-env steps.")
    parser.add_argument("--vector-steps", type=int, default=200, help="Vector-env iterations.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of envs for vector benchmark.")
    args = parser.parse_args()

    results = {
        "single_auto_mask": run_single_env_benchmark(args.single_steps, consistent_mask=False),
        "single_consistent_mask": run_single_env_benchmark(args.single_steps, consistent_mask=True),
        "vector_auto_mask": run_vector_env_benchmark(args.vector_steps, args.num_envs),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
