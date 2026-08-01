"""Measure Python-visible simulator throughput and latency.

Usage: python train/benchmark.py --steps 100000 --output benchmark-results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cpp_gym_nav  # noqa: F401 - registers the environment
import gymnasium as gym
import numpy as np
from cpp_gym_nav import _core


def measure_single(steps: int, seed: int) -> dict[str, float | int | str]:
    env = gym.make(cpp_gym_nav.ENV_ID)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    latencies_ns: list[int] = []
    completed = 0
    started = time.perf_counter_ns()
    for _ in range(steps):
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        before = time.perf_counter_ns()
        _, _, terminated, truncated, _ = env.step(action)
        latencies_ns.append(time.perf_counter_ns() - before)
        completed += 1
        if terminated or truncated:
            env.reset(seed=seed + completed)
    elapsed_ns = time.perf_counter_ns() - started
    env.close()
    ordered = sorted(latencies_ns)
    return {
        "configuration": "Gymnasium wrapper, single environment",
        "steps": completed,
        "steps_per_second": completed / (elapsed_ns / 1e9),
        "mean_step_us": statistics.fmean(latencies_ns) / 1e3,
        "p50_step_us": ordered[int(0.50 * (len(ordered) - 1))] / 1e3,
        "p95_step_us": ordered[int(0.95 * (len(ordered) - 1))] / 1e3,
    }


def measure_vector(steps: int, count: int, seed: int) -> dict[str, float | int | str]:
    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make(cpp_gym_nav.ENV_ID) for _ in range(count)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    batches = max(1, steps // count)
    latencies_ns: list[int] = []
    started = time.perf_counter_ns()
    for _ in range(batches):
        actions = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        before = time.perf_counter_ns()
        env.step(actions)
        latencies_ns.append(time.perf_counter_ns() - before)
    elapsed_ns = time.perf_counter_ns() - started
    completed = batches * count
    env.close()
    ordered = sorted(latencies_ns)
    return {
        "configuration": f"Gymnasium SyncVectorEnv, {count} environments",
        "steps": completed,
        "steps_per_second": completed / (elapsed_ns / 1e9),
        "mean_step_us": statistics.fmean(latencies_ns) / (1e3 * count),
        "p50_step_us": ordered[int(0.50 * (len(ordered) - 1))] / (1e3 * count),
        "p95_step_us": ordered[int(0.95 * (len(ordered) - 1))] / (1e3 * count),
    }


def measure_native_batch(steps: int, count: int, seed: int) -> dict[str, float | int | str]:
    batch = _core.BatchWorld(count)
    batch.reset(seed)
    rng = np.random.default_rng(seed)
    batches = max(1, steps // count)
    latencies_ns: list[int] = []
    started = time.perf_counter_ns()
    for index in range(batches):
        actions = rng.uniform(-1.0, 1.0, size=(count, 2)).astype(np.float32)
        before = time.perf_counter_ns()
        result = batch.step(actions)
        latencies_ns.append(time.perf_counter_ns() - before)
        ended = np.logical_or(result["successes"], result["collisions"])
        if ended.any():
            # Resetting all worlds keeps this microbenchmark simple and deterministic.
            batch.reset(seed + index + 1)
    elapsed_ns = time.perf_counter_ns() - started
    completed = batches * count
    ordered = sorted(latencies_ns)
    return {
        "configuration": f"Native C++ batch, {count} environments",
        "steps": completed,
        "steps_per_second": completed / (elapsed_ns / 1e9),
        "mean_step_us": statistics.fmean(latencies_ns) / (1e3 * count),
        "p50_step_us": ordered[int(0.50 * (len(ordered) - 1))] / (1e3 * count),
        "p95_step_us": ordered[int(0.95 * (len(ordered) - 1))] / (1e3 * count),
    }


def measure_training_throughput(steps: int, count: int, seed: int) -> list[dict[str, float | str]]:
    """Compare PPO training throughput across environment backends.

    The same PPO recipe trains on a single Gymnasium env, SB3's Python-loop
    ``DummyVecEnv``, and the native ``BatchWorldVecEnv``. Throughput uses the
    actual number of steps SB3 collected (``model.num_timesteps``), not the
    requested total, so the rollout rounding of vectorized runs is accounted for.
    """
    from cpp_gym_nav import BatchWorldVecEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def throughput(build) -> float:
        model = PPO(
            "MlpPolicy",
            build(),
            seed=seed,
            verbose=0,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
        )
        started = time.perf_counter()
        model.learn(total_timesteps=steps, progress_bar=False)
        return model.num_timesteps / (time.perf_counter() - started)

    backends = [
        ("single Gymnasium env", lambda: Monitor(gym.make(cpp_gym_nav.ENV_ID))),
        (
            f"SB3 DummyVecEnv, {count} envs",
            lambda: make_vec_env(
                lambda: gym.make(cpp_gym_nav.ENV_ID), n_envs=count, vec_env_cls=DummyVecEnv
            ),
        ),
        (
            f"Native BatchWorldVecEnv, {count} envs",
            lambda: VecMonitor(BatchWorldVecEnv(count, seed=seed)),
        ),
    ]
    rows: list[dict[str, float | str]] = []
    baseline = 0.0
    for name, build in backends:
        steps_per_second = throughput(build)
        baseline = baseline or steps_per_second
        rows.append(
            {
                "configuration": name,
                "train_steps_per_second": steps_per_second,
                "speedup_vs_single": steps_per_second / baseline,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--vector-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=5951)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    parser.add_argument(
        "--training",
        action="store_true",
        help="compare PPO training throughput across env backends instead of raw stepping",
    )
    parser.add_argument("--training-steps", type=int, default=20_000)
    args = parser.parse_args()
    if args.steps <= 0 or args.vector_envs <= 0:
        parser.error("--steps and --vector-envs must be positive")
    if args.training_steps <= 0:
        parser.error("--training-steps must be positive")

    if args.training:
        for row in measure_training_throughput(args.training_steps, args.vector_envs, args.seed):
            print(
                f"{row['configuration']}: {row['train_steps_per_second']:.0f} train steps/s "
                f"({row['speedup_vs_single']:.2f}x single)"
            )
        return

    results = {
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "note": "Python-visible wall-clock measurements; lower latency is better.",
        },
        "results": [
            measure_single(args.steps, args.seed),
            measure_vector(args.steps, args.vector_envs, args.seed),
            measure_native_batch(args.steps, args.vector_envs, args.seed),
        ],
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for row in results["results"]:
        print(
            f"{row['configuration']}: {row['steps_per_second']:.0f} steps/s, "
            f"p50={row['p50_step_us']:.2f} us, p95={row['p95_step_us']:.2f} us"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
