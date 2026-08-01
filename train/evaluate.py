"""Evaluate a saved PPO policy and write an animated GIF of a successful run."""

from __future__ import annotations

import argparse
from pathlib import Path

import cpp_gym_nav
import gymnasium as gym
import imageio.v3 as iio
from stable_baselines3 import PPO


def rollout(model: PPO, env: gym.Env, seed: int) -> tuple[list, dict, float]:
    observation, _ = env.reset(seed=seed)
    frames = [env.render()]
    terminated = truncated = False
    total_reward = 0.0
    info: dict = {}
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        frames.append(env.render())
    return frames, info, total_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--seed", type=int, default=5951)
    parser.add_argument("--gif", type=Path, default=Path("assets/generated/trained_agent.gif"))
    parser.add_argument(
        "--search",
        type=int,
        default=30,
        help="scan up to N consecutive seeds from --seed for a successful episode to render",
    )
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = gym.make(cpp_gym_nav.ENV_ID, render_mode="rgb_array")

    # Render a representative *successful* episode: deterministically scan seeds
    # and keep the first success, falling back to the highest-return episode if
    # none succeed. The honest success rate lives in evaluation.json; this only
    # makes the demo GIF show the policy working rather than a random episode
    # that may land in the policy's failure fraction.
    best: tuple[list, dict, float, int] | None = None
    chosen: tuple[list, dict, float, int] | None = None
    for offset in range(max(1, args.search)):
        seed = args.seed + offset
        frames, info, total_reward = rollout(model, env, seed)
        if best is None or total_reward > best[2]:
            best = (frames, info, total_reward, seed)
        if info.get("is_success"):
            chosen = (frames, info, total_reward, seed)
            break
    env.close()

    assert best is not None  # the loop always runs at least once
    frames, info, total_reward, seed = chosen or best
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.gif, frames, duration=100, loop=0)
    status = "success" if info.get("is_success") else "best-effort (no success in scan)"
    print(
        f"Wrote {args.gif} [{status}]: seed={seed}, success={info['is_success']}, "
        f"collision={info['collision']}, return={total_reward:.3f}, frames={len(frames)}"
    )


if __name__ == "__main__":
    main()
