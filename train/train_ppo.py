"""Train and evaluate PPO while logging success rate and a learning curve."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cpp_gym_nav
import gymnasium as gym
import numpy as np
from cpp_gym_nav import BatchWorldVecEnv, NavEnvConfig
from matplotlib.figure import Figure
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor


class EpisodeMetrics(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.timesteps: list[int] = []
        self.returns: list[float] = []
        self.successes: list[bool] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.timesteps.append(self.num_timesteps)
                self.returns.append(float(episode["r"]))
                self.successes.append(bool(info.get("is_success", False)))
        return True


def evaluate(
    model: PPO | None, episodes: int, seed: int, config: NavEnvConfig
) -> dict[str, float | int]:
    env = gym.make(cpp_gym_nav.ENV_ID, config=config)
    successes = 0
    collisions = 0
    returns = []
    lengths = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            done = terminated or truncated
        successes += int(info.get("is_success", False))
        collisions += int(info.get("collision", False))
        returns.append(episode_return)
        lengths.append(episode_length)
    env.close()
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=5951)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/generated"))
    parser.add_argument("--realism", action="store_true")
    parser.add_argument(
        "--native-vec",
        type=int,
        default=0,
        metavar="N",
        help="train on N environments stepped together in C++ (0 = single Gymnasium env)",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_episodes <= 0:
        parser.error("--steps and --eval-episodes must be positive")
    if args.native_vec < 0:
        parser.error("--native-vec must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = NavEnvConfig(
        sensor_noise=args.realism,
        actuator_limits=args.realism,
        action_delay=args.realism,
        dynamics_randomization=args.realism,
    )
    if args.native_vec > 0:
        env = VecMonitor(
            BatchWorldVecEnv(
                args.native_vec,
                config=config,
                max_episode_steps=config.max_episode_steps,
                seed=args.seed,
            )
        )
    else:
        env = Monitor(gym.make(cpp_gym_nav.ENV_ID, config=config))
    callback = EpisodeMetrics()
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=0,
        tensorboard_log=str(args.output_dir / "tensorboard"),
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
    )
    training_started = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
    training_wall_seconds = time.perf_counter() - training_started
    model.save(args.output_dir / "ppo_nav")
    env.close()

    evaluation_seed = args.seed + 100_000
    metrics = {
        "requested_training_steps": args.steps,
        "training_steps": model.num_timesteps,
        "training_wall_seconds": training_wall_seconds,
        "training_steps_per_second": model.num_timesteps / training_wall_seconds,
        "native_vec_envs": args.native_vec,
        "evaluation_seed_start": evaluation_seed,
        "realism": args.realism,
        "ppo": evaluate(model, args.eval_episodes, evaluation_seed, config),
        "random": evaluate(None, args.eval_episodes, evaluation_seed, config),
    }
    if args.realism:
        # Generalization check: score the domain-randomization-trained policy on
        # the *clean* dynamics, over the same seeded episodes, to see whether
        # robustness training costs anything on the noise-free task.
        metrics["ppo_clean_transfer"] = evaluate(
            model, args.eval_episodes, evaluation_seed, NavEnvConfig()
        )
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    if callback.returns:
        window = min(50, len(callback.returns))
        kernel = np.ones(window) / window
        smoothed = np.convolve(callback.returns, kernel, mode="valid")
        figure = Figure(figsize=(8, 4.5))
        axes = figure.subplots()
        axes.plot(callback.timesteps[window - 1 :], smoothed)
        axes.set_xlabel("Environment steps")
        axes.set_ylabel(f"Mean episode return ({window}-episode window)")
        axes.set_title("PPO learning curve")
        axes.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.output_dir / "learning_curve.png", dpi=160)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
