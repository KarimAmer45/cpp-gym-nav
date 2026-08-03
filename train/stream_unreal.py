"""Stream a trained policy's world state to an Unreal Engine renderer.

This is the spec's "UE as renderer" first step: the physics and RL stay in the
proven C++ core (`NavEnv`), and Unreal only visualizes. This script runs the
trained PPO policy, and for every step sends the world state (robot pose, goal,
obstacles) over a TCP socket to a UE process that moves matching actors.

Run order: start this script (it listens), then press Play in Unreal (the
renderer connects out on BeginPlay), then screen-record the UE viewport.

Wire protocol (newline-delimited JSON, this script -> Unreal):

    {"type": "config", "arena_half_extent": 5.0, "robot_radius": 0.22, "goal_radius": 0.35}
    {"type": "reset", "goal": [x, y], "obstacles": [[x, y, r], ...]}
    {"type": "step", "robot": [x, y, heading], "success": bool, "collision": bool}
    {"type": "done"}
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

from cpp_gym_nav import NavEnv, NavEnvConfig
from stable_baselines3 import PPO


def _send(conn: socket.socket, payload: dict) -> None:
    conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a trained policy to an Unreal renderer")
    parser.add_argument("model", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8920)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=5951)
    parser.add_argument(
        "--fps", type=float, default=15.0, help="playback rate (0 = as fast as possible)"
    )
    parser.add_argument("--fixed-obstacles", action="store_true")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = NavEnv(config=NavEnvConfig(random_obstacles=not args.fixed_obstacles))
    world = env._world
    delay = 1.0 / args.fps if args.fps > 0 else 0.0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"waiting for Unreal to connect on {args.host}:{args.port} ...")
        conn, address = server.accept()
        print(f"Unreal connected: {address}")
        with conn:
            _send(
                conn,
                {
                    "type": "config",
                    "arena_half_extent": float(env._core_config.arena_half_extent),
                    "robot_radius": float(env._core_config.robot_radius),
                    "goal_radius": float(env._core_config.goal_radius),
                },
            )
            for episode in range(args.episodes):
                observation, _ = env.reset(seed=args.seed + episode)
                goal_x, goal_y = world.goal
                _send(
                    conn,
                    {
                        "type": "reset",
                        "goal": [float(goal_x), float(goal_y)],
                        "obstacles": [
                            [float(x), float(y), float(r)] for x, y, r in world.obstacles
                        ],
                    },
                )
                done = False
                while not done:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = env.step(action)
                    x, y, heading = world.pose
                    _send(
                        conn,
                        {
                            "type": "step",
                            "robot": [float(x), float(y), float(heading)],
                            "success": bool(info["is_success"]),
                            "collision": bool(info["collision"]),
                        },
                    )
                    done = terminated or truncated
                    if delay:
                        time.sleep(delay)
            try:
                _send(conn, {"type": "done"})
            except OSError:
                pass
    env.close()
    print("stream complete")


if __name__ == "__main__":
    main()
