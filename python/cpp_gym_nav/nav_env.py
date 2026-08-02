"""A standards-compliant Gymnasium wrapper around the C++ navigation core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cpp_gym_nav import _core


@dataclass(frozen=True, slots=True)
class NavEnvConfig:
    """Python-facing configuration for episode and realism behavior."""

    lidar_beams: int = 16
    max_episode_steps: int = 300
    random_obstacles: bool = True
    sensor_noise: bool = False
    actuator_limits: bool = False
    action_delay: bool = False
    action_delay_steps: int = 1
    dynamics_randomization: bool = False

    def to_core(self) -> _core.Config:
        config = _core.Config()
        config.lidar_beams = self.lidar_beams
        config.random_obstacles = self.random_obstacles
        config.sensor_noise = self.sensor_noise
        config.actuator_limits = self.actuator_limits
        config.action_delay = self.action_delay
        config.action_delay_steps = self.action_delay_steps
        config.dynamics_randomization = self.dynamics_randomization
        return config


class NavEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous-control 2-D navigation environment.

    Actions are ``[linear_velocity, angular_velocity]``. Observations contain
    robot-frame goal coordinates, normalized goal distance and bearing, body
    velocities, and normalized lidar ranges.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        *,
        config: NavEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NavEnvConfig()
        if self.config.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode
        self._core_config = self.config.to_core()
        self._world = _core.World(self._core_config)
        self._elapsed_steps = 0
        self._last_observation: np.ndarray | None = None
        self._closed = False

        # The policy-facing action is normalized; conversion to physical units
        # happens immediately before the binding call.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation layout mirrors the C++ core's fill_observation():
        #   [0]  robot-frame goal x          in [-1, 1]
        #   [1]  robot-frame goal y          in [-1, 1]
        #   [2]  normalized goal distance    in [ 0, 1]
        #   [3]  goal bearing / pi           in [-1, 1]
        #   [4]  normalized linear velocity  in [-1, 1]
        #   [5]  normalized angular velocity in [-1, 1]
        #   [6:] normalized lidar ranges     in [ 0, 1]
        obs_size = self._world.observation_size
        low = np.full(obs_size, -1.0, dtype=np.float32)
        low[2] = 0.0  # distance is non-negative
        low[6:] = 0.0  # lidar ranges are non-negative
        self.observation_space = spaces.Box(
            low=low,
            high=np.ones(obs_size, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is None:
            # Keep info values vector-env friendly: Gymnasium aggregates Python
            # integers into signed NumPy arrays.
            seed = int(self.np_random.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        observation = self._validated_observation(self._world.reset(seed))
        self._elapsed_steps = 0
        self._last_observation = observation
        self._closed = False
        robot_x, robot_y, _ = self._world.pose
        goal_x, goal_y = self._world.goal
        info = {
            "seed": int(seed),
            "distance_to_goal": float(np.hypot(goal_x - robot_x, goal_y - robot_y)),
        }
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_observation is None:
            raise RuntimeError("reset() must be called before step()")
        raw_action = np.asarray(action, dtype=np.float32)
        if raw_action.shape != (2,):
            raise ValueError(f"action must have shape (2,), got {raw_action.shape}")
        invalid_action = not bool(np.all(np.isfinite(raw_action)))
        safe_action = np.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
        clipped_action = np.clip(safe_action, self.action_space.low, self.action_space.high).astype(
            np.float32
        )
        physical_action = clipped_action * np.array(
            [self._core_config.max_linear_speed, self._core_config.max_angular_speed],
            dtype=np.float32,
        )

        result = self._world.step(physical_action)
        observation = self._validated_observation(result["observation"])
        self._elapsed_steps += 1
        terminated = bool(result["success"] or result["collision"])
        truncated = bool(self._elapsed_steps >= self.config.max_episode_steps and not terminated)
        self._last_observation = observation
        info = {
            "is_success": bool(result["success"]),
            "collision": bool(result["collision"]),
            "distance_to_goal": float(result["distance_to_goal"]),
            "applied_action": np.asarray(result["applied_action"], dtype=np.float32),
            "action_clipped": bool(
                invalid_action or not np.array_equal(raw_action, clipped_action)
            ),
        }
        return observation, float(result["reward"]), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        frame = self._render_frame()
        if self.render_mode == "human":
            # Avoid a heavyweight GUI dependency; ANSI feedback still makes interactive use useful.
            x, y, heading = self._world.pose
            print(f"robot=({x:+.2f}, {y:+.2f}, heading={heading:+.2f})")
            return None
        return frame

    def close(self) -> None:
        self._closed = True

    def _validated_observation(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != self.observation_space.shape:
            expected_shape = self.observation_space.shape
            raise RuntimeError(
                f"core returned observation shape {value.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(value)):
            raise RuntimeError("core returned NaN or infinite observation values")
        if not self.observation_space.contains(value):
            raise RuntimeError("core returned an out-of-range observation")
        return value

    def _render_frame(self) -> np.ndarray:
        size = 512
        frame = np.full((size, size, 3), 247, dtype=np.uint8)
        extent = float(self._core_config.arena_half_extent)

        def point(x: float, y: float) -> tuple[int, int]:
            px = int(np.clip((x + extent) / (2.0 * extent) * (size - 1), 0, size - 1))
            py = int(np.clip((extent - y) / (2.0 * extent) * (size - 1), 0, size - 1))
            return px, py

        def disk(x: float, y: float, radius: float, color: tuple[int, int, int]) -> None:
            cx, cy = point(x, y)
            radius_px = max(1, int(radius / (2.0 * extent) * size))
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2
            frame[mask] = color

        frame[[0, -1], :, :] = 30
        frame[:, [0, -1], :] = 30
        for x, y, radius in self._world.obstacles:
            disk(float(x), float(y), float(radius), (80, 85, 95))
        goal_x, goal_y = self._world.goal
        disk(float(goal_x), float(goal_y), self._core_config.goal_radius, (50, 190, 90))
        robot_x, robot_y, heading = self._world.pose
        disk(float(robot_x), float(robot_y), self._core_config.robot_radius, (35, 105, 210))
        cx, cy = point(float(robot_x), float(robot_y))
        tip_x, tip_y = point(
            float(robot_x) + 0.45 * float(np.cos(heading)),
            float(robot_y) + 0.45 * float(np.sin(heading)),
        )
        samples = max(abs(tip_x - cx), abs(tip_y - cy), 1)
        xs = np.linspace(cx, tip_x, samples + 1).astype(int)
        ys = np.linspace(cy, tip_y, samples + 1).astype(int)
        frame[ys, xs] = (255, 255, 255)
        return frame
