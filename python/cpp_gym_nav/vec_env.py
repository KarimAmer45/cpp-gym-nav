"""A Stable-Baselines3 ``VecEnv`` backed by the native C++ ``BatchWorld``.

``SyncVectorEnv`` steps environments in a Python loop, paying one binding
crossing per environment per step. This adapter advances all ``num_envs`` worlds
in a single C++ call (``BatchWorld.step``), releasing the GIL for the compute, so
the environment stops being the throughput bottleneck as ``num_envs`` grows.

Only the environments that terminate or time out are reset (via
``BatchWorld.reset_at``), matching the SB3 auto-reset contract: the true terminal
observation is returned in ``info["terminal_observation"]`` and the fresh reset
observation takes its place in the batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from cpp_gym_nav import _core
from cpp_gym_nav.nav_env import NavEnvConfig


class BatchWorldVecEnv(VecEnv):
    """Vectorized navigation environment stepped entirely in C++."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int,
        *,
        config: NavEnvConfig | None = None,
        max_episode_steps: int = 300,
        seed: int | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self._config = config or NavEnvConfig()
        self._core_config = self._config.to_core()
        self._batch = _core.BatchWorld(num_envs, self._core_config)
        self._max_steps = max_episode_steps
        self._obs_size = self._batch.observation_size
        self._elapsed = np.zeros(num_envs, dtype=np.int64)
        self._actions = np.zeros((num_envs, 2), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._closed = False

        low = np.full(self._obs_size, -1.0, dtype=np.float32)
        low[2] = 0.0  # normalized goal distance is non-negative
        low[6:] = 0.0  # lidar ranges are non-negative
        observation_space = spaces.Box(
            low=low, high=np.ones(self._obs_size, dtype=np.float32), dtype=np.float32
        )
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Set before super().__init__: SB3's VecEnv probes get_attr("render_mode")
        # during construction and warns if the attribute is missing.
        self.render_mode = None
        super().__init__(num_envs, observation_space, action_space)

    def _draw_seed(self) -> int:
        return int(self._rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))

    def reset(self) -> np.ndarray:
        observations = np.empty((self.num_envs, self._obs_size), dtype=np.float32)
        for index in range(self.num_envs):
            observations[index] = self._batch.reset_at(index, self._draw_seed())
        self._elapsed[:] = 0
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, 2)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        result = self._batch.step(self._actions)
        observations = result["observations"]
        rewards = result["rewards"].astype(np.float32)
        successes = result["successes"]
        collisions = result["collisions"]

        self._elapsed += 1
        terminated = np.logical_or(successes, collisions)
        truncated = np.logical_and(self._elapsed >= self._max_steps, ~terminated)
        dones = np.logical_or(terminated, truncated)

        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for index in np.nonzero(dones)[0]:
            index = int(index)
            infos[index] = {
                "is_success": bool(successes[index]),
                "collision": bool(collisions[index]),
                "TimeLimit.truncated": bool(truncated[index]),
                "terminal_observation": observations[index].copy(),
            }
            observations[index] = self._batch.reset_at(index, self._draw_seed())
            self._elapsed[index] = 0
        return observations, rewards, dones, infos

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._rng = np.random.default_rng(seed)
        return [seed] * self.num_envs

    def close(self) -> None:
        self._closed = True

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        setattr(self, attr_name, value)

    def env_method(
        self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any
    ) -> list[Any]:
        raise NotImplementedError("BatchWorldVecEnv has no per-environment Python methods")

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]
