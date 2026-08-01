from __future__ import annotations

import cpp_gym_nav
import gymnasium as gym
import numpy as np
import pytest
from cpp_gym_nav import NavEnv, NavEnvConfig, _core
from gymnasium.utils.env_checker import check_env as gym_check_env


def test_gymnasium_checker_and_registration() -> None:
    registered = gym.make(cpp_gym_nav.ENV_ID, disable_env_checker=True)
    gym_check_env(registered.unwrapped, skip_render_check=False)
    observation, info = registered.reset(seed=5951)
    assert registered.observation_space.contains(observation)
    assert info["seed"] == 5951
    registered.close()


def test_stable_baselines_checker() -> None:
    checker = pytest.importorskip("stable_baselines3.common.env_checker")
    checker.check_env(NavEnv(), warn=True)


def test_deterministic_replay() -> None:
    first = NavEnv()
    second = NavEnv()
    observation_a, _ = first.reset(seed=42)
    observation_b, _ = second.reset(seed=42)
    np.testing.assert_array_equal(observation_a, observation_b)

    actions = [
        np.array([0.4, 0.2], dtype=np.float32),
        np.array([0.6, -0.1], dtype=np.float32),
        np.array([0.3, 0.0], dtype=np.float32),
    ]
    for action in actions:
        transition_a = first.step(action)
        transition_b = second.step(action)
        np.testing.assert_array_equal(transition_a[0], transition_b[0])
        assert transition_a[1:4] == transition_b[1:4]


def test_reused_world_replays_identically_to_fresh_world() -> None:
    # A reused World must replay a seed bit-for-bit, exactly like a fresh one.
    # std::normal_distribution caches the second Box-Muller value, so an odd
    # number of noise draws leaves the cache occupied; reset() must clear it,
    # otherwise deterministic replay silently breaks after the first episode
    # (which is exactly what happens during training).
    #
    # An odd, dropout-free beam count guarantees the prior reset draws an odd
    # number of normals, so the trigger fires regardless of the C++ standard
    # library's Box-Muller ordering (portable across the CI matrix).
    def make() -> _core.World:
        config = _core.Config()
        config.sensor_noise = True
        config.lidar_beams = 15
        config.lidar_dropout_probability = 0.0
        return _core.World(config)

    fresh = np.asarray(make().reset(7))
    reused_world = make()
    reused_world.reset(1)  # leaves an odd number of cached normal draws
    reused = np.asarray(reused_world.reset(7))
    np.testing.assert_array_equal(fresh, reused)


def test_terminated_and_truncated_are_distinct() -> None:
    env = NavEnv(config=NavEnvConfig(max_episode_steps=1))
    env.reset(seed=7)
    _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert not terminated
    assert truncated


def test_invalid_actions_are_safely_normalized() -> None:
    env = NavEnv()
    env.reset(seed=12)
    observation, _, _, _, info = env.step(np.array([np.nan, np.inf], dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert info["action_clipped"]
    np.testing.assert_array_equal(info["applied_action"], np.zeros(2, dtype=np.float32))


def test_rgb_render_contract() -> None:
    env = NavEnv(render_mode="rgb_array")
    env.reset(seed=3)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (512, 512, 3)
    assert frame.dtype == np.uint8


def test_sync_vector_env_autoreset() -> None:
    vector_env = gym.vector.SyncVectorEnv(
        [lambda: NavEnv(config=NavEnvConfig(max_episode_steps=1)) for _ in range(2)]
    )
    vector_env.reset(seed=123)
    observations, _, _, truncations, infos = vector_env.step(np.zeros((2, 2), dtype=np.float32))
    assert observations.shape == (2, 22)
    assert truncations.all()
    assert "seed" not in infos
    vector_env.step(np.zeros((2, 2), dtype=np.float32))
    vector_env.close()


def test_native_batch_contract_and_replay() -> None:
    first = _core.BatchWorld(4)
    second = _core.BatchWorld(4)
    np.testing.assert_array_equal(first.reset(5951), second.reset(5951))
    actions = np.full((4, 2), [0.25, -0.1], dtype=np.float32)
    result_a = first.step(actions)
    result_b = second.step(actions)
    assert result_a["observations"].shape == (4, 22)
    for key in ("observations", "rewards", "successes", "collisions"):
        np.testing.assert_array_equal(result_a[key], result_b[key])


def test_reward_matches_potential_shaping_decomposition() -> None:
    # reward = 2*(old_dist - new_dist) - 0.01 (step) - 0.005 * effort, plus terminal
    # bonuses. Verifying the decomposition per step proves the progress term is a
    # telescoping potential, so a path that returns to the same distance-to-goal
    # cannot farm net progress reward (the anti-reward-hacking property).
    env = NavEnv()
    _, info = env.reset(seed=17)
    max_lin = env._core_config.max_linear_speed
    max_ang = env._core_config.max_angular_speed
    prev_dist = dist_start = info["distance_to_goal"]
    progress_sum = 0.0
    steps = 0
    actions = [
        np.array([0.6, 0.3], dtype=np.float32),
        np.array([0.5, -0.4], dtype=np.float32),
        np.array([-0.5, 0.2], dtype=np.float32),
        np.array([-0.4, -0.3], dtype=np.float32),
    ]
    for action in actions:
        _, reward, terminated, _, info = env.step(action)
        if terminated:  # a goal/collision adds a terminal bonus outside this decomposition
            break
        lin, ang = info["applied_action"]
        effort = 0.5 * (abs(lin) / max_lin + abs(ang) / max_ang)
        new_dist = info["distance_to_goal"]
        expected = 2.0 * (prev_dist - new_dist) - 0.01 - 0.005 * effort
        assert abs(reward - expected) < 1e-3, (reward, expected)
        progress_sum += 2.0 * (prev_dist - new_dist)
        prev_dist = new_dist
        steps += 1
    assert steps >= 2, "expected several non-terminal steps to validate the decomposition"
    # The progress terms telescope to 2 * (start_distance - final_distance).
    assert abs(progress_sum - 2.0 * (dist_start - prev_dist)) < 1e-4
