from __future__ import annotations

import numpy as np
from cpp_gym_nav import NavEnv, NavEnvConfig


def rollout(config: NavEnvConfig, seed: int = 99) -> list[np.ndarray]:
    env = NavEnv(config=config)
    observations = [env.reset(seed=seed)[0]]
    action = np.array([1.0, 0.4], dtype=np.float32)
    for _ in range(3):
        observations.append(env.step(action)[0])
    return observations


def test_all_realism_flags_off_preserve_baseline() -> None:
    baseline = rollout(NavEnvConfig())
    explicit_off = rollout(
        NavEnvConfig(
            sensor_noise=False,
            actuator_limits=False,
            action_delay=False,
            dynamics_randomization=False,
        )
    )
    for left, right in zip(baseline, explicit_off, strict=True):
        np.testing.assert_array_equal(left, right)


def test_sensor_noise_is_seeded_and_changes_lidar() -> None:
    noisy_config = NavEnvConfig(sensor_noise=True)
    first = rollout(noisy_config)
    second = rollout(noisy_config)
    baseline = rollout(NavEnvConfig())
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert any(not np.array_equal(a[6:], b[6:]) for a, b in zip(first, baseline, strict=True))


def test_actuator_limits_bound_first_command() -> None:
    env = NavEnv(config=NavEnvConfig(actuator_limits=True))
    env.reset(seed=4)
    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(info["applied_action"], [0.2, 0.4], atol=1e-6)


def test_action_delay_defers_command() -> None:
    env = NavEnv(config=NavEnvConfig(action_delay=True, action_delay_steps=1))
    env.reset(seed=5)
    _, _, _, _, first = env.step(np.array([1.0, 0.5], dtype=np.float32))
    _, _, _, _, second = env.step(np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(first["applied_action"], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(second["applied_action"], [1.25, 1.0])


def test_domain_randomization_is_reproducible() -> None:
    randomized = NavEnvConfig(dynamics_randomization=True)
    first = rollout(randomized, seed=123)
    second = rollout(randomized, seed=123)
    baseline = rollout(NavEnvConfig(), seed=123)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert any(not np.array_equal(a, b) for a, b in zip(first, baseline, strict=True))
