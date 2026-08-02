from __future__ import annotations

import cpp_gym_nav
import gymnasium as gym
import numpy as np
from cpp_gym_nav import NavEnv, NavEnvUnreal
from gymnasium.utils.env_checker import check_env as gym_check_env


def test_unreal_bridge_passes_gymnasium_checker_and_registration() -> None:
    registered = gym.make(cpp_gym_nav.UNREAL_ENV_ID, disable_env_checker=True)
    gym_check_env(registered.unwrapped, skip_render_check=True)
    observation, _ = registered.reset(seed=5951)
    assert registered.observation_space.contains(observation)
    registered.close()


def test_unreal_bridge_matches_direct_env_bitwise() -> None:
    # The socket backend is the C++ core, so a trajectory over the wire must
    # replay bit-for-bit against the in-process env for the same seed/actions.
    direct = NavEnv()
    bridged = NavEnvUnreal()
    try:
        obs_direct, _ = direct.reset(seed=42)
        obs_bridged, _ = bridged.reset(seed=42)
        np.testing.assert_array_equal(obs_direct, obs_bridged)

        for action in (
            np.array([0.4, 0.2], dtype=np.float32),
            np.array([0.6, -0.1], dtype=np.float32),
            np.array([-0.3, 0.5], dtype=np.float32),
        ):
            d_obs, d_rew, d_term, d_trunc, _ = direct.step(action)
            b_obs, b_rew, b_term, b_trunc, _ = bridged.step(action)
            np.testing.assert_array_equal(d_obs, b_obs)
            assert (d_rew, d_term, d_trunc) == (b_rew, b_term, b_trunc)
    finally:
        direct.close()
        bridged.close()


def test_unreal_bridge_spaces_match_backend() -> None:
    direct = NavEnv()
    bridged = NavEnvUnreal()
    try:
        np.testing.assert_array_equal(direct.observation_space.low, bridged.observation_space.low)
        np.testing.assert_array_equal(direct.observation_space.high, bridged.observation_space.high)
        assert direct.action_space.shape == bridged.action_space.shape
    finally:
        direct.close()
        bridged.close()
