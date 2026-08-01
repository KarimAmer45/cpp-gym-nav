"""Gymnasium registration and public API for cpp-gym-nav."""

from gymnasium.envs.registration import register, registry

from cpp_gym_nav.nav_env import NavEnv, NavEnvConfig

ENV_ID = "CppGymNav-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="cpp_gym_nav.nav_env:NavEnv", max_episode_steps=None)

__all__ = ["ENV_ID", "NavEnv", "NavEnvConfig"]
