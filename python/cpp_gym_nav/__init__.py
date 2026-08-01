"""Gymnasium registration and public API for cpp-gym-nav."""

from gymnasium.envs.registration import register, registry

from cpp_gym_nav.nav_env import NavEnv, NavEnvConfig

ENV_ID = "CppGymNav-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="cpp_gym_nav.nav_env:NavEnv", max_episode_steps=None)

__all__ = ["ENV_ID", "NavEnv", "NavEnvConfig", "BatchWorldVecEnv"]


def __getattr__(name: str) -> object:
    # Lazily import the SB3-dependent adapter so the base package does not
    # require stable-baselines3 to be installed.
    if name == "BatchWorldVecEnv":
        from cpp_gym_nav.vec_env import BatchWorldVecEnv

        return BatchWorldVecEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
