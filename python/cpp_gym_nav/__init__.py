"""Gymnasium registration and public API for cpp-gym-nav."""

from gymnasium.envs.registration import register, registry

from cpp_gym_nav.nav_env import NavEnv, NavEnvConfig

ENV_ID = "CppGymNav-v0"
UNREAL_ENV_ID = "CppGymNavUnreal-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="cpp_gym_nav.nav_env:NavEnv", max_episode_steps=None)
if UNREAL_ENV_ID not in registry:
    register(
        id=UNREAL_ENV_ID,
        entry_point="cpp_gym_nav.nav_env_unreal:NavEnvUnreal",
        max_episode_steps=None,
    )

__all__ = [
    "ENV_ID",
    "UNREAL_ENV_ID",
    "NavEnv",
    "NavEnvConfig",
    "NavEnvUnreal",
    "BatchWorldVecEnv",
]


def __getattr__(name: str) -> object:
    # Lazy imports keep optional SB3 out of the base import path and avoid
    # preloading backend_server when it is executed with ``python -m``.
    if name == "NavEnvUnreal":
        from cpp_gym_nav.nav_env_unreal import NavEnvUnreal

        return NavEnvUnreal
    if name == "BatchWorldVecEnv":
        from cpp_gym_nav.vec_env import BatchWorldVecEnv

        return BatchWorldVecEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
