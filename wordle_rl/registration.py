from __future__ import annotations

from typing import Optional

try:
    import gymnasium as gym

    _HAS_GYMNASIUM = True
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    _HAS_GYMNASIUM = False


def register_envs(*, force: bool = False) -> None:
    """
    Register Gymnasium entry points for this package.
    """
    if not _HAS_GYMNASIUM:
        return

    env_id = "WordleRL-v0"
    if env_id in gym.registry and not force:
        return
    if env_id in gym.registry and force:
        del gym.registry[env_id]

    gym.register(
        id=env_id,
        entry_point="wordle_rl.env:WordleEnv",
        max_episode_steps=6,
    )
