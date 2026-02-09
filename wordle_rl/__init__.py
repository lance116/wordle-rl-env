"""Wordle reinforcement learning environment."""

from .env import ActionMaskMode, Feedback, RewardConfig, WordleEnv
from .lexicon import Lexicon, load_nyt_lexicon, load_word_list, make_lexicon
from .registration import register_envs
from .vector_env import WordleVectorEnv

try:
    from .wrappers import ActionMaskToInfoWrapper, FlattenWordleObservation
except ModuleNotFoundError:  # pragma: no cover - wrappers need gymnasium
    ActionMaskToInfoWrapper = None  # type: ignore[assignment]
    FlattenWordleObservation = None  # type: ignore[assignment]

__all__ = [
    "ActionMaskMode",
    "Feedback",
    "RewardConfig",
    "WordleEnv",
    "WordleVectorEnv",
    "Lexicon",
    "load_nyt_lexicon",
    "load_word_list",
    "make_lexicon",
    "register_envs",
    "ActionMaskToInfoWrapper",
    "FlattenWordleObservation",
]

# Register gym entry points on import when gymnasium is installed.
register_envs()
