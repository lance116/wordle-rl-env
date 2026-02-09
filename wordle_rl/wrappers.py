from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYMNASIUM = True
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _HAS_GYMNASIUM = False


if _HAS_GYMNASIUM:

    class ActionMaskToInfoWrapper(gym.Wrapper):
        """
        Copies `obs['action_mask']` into info for frameworks expecting masks there.
        """

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            if isinstance(obs, dict) and "action_mask" in obs:
                info = dict(info)
                info["action_mask"] = obs["action_mask"]
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if isinstance(obs, dict) and "action_mask" in obs:
                info = dict(info)
                info["action_mask"] = obs["action_mask"]
            return obs, reward, terminated, truncated, info


    class FlattenWordleObservation(gym.ObservationWrapper):
        """
        Flattens dict observations into a single float32 vector.

        By default, `action_mask` is excluded from flattening because most
        policies consume it separately.
        """

        def __init__(
            self,
            env: gym.Env,
            *,
            include_action_mask: bool = False,
            keys: Optional[Sequence[str]] = None,
        ) -> None:
            super().__init__(env)
            if not isinstance(self.observation_space, spaces.Dict):
                raise TypeError("FlattenWordleObservation expects a Dict observation space.")

            all_keys = list(self.observation_space.spaces.keys())
            if keys is None:
                chosen = [k for k in all_keys if include_action_mask or k != "action_mask"]
            else:
                chosen = list(keys)

            for key in chosen:
                if key not in self.observation_space.spaces:
                    raise KeyError(f"Unknown observation key: {key}")

            self._keys = chosen
            self._sizes: Dict[str, int] = {}
            total = 0
            for key in self._keys:
                space = self.observation_space.spaces[key]
                size = int(np.prod(space.shape))
                self._sizes[key] = size
                total += size

            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total,),
                dtype=np.float32,
            )

        def observation(self, observation):
            flat_parts: List[np.ndarray] = []
            for key in self._keys:
                value = np.asarray(observation[key], dtype=np.float32).reshape(-1)
                flat_parts.append(value)
            return np.concatenate(flat_parts, axis=0)
