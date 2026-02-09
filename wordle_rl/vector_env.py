from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .env import WordleEnv


class WordleVectorEnv:
    """
    Lightweight synchronous batched Wordle environment.

    This is useful for RL collectors that want N parallel episodes without
    depending on Gymnasium's vector API contracts.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        auto_reset: bool = True,
        **env_kwargs: object,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.auto_reset = bool(auto_reset)
        self.envs: List[WordleEnv] = [WordleEnv(**env_kwargs) for _ in range(self.num_envs)]

        # Best-effort compatibility with common RL collector expectations.
        self.single_action_space = getattr(self.envs[0], "action_space", None)
        self.single_observation_space = getattr(self.envs[0], "observation_space", None)

    def reset(
        self,
        *,
        seed: Optional[Union[int, Sequence[Optional[int]]]] = None,
        options: Optional[Union[Dict[str, object], Sequence[Optional[Dict[str, object]]]]] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
        seeds = self._normalize_seeds(seed)
        options_list = self._normalize_options(options)

        obs_list: List[Dict[str, np.ndarray]] = []
        infos: List[Dict[str, object]] = []
        for env, env_seed, env_opts in zip(self.envs, seeds, options_list):
            obs, info = env.reset(seed=env_seed, options=env_opts)
            obs_list.append(obs)
            infos.append(info)

        return self._stack_observations(obs_list), infos

    def step(
        self, actions: Sequence[Union[int, np.integer, str]]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
        if len(actions) != self.num_envs:
            raise ValueError("actions length must match num_envs.")

        obs_list: List[Dict[str, np.ndarray]] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, object]] = []

        for idx, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, term, trunc, info = env.step(action)

            if self.auto_reset and (term or trunc):
                final_obs = obs
                final_info = dict(info)
                obs, reset_info = env.reset()
                info = dict(info)
                info["final_observation"] = final_obs
                info["final_info"] = final_info
                info["reset_info"] = reset_info

            obs_list.append(obs)
            rewards[idx] = reward
            terminated[idx] = term
            truncated[idx] = trunc
            infos.append(info)

        return self._stack_observations(obs_list), rewards, terminated, truncated, infos

    def close(self) -> None:
        for env in self.envs:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()

    def _stack_observations(self, obs_list: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        keys = obs_list[0].keys()
        return {key: np.stack([obs[key] for obs in obs_list], axis=0) for key in keys}

    def _normalize_seeds(self, seed: Optional[Union[int, Sequence[Optional[int]]]]) -> List[Optional[int]]:
        if seed is None:
            return [None] * self.num_envs
        if isinstance(seed, int):
            return [seed + idx for idx in range(self.num_envs)]
        seeds = list(seed)
        if len(seeds) != self.num_envs:
            raise ValueError("seed sequence length must match num_envs.")
        return seeds

    def _normalize_options(
        self,
        options: Optional[Union[Dict[str, object], Sequence[Optional[Dict[str, object]]]]],
    ) -> List[Optional[Dict[str, object]]]:
        if options is None:
            return [None] * self.num_envs
        if isinstance(options, dict):
            return [options] * self.num_envs
        opts = list(options)
        if len(opts) != self.num_envs:
            raise ValueError("options sequence length must match num_envs.")
        return opts
