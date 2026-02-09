from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYMNASIUM = True
except ImportError:  # pragma: no cover - optional dependency
    gym = object  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _HAS_GYMNASIUM = False


class Feedback(IntEnum):
    GREY = 0
    YELLOW = 1
    GREEN = 2


@dataclass(frozen=True)
class RewardConfig:
    step_penalty: float = -0.01
    invalid_guess_penalty: float = -0.05
    green_reward: float = 0.05
    yellow_reward: float = 0.02
    win_reward: float = 1.0
    lose_reward: float = -1.0
    dense: bool = True


class WordleEnv(gym.Env if _HAS_GYMNASIUM else object):  # type: ignore[misc]
    """
    RL-ready Wordle environment.

    Action:
    - Discrete index into `allowed_guesses`.

    Observation dict:
    - guesses:        (max_attempts, word_length), uint8 in [0, 26] where 26 is padding.
    - feedback:       (max_attempts, word_length), uint8 in [0, 3] where 3 is padding.
    - alphabet_status:(26,), uint8 in [0, 3] (0 unknown, 1 absent, 2 present, 3 fixed/green).
    - attempts_used:  (1,), int32.
    - action_mask:    (num_actions,), uint8 valid action mask (1 valid, 0 invalid).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    DEFAULT_ANSWERS: Tuple[str, ...] = (
        "cigar",
        "rebut",
        "sissy",
        "humph",
        "awake",
        "blush",
        "focal",
        "evade",
        "naval",
        "serve",
        "heath",
        "dwarf",
        "model",
        "karma",
        "stink",
        "grade",
        "quiet",
        "bench",
        "abate",
        "feign",
        "major",
        "death",
        "fresh",
        "crust",
        "stool",
        "colon",
        "abase",
        "marry",
        "react",
        "batty",
    )

    def __init__(
        self,
        *,
        answers: Optional[Sequence[str]] = None,
        allowed_guesses: Optional[Sequence[str]] = None,
        max_attempts: int = 6,
        word_length: int = 5,
        hard_mode: bool = False,
        reward_config: Optional[RewardConfig] = None,
        include_action_mask: bool = True,
        render_mode: Optional[str] = None,
    ) -> None:
        self.word_length = word_length
        self.max_attempts = max_attempts
        self.hard_mode = hard_mode
        self.reward_config = reward_config or RewardConfig()
        self.include_action_mask = include_action_mask
        self.render_mode = render_mode

        raw_answers = list(answers) if answers is not None else list(self.DEFAULT_ANSWERS)
        self.answers = self._normalize_words(raw_answers, word_length=word_length, name="answers")

        raw_allowed = list(allowed_guesses) if allowed_guesses is not None else list(self.answers)
        self.allowed_guesses = self._normalize_words(
            raw_allowed, word_length=word_length, name="allowed_guesses"
        )

        missing_answers = set(self.answers) - set(self.allowed_guesses)
        if missing_answers:
            raise ValueError(
                "Every answer must exist in allowed_guesses. Missing: "
                + ", ".join(sorted(missing_answers))
            )

        self._word_to_action = {word: idx for idx, word in enumerate(self.allowed_guesses)}
        self._rng = Random()

        self._answer = ""
        self._attempts_used = 0
        self._done = False
        self._guesses = np.full(
            (self.max_attempts, self.word_length), fill_value=26, dtype=np.uint8
        )
        self._feedback = np.full(
            (self.max_attempts, self.word_length), fill_value=3, dtype=np.uint8
        )
        self._alphabet_status = np.zeros(26, dtype=np.uint8)

        if _HAS_GYMNASIUM:
            self.action_space = spaces.Discrete(len(self.allowed_guesses))  # type: ignore[attr-defined]
            self.observation_space = spaces.Dict(  # type: ignore[attr-defined]
                {
                    "guesses": spaces.Box(
                        low=0,
                        high=26,
                        shape=(self.max_attempts, self.word_length),
                        dtype=np.uint8,
                    ),
                    "feedback": spaces.Box(
                        low=0,
                        high=3,
                        shape=(self.max_attempts, self.word_length),
                        dtype=np.uint8,
                    ),
                    "alphabet_status": spaces.Box(low=0, high=3, shape=(26,), dtype=np.uint8),
                    "attempts_used": spaces.Box(low=0, high=self.max_attempts, shape=(1,), dtype=np.int32),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(len(self.allowed_guesses),), dtype=np.uint8
                    ),
                }
            )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        if seed is not None:
            self._rng.seed(seed)

        chosen_answer: Optional[str] = None
        if options:
            forced = options.get("answer")
            if forced is not None:
                normalized = self._validate_word(forced, self.word_length)
                if normalized not in self.answers:
                    raise ValueError(f"Forced answer must be in answers list: {forced}")
                chosen_answer = normalized

        self._answer = chosen_answer if chosen_answer is not None else self._rng.choice(self.answers)
        self._attempts_used = 0
        self._done = False
        self._guesses.fill(26)
        self._feedback.fill(3)
        self._alphabet_status.fill(0)

        obs = self._build_observation()
        info: Dict[str, object] = {"answer_length": self.word_length}
        return obs, info

    def step(
        self, action: Union[int, np.integer, str]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, object]]:
        if self._answer == "":
            raise RuntimeError("Environment must be reset before stepping.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before calling step() again.")

        guess = self._action_to_guess(action)
        hard_mode_ok, hard_mode_reason = self._validate_hard_mode_guess(guess)

        if not hard_mode_ok:
            reward = self.reward_config.invalid_guess_penalty
            obs = self._build_observation()
            info = {
                "guess": guess,
                "is_valid_guess": False,
                "reason": hard_mode_reason,
                "feedback": None,
            }
            return obs, reward, False, False, info

        row = self._attempts_used
        feedback = self._score_guess(guess, self._answer)
        self._guesses[row] = self._encode_word(guess)
        self._feedback[row] = np.array(feedback, dtype=np.uint8)
        self._update_alphabet_status(guess, feedback)
        self._attempts_used += 1

        solved = guess == self._answer
        exhausted = self._attempts_used >= self.max_attempts
        terminated = solved or exhausted
        truncated = False
        self._done = terminated

        reward = self._compute_reward(feedback, solved=solved, exhausted=exhausted)
        obs = self._build_observation()
        info = {
            "guess": guess,
            "is_valid_guess": True,
            "feedback": np.array(feedback, dtype=np.uint8),
            "answer": self._answer if terminated else None,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> str:
        if self._attempts_used == 0:
            return ""

        tokens = {Feedback.GREY: "⬛", Feedback.YELLOW: "🟨", Feedback.GREEN: "🟩"}
        rows: List[str] = []
        for row in range(self._attempts_used):
            guess = "".join(chr(x + ord("a")) for x in self._guesses[row] if x <= 25)
            fb = [Feedback(int(v)) for v in self._feedback[row]]
            marks = "".join(tokens[c] for c in fb)
            rows.append(f"{guess} {marks}")
        board = "\n".join(rows)
        if self.render_mode == "ansi":
            return board
        return board

    def valid_action_mask(self) -> np.ndarray:
        if not self.include_action_mask:
            return np.ones(len(self.allowed_guesses), dtype=np.uint8)

        if not self.hard_mode or self._attempts_used == 0:
            return np.ones(len(self.allowed_guesses), dtype=np.uint8)

        mask = np.zeros(len(self.allowed_guesses), dtype=np.uint8)
        for idx, guess in enumerate(self.allowed_guesses):
            ok, _ = self._validate_hard_mode_guess(guess)
            mask[idx] = 1 if ok else 0
        return mask

    @staticmethod
    def _normalize_words(words: Sequence[str], *, word_length: int, name: str) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for word in words:
            clean = WordleEnv._validate_word(word, word_length)
            if clean not in seen:
                seen.add(clean)
                normalized.append(clean)
        if not normalized:
            raise ValueError(f"{name} must contain at least one valid word.")
        return normalized

    @staticmethod
    def _validate_word(word: str, word_length: int) -> str:
        clean = word.strip().lower()
        if len(clean) != word_length or not clean.isalpha():
            raise ValueError(f"Invalid word '{word}'. Must be {word_length} alphabetic letters.")
        return clean

    def _action_to_guess(self, action: Union[int, np.integer, str]) -> str:
        if isinstance(action, (int, np.integer)):
            idx = int(action)
            if idx < 0 or idx >= len(self.allowed_guesses):
                raise ValueError(f"Action index out of range: {idx}")
            return self.allowed_guesses[idx]

        if isinstance(action, str):
            guess = self._validate_word(action, self.word_length)
            if guess not in self._word_to_action:
                return guess
            return guess

        raise TypeError(f"Unsupported action type: {type(action)!r}")

    @staticmethod
    def _encode_word(word: str) -> np.ndarray:
        return np.array([ord(ch) - ord("a") for ch in word], dtype=np.uint8)

    @staticmethod
    def _score_guess(guess: str, answer: str) -> List[int]:
        n = len(answer)
        out = [int(Feedback.GREY)] * n
        remaining = Counter()

        for i, (g, a) in enumerate(zip(guess, answer)):
            if g == a:
                out[i] = int(Feedback.GREEN)
            else:
                remaining[a] += 1

        for i, g in enumerate(guess):
            if out[i] == int(Feedback.GREEN):
                continue
            if remaining[g] > 0:
                out[i] = int(Feedback.YELLOW)
                remaining[g] -= 1

        return out

    def _update_alphabet_status(self, guess: str, feedback: Sequence[int]) -> None:
        for ch, fb in zip(guess, feedback):
            idx = ord(ch) - ord("a")
            if fb == int(Feedback.GREEN):
                self._alphabet_status[idx] = 3
            elif fb == int(Feedback.YELLOW):
                self._alphabet_status[idx] = max(2, self._alphabet_status[idx])
            elif self._alphabet_status[idx] == 0:
                self._alphabet_status[idx] = 1

    def _build_observation(self) -> Dict[str, np.ndarray]:
        return {
            "guesses": self._guesses.copy(),
            "feedback": self._feedback.copy(),
            "alphabet_status": self._alphabet_status.copy(),
            "attempts_used": np.array([self._attempts_used], dtype=np.int32),
            "action_mask": self.valid_action_mask(),
        }

    def _validate_hard_mode_guess(self, guess: str) -> Tuple[bool, Optional[str]]:
        if guess not in self._word_to_action:
            return False, "guess_not_in_allowed_list"

        if not self.hard_mode or self._attempts_used == 0:
            return True, None

        fixed_positions: Dict[int, str] = {}
        banned_positions: List[set[str]] = [set() for _ in range(self.word_length)]
        required_counts = Counter()

        for row in range(self._attempts_used):
            row_guess = "".join(chr(x + ord("a")) for x in self._guesses[row])
            row_fb = self._feedback[row].tolist()

            present_count = Counter()
            for i, (ch, fb) in enumerate(zip(row_guess, row_fb)):
                if fb == int(Feedback.GREEN):
                    fixed_positions[i] = ch
                    present_count[ch] += 1
                elif fb == int(Feedback.YELLOW):
                    banned_positions[i].add(ch)
                    present_count[ch] += 1

            for ch, c in present_count.items():
                required_counts[ch] = max(required_counts[ch], c)

        for idx, ch in fixed_positions.items():
            if guess[idx] != ch:
                return False, "hard_mode_missing_green"

        for idx, ch in enumerate(guess):
            if ch in banned_positions[idx]:
                return False, "hard_mode_yellow_position_violation"

        guess_counts = Counter(guess)
        for ch, needed in required_counts.items():
            if guess_counts[ch] < needed:
                return False, "hard_mode_missing_present_letter"

        return True, None

    def _compute_reward(self, feedback: Sequence[int], *, solved: bool, exhausted: bool) -> float:
        cfg = self.reward_config
        reward = cfg.step_penalty

        if cfg.dense:
            greens = sum(1 for x in feedback if x == int(Feedback.GREEN))
            yellows = sum(1 for x in feedback if x == int(Feedback.YELLOW))
            reward += cfg.green_reward * greens + cfg.yellow_reward * yellows

        if solved:
            reward += cfg.win_reward
        elif exhausted:
            reward += cfg.lose_reward

        return float(reward)
