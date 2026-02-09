from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum, IntEnum
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .lexicon import Lexicon, load_nyt_lexicon, make_lexicon, validate_word

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


class ActionMaskMode(str, Enum):
    ALL = "all"
    HARD = "hard"
    CONSISTENT = "consistent"
    AUTO = "auto"


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
    High-fidelity, RL-ready Wordle environment.

    Action:
    - int index into `allowed_guesses`, or a guess string.

    Observation dict:
    - guesses:            (max_attempts, word_length), uint8 [0..26], 26=padding.
    - feedback:           (max_attempts, word_length), uint8 [0..3], 3=padding.
    - alphabet_status:    (26,), uint8 [0..3] where 0=unknown, 1=absent, 2=present, 3=green-seen.
    - attempts_used:      (1,), int32.
    - candidate_count:    (1,), int32.
    - min_letter_counts:  (26,), uint8 lower bound of occurrences from feedback history.
    - max_letter_counts:  (26,), uint8 upper bound of occurrences from feedback history.
    - position_mask:      (word_length, 26), uint8 with allowed letters by position.
    - action_mask:        (num_actions,), uint8 valid action mask.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

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
        action_mask_mode: Union[str, ActionMaskMode] = "auto",
        include_constraints: bool = True,
        render_mode: Optional[str] = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be >= 1")
        if word_length <= 0:
            raise ValueError("word_length must be >= 1")

        self.word_length = int(word_length)
        self.max_attempts = int(max_attempts)
        self.hard_mode = bool(hard_mode)
        self.reward_config = reward_config or RewardConfig()
        self.include_action_mask = include_action_mask
        self.include_constraints = include_constraints
        self.render_mode = render_mode

        self._action_mask_mode = self._parse_mask_mode(action_mask_mode)

        if answers is None and allowed_guesses is None:
            default_lexicon = load_nyt_lexicon(word_length=self.word_length)
            self.answers = list(default_lexicon.answers)
            self.allowed_guesses = list(default_lexicon.allowed_guesses)
        else:
            if answers is None:
                raise ValueError("answers must be provided when allowed_guesses is provided.")
            custom_lexicon = make_lexicon(
                answers=answers,
                allowed_guesses=allowed_guesses,
                word_length=self.word_length,
            )
            self.answers = list(custom_lexicon.answers)
            self.allowed_guesses = list(custom_lexicon.allowed_guesses)

        self._word_to_action = {word: idx for idx, word in enumerate(self.allowed_guesses)}
        self._rng = Random()

        self._answer = ""
        self._attempts_used = 0
        self._done = False

        self._guesses = np.full((self.max_attempts, self.word_length), fill_value=26, dtype=np.uint8)
        self._feedback = np.full((self.max_attempts, self.word_length), fill_value=3, dtype=np.uint8)
        self._alphabet_status = np.zeros(26, dtype=np.uint8)
        self._green_seen = np.zeros(26, dtype=np.uint8)

        self._known_greens = np.full(self.word_length, fill_value=26, dtype=np.uint8)
        self._banned_positions = np.zeros((self.word_length, 26), dtype=np.uint8)
        self._min_letter_counts = np.zeros(26, dtype=np.uint8)
        self._max_letter_counts = np.full(26, fill_value=self.word_length, dtype=np.uint8)

        self._hard_fixed_positions = np.full(self.word_length, fill_value=26, dtype=np.uint8)
        self._hard_banned_positions = np.zeros((self.word_length, 26), dtype=np.uint8)
        self._hard_required_counts = np.zeros(26, dtype=np.uint8)

        self._guess_history: List[str] = []
        self._feedback_history: List[np.ndarray] = []
        self._candidate_answer_mask = np.ones(len(self.answers), dtype=np.uint8)

        if _HAS_GYMNASIUM:
            self.action_space = spaces.Discrete(len(self.allowed_guesses))  # type: ignore[attr-defined]

            obs_spaces: Dict[str, object] = {
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
                "attempts_used": spaces.Box(
                    low=0,
                    high=self.max_attempts,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "candidate_count": spaces.Box(
                    low=0,
                    high=len(self.answers),
                    shape=(1,),
                    dtype=np.int32,
                ),
            }

            if self.include_constraints:
                obs_spaces["min_letter_counts"] = spaces.Box(
                    low=0,
                    high=self.word_length,
                    shape=(26,),
                    dtype=np.uint8,
                )
                obs_spaces["max_letter_counts"] = spaces.Box(
                    low=0,
                    high=self.word_length,
                    shape=(26,),
                    dtype=np.uint8,
                )
                obs_spaces["position_mask"] = spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.word_length, 26),
                    dtype=np.uint8,
                )

            if self.include_action_mask:
                obs_spaces["action_mask"] = spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(self.allowed_guesses),),
                    dtype=np.uint8,
                )

            self.observation_space = spaces.Dict(obs_spaces)  # type: ignore[attr-defined]

    @property
    def answer(self) -> str:
        return self._answer

    @property
    def attempts_used(self) -> int:
        return self._attempts_used

    @property
    def attempts_left(self) -> int:
        return self.max_attempts - self._attempts_used

    @property
    def candidate_count(self) -> int:
        return int(self._candidate_answer_mask.sum())

    @property
    def lexicon(self) -> Lexicon:
        return Lexicon(
            answers=tuple(self.answers),
            allowed_guesses=tuple(self.allowed_guesses),
        )

    @staticmethod
    def score_guess(guess: str, answer: str) -> np.ndarray:
        clean_guess = guess.strip().lower()
        clean_answer = answer.strip().lower()
        if len(clean_guess) != len(clean_answer) or not clean_guess.isalpha() or not clean_answer.isalpha():
            raise ValueError("guess and answer must be alphabetic words with matching length.")
        return np.array(WordleEnv._score_guess(clean_guess, clean_answer), dtype=np.uint8)

    def feedback_for(self, guess: str, answer: Optional[str] = None) -> np.ndarray:
        target = self._answer if answer is None else answer
        if target == "":
            raise RuntimeError("Environment has no active answer. Call reset() first.")
        return self.score_guess(guess, target)

    def action_to_word(self, action: int) -> str:
        if action < 0 or action >= len(self.allowed_guesses):
            raise ValueError(f"Action index out of range: {action}")
        return self.allowed_guesses[action]

    def word_to_action(self, word: str) -> int:
        clean = validate_word(word, self.word_length)
        if clean not in self._word_to_action:
            raise ValueError(f"Word not in allowed_guesses: {word}")
        return self._word_to_action[clean]

    def sample_valid_action(self) -> int:
        mask = self.valid_action_mask()
        valid_indices = np.flatnonzero(mask)
        if valid_indices.size == 0:
            return int(self._rng.randrange(len(self.allowed_guesses)))
        return int(valid_indices[self._rng.randrange(valid_indices.size)])

    def candidate_answers(self) -> List[str]:
        return [w for w, keep in zip(self.answers, self._candidate_answer_mask) if keep == 1]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Union[str, int]]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        if seed is not None:
            self._rng.seed(seed)

        chosen_answer: Optional[str] = None
        if options:
            forced = options.get("answer")
            if forced is not None:
                forced_str = str(forced)
                normalized = validate_word(forced_str, self.word_length)
                if normalized not in self.answers:
                    raise ValueError(f"Forced answer must be in answers list: {forced_str}")
                chosen_answer = normalized

            answer_index = options.get("answer_index")
            if answer_index is not None:
                idx = int(answer_index)
                if idx < 0 or idx >= len(self.answers):
                    raise ValueError("answer_index out of range.")
                chosen_answer = self.answers[idx]

        self._answer = chosen_answer if chosen_answer is not None else self._rng.choice(self.answers)
        self._attempts_used = 0
        self._done = False

        self._guesses.fill(26)
        self._feedback.fill(3)
        self._alphabet_status.fill(0)
        self._green_seen.fill(0)

        self._known_greens.fill(26)
        self._banned_positions.fill(0)
        self._min_letter_counts.fill(0)
        self._max_letter_counts.fill(self.word_length)

        self._hard_fixed_positions.fill(26)
        self._hard_banned_positions.fill(0)
        self._hard_required_counts.fill(0)

        self._guess_history.clear()
        self._feedback_history.clear()
        self._candidate_answer_mask.fill(1)

        obs = self._build_observation()
        info: Dict[str, object] = {
            "answer_length": self.word_length,
            "allowed_guesses": len(self.allowed_guesses),
            "answers": len(self.answers),
        }
        return obs, info

    def step(
        self,
        action: Union[int, np.integer, str],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, object]]:
        if self._answer == "":
            raise RuntimeError("Environment must be reset before stepping.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before calling step() again.")

        guess = self._action_to_guess(action)
        is_valid, reason = self._validate_guess(guess)
        if not is_valid:
            reward = float(self.reward_config.invalid_guess_penalty)
            obs = self._build_observation()
            info = {
                "guess": guess,
                "is_valid_guess": False,
                "reason": reason,
                "feedback": None,
                "candidate_count": self.candidate_count,
            }
            return obs, reward, False, False, info

        row = self._attempts_used
        feedback = np.array(self._score_guess(guess, self._answer), dtype=np.uint8)

        self._guesses[row] = self._encode_word(guess)
        self._feedback[row] = feedback
        self._guess_history.append(guess)
        self._feedback_history.append(feedback.copy())

        self._update_alphabet_status(guess, feedback)
        self._update_constraint_state(guess, feedback)
        self._update_hard_mode_state(guess, feedback)
        self._refresh_candidates()

        self._attempts_used += 1

        solved = guess == self._answer
        exhausted = self._attempts_used >= self.max_attempts
        terminated = solved or exhausted
        self._done = terminated

        reward = self._compute_reward(feedback, solved=solved, exhausted=exhausted)
        obs = self._build_observation()
        info = {
            "guess": guess,
            "is_valid_guess": True,
            "feedback": feedback.copy(),
            "answer": self._answer if terminated else None,
            "candidate_count": self.candidate_count,
        }
        return obs, reward, terminated, False, info

    def render(self) -> str:
        if self._attempts_used == 0:
            return ""

        tokens = {Feedback.GREY: "⬛", Feedback.YELLOW: "🟨", Feedback.GREEN: "🟩"}
        rows: List[str] = []
        for row in range(self._attempts_used):
            guess = "".join(chr(v + ord("a")) for v in self._guesses[row] if v <= 25)
            fb = [Feedback(int(v)) for v in self._feedback[row]]
            marks = "".join(tokens[c] for c in fb)
            rows.append(f"{guess} {marks}")
        return "\n".join(rows)

    def valid_action_mask(self) -> np.ndarray:
        mode = self._effective_mask_mode()

        if mode == ActionMaskMode.ALL:
            return np.ones(len(self.allowed_guesses), dtype=np.uint8)

        mask = np.zeros(len(self.allowed_guesses), dtype=np.uint8)

        if mode == ActionMaskMode.HARD:
            for idx, word in enumerate(self.allowed_guesses):
                ok, _ = self._validate_hard_mode_guess(word)
                mask[idx] = 1 if ok else 0
            return mask

        for idx, word in enumerate(self.allowed_guesses):
            mask[idx] = 1 if self._is_consistent_with_history(word) else 0
        return mask

    def _effective_mask_mode(self) -> ActionMaskMode:
        mode = self._action_mask_mode
        if mode == ActionMaskMode.AUTO:
            return ActionMaskMode.HARD if self.hard_mode else ActionMaskMode.ALL
        return mode

    @staticmethod
    def _parse_mask_mode(mode: Union[str, ActionMaskMode]) -> ActionMaskMode:
        if isinstance(mode, ActionMaskMode):
            return mode

        clean = mode.strip().lower()
        if clean == "all":
            return ActionMaskMode.ALL
        if clean == "hard":
            return ActionMaskMode.HARD
        if clean == "consistent":
            return ActionMaskMode.CONSISTENT
        if clean == "auto":
            return ActionMaskMode.AUTO
        raise ValueError("action_mask_mode must be one of: auto, all, hard, consistent.")

    def _build_observation(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {
            "guesses": self._guesses.copy(),
            "feedback": self._feedback.copy(),
            "alphabet_status": self._alphabet_status.copy(),
            "attempts_used": np.array([self._attempts_used], dtype=np.int32),
            "candidate_count": np.array([self.candidate_count], dtype=np.int32),
        }

        if self.include_constraints:
            obs["min_letter_counts"] = self._min_letter_counts.copy()
            obs["max_letter_counts"] = self._max_letter_counts.copy()
            obs["position_mask"] = self._position_mask()

        if self.include_action_mask:
            obs["action_mask"] = self.valid_action_mask()

        return obs

    def _position_mask(self) -> np.ndarray:
        mask = np.ones((self.word_length, 26), dtype=np.uint8)
        globally_absent = self._max_letter_counts == 0

        for idx in range(self.word_length):
            if self._known_greens[idx] <= 25:
                mask[idx].fill(0)
                mask[idx, self._known_greens[idx]] = 1
            else:
                banned = self._banned_positions[idx] == 1
                mask[idx, banned] = 0
                mask[idx, globally_absent] = 0

        return mask

    def _validate_guess(self, guess: str) -> Tuple[bool, Optional[str]]:
        if guess not in self._word_to_action:
            return False, "guess_not_in_allowed_list"
        if not self.hard_mode:
            return True, None
        return self._validate_hard_mode_guess(guess)

    def _validate_hard_mode_guess(self, guess: str) -> Tuple[bool, Optional[str]]:
        if guess not in self._word_to_action:
            return False, "guess_not_in_allowed_list"

        if self._attempts_used == 0:
            return True, None

        for idx in range(self.word_length):
            fixed = self._hard_fixed_positions[idx]
            if fixed <= 25 and (ord(guess[idx]) - ord("a")) != fixed:
                return False, "hard_mode_missing_green"

        for idx in range(self.word_length):
            letter_idx = ord(guess[idx]) - ord("a")
            if self._hard_banned_positions[idx, letter_idx] == 1:
                return False, "hard_mode_yellow_position_violation"

        guess_counts = Counter(guess)
        for letter_idx in range(26):
            needed = int(self._hard_required_counts[letter_idx])
            if needed == 0:
                continue
            letter = chr(letter_idx + ord("a"))
            if guess_counts[letter] < needed:
                return False, "hard_mode_missing_present_letter"

        return True, None

    def _update_hard_mode_state(self, guess: str, feedback: np.ndarray) -> None:
        present_counts = Counter()

        for idx, (ch, fb) in enumerate(zip(guess, feedback.tolist())):
            letter_idx = ord(ch) - ord("a")
            if fb == int(Feedback.GREEN):
                self._hard_fixed_positions[idx] = letter_idx
                present_counts[ch] += 1
            elif fb == int(Feedback.YELLOW):
                self._hard_banned_positions[idx, letter_idx] = 1
                present_counts[ch] += 1

        for ch, count in present_counts.items():
            letter_idx = ord(ch) - ord("a")
            if count > self._hard_required_counts[letter_idx]:
                self._hard_required_counts[letter_idx] = count

    def _update_constraint_state(self, guess: str, feedback: np.ndarray) -> None:
        present_counts = Counter()
        total_counts = Counter(guess)

        for idx, (ch, fb) in enumerate(zip(guess, feedback.tolist())):
            letter_idx = ord(ch) - ord("a")
            if fb == int(Feedback.GREEN):
                self._known_greens[idx] = letter_idx
                present_counts[ch] += 1
            else:
                self._banned_positions[idx, letter_idx] = 1
                if fb == int(Feedback.YELLOW):
                    present_counts[ch] += 1

        for ch, present in present_counts.items():
            letter_idx = ord(ch) - ord("a")
            if present > self._min_letter_counts[letter_idx]:
                self._min_letter_counts[letter_idx] = present

        for ch, total in total_counts.items():
            letter_idx = ord(ch) - ord("a")
            present = present_counts.get(ch, 0)
            if present < total and present < self._max_letter_counts[letter_idx]:
                self._max_letter_counts[letter_idx] = present

    def _refresh_candidates(self) -> None:
        if not self._guess_history:
            self._candidate_answer_mask.fill(1)
            return

        for idx, answer in enumerate(self.answers):
            self._candidate_answer_mask[idx] = 1 if self._is_consistent_with_history(answer) else 0

    def _is_consistent_with_history(self, candidate_word: str) -> bool:
        for guess, feedback in zip(self._guess_history, self._feedback_history):
            scored = self._score_guess(guess, candidate_word)
            if any(a != b for a, b in zip(scored, feedback.tolist())):
                return False
        return True

    def _action_to_guess(self, action: Union[int, np.integer, str]) -> str:
        if isinstance(action, (int, np.integer)):
            idx = int(action)
            if idx < 0 or idx >= len(self.allowed_guesses):
                raise ValueError(f"Action index out of range: {idx}")
            return self.allowed_guesses[idx]

        if isinstance(action, str):
            return validate_word(action, self.word_length)

        raise TypeError(f"Unsupported action type: {type(action)!r}")

    @staticmethod
    def _encode_word(word: str) -> np.ndarray:
        return np.array([ord(ch) - ord("a") for ch in word], dtype=np.uint8)

    @staticmethod
    def _score_guess(guess: str, answer: str) -> List[int]:
        n = len(answer)
        out = [int(Feedback.GREY)] * n
        remaining = Counter()

        for idx, (g, a) in enumerate(zip(guess, answer)):
            if g == a:
                out[idx] = int(Feedback.GREEN)
            else:
                remaining[a] += 1

        for idx, g in enumerate(guess):
            if out[idx] == int(Feedback.GREEN):
                continue
            if remaining[g] > 0:
                out[idx] = int(Feedback.YELLOW)
                remaining[g] -= 1

        return out

    def _update_alphabet_status(self, guess: str, feedback: np.ndarray) -> None:
        for ch, fb in zip(guess, feedback.tolist()):
            letter_idx = ord(ch) - ord("a")
            if fb == int(Feedback.GREEN):
                self._green_seen[letter_idx] = 1
                self._alphabet_status[letter_idx] = 3
                continue

            if fb == int(Feedback.YELLOW):
                if self._green_seen[letter_idx] == 0:
                    self._alphabet_status[letter_idx] = max(2, self._alphabet_status[letter_idx])
                continue

            if (
                self._green_seen[letter_idx] == 0
                and self._min_letter_counts[letter_idx] == 0
                and self._alphabet_status[letter_idx] == 0
            ):
                self._alphabet_status[letter_idx] = 1

    def _compute_reward(self, feedback: np.ndarray, *, solved: bool, exhausted: bool) -> float:
        cfg = self.reward_config
        reward = cfg.step_penalty

        if cfg.dense:
            greens = int((feedback == int(Feedback.GREEN)).sum())
            yellows = int((feedback == int(Feedback.YELLOW)).sum())
            reward += cfg.green_reward * greens + cfg.yellow_reward * yellows

        if solved:
            reward += cfg.win_reward
        elif exhausted:
            reward += cfg.lose_reward

        return float(reward)
