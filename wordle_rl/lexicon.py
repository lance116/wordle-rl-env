from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Lexicon:
    answers: Tuple[str, ...]
    allowed_guesses: Tuple[str, ...]


def normalize_words(words: Sequence[str], *, word_length: int, name: str) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for word in words:
        clean = validate_word(word, word_length)
        if clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    if not normalized:
        raise ValueError(f"{name} must contain at least one valid word.")
    return normalized


def validate_word(word: str, word_length: int) -> str:
    clean = word.strip().lower()
    if len(clean) != word_length or not clean.isalpha():
        raise ValueError(f"Invalid word '{word}'. Must be {word_length} alphabetic letters.")
    return clean


def load_word_list(path: str, *, word_length: int = 5) -> List[str]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8").splitlines()
    words = [w for w in raw if w.strip()]
    return normalize_words(words, word_length=word_length, name=str(p))


def make_lexicon(
    *,
    answers: Sequence[str],
    allowed_guesses: Optional[Sequence[str]] = None,
    word_length: int = 5,
) -> Lexicon:
    clean_answers = normalize_words(answers, word_length=word_length, name="answers")
    if allowed_guesses is None:
        clean_allowed = clean_answers
    else:
        clean_allowed = normalize_words(
            allowed_guesses,
            word_length=word_length,
            name="allowed_guesses",
        )
        clean_allowed_set = set(clean_allowed)
        merged = list(clean_allowed) + [w for w in clean_answers if w not in clean_allowed_set]
        clean_allowed = normalize_words(
            merged,
            word_length=word_length,
            name="allowed_guesses",
        )

    return Lexicon(
        answers=tuple(clean_answers),
        allowed_guesses=tuple(clean_allowed),
    )


def load_nyt_lexicon(*, word_length: int = 5) -> Lexicon:
    if word_length != 5:
        raise ValueError("Bundled NYT lexicon only supports 5-letter Wordle.")

    data_dir = files("wordle_rl.data")
    answers_txt = data_dir.joinpath("nyt_answers.txt").read_text(encoding="utf-8").splitlines()
    guesses_txt = data_dir.joinpath("nyt_guesses.txt").read_text(encoding="utf-8").splitlines()

    answers = normalize_words(answers_txt, word_length=5, name="nyt_answers")
    guesses = normalize_words(guesses_txt, word_length=5, name="nyt_guesses")
    return make_lexicon(answers=answers, allowed_guesses=[*answers, *guesses], word_length=5)
