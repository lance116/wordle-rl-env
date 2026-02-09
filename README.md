# Wordle RL Environment

High-fidelity Wordle environment for reinforcement learning with full-scale default lexicons and training-ready observations.

## What is included
- Exact Wordle scoring semantics with duplicate-letter correctness.
- Built-in NYT-scale vocabularies by default:
  - Answers: 2315
  - Allowed guesses: 12953
- Optional hard mode validation.
- Finite invalid-guess budget (`max_invalid_guesses`, default `2 * max_attempts`) to prevent non-terminating episodes during RL exploration.
- Configurable action masks:
  - `all`: all guesses are mask-valid.
  - `hard`: guesses satisfying hard mode hints.
  - `consistent`: guesses fully consistent with all prior feedback.
  - `auto`: `hard` if `hard_mode=True`, else `all`.
- Rich observations for modern policies:
  - Full board history
  - Alphabet state
  - Constraint tensors (`min/max_letter_counts`, `position_mask`)
  - Remaining candidate-answer count
  - Optional `action_mask`
- Candidate tracking API: exact feedback-consistent answer set over all previous guesses.
- Lightweight batched vector API via `WordleVectorEnv`.
- Gymnasium registration + wrappers for easier integration with external trainers.

## Installation
```bash
pip install -e .
```

Optional Gymnasium support:
```bash
pip install -e ".[gym]"
```

## Quick Start
```python
from wordle_rl import ActionMaskMode, WordleEnv

env = WordleEnv(
    hard_mode=True,
    action_mask_mode=ActionMaskMode.HARD,
    max_invalid_guesses=12,
)

obs, info = env.reset(seed=7)
done = False
while not done:
    action = env.sample_valid_action()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## Batched collection
```python
from wordle_rl import WordleVectorEnv

vec = WordleVectorEnv(num_envs=8, auto_reset=True)
obs, infos = vec.reset(seed=123)

actions = [env.sample_valid_action() for env in vec.envs]
obs, rewards, terminated, truncated, infos = vec.step(actions)
```

## Gymnasium registration and wrappers
`WordleRL-v0` is auto-registered when `wordle_rl` is imported and Gymnasium is installed.

```python
import gymnasium as gym
import wordle_rl
from wordle_rl import ActionMaskToInfoWrapper, FlattenWordleObservation

env = gym.make("WordleRL-v0")
env = ActionMaskToInfoWrapper(env)
flat_env = FlattenWordleObservation(env)
```

## Observation schema
- `guesses`: `(max_attempts, word_length)` with `a=0..z=25`, pad=`26`
- `feedback`: `(max_attempts, word_length)` with `GREY=0`, `YELLOW=1`, `GREEN=2`, pad=`3`
- `alphabet_status`: `(26,)` (`0=unknown, 1=absent, 2=present, 3=green-seen`)
- `attempts_used`: `(1,)`
- `candidate_count`: `(1,)`
- `min_letter_counts`: `(26,)` (optional)
- `max_letter_counts`: `(26,)` (optional)
- `position_mask`: `(word_length, 26)` (optional)
- `action_mask`: `(num_actions,)` (optional)

## Core APIs
- `WordleEnv.score_guess(guess, answer)` for standalone scoring
- `env.candidate_answers()` for exact remaining solutions
- `env.word_to_action(word)` / `env.action_to_word(idx)`
- `env.feedback_for(guess, answer=None)`
- `load_nyt_lexicon()`, `load_word_list(path)`, `make_lexicon(...)`

## Tests
```bash
python3 -m unittest discover -s tests -v
```

## Benchmarks
```bash
python3 benchmarks/benchmark_env.py
python3 scripts/perf_guardrail.py
```
