# Wordle RL Environment

High-fidelity, RL-ready Wordle environment for training modern agents.

## Features
- Exact Wordle feedback semantics:
  - `GREEN`: right letter, right position.
  - `YELLOW`: right letter, wrong position.
  - `GREY`: letter not present (after duplicate accounting).
- Correct duplicate-letter handling (Wordle two-pass scoring logic).
- Optional hard mode constraints on subsequent guesses.
- Invalid-guess behavior that does not consume an attempt.
- Action masking for policy-gradient and masked-action methods.
- Gymnasium-compatible `reset()` / `step()` interface.

## Quick Start
```python
from wordle_rl import WordleEnv

env = WordleEnv(
    answers=["cigar", "rebut", "sissy"],
    allowed_guesses=["cigar", "rebut", "sissy", "raise", "arise", "stare"],
    hard_mode=False,
)

obs, info = env.reset(seed=7)
done = False
while not done:
    action = env.action_space.sample() if hasattr(env, "action_space") else 0
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## Observation / Action
- Action: `Discrete(len(allowed_guesses))`
- Observation dictionary:
  - `guesses`: `(max_attempts, word_length)` letter indices (`a=0..z=25`, pad=`26`)
  - `feedback`: `(max_attempts, word_length)` (`GREY=0`, `YELLOW=1`, `GREEN=2`, pad=`3`)
  - `alphabet_status`: `(26,)` (`0=unknown, 1=absent, 2=present, 3=green`)
  - `attempts_used`: `(1,)`
  - `action_mask`: `(num_actions,)` (`1=valid`, `0=invalid`)

## Notes for SOTA Training
- Use a large, realistic `allowed_guesses` vocabulary.
- Keep `answers` as the subset of valid target words.
- Use `action_mask` for masked PPO / A2C / transformer policies to avoid impossible actions.
- Start with dense rewards, then move to sparse rewards for robust final policies.

## Tests
```bash
python3 -m unittest discover -s tests -v
```
