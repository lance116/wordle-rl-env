import unittest

from wordle_rl import Feedback, RewardConfig, WordleEnv


class WordleEnvTests(unittest.TestCase):
    def test_duplicate_letter_scoring_is_exact(self) -> None:
        env = WordleEnv(answers=["apple"], allowed_guesses=["allee", "apple"])
        env.reset(options={"answer": "apple"})

        _, _, _, _, info = env.step("allee")
        self.assertEqual(
            info["feedback"].tolist(),
            [
                int(Feedback.GREEN),
                int(Feedback.YELLOW),
                int(Feedback.GREY),
                int(Feedback.GREY),
                int(Feedback.GREEN),
            ],
        )

    def test_solve_and_terminate(self) -> None:
        env = WordleEnv(
            answers=["cigar"],
            allowed_guesses=["cigar", "rebut"],
            reward_config=RewardConfig(dense=False, win_reward=1.0),
        )
        env.reset(options={"answer": "cigar"})
        _, reward, terminated, truncated, info = env.step("cigar")

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 1.0 + env.reward_config.step_penalty)
        self.assertEqual(info["answer"], "cigar")

    def test_invalid_guess_not_in_vocab_penalized_without_consuming_attempt(self) -> None:
        env = WordleEnv(
            answers=["cigar"],
            allowed_guesses=["cigar", "rebut"],
            reward_config=RewardConfig(invalid_guess_penalty=-0.123),
        )
        obs, _ = env.reset(options={"answer": "cigar"})
        self.assertEqual(obs["attempts_used"].item(), 0)

        obs2, reward, terminated, truncated, info = env.step("zzzzz")
        self.assertEqual(reward, -0.123)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["is_valid_guess"])
        self.assertEqual(obs2["attempts_used"].item(), 0)

    def test_hard_mode_enforces_hints(self) -> None:
        env = WordleEnv(
            answers=["cigar"],
            allowed_guesses=["cairn", "cigar", "sugar", "rebut"],
            hard_mode=True,
        )
        env.reset(options={"answer": "cigar"})
        env.step("cairn")  # c green, a yellow, i yellow.

        obs, reward, terminated, _, info = env.step("rebut")
        self.assertFalse(terminated)
        self.assertLess(reward, 0.0)
        self.assertFalse(info["is_valid_guess"])
        self.assertEqual(info["reason"], "hard_mode_missing_green")
        self.assertEqual(obs["attempts_used"].item(), 1)

    def test_action_mask_respects_hard_mode(self) -> None:
        env = WordleEnv(
            answers=["cigar"],
            allowed_guesses=["cairn", "cigar", "sugar", "rebut"],
            hard_mode=True,
        )
        env.reset(options={"answer": "cigar"})
        env.step("cairn")

        mask = env.valid_action_mask()
        idx = {w: i for i, w in enumerate(env.allowed_guesses)}
        self.assertEqual(mask[idx["rebut"]], 0)
        self.assertEqual(mask[idx["cigar"]], 1)


if __name__ == "__main__":
    unittest.main()
