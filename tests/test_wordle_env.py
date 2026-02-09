import unittest

from wordle_rl import (
    ActionMaskMode,
    Feedback,
    RewardConfig,
    WordleEnv,
    load_nyt_lexicon,
)


class WordleEnvTests(unittest.TestCase):
    def test_invalid_guess_budget_truncates_episode(self) -> None:
        env = WordleEnv(
            answers=["cigar"],
            allowed_guesses=["cigar", "cairn", "rebut"],
            hard_mode=True,
            max_invalid_guesses=2,
        )
        env.reset(options={"answer": "cigar"})
        env.step("cairn")

        _, _, terminated1, truncated1, info1 = env.step("rebut")
        self.assertFalse(terminated1)
        self.assertFalse(truncated1)
        self.assertEqual(info1["invalid_guesses_used"], 1)

        _, _, terminated2, truncated2, info2 = env.step("rebut")
        self.assertFalse(terminated2)
        self.assertTrue(truncated2)
        self.assertEqual(info2["invalid_guesses_used"], 2)
        self.assertEqual(info2["answer"], "cigar")

    def test_bundled_nyt_lexicon_loaded(self) -> None:
        lexicon = load_nyt_lexicon()
        self.assertEqual(len(lexicon.answers), 2315)
        self.assertEqual(len(lexicon.allowed_guesses), 12953)
        self.assertIn("cigar", lexicon.answers)
        self.assertIn("zymic", lexicon.allowed_guesses)

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

    def test_constraints_and_candidate_count_update(self) -> None:
        env = WordleEnv(
            answers=["apple", "ample", "alien", "angle"],
            allowed_guesses=["allee", "apple", "ample", "alien", "angle"],
        )
        obs, _ = env.reset(options={"answer": "apple"})
        self.assertEqual(obs["candidate_count"].item(), 4)

        obs2, _, _, _, _ = env.step("allee")
        l_idx = ord("l") - ord("a")
        a_idx = ord("a") - ord("a")
        self.assertEqual(obs2["max_letter_counts"][l_idx], 1)
        self.assertEqual(obs2["min_letter_counts"][a_idx], 1)
        self.assertEqual(obs2["candidate_count"].item(), 3)
        self.assertEqual(env.candidate_answers(), ["apple", "ample", "angle"])

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

    def test_action_mask_consistent_mode(self) -> None:
        env = WordleEnv(
            answers=["apple", "ample", "alien", "angle"],
            allowed_guesses=["allee", "apple", "ample", "alien", "angle", "rebut"],
            action_mask_mode=ActionMaskMode.CONSISTENT,
        )
        env.reset(options={"answer": "apple"})
        env.step("allee")

        idx = {w: i for i, w in enumerate(env.allowed_guesses)}
        mask = env.valid_action_mask()
        self.assertEqual(mask[idx["apple"]], 1)
        self.assertEqual(mask[idx["ample"]], 1)
        self.assertEqual(mask[idx["rebut"]], 0)
        self.assertEqual(mask[idx["angle"]], 1)

    def test_observation_flags_can_remove_optional_keys(self) -> None:
        env = WordleEnv(
            answers=["cigar"],
            allowed_guesses=["cigar"],
            include_action_mask=False,
            include_constraints=False,
        )
        obs, _ = env.reset(options={"answer": "cigar"})
        self.assertNotIn("action_mask", obs)
        self.assertNotIn("position_mask", obs)
        self.assertNotIn("min_letter_counts", obs)
        self.assertNotIn("max_letter_counts", obs)

    def test_reset_with_answer_index(self) -> None:
        env = WordleEnv(answers=["cigar", "rebut"], allowed_guesses=["cigar", "rebut"])
        env.reset(options={"answer_index": 1})
        self.assertEqual(env.answer, "rebut")

    def test_seeded_reset_is_deterministic(self) -> None:
        env = WordleEnv(answers=["cigar", "rebut", "sissy"], allowed_guesses=["cigar", "rebut", "sissy"])
        env.reset(seed=123)
        first = env.answer
        env.reset(seed=123)
        second = env.answer
        self.assertEqual(first, second)

    def test_gym_checker_passes_when_available(self) -> None:
        try:
            from gymnasium.utils.env_checker import check_env
        except ModuleNotFoundError:
            self.skipTest("gymnasium not installed")

        env = WordleEnv(answers=["cigar", "rebut"], allowed_guesses=["cigar", "rebut"])
        check_env(env)


if __name__ == "__main__":
    unittest.main()
