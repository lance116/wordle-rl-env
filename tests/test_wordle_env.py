import unittest

import numpy as np

from wordle_rl import (
    ActionMaskToInfoWrapper,
    ActionMaskMode,
    Feedback,
    FlattenWordleObservation,
    RewardConfig,
    WordleEnv,
    WordleVectorEnv,
    load_nyt_lexicon,
    register_envs,
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

    def test_vectorized_batch_scoring_matches_scalar(self) -> None:
        env = WordleEnv(answers=["apple"], allowed_guesses=["apple"])
        words = env._encode_words(["apple", "ample", "alien", "angle", "cigar"])
        guess = env._encode_word("allee")
        batch = env._score_guess_encoded_batch(guess, words)
        scalar = np.array(
            [
                env._score_guess("allee", "apple"),
                env._score_guess("allee", "ample"),
                env._score_guess("allee", "alien"),
                env._score_guess("allee", "angle"),
                env._score_guess("allee", "cigar"),
            ],
            dtype=np.uint8,
        )
        self.assertTrue((batch == scalar).all())

    def test_vector_env_batch_shapes(self) -> None:
        vec = WordleVectorEnv(
            num_envs=3,
            answers=["cigar", "rebut"],
            allowed_guesses=["cigar", "rebut"],
            max_invalid_guesses=1,
        )
        obs, infos = vec.reset(seed=10)
        self.assertEqual(len(infos), 3)
        self.assertEqual(obs["guesses"].shape[0], 3)
        actions = [0, 1, 0]
        next_obs, rewards, terminated, truncated, infos2 = vec.step(actions)
        self.assertEqual(next_obs["feedback"].shape[0], 3)
        self.assertEqual(rewards.shape, (3,))
        self.assertEqual(terminated.shape, (3,))
        self.assertEqual(truncated.shape, (3,))
        self.assertEqual(len(infos2), 3)

    def test_vector_env_input_validation(self) -> None:
        vec = WordleVectorEnv(
            num_envs=2,
            answers=["cigar", "rebut"],
            allowed_guesses=["cigar", "rebut"],
        )
        vec.reset(seed=0)
        with self.assertRaises(TypeError):
            vec.step("cigar")
        with self.assertRaises(TypeError):
            vec.reset(options="not-a-dict")

    def test_register_envs_and_make(self) -> None:
        try:
            import gymnasium as gym
        except ModuleNotFoundError:
            self.skipTest("gymnasium not installed")
        register_envs(force=True)
        env = gym.make("WordleRL-v0", answers=["cigar"], allowed_guesses=["cigar"])
        obs, info = env.reset(seed=0)
        self.assertIn("guesses", obs)
        env.close()

    def test_registered_env_respects_custom_max_attempts(self) -> None:
        try:
            import gymnasium as gym
        except ModuleNotFoundError:
            self.skipTest("gymnasium not installed")

        register_envs(force=True)
        env = gym.make(
            "WordleRL-v0",
            answers=["rebut"],
            allowed_guesses=["cigar", "rebut"],
            max_attempts=8,
            max_invalid_guesses=100,
        )
        env.reset(seed=0, options={"answer": "rebut"})

        ended_at = None
        for idx in range(8):
            _, _, terminated, truncated, _ = env.step(0)  # always wrong guess: cigar
            if terminated or truncated:
                ended_at = idx + 1
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                break

        self.assertEqual(ended_at, 8)
        env.close()

    def test_wrappers_work_when_gym_available(self) -> None:
        if ActionMaskToInfoWrapper is None or FlattenWordleObservation is None:
            self.skipTest("gymnasium wrappers unavailable")
        try:
            import gymnasium as gym
        except ModuleNotFoundError:
            self.skipTest("gymnasium not installed")

        base = WordleEnv(answers=["cigar"], allowed_guesses=["cigar"])
        wrapped = ActionMaskToInfoWrapper(base)
        obs, info = wrapped.reset(seed=0, options={"answer": "cigar"})
        self.assertIn("action_mask", info)

        flat = FlattenWordleObservation(WordleEnv(answers=["cigar"], allowed_guesses=["cigar"]))
        flat_obs, _ = flat.reset(seed=0, options={"answer": "cigar"})
        self.assertEqual(len(flat_obs.shape), 1)


if __name__ == "__main__":
    unittest.main()
