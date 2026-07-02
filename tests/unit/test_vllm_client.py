import unittest

from email_system.models.vllm_client import truncate_token_ids


class VLLMClientTest(unittest.TestCase):
    def test_short_prompt_is_unchanged(self):
        tokens = [1, 2, 3]
        self.assertIs(truncate_token_ids(tokens, 3), tokens)

    def test_long_prompt_keeps_prefix_and_suffix(self):
        tokens = list(range(20))
        self.assertEqual(truncate_token_ids(tokens, 8), [0, 1, 2, 3, 4, 5, 18, 19])

    def test_rejects_no_input_budget(self):
        with self.assertRaises(ValueError):
            truncate_token_ids([1], 0)


if __name__ == "__main__":
    unittest.main()
