import unittest

from email_system.models import MockLLMClient, build_llm_client


class ModelFactoryTest(unittest.TestCase):
    def test_build_mock_client(self):
        self.assertIsInstance(build_llm_client("mock"), MockLLMClient)

    def test_reject_unknown_backend(self):
        with self.assertRaises(ValueError):
            build_llm_client("missing")


if __name__ == "__main__":
    unittest.main()
