import unittest
from unittest.mock import patch

from email_system.models import MockLLMClient, build_llm_client


class ModelFactoryTest(unittest.TestCase):
    def test_build_mock_client(self):
        self.assertIsInstance(build_llm_client("mock"), MockLLMClient)

    def test_build_vllm_client_branch(self):
        with patch("email_system.models.factory.VLLMClient") as client_cls:
            build_llm_client(
                "vllm",
                model_path="models/Qwen3-4B",
                max_model_len=4096,
                tensor_parallel_size=2,
                gpu_memory_utilization=0.8,
                enforce_eager=True,
            )

        client_cls.assert_called_once_with(
            "models/Qwen3-4B",
            dtype="auto",
            max_model_len=4096,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
        )

    def test_reject_unknown_backend(self):
        with self.assertRaises(ValueError):
            build_llm_client("missing")


if __name__ == "__main__":
    unittest.main()
