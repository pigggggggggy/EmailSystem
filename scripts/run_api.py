#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EmailSystem FastAPI web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", default="mock", choices=["mock", "transformers", "vllm"])
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--eagle3-model-path", default=None)
    parser.add_argument("--speculative-tokens", type=int, default=3)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["EMAILSYSTEM_API_BACKEND"] = args.backend
    os.environ["EMAILSYSTEM_API_MODEL_PATH"] = args.model_path
    if args.eagle3_model_path:
        os.environ["EMAILSYSTEM_API_EAGLE3_MODEL_PATH"] = args.eagle3_model_path
    os.environ["EMAILSYSTEM_API_SPECULATIVE_TOKENS"] = str(args.speculative_tokens)
    os.environ["EMAILSYSTEM_API_TORCH_DTYPE"] = args.torch_dtype
    os.environ["EMAILSYSTEM_API_MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["EMAILSYSTEM_API_TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)
    os.environ["EMAILSYSTEM_API_GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)
    os.environ["EMAILSYSTEM_API_ENFORCE_EAGER"] = "false" if args.no_enforce_eager else "true"

    import uvicorn

    uvicorn.run("email_system.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
