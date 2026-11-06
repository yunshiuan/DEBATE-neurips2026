"""Launch an OpenAI-compatible vLLM endpoint for a base model or LoRA adapter."""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-name", default="debate-sft")
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    command = ["vllm", "serve", args.model, "--dtype", "bfloat16", "--max-model-len", str(args.max_model_len),
               "--tensor-parallel-size", str(args.tensor_parallel_size), "--port", str(args.port)]
    if args.adapter:
        if not (args.adapter / "adapter_config.json").exists():
            raise ValueError("--adapter must contain adapter_config.json from a saved LoRA run")
        command += ["--enable-lora", "--max-lora-rank", "16", "--lora-modules", f"{args.adapter_name}={args.adapter.resolve()}"]
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
