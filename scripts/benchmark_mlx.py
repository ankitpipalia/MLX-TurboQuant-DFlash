#!/usr/bin/env python3
"""Benchmark MLX-LM with standard and quantized KV cache settings."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / "Models/mlx/Qwen3.6-35B-A3B-4bit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    prompt = "Explain in three concise points why unified memory helps local inference."
    trials = []
    for kv_bits in (None, 8, 4):
        command = [
            str(ROOT / ".venv/bin/python"), "-m", "mlx_lm", "generate",
            "--model", str(args.model), "--prompt", prompt,
            "--max-tokens", "128", "--temp", "0", "--verbose", "True",
            "--chat-template-config", '{"enable_thinking":false}',
        ]
        if kv_bits is not None:
            command += [
                "--kv-bits", str(kv_bits),
                "--kv-group-size", "64",
                "--quantized-kv-start", "0",
            ]
        started = time.monotonic()
        completed = subprocess.run(command, text=True, capture_output=True)
        combined = completed.stdout + "\n" + completed.stderr
        rates = {
            key: float(value)
            for key, value in re.findall(
                r"(Prompt|Generation):\s+[^\n]*?([0-9.]+) tokens-per-sec", combined
            )
        }
        trials.append(
            {
                "kv_bits": kv_bits,
                "returncode": completed.returncode,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "rates": rates,
                "output": combined[-12000:],
            }
        )
        print(f"MLX kv_bits={kv_bits}: rc={completed.returncode}", flush=True)

    output = ROOT / "benchmark-results"
    output.mkdir(exist_ok=True)
    destination = output / f"mlx-bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    destination.write_text(json.dumps({"model": str(args.model), "trials": trials}, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
