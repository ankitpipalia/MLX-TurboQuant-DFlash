#!/usr/bin/env python3
"""Run comparable Metal/TurboQuant llama-bench trials and save raw JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = ROOT / "vendor/llama-cpp-turboquant/build/bin/llama-bench"
DEFAULT_MODEL = Path.home() / (
    "Models/gguf/Qwen3.6-35B-A3B-Uncensored-Aggressive/"
    "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
)
CONFIGS = {
    "q8_0/q8_0": ("q8_0", "q8_0"),
    "q8_0/turbo4": ("q8_0", "turbo4"),
    "turbo4/turbo4": ("turbo4", "turbo4"),
    "turbo3/turbo3": ("turbo3", "turbo3"),
}


def run(command: list[str], *, force_symmetric: bool) -> tuple[list[dict], str]:
    env = os.environ.copy()
    if force_symmetric:
        env["TURBO_AUTO_ASYMMETRIC"] = "0"
    completed = subprocess.run(
        command, text=True, capture_output=True, check=True, env=env
    )
    return json.loads(completed.stdout), completed.stderr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--prompts", default="512,8192,32768")
    parser.add_argument("--generation", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument(
        "--config",
        action="append",
        choices=tuple(CONFIGS),
        help="cache pair to test; repeat to select multiple (default: all)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    output_dir = ROOT / "benchmark-results"
    output_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    document = {
        "created_at": stamp,
        "machine": platform.platform(),
        "memory_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "model": str(args.model),
        "tests": [],
    }

    selected = args.config or list(CONFIGS)
    for config_name in selected:
        cache_k, cache_v = CONFIGS[config_name]
        command = [
            str(args.binary),
            "-m", str(args.model),
            "-p", args.prompts,
            "-n", str(args.generation),
            "-r", str(args.repetitions),
            "-b", str(args.batch), "-ub", str(args.ubatch),
            "-t", str(args.threads), "-ngl", "99", "-ncmoe", "0",
            "-fa", "on", "--no-host", "1",
            "-ctk", cache_k, "-ctv", cache_v,
            "-o", "json",
        ]
        started = time.monotonic()
        try:
            rows, stderr = run(
                command, force_symmetric=cache_k.startswith("turbo")
            )
            result = {"status": "ok", "rows": rows, "stderr": stderr[-8000:]}
        except subprocess.CalledProcessError as exc:
            result = {
                "status": "failed",
                "returncode": exc.returncode,
                "stdout": exc.stdout[-8000:],
                "stderr": exc.stderr[-8000:],
            }
        result.update(
            cache_k=cache_k,
            cache_v=cache_v,
            auto_asymmetric_disabled=cache_k.startswith("turbo"),
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
        document["tests"].append(result)
        print(f"{cache_k}/{cache_v}: {result['status']}", flush=True)

    destination = output_dir / f"llama-bench-{stamp}.json"
    destination.write_text(json.dumps(document, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
