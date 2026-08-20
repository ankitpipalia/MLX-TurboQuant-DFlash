#!/usr/bin/env python3
"""Long-context needle probe for the MLX OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


NEEDLE = "BLUE-OTTER-7741"
FILLER = (
    "The archive contains routine operational notes. "
    "Every paragraph is independent and should be read as ordinary filler. "
)


def _emit(result: dict, output: Path | None) -> None:
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path.home() / "Models/mlx/Qwen3.6-35B-A3B-4bit",
    )
    parser.add_argument("--target-tokens", type=int, default=32768)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10800,
        help="HTTP timeout for very long cold prefills (default: 3 hours)",
    )
    parser.add_argument("--output", type=Path, help="also save the result as JSON")
    args = parser.parse_args()

    started = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, fix_mistral_regex=True
        )
        unit_tokens = len(tokenizer.encode(FILLER, add_special_tokens=False))
        repeats = max(1, args.target_tokens // max(unit_tokens, 1))
        prompt = (
            FILLER * (repeats // 2)
            + f"\nCritical retrieval key: {NEEDLE}. Remember it exactly.\n"
            + FILLER * (repeats - repeats // 2)
            + "\nWhat is the exact critical retrieval key? Answer with only the key."
        )
        actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        with httpx.Client(timeout=args.timeout_seconds) as client:
            response = client.post(
                f"{args.base_url}/v1/chat/completions",
                json={
                    "model": str(args.tokenizer),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 64,
                    "reasoning_effort": "none",
                },
            )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        answer = message.get("content") or message.get("reasoning_content") or ""
        result = {
            "status": "passed" if NEEDLE in answer else "failed",
            "target_tokens": args.target_tokens,
            "actual_tokens": actual_tokens,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "needle_found": NEEDLE in answer,
            "answer": answer,
            "usage": payload.get("usage", {}),
            "timings": payload.get("timings", {}),
        }
        _emit(result, args.output)
        return 0 if result["needle_found"] else 2
    except Exception as exc:
        _emit(
            {
                "status": "error",
                "target_tokens": args.target_tokens,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "needle_found": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
            args.output,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
