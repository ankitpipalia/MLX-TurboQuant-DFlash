#!/usr/bin/env python3
"""Verify append-only OpenAI chat prefix reuse against the MLX server."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


NEEDLE = "BLUE-OTTER-7741"
FILLER = (
    "The archive contains routine operational notes. "
    "Every paragraph is independent and should be read as ordinary filler. "
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, fix_mistral_regex=True)
    unit = len(tokenizer.encode(FILLER, add_special_tokens=False))
    repeats = max(1, args.target_tokens // max(unit, 1))
    user_prompt = (
        FILLER * (repeats // 2)
        + f"\nCritical retrieval key: {NEEDLE}. Remember it exactly.\n"
        + FILLER * (repeats - repeats // 2)
        + "\nWhat is the exact critical retrieval key? Answer with only the key."
    )
    messages = [{"role": "user", "content": user_prompt}]
    common = {
        "model": str(args.tokenizer),
        "temperature": 0,
        "max_tokens": 128,
        "reasoning_effort": "none",
    }

    with httpx.Client(timeout=3600) as client:
        cold_started = time.monotonic()
        cold_response = client.post(
            f"{args.base_url}/v1/chat/completions",
            json={**common, "messages": messages},
        )
        cold_response.raise_for_status()
        cold_elapsed = time.monotonic() - cold_started
        cold = cold_response.json()
        cold_message = cold["choices"][0]["message"]
        assistant = (
            cold_message.get("content")
            or cold_message.get("reasoning_content")
            or ""
        )

        # Reuse requires a byte-stable rendered prefix. Preserve every field
        # emitted by the server (notably separate Qwen reasoning content).
        append_messages = messages + [
            cold_message,
            {
                "role": "user",
                "content": "Repeat the exact key again. Answer with only the key.",
            },
        ]
        warm_started = time.monotonic()
        warm_response = client.post(
            f"{args.base_url}/v1/chat/completions",
            json={**common, "messages": append_messages},
        )
        warm_response.raise_for_status()
        warm_elapsed = time.monotonic() - warm_started
        warm = warm_response.json()

    warm_message = warm["choices"][0]["message"]
    warm_answer = (
        warm_message.get("content")
        or warm_message.get("reasoning_content")
        or ""
    )
    cached = int(
        warm.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        or warm.get("timings", {}).get("cache_n", 0)
        or 0
    )
    result = {
        "status": "passed" if NEEDLE in warm_answer and cached > 0 else "failed",
        "target_tokens": args.target_tokens,
        "cold_elapsed_seconds": round(cold_elapsed, 3),
        "append_elapsed_seconds": round(warm_elapsed, 3),
        "speedup": round(cold_elapsed / warm_elapsed, 3) if warm_elapsed else None,
        "cold_answer": assistant,
        "append_answer": warm_answer,
        "cold_usage": cold.get("usage", {}),
        "append_usage": warm.get("usage", {}),
        "cold_timings": cold.get("timings", {}),
        "append_timings": warm.get("timings", {}),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
