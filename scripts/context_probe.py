#!/usr/bin/env python3
"""Create a long prompt, call an OpenAI-compatible server, and verify a needle."""

from __future__ import annotations

import argparse
import json
import time

import httpx


NEEDLE = "BLUE-OTTER-7741"
FILLER = (
    "The archive contains routine operational notes. "
    "Every paragraph is independent and should be read as ordinary filler. "
)


def token_count(client: httpx.Client, base_url: str, text: str) -> int:
    response = client.post(f"{base_url}/tokenize", json={"content": text})
    response.raise_for_status()
    payload = response.json()
    return len(payload.get("tokens", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8097")
    parser.add_argument("--target-tokens", type=int, default=32768)
    args = parser.parse_args()

    with httpx.Client(timeout=3600) as client:
        unit_tokens = token_count(client, args.base_url, FILLER)
        repeats = max(1, args.target_tokens // max(unit_tokens, 1))
        left = FILLER * (repeats // 2)
        prompt = (
            left
            + f"\nCritical retrieval key: {NEEDLE}. Remember it exactly.\n"
            + FILLER * (repeats - repeats // 2)
            + "\nWhat is the exact critical retrieval key? Answer with only the key."
        )
        actual_tokens = token_count(client, args.base_url, prompt)
        started = time.monotonic()
        response = client.post(
            f"{args.base_url}/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        elapsed = time.monotonic() - started
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        print(
            json.dumps(
                {
                    "target_tokens": args.target_tokens,
                    "actual_tokens": actual_tokens,
                    "elapsed_seconds": round(elapsed, 3),
                    "needle_found": NEEDLE in answer,
                    "answer": answer,
                    "usage": usage,
                    "timings": payload.get("timings", {}),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
