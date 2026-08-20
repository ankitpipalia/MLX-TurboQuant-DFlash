#!/usr/bin/env python3
"""Measure cold versus incremental multi-turn MLX server latency."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


FILLER = "Stable project context for prefix-cache measurement. "


def cached_tokens(payload: dict) -> int:
    return int(
        payload.get("usage", {})
        .get("prompt_tokens_details", {})
        .get("cached_tokens", 0)
        or 0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, fix_mistral_regex=True
        )
        unit = max(1, len(tokenizer.encode(FILLER, add_special_tokens=False)))
        context = FILLER * max(1, args.target_tokens // unit)
        messages = [{"role": "user", "content": context + "\nReply with READY only."}]

        def request(turns: list[dict]) -> tuple[dict, float]:
            tick = time.monotonic()
            response = httpx.post(
                f"{args.base_url}/v1/chat/completions",
                json={
                    "model": str(args.tokenizer),
                    "messages": turns,
                    "temperature": 0,
                    "max_tokens": 16,
                    "reasoning_effort": "none",
                },
                timeout=3600,
            )
            response.raise_for_status()
            return response.json(), time.monotonic() - tick

        cold, cold_seconds = request(messages)
        answer = cold["choices"][0]["message"]["content"]
        warm_messages = [
            *messages,
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Now reply with WARM only."},
        ]
        warm, warm_seconds = request(warm_messages)
        result = {
            "status": "passed",
            "target_tokens": args.target_tokens,
            "cold_seconds": round(cold_seconds, 3),
            "warm_seconds": round(warm_seconds, 3),
            "speedup": round(cold_seconds / warm_seconds, 2) if warm_seconds else None,
            "cold_usage": cold.get("usage", {}),
            "warm_usage": warm.get("usage", {}),
            "cold_timings": cold.get("timings", {}),
            "warm_timings": warm.get("timings", {}),
            "cold_speculative": cold.get("x_mlx_dspark"),
            "warm_speculative": warm.get("x_mlx_dspark"),
            "warm_cached_tokens": cached_tokens(warm),
            "cold_answer": answer,
            "warm_answer": warm["choices"][0]["message"]["content"],
        }
        code = 0
    except Exception as exc:
        result = {
            "status": "error",
            "target_tokens": args.target_tokens,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
        code = 1
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
