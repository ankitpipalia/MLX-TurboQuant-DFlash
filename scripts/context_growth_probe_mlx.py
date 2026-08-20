#!/usr/bin/env python3
"""Grow one MLX chat session in append-only chunks and verify an early needle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


NEEDLE = "BLUE-OTTER-7741"
FILLER = (
    "The project archive contains routine source-control notes, test summaries, "
    "and ordinary implementation details. Treat this as retained coding context. "
)


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
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--tokens-per-stage", type=int, default=60000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, fix_mistral_regex=True
    )
    filler_tokens = max(
        1, len(tokenizer.encode(FILLER, add_special_tokens=False))
    )
    repeats = max(1, args.tokens_per_stage // filler_tokens)
    messages: list[dict[str, str]] = []
    stages: list[dict] = []

    with httpx.Client(timeout=7200) as client:
        for stage in range(args.stages):
            segment = FILLER * repeats
            if stage == 0:
                midpoint = len(segment) // 2
                segment = (
                    segment[:midpoint]
                    + f"\nPersistent critical retrieval key: {NEEDLE}.\n"
                    + segment[midpoint:]
                )
            final = stage == args.stages - 1
            instruction = (
                "\nWhat is the exact persistent critical retrieval key? "
                "Answer with only the key."
                if final
                else f"\nRetain archive segment {stage + 1}. Reply with ACK-{stage + 1} only."
            )
            messages.append({"role": "user", "content": segment + instruction})
            tick = time.monotonic()
            response = client.post(
                f"{args.base_url}/v1/chat/completions",
                json={
                    "model": str(args.tokenizer),
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 32,
                    "reasoning_effort": "none",
                },
            )
            elapsed = time.monotonic() - tick
            response.raise_for_status()
            payload = response.json()
            answer = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage", {})
            stages.append({
                "stage": stage + 1,
                "elapsed_seconds": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": cached_tokens(payload),
                "timings": payload.get("timings", {}),
                "speculative": payload.get("x_mlx_dspark"),
                "answer": answer,
            })
            print(json.dumps(stages[-1]), flush=True)
            if not final:
                messages.append({"role": "assistant", "content": answer})

    result = {
        "status": "passed" if NEEDLE in stages[-1]["answer"] else "failed",
        "needle_found": NEEDLE in stages[-1]["answer"],
        "target_tokens": args.stages * args.tokens_per_stage,
        "stages": stages,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if result["needle_found"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
