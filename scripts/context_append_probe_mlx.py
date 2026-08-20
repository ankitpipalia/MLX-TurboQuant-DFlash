#!/usr/bin/env python3
"""Reconstruct a completed growth probe and append near the fixed KV limit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from context_growth_probe_mlx import FILLER, NEEDLE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--completed-stages", type=int, default=5)
    parser.add_argument("--tokens-per-stage", type=int, default=50000)
    parser.add_argument("--append-tokens", type=int, default=20000)
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
    for stage in range(args.completed_stages):
        segment = FILLER * repeats
        if stage == 0:
            midpoint = len(segment) // 2
            segment = (
                segment[:midpoint]
                + f"\nPersistent critical retrieval key: {NEEDLE}.\n"
                + segment[midpoint:]
            )
        final = stage == args.completed_stages - 1
        instruction = (
            "\nWhat is the exact persistent critical retrieval key? "
            "Answer with only the key."
            if final
            else (
                f"\nRetain archive segment {stage + 1}. "
                f"Reply with ACK-{stage + 1} only."
            )
        )
        messages.append({"role": "user", "content": segment + instruction})
        messages.append({
            "role": "assistant",
            "content": NEEDLE if final else f"ACK-{stage + 1}",
        })

    append_repeats = max(1, args.append_tokens // filler_tokens)
    messages.append({
        "role": "user",
        "content": (
            FILLER * append_repeats
            + "\nWhat is the exact persistent critical retrieval key? "
            "Answer with only the key."
        ),
    })
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = (
        rendered.get("input_ids", rendered)
        if isinstance(rendered, dict)
        else rendered
    )
    if (
        isinstance(token_ids, list)
        and token_ids
        and isinstance(token_ids[0], list)
    ):
        token_ids = token_ids[0]
    expected_prompt_tokens = len(token_ids)

    started = time.monotonic()
    result: dict
    try:
        with httpx.Client(timeout=7200) as client:
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
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        result = {
            "status": "passed" if NEEDLE in answer else "failed",
            "needle_found": NEEDLE in answer,
            "expected_prompt_tokens": expected_prompt_tokens,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "answer": answer,
            "usage": usage,
        }
    except Exception as exc:
        result = {
            "status": "error",
            "needle_found": False,
            "expected_prompt_tokens": expected_prompt_tokens,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }

    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
