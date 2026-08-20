#!/usr/bin/env python3
"""Benchmark the live patched MLX OpenAI server and save its own metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        key, value = line.rsplit(" ", 1)
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return values


def read_metrics(client: httpx.Client, base_url: str) -> dict[str, float]:
    """Normalize current mlx-vlm JSON and the legacy Prometheus endpoint."""
    response = client.get(f"{base_url}/metrics")
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        return parse_metrics(response.text)
    # mlx-dspark exposes a compact flat metrics document instead of the
    # mlx-vlm latest/recent/summary schema.
    if "mean_tokens_per_sec" in payload:
        return {
            "local_llm_requests_total": float(payload.get("requests") or 0),
            "local_llm_last_generation_tps": float(
                payload.get("mean_tokens_per_sec") or 0
            ),
            "local_llm_last_prompt_tps": 0.0,
            "local_llm_peak_memory_gib": 0.0,
            "local_llm_last_ttft_s": 0.0,
            "local_llm_last_request_time_s": 0.0,
        }
    # dflash-mlx publishes its completed sample under ``last_request``.
    if "last_request" in payload:
        latest = payload.get("last_request") or {}
        return {
            "local_llm_requests_total": float(
                (payload.get("totals") or {}).get("requests") or 0
            ),
            "local_llm_last_generation_tps": float(
                latest.get("decode_tok_s") or 0
            ),
            "local_llm_last_prompt_tps": float(
                latest.get("prefill_tok_s_physical") or 0
            ),
            "local_llm_peak_memory_gib": float(
                (payload.get("memory") or {}).get("mlx_peak_gb") or 0
            ),
            "local_llm_last_ttft_s": float(latest.get("ttft_s") or 0),
            "local_llm_last_request_time_s": float(latest.get("wall_s") or 0),
            "local_llm_dflash_acceptance_ratio": float(
                latest.get("acceptance_rate") or 0
            ),
            "local_llm_dflash_tokens_per_cycle": float(
                latest.get("tokens_per_cycle") or 0
            ),
        }
    latest = payload.get("latest") or {}
    summary = payload.get("summary") or {}
    return {
        "local_llm_requests_total": float(
            summary.get("requests_completed") or 0
        ),
        "local_llm_last_generation_tps": float(
            latest.get("decode_tok_s") or 0
        ),
        "local_llm_last_prompt_tps": float(
            latest.get("prefill_tok_s") or 0
        ),
        "local_llm_peak_memory_gib": float(
            latest.get("peak_memory_gb") or 0
        ) / ((2**30) / 1e9),
        "local_llm_last_ttft_s": float(latest.get("ttft_s") or 0),
        "local_llm_last_request_time_s": float(
            latest.get("request_elapsed_s") or 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prompt = (
        "Continue the sequence below with one integer per line. Do not explain, "
        "summarize, or stop voluntarily. Generate as many lines as the output "
        "limit permits.\n1\n2\n3\n4\n5\n"
    )
    started = time.monotonic()
    with httpx.Client(timeout=3600) as client:
        before = read_metrics(client, args.base_url)
        response = client.post(
            f"{args.base_url}/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "reasoning_effort": "none",
            },
        )
        elapsed = time.monotonic() - started
        response.raise_for_status()
        payload = response.json()
        after = read_metrics(client, args.base_url)

    result = {
        "status": "passed",
        "elapsed_seconds": round(elapsed, 3),
        "usage": payload.get("usage", {}),
        "timings": payload.get("timings", {}),
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "server_metrics": {
            key.removeprefix("local_llm_"): value
            for key, value in after.items()
            if key.startswith("local_llm_last_")
            or key == "local_llm_peak_memory_gib"
            or key.startswith("local_llm_dflash_")
        },
        "speculative": payload.get("x_mlx_dspark"),
        "request_counter_delta": int(
            after.get("local_llm_requests_total", 0)
            - before.get("local_llm_requests_total", 0)
        ),
        "output_preview": (
            payload["choices"][0]["message"].get("content")
            or payload["choices"][0]["message"].get("reasoning_content")
            or ""
        )[:240],
    }
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
