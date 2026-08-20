#!/usr/bin/env python3
"""Run a small, repeatable coding-agent benchmark against the MLX server."""

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
            pass
    return values


def read_metrics(client: httpx.Client, base_url: str) -> dict[str, float]:
    """Normalize current mlx-vlm JSON and the legacy Prometheus endpoint."""
    response = client.get(f"{base_url}/metrics")
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        return parse_metrics(response.text)
    if "mean_tokens_per_sec" in payload:
        return {
            "local_llm_last_generation_tps": float(
                payload.get("mean_tokens_per_sec") or 0
            ),
            "local_llm_last_prompt_tps": 0.0,
            "local_llm_last_ttft_s": 0.0,
        }
    if "last_request" in payload:
        latest = payload.get("last_request") or {}
        return {
            "local_llm_last_generation_tps": float(
                latest.get("decode_tok_s") or 0
            ),
            "local_llm_last_prompt_tps": float(
                latest.get("prefill_tok_s_physical") or 0
            ),
            "local_llm_last_ttft_s": float(latest.get("ttft_s") or 0),
        }
    latest = payload.get("latest") or {}
    return {
        "local_llm_last_generation_tps": float(
            latest.get("decode_tok_s") or 0
        ),
        "local_llm_last_prompt_tps": float(
            latest.get("prefill_tok_s") or 0
        ),
        "local_llm_last_ttft_s": float(latest.get("ttft_s") or 0),
    }


TASKS = [
    {
        "name": "implementation",
        "max_tokens": 384,
        "messages": [{
            "role": "user",
            "content": (
                "Return only Python code. Implement merge_intervals(intervals), "
                "accepting an iterable of [start, end] integer pairs. Normalize "
                "reversed endpoints, merge touching intervals, do not mutate the "
                "input, and return a sorted list of [start, end] lists."
            ),
        }],
    },
    {
        "name": "debugging",
        "max_tokens": 256,
        "messages": [{
            "role": "user",
            "content": (
                "Diagnose this Python bug and give the smallest corrected code "
                "snippet. Be concise.\n\n"
                "def consume(items=[]):\n"
                "    items.append('x')\n"
                "    return items\n\n"
                "The second call unexpectedly returns ['x', 'x']."
            ),
        }],
    },
    {
        "name": "tool_call",
        "max_tokens": 128,
        "messages": [{
            "role": "user",
            "content": (
                "Inspect /workspace/app.py with the available tool before making "
                "any recommendation. Do not guess its contents."
            ),
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }],
        "tool_choice": "auto",
    },
]


def tool_arguments_contain(arguments: object, expected: str) -> bool:
    """Accept both OpenAI's JSON string and already-decoded argument objects."""
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return expected in arguments
    else:
        decoded = arguments
    if isinstance(decoded, dict):
        return any(expected in str(value) for value in decoded.values())
    return expected in str(decoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results: list[dict] = []
    with httpx.Client(timeout=3600) as client:
        for task in TASKS:
            request = {
                "model": args.model,
                "messages": task["messages"],
                "temperature": 0,
                "max_tokens": task["max_tokens"],
                # The current server normalizes this standard field. The old
                # chat_template_kwargs extra did not reliably override a
                # model's native reasoning default.
                "reasoning_effort": "none",
            }
            if "tools" in task:
                request["tools"] = task["tools"]
                request["tool_choice"] = task["tool_choice"]
            started = time.monotonic()
            response = client.post(
                f"{args.base_url}/v1/chat/completions",
                json=request,
            )
            elapsed = time.monotonic() - started
            response.raise_for_status()
            payload = response.json()
            message = payload["choices"][0]["message"]
            metrics = read_metrics(client, args.base_url)
            tool_calls = message.get("tool_calls") or []
            results.append({
                "name": task["name"],
                "elapsed_seconds": round(elapsed, 3),
                "usage": payload.get("usage", {}),
                "timings": payload.get("timings", {}),
                "finish_reason": payload["choices"][0].get("finish_reason"),
                "generation_tps": round(
                    metrics.get("local_llm_last_generation_tps", 0), 3
                ),
                "prompt_tps": round(
                    metrics.get("local_llm_last_prompt_tps", 0), 3
                ),
                "tool_call_valid": (
                    task["name"] != "tool_call"
                    or (
                        bool(tool_calls)
                        and tool_calls[0].get("function", {}).get("name")
                        == "read_file"
                        and tool_arguments_contain(
                            tool_calls[0].get("function", {}).get("arguments", ""),
                            "/workspace/app.py",
                        )
                    )
                ),
                "tool_calls": tool_calls,
                "speculative": payload.get("x_mlx_dspark"),
                "content": message.get("content") or "",
                "reasoning_content": message.get("reasoning_content") or "",
            })
            print(json.dumps({
                "name": results[-1]["name"],
                "elapsed_seconds": results[-1]["elapsed_seconds"],
                "generation_tps": results[-1]["generation_tps"],
                "tool_call_valid": results[-1]["tool_call_valid"],
            }), flush=True)

    result = {
        "status": (
            "passed"
            if all(item["tool_call_valid"] for item in results)
            else "failed"
        ),
        "model": args.model,
        "tasks": results,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
