"""Low-overhead process metrics for the patched MLX-LM server."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from threading import Lock
from typing import Any


_LOCK = Lock()
_VALUES: dict[str, float] = {
    "requests_total": 0,
    "errors_total": 0,
    "rejected_total": 0,
    "cache_hits_total": 0,
    "cached_tokens_total": 0,
    "forced_reprocess_total": 0,
    "tool_loops_broken_total": 0,
    "prompt_boundary_checkpoints_total": 0,
    "prompt_boundary_mismatches_total": 0,
    "checkpoints_created_total": 0,
    "checkpoints_restored_total": 0,
    "prompt_tokens_total": 0,
    "tokens_generated_total": 0,
    "last_cached_tokens": 0,
    "last_replayed_tokens": 0,
    "last_prompt_tokens": 0,
    "last_generation_tokens": 0,
    "last_prompt_tps": 0,
    "last_generation_tps": 0,
    "last_prefill_step": 0,
    "last_target_depth": 0,
    "peak_memory_gib": 0,
}


def _increment(name: str, amount: float = 1) -> None:
    with _LOCK:
        _VALUES[name] = _VALUES.get(name, 0) + amount


def record_cache_fetch(
    *, reused: int, replayed: int, had_session: bool, restored_checkpoint: bool
) -> None:
    with _LOCK:
        _VALUES["last_cached_tokens"] = reused
        _VALUES["last_replayed_tokens"] = replayed
        _VALUES["cached_tokens_total"] += reused
        if reused:
            _VALUES["cache_hits_total"] += 1
        elif had_session:
            _VALUES["forced_reprocess_total"] += 1
        if restored_checkpoint:
            _VALUES["checkpoints_restored_total"] += 1


def record_checkpoint() -> None:
    _increment("checkpoints_created_total")


def record_rejection() -> None:
    _increment("rejected_total")


def record_error() -> None:
    _increment("errors_total")


def record_tool_loop_broken() -> None:
    _increment("tool_loops_broken_total")


def record_prompt_boundary_checkpoint(*, matched: bool) -> None:
    _increment(
        "prompt_boundary_checkpoints_total"
        if matched
        else "prompt_boundary_mismatches_total"
    )


def record_generation(response: Any) -> None:
    """Record the final ``GenerationResponse`` emitted by MLX-LM."""
    with _LOCK:
        _VALUES["requests_total"] += 1
        prompt_tokens = int(getattr(response, "prompt_tokens", 0) or 0)
        generated = int(getattr(response, "generation_tokens", 0) or 0)
        _VALUES["prompt_tokens_total"] += prompt_tokens
        _VALUES["tokens_generated_total"] += generated
        _VALUES["last_prompt_tokens"] = prompt_tokens
        _VALUES["last_generation_tokens"] = generated
        _VALUES["last_prompt_tps"] = float(
            getattr(response, "prompt_tps", 0) or 0
        )
        _VALUES["last_generation_tps"] = float(
            getattr(response, "generation_tps", 0) or 0
        )
        _VALUES["peak_memory_gib"] = max(
            _VALUES["peak_memory_gib"],
            float(getattr(response, "peak_memory", 0) or 0),
        )


def record_prefill_plan(*, step: int, target_depth: int) -> None:
    with _LOCK:
        _VALUES["last_prefill_step"] = step
        _VALUES["last_target_depth"] = target_depth


def instrument_stream_generate(
    original: Callable[..., Iterator[Any]],
) -> Callable[..., Iterator[Any]]:
    """Wrap MLX generation without adding work to the token hot path."""

    def measured(*args: Any, **kwargs: Any) -> Iterator[Any]:
        final = None
        recorded = False
        try:
            for response in original(*args, **kwargs):
                final = response
                # MLX-LM's server breaks its outer loop as soon as it observes
                # a finish reason, so the wrapped generator is not necessarily
                # resumed after its final yield. Record before yielding it.
                if getattr(response, "finish_reason", None) is not None:
                    record_generation(response)
                    recorded = True
                yield response
            if final is not None and not recorded:
                record_generation(final)
        except Exception:
            record_error()
            raise

    return measured


def snapshot() -> dict[str, float]:
    with _LOCK:
        return dict(_VALUES)


def prometheus_text() -> str:
    """Return a Prometheus text exposition understood by the workbench."""
    lines = [
        "# HELP local_llm_requests_total Completed MLX generation requests.",
        "# TYPE local_llm_requests_total counter",
    ]
    for name, value in snapshot().items():
        metric = f"local_llm_{name}"
        lines.append(f"{metric} {value}")
    return "\n".join(lines) + "\n"


def install_metrics_endpoint(handler_class: type[Any]) -> None:
    """Add ``GET /metrics`` to MLX-LM's existing API handler."""
    original = handler_class.do_GET

    def do_GET(self: Any) -> None:
        if self.path == "/metrics":
            payload = prometheus_text().encode()
            self._set_completion_headers(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            return
        original(self)

    handler_class.do_GET = do_GET
