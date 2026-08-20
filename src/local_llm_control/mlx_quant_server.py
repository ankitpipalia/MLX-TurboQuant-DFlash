"""MLX-LM OpenAI server with the KV quantization controls its CLI omits."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections.abc import Callable
from typing import Any

import mlx_lm.server as mlx_server

from .mlx_metrics import (
    install_metrics_endpoint,
    instrument_stream_generate,
    record_error,
    record_prefill_plan,
    record_rejection,
    record_tool_loop_broken,
)


_UNSTABLE_QWEN_ASSISTANT = """{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}"""
_STABLE_QWEN_ASSISTANT = """{%- if enable_thinking is defined and enable_thinking is false %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n\\n</think>\\n\\n' + content }}
        {%- elif (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}"""


def stabilize_qwen_chat_template(template: str | None) -> str | None:
    """Keep no-thinking assistant turns byte-stable for prefix-cache reuse."""
    if not template or _STABLE_QWEN_ASSISTANT in template:
        return template
    return template.replace(_UNSTABLE_QWEN_ASSISTANT, _STABLE_QWEN_ASSISTANT)


def stable_template_loader(original: Callable[..., Any]) -> Callable[..., Any]:
    def load(provider: Any, *args: Any, **kwargs: Any) -> Any:
        # Qwen3.5/3.6 tokenizers can inherit the historical Mistral regex.
        # Transformers detects it but only warns unless this opt-in is passed;
        # incorrect token boundaries would invalidate context measurements and
        # byte-stable session reuse.
        tokenizer_config = getattr(provider, "_tokenizer_config", None)
        if isinstance(tokenizer_config, dict):
            tokenizer_config.setdefault("fix_mistral_regex", True)
        result = original(provider, *args, **kwargs)
        tokenizer = getattr(provider, "tokenizer", None)
        template = getattr(tokenizer, "chat_template", None)
        stable = stabilize_qwen_chat_template(template)
        if tokenizer is not None and stable != template:
            tokenizer.chat_template = stable
            print(
                "local-llm: enabled byte-stable Qwen no-thinking chat template",
                file=sys.stderr,
                flush=True,
            )
        print(
            "local-llm: model load complete",
            file=sys.stderr,
            flush=True,
        )
        return result

    return load


def process_message_content_idempotent(messages: list[dict[str, Any]]) -> None:
    """Normalize chat messages without double-decoding tool arguments.

    OpenCode can send historical ``function.arguments`` as an object.
    MLX-LM 0.31.3 unconditionally calls ``json.loads`` and rejects that valid
    representation. Fixed-capacity mode also tokenizes once for the capacity
    guard and once for inference, so normalization must be idempotent.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text_fragments = [
                fragment["text"]
                for fragment in content
                if (
                    isinstance(fragment, dict)
                    and fragment.get("type") == "text"
                    and "text" in fragment
                )
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            if not arguments:
                continue
            if isinstance(arguments, (str, bytes, bytearray)):
                function["arguments"] = json.loads(arguments)
            elif not isinstance(arguments, (dict, list)):
                raise ValueError(
                    "tool-call function.arguments must be JSON text or an object"
                )


class DeduplicatingToolCallFormatter(mlx_server.ToolCallFormatter):
    """Suppress identical parallel tool calls from one assistant response.

    Repeating the same tool with identical arguments cannot add information:
    parallel tool calls do not observe each other's results. Some local coding
    checkpoints can nevertheless emit the same Read call until their output
    budget is exhausted. Keep the first call and prevent OpenCode from
    executing the redundant copies.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._seen_calls: set[str] = set()
        self.duplicates_suppressed = 0

    def _format(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        fingerprint = json.dumps(
            {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments"),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        if fingerprint in self._seen_calls:
            self.duplicates_suppressed += 1
            if self.duplicates_suppressed == 1:
                logging.warning(
                    "local-llm: suppressing duplicate tool calls in one "
                    "assistant response"
                )
            return None
        self._seen_calls.add(fingerprint)
        return super()._format(tool_call)

    def __call__(self, tool_calls: list[str]) -> list[dict[str, Any]]:
        return [
            formatted
            for formatted in super().__call__(tool_calls)
            if formatted is not None
        ]


def _tool_call_fingerprint(tool_call: dict[str, Any]) -> str:
    """Return one stable identity for string- and object-valued arguments."""
    function = tool_call.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, (str, bytes, bytearray)):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError, json.JSONDecodeError):
            arguments = str(arguments)
    return json.dumps(
        {
            "name": function.get("name"),
            "arguments": arguments,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def consecutive_tool_loop(
    messages: list[dict[str, Any]], *, minimum_repeats: int = 3
) -> tuple[str, int] | None:
    """Detect one identical tool call repeated across consecutive tool turns.

    Tool-result messages between assistant turns are expected and ignored.
    Any user/system turn, textual assistant turn, distinct call, or parallel
    call ends the sequence. This deliberately permits one retry before
    intervening on the third identical call.
    """
    fingerprint: str | None = None
    repeats = 0
    for message in reversed(messages):
        role = message.get("role")
        if role == "tool":
            continue
        if role != "assistant":
            break
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            break
        candidate = _tool_call_fingerprint(calls[0])
        if fingerprint is None:
            fingerprint = candidate
        elif candidate != fingerprint:
            break
        repeats += 1
    if fingerprint is not None and repeats >= minimum_repeats:
        return fingerprint, repeats
    return None


def tool_loop_safe_generate(
    original: Callable[..., Any], *, minimum_repeats: int = 3
) -> Callable[..., Any]:
    """End a cross-turn tool loop with a normal, immediate assistant response."""

    def generate(
        generator: Any,
        request: Any,
        generation_args: Any,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Any:
        loop = consecutive_tool_loop(
            request.messages, minimum_repeats=minimum_repeats
        )
        if loop is None:
            return original(
                generator,
                request,
                generation_args,
                progress_callback=progress_callback,
            )

        fingerprint, repeats = loop
        try:
            name = json.loads(fingerprint).get("name") or "unknown"
        except (TypeError, ValueError, json.JSONDecodeError):
            name = "unknown"
        logging.warning(
            "local-llm: stopped cross-turn tool loop after %d identical %s "
            "calls",
            repeats,
            name,
        )
        record_tool_loop_broken()
        context = mlx_server.GenerationContext(
            has_tool_calling=False,
            has_thinking=False,
            tool_parser=lambda _text, _tools: {},
            sequences={(0,): ""},
            prompt=[],
        )
        response = mlx_server.Response(
            text=(
                f"Tool loop stopped: the identical {name} call was attempted "
                f"{repeats} consecutive times without progress. Review the "
                "previous tool error or change the arguments before retrying."
            ),
            token=0,
            state="normal",
            match=None,
            logprob=0.0,
            finish_reason="stop",
            top_tokens=(),
        )
        return context, iter((response,))

    return generate


def quantized_cache_factory(
    original: Callable[[Any], list[Any]], bits: int, group_size: int
) -> Callable[[Any], list[Any]]:
    def make(model: Any) -> list[Any]:
        caches = original(model)
        for index, cache in enumerate(caches):
            if hasattr(cache, "to_quantized"):
                caches[index] = cache.to_quantized(
                    bits=bits, group_size=group_size
                )
        return caches

    return make


def fixed_capacity_guard(
    original: Callable[[Any, Any], None], max_size: int
) -> Callable[[Any, Any], None]:
    """Reject oversized requests before MLX sends successful HTTP headers."""

    def serve(generator: Any, item: Any) -> None:
        response_queue, request, args = item
        try:
            # MLX-LM normalizes chat messages in place. Isolate the capacity
            # check so the actual inference pass sees the original request.
            prompt, _, _, _ = generator._tokenize(
                generator.model_provider.tokenizer,
                copy.deepcopy(request),
                args,
            )
            requested = len(prompt) + int(args.max_tokens)
            if requested > max_size:
                record_rejection()
                response_queue.put(
                    ValueError(
                        "MLX fixed context capacity exceeded: "
                        f"prompt {len(prompt):,} + output {int(args.max_tokens):,} "
                        f"= {requested:,} tokens, reserved {max_size:,}"
                    )
                )
                return
        except Exception as exc:
            record_error()
            response_queue.put(exc)
            return
        original(generator, item)

    return serve


def adaptive_prefill_stream(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Reduce Metal attention workspace as the retained context gets deeper."""

    def generate(*args: Any, **kwargs: Any):
        prompt = kwargs.get("prompt") or []
        caches = kwargs.get("prompt_cache") or []
        offset = max(
            (int(getattr(cache, "offset", 0) or 0) for cache in caches),
            default=0,
        )
        target_depth = offset + len(prompt)
        configured = int(kwargs.get("prefill_step_size", 2048))
        # The 22.2 GiB Qwopus 5-bit checkpoint is dominated by transient
        # attention workspace rather than its small hybrid KV arena.  Use
        # progressively smaller chunks at depth so a cold/append prefill does
        # not briefly consume all unified memory and take SSH down with it.
        if target_depth >= 220_000:
            selected = min(configured, 128)
        elif target_depth >= 100_000:
            selected = min(configured, 256)
        elif target_depth >= 64_000:
            selected = min(configured, 512)
        else:
            selected = configured
        kwargs["prefill_step_size"] = selected
        record_prefill_plan(step=selected, target_depth=target_depth)
        if selected != configured:
            print(
                "local-llm: adaptive prefill "
                f"{configured} -> {selected} at target depth {target_depth:,}",
                file=sys.stderr,
                flush=True,
            )
        yield from original(*args, **kwargs)

    return generate


def prompt_boundary_checkpoint_stream(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Save a rewind point before mutable assistant/tool suffixes grow."""

    def generate(*args: Any, **kwargs: Any):
        from .mlx_session_cache import checkpoint_prompt_boundary

        prompt_cache = kwargs.get("prompt_cache")
        first = True
        for response in original(*args, **kwargs):
            if first:
                checkpoint_prompt_boundary(prompt_cache, int(response.token))
                first = False
            yield response

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kv-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--kv-mode",
        choices=("native4", "native8", "turbo3", "turbo4"),
        default=None,
    )
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument(
        "--preallocate-kv-size",
        type=int,
        default=0,
        help=(
            "reserve a fixed quantized KV arena for this many tokens; "
            "supported by native4/native8 and experimental Turbo4"
        ),
    )
    parser.add_argument(
        "--session-reuse",
        action="store_true",
        help=(
            "keep the single KV slot across requests and reuse the shared "
            "token prefix (llama.cpp-style checkpoints) so follow-up prompts "
            "skip re-prefilling unchanged context"
        ),
    )
    parser.add_argument(
        "--session-checkpoints",
        type=int,
        default=2,
        help="recurrent-state snapshots kept for rewinding on divergence",
    )
    parser.add_argument(
        "--adaptive-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reduce prefill chunks to 512 above 64K, 256 above 100K, and 128 "
            "above 220K total depth to bound Metal attention workspace"
        ),
    )
    args, remaining = parser.parse_known_args()
    if args.kv_group_size <= 0:
        parser.error("--kv-group-size must be positive")
    if args.preallocate_kv_size < 0:
        parser.error("--preallocate-kv-size cannot be negative")

    mode = args.kv_mode or f"native{args.kv_bits}"
    if args.preallocate_kv_size and mode == "turbo3":
        parser.error(
            "fixed preallocation is not available for Turbo3's fused cache"
        )
    if mode.startswith("turbo"):
        from .mlx_turboquant_adapter import install_turboquant

        mlx_server.make_prompt_cache = install_turboquant(
            mlx_server.make_prompt_cache,
            mode=mode,
            group_size=args.kv_group_size,
            preallocate_kv_size=args.preallocate_kv_size,
        )
    else:
        bits = int(mode.removeprefix("native"))
        if args.preallocate_kv_size:
            from .mlx_preallocated_cache import fixed_native_cache_factory

            mlx_server.make_prompt_cache = fixed_native_cache_factory(
                mlx_server.make_prompt_cache,
                max_size=args.preallocate_kv_size,
                bits=bits,
                group_size=args.kv_group_size,
            )
        else:
            mlx_server.make_prompt_cache = quantized_cache_factory(
                mlx_server.make_prompt_cache,
                bits=bits,
                group_size=args.kv_group_size,
            )
    if args.preallocate_kv_size:
        # A retained cache would hold a reference to the shared arena and
        # could be deep-copied.  Fixed mode deliberately operates one slot.
        remaining = [
            value
            for index, value in enumerate(remaining)
            if not (
                value in ("--prompt-cache-size", "--prompt-cache-bytes")
                or (
                    index > 0
                    and remaining[index - 1]
                    in ("--prompt-cache-size", "--prompt-cache-bytes")
                )
            )
        ]
        remaining.extend(
            ["--prompt-cache-size", "0", "--prompt-cache-bytes", "0"]
        )
        mlx_server.ResponseGenerator._serve_single = fixed_capacity_guard(
            mlx_server.ResponseGenerator._serve_single,
            args.preallocate_kv_size,
        )
    if args.session_reuse:
        # Replace the LRU prompt cache with a single-slot, in-place session
        # cache.  The stock cache deep-copies the whole KV on every reuse,
        # which doubles multi-GiB caches at large context; the session cache
        # keeps one arena and reuses the shared prefix without copying.
        from .mlx_session_cache import SessionPromptCache

        # One end-of-turn snapshot is insufficient for agentic tool traffic:
        # the rendered historical assistant response can diverge from the raw
        # generated suffix. Keep the prompt boundary plus the live turn.
        checkpoints = max(2, args.session_checkpoints)

        def _make_session_cache(_size: int = 0) -> SessionPromptCache:
            return SessionPromptCache(max_checkpoints=checkpoints)

        mlx_server.LRUPromptCache = _make_session_cache
        original_make_prompt_cache = mlx_server.make_prompt_cache

        def _make_bound_prompt_cache(model: Any) -> list[Any]:
            from .mlx_session_cache import bind_new_prompt_cache

            caches = original_make_prompt_cache(model)
            bind_new_prompt_cache(caches)
            return caches

        mlx_server.make_prompt_cache = _make_bound_prompt_cache
    mlx_server.ModelProvider._load = stable_template_loader(
        mlx_server.ModelProvider._load
    )
    mlx_server.process_message_content = process_message_content_idempotent
    mlx_server.ToolCallFormatter = DeduplicatingToolCallFormatter
    mlx_server.ResponseGenerator.generate = tool_loop_safe_generate(
        mlx_server.ResponseGenerator.generate
    )
    stream_generate = mlx_server.stream_generate
    if args.preallocate_kv_size and args.adaptive_prefill:
        stream_generate = adaptive_prefill_stream(stream_generate)
    if args.session_reuse:
        stream_generate = prompt_boundary_checkpoint_stream(stream_generate)
    mlx_server.stream_generate = instrument_stream_generate(stream_generate)
    install_metrics_endpoint(mlx_server.APIHandler)
    sys.argv = [sys.argv[0], *remaining]
    mlx_server.main()


if __name__ == "__main__":
    main()
