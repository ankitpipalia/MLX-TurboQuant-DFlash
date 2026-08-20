"""Run mlx-vlm with a text-first, lazy vision-weight load policy.

The stock server eagerly materializes every VLM parameter before reporting
ready.  Coding agents do not send images, so that needlessly makes the vision
tower resident.  This wrapper leaves non-language weights lazy while eagerly
evaluating the complete language model.  Multimodal requests remain supported;
their vision weights materialize on first use and therefore need extra memory.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

import mlx.core as mx


_ONYX_GENERATION_PROMPT = (
    "{%- if add_generation_prompt -%}"
    "{{- '<|start|>assistant' -}}"
    "{%- endif -%}"
)
_ONYX_REASONING_AWARE_GENERATION_PROMPT = (
    "{%- if add_generation_prompt -%}"
    "{%- if enable_thinking or tools -%}"
    "{{- '<|start|>assistant' -}}"
    "{%- else -%}"
    "{{- '<|start|>assistant to=user<|message|>' -}}"
    "{%- endif -%}"
    "{%- endif -%}"
)


def patch_onyx_generation_prompt(processor: Any) -> bool:
    """Make Muse's Onyx template honor the normalized thinking switch.

    The published template accepts ``enable_thinking`` but its final
    generation prompt ignores it, so the model enters ``to=self`` reasoning
    even for an explicit OpenAI ``reasoning_effort: none`` request. Prefixing
    the user recipient is the native Onyx protocol for a direct answer. Tool
    requests retain the self channel because Onyx uses it to select and emit
    ATEM calls before returning to the user.
    """
    changed = False
    targets = [processor]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer is not processor:
        targets.append(tokenizer)
    for target in targets:
        template = getattr(target, "chat_template", None)
        if not isinstance(template, str):
            continue
        if _ONYX_REASONING_AWARE_GENERATION_PROMPT in template:
            changed = True
            continue
        if (
            _ONYX_GENERATION_PROMPT not in template
            or "<atem:function_calls>" not in template
        ):
            continue
        target.chat_template = template.replace(
            _ONYX_GENERATION_PROMPT,
            _ONYX_REASONING_AWARE_GENERATION_PROMPT,
            1,
        )
        changed = True
    return changed


def load_text_first(
    original_load: Callable[..., tuple[Any, Any]],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load a VLM lazily, then make only its language model resident."""
    kwargs["lazy"] = True
    # EigenLabs' benchmark backbone intentionally publishes only the language
    # tensors while retaining Qwen's multimodal config. Its files are
    # separately SHA-256 pinned by our preparation script; scope the relaxed
    # load to an explicit profile flag instead of weakening every VLM load.
    if os.getenv("LOCAL_LLM_ALLOW_MISSING_VISION") == "1":
        kwargs["strict"] = False
    model, processor = original_load(*args, **kwargs)
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise TypeError("text-first mode requires a VLM with language_model")
    patch_onyx_generation_prompt(processor)
    mx.eval(language_model.parameters())
    return model, processor


def install_text_first_loader() -> None:
    """Patch the server generation module once, before Uvicorn starts."""
    from mlx_vlm.server import generation

    if getattr(generation.load, "_local_llm_text_first", False):
        return
    original_load = generation.load

    def patched_load(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        return load_text_first(original_load, *args, **kwargs)

    patched_load._local_llm_text_first = True  # type: ignore[attr-defined]
    generation.load = patched_load


def install_reasoning_strength_bridge() -> None:
    """Map the OpenAI reasoning field to Muse/Onyx's template variable.

    mlx-vlm normalizes ``reasoning_effort`` but its generic template kwargs do
    not expose the ``reasoning_strength`` name used by Muse's Onyx template.
    Without this bridge every request silently renders ``Reasoning strength:
    high`` -- including requests that explicitly send ``none``.
    """
    from mlx_vlm.server.generation import GenerationArguments

    original = GenerationArguments.to_template_kwargs
    if getattr(original, "_local_llm_reasoning_strength", False):
        return

    def with_reasoning_strength(self: Any) -> dict:
        kwargs = original(self)
        if self.reasoning_effort is not None:
            kwargs["reasoning_strength"] = self.reasoning_effort
        return kwargs

    with_reasoning_strength._local_llm_reasoning_strength = True  # type: ignore[attr-defined]
    with_reasoning_strength._local_llm_original = original  # type: ignore[attr-defined]
    GenerationArguments.to_template_kwargs = with_reasoning_strength


QWEN_REASONING_LEVELS = ("xhigh", "medium", "low")
_QWEN_REASONING_ALIASES = {
    "minimal": "low",
    "none": "low",
    "off": "low",
    "high": "xhigh",
    "max": "xhigh",
    "maximum": "xhigh",
}


def normalize_qwen_reasoning_effort(effort: Any) -> Any:
    """Translate OpenAI reasoning vocabulary into Qwen3.8's accepted set.

    Qwen3.8's chat template accepts only ``xhigh``, ``medium`` and ``low``, and
    calls ``raise_exception`` on anything else -- so a client sending the
    OpenAI-standard ``high`` or ``minimal`` fails the entire request rather
    than degrading. Map the well-known aliases; pass anything unrecognized
    through so a genuine typo still surfaces instead of being silently
    reinterpreted as a valid level.
    """
    if effort is None:
        return None
    lowered = str(effort).strip().lower()
    if lowered in QWEN_REASONING_LEVELS:
        return lowered
    return _QWEN_REASONING_ALIASES.get(lowered, effort)


def install_qwen_reasoning_normalizer() -> None:
    """Alias OpenAI reasoning levels, and optionally pick a default level.

    Opt-in via ``LOCAL_LLM_QWEN_REASONING_ALIASES=1`` so the Muse/Onyx
    profiles, whose template reads a different vocabulary, are unaffected.
    ``LOCAL_LLM_QWEN_REASONING_DEFAULT`` overrides the template's own default
    of ``xhigh``, which otherwise spends most of the output budget thinking on
    this hardware.
    """
    if os.getenv("LOCAL_LLM_QWEN_REASONING_ALIASES", "").strip() != "1":
        return
    default = (
        os.getenv("LOCAL_LLM_QWEN_REASONING_DEFAULT", "").strip().lower() or None
    )
    if default is not None and default not in QWEN_REASONING_LEVELS:
        raise ValueError(
            "LOCAL_LLM_QWEN_REASONING_DEFAULT must be one of "
            f"{', '.join(QWEN_REASONING_LEVELS)}"
        )

    from mlx_vlm.server.generation import GenerationArguments

    original = GenerationArguments.to_template_kwargs
    if getattr(original, "_local_llm_qwen_reasoning", False):
        return

    def with_normalized_reasoning(self: Any) -> dict:
        kwargs = original(self)
        effort = normalize_qwen_reasoning_effort(kwargs.get("reasoning_effort"))
        if effort is None:
            effort = default
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        return kwargs

    with_normalized_reasoning._local_llm_qwen_reasoning = True  # type: ignore[attr-defined]
    with_normalized_reasoning._local_llm_original = original  # type: ignore[attr-defined]
    GenerationArguments.to_template_kwargs = with_normalized_reasoning


def install_compressed_turboquant_apc_bridge() -> None:
    """Keep exact-prefix snapshots compressed when TurboQuant KV is active.

    Upstream APC deliberately normalizes any cache exposing
    ``dequantize_for_apc`` to a float ``KVCache``. That is conservative, but a
    Qwen3.8 32K snapshot expands to several GiB and makes long-context reuse
    impossible on a 32 GiB Mac. TurboQuant's state tree is independently
    copyable, so retain that tree and rebuild the one-row batch cache directly.
    """
    from mlx_vlm import apc_adapters
    from mlx_vlm.turboquant import (
        BatchTurboQuantKVCache,
        TurboQuantKVCache,
        _map_state,
    )

    original_clone = apc_adapters.clone_cache_entry
    if getattr(original_clone, "_local_llm_turboquant_compressed", False):
        return
    original_merge = apc_adapters.merge_cache_entries

    def clone_cache_entry(c: Any, *, min_capacity_tokens: Any, eval_targets: list):
        if not isinstance(c, TurboQuantKVCache):
            return original_clone(
                c,
                min_capacity_tokens=min_capacity_tokens,
                eval_targets=eval_targets,
            )
        out = TurboQuantKVCache(
            bits=c.bits,
            seed=c.seed,
            key_bits=c.key_bits,
            value_bits=c.value_bits,
        )
        out.key_codec = c.key_codec
        out.value_codec = c.value_codec
        def copy_array(a: Any, _ndim: int):
            return mx.contiguous(mx.array(a, dtype=a.dtype))

        # _map_state preserves TurboQuant's NamedTuple state types; the
        # generic APC tree copier turns them into plain tuples, which the
        # codec cannot subsequently slice or decode.
        out.keys = _map_state(c.keys, copy_array) if c.keys is not None else None
        out.values = _map_state(c.values, copy_array) if c.values is not None else None
        out.offset = int(c.offset)
        apc_adapters._eval_tree(out.keys, eval_targets)
        apc_adapters._eval_tree(out.values, eval_targets)
        return out

    def merge_cache_entries(entries: Any, prefix_lens: Any):
        if not entries or not all(isinstance(c, TurboQuantKVCache) for c in entries):
            return original_merge(entries, prefix_lens)
        merged = None
        for c in entries:
            row = BatchTurboQuantKVCache(
                [0],
                bits=c.bits,
                seed=c.seed,
                key_bits=c.key_bits,
                value_bits=c.value_bits,
            )
            row.key_codec = c.key_codec
            row.value_codec = c.value_codec
            row.keys = c.keys
            row.values = c.values
            row._idx = int(c.offset)
            row.offset = mx.array([int(c.offset)])
            if merged is None:
                merged = row
            else:
                merged.extend(row)
        return merged

    clone_cache_entry._local_llm_turboquant_compressed = True
    clone_cache_entry._local_llm_original = original_clone
    merge_cache_entries._local_llm_turboquant_compressed = True
    merge_cache_entries._local_llm_original = original_merge
    apc_adapters.clone_cache_entry = clone_cache_entry
    apc_adapters.merge_cache_entries = merge_cache_entries


def main() -> None:
    install_text_first_loader()
    install_reasoning_strength_bridge()
    install_qwen_reasoning_normalizer()
    install_compressed_turboquant_apc_bridge()
    from mlx_vlm.server.cli import main as server_main

    server_main()


if __name__ == "__main__":
    main()
