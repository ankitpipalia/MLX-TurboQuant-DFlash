"""TurboQuant integration that preserves hybrid MLX-LM cache layouts.

The upstream adapter's server helper assumes every model layer owns a normal
KV cache.  Qwen3.5/3.6 instead mixes full attention with recurrent
``ArraysCache`` layers, so replacing the entire list corrupts the model.  This
module converts only genuine ``KVCache`` entries and leaves recurrent state
untouched.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from turboquant_mlx.cache import TurboQuantKVCache as FusedTurboQuantKVCache

from .mlx_preallocated_cache import (
    FixedQuantizedKVCache,
    model_text_args,
    reserve_recurrent_state,
    reset_cache,
)


_PATCHED_QUANT_UPDATE = False


class PrefillSafeFusedTurboQuantCache(FusedTurboQuantKVCache):
    """Use packed Metal attention for decode without breaking chunked prefill.

    The upstream fused cache returns placeholder K/V tensors because its decode
    kernel reads packed storage directly.  Multi-token prefill still uses MLX's
    regular attention path and therefore needs real dequantized tensors.  Toggle
    the storage fast path from the incoming K/V chunk length while keeping the
    cache's public ``fused`` flag enabled for attention dispatch.
    """

    def update_and_fetch(self, keys: Any, values: Any):
        configured = self.fused
        use_fused_decode = configured and keys.shape[2] <= 4
        self.fused = use_fused_decode
        if use_fused_decode:
            # Chunked prefill needs materialized K/V for regular SDPA, but the
            # packed decode kernel reads k_packed/v_packed directly.  Keeping
            # the full dequant buffers after this transition erases the memory
            # saving and makes LRUPromptCache under-report retained bytes.
            self._k_deq_buf = None
            self._v_deq_buf = None
            self._deq_offset = 0
            self._deq_alloc = 0
        try:
            return super().update_and_fetch(keys, values)
        finally:
            self.fused = configured

    def __deepcopy__(self, memo: dict[int, Any]):
        """Copy prompt-cache state while preserving MLX's immutable dtype.

        ``mlx.core.Dtype`` is immutable but not pickleable, so Python's default
        deepcopy fails when MLX-LM reuses an LRU prompt prefix.  Arrays and
        mutable cache metadata are still copied normally.
        """
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        dtype_type = type(mx.float16)
        for name, value in self.__dict__.items():
            copied = value if isinstance(value, dtype_type) else copy.deepcopy(
                value, memo
            )
            setattr(clone, name, copied)
        return clone


def _compatible_quantized_update(self: QuantizedKVCache, keys: Any, values: Any):
    """QuantizedKVCache update that correctly allocates non-divisor bit widths.

    MLX itself supports 3-bit quantization, but MLX-LM 0.31.3 estimates the
    packed dimension using ``32 // bits``.  That is wrong when the bit width
    does not divide 32 (256 dimensions become 25 words instead of the 24 words
    returned by ``mx.quantize``).  Allocate from the actual quantized tensor
    shapes instead.
    """
    num_steps = keys.shape[2]
    previous = self.offset
    q_keys = mx.quantize(
        keys, group_size=self.group_size, bits=self.bits
    )
    q_values = mx.quantize(
        values, group_size=self.group_size, bits=self.bits
    )

    if self.keys is None or previous + num_steps > self.keys[0].shape[-2]:
        new_steps = (self.step + num_steps - 1) // self.step * self.step

        def initialize(parts: tuple[Any, ...]) -> tuple[Any, ...]:
            return tuple(
                mx.zeros(
                    (*part.shape[:-2], new_steps, part.shape[-1]),
                    dtype=part.dtype,
                )
                for part in parts
            )

        def expand(part: Any) -> Any:
            if previous % self.step:
                part = part[..., :previous, :]
            extension = mx.zeros(
                (*part.shape[:-2], new_steps, part.shape[-1]),
                dtype=part.dtype,
            )
            return mx.concatenate([part, extension], axis=-2)

        if self.keys is None:
            self.keys, self.values = initialize(q_keys), initialize(q_values)
        else:
            self.keys, self.values = tree_map(
                expand, (self.keys, self.values)
            )

    self.offset += num_steps
    for index in range(len(self.keys)):
        self.keys[index][..., previous : self.offset, :] = q_keys[index]
        self.values[index][..., previous : self.offset, :] = q_values[index]
    return tree_map(
        lambda part: part[..., : self.offset, :],
        (self.keys, self.values),
    )


def patch_three_bit_allocation() -> None:
    """Install the MLX-LM 3-bit packed-shape correction once."""
    global _PATCHED_QUANT_UPDATE
    if _PATCHED_QUANT_UPDATE:
        return
    QuantizedKVCache.update_and_fetch = _compatible_quantized_update
    _PATCHED_QUANT_UPDATE = True


def turboquant_cache_factory(
    original: Callable[[Any], list[Any]],
    *,
    bits: int,
    group_size: int,
    qjl: bool,
) -> Callable[[Any], list[Any]]:
    """Wrap MLX-LM's factory and replace only full-attention KV caches."""
    from mlx_turboquant.kv_cache import TurboQuantKVCache

    def make(model: Any) -> list[Any]:
        caches = original(model)
        converted = 0
        for index, cache in enumerate(caches):
            if isinstance(cache, KVCache):
                caches[index] = TurboQuantKVCache(
                    group_size=group_size,
                    bits=bits,
                    qjl=qjl,
                )
                converted += 1
        if not converted:
            raise RuntimeError(
                "TurboQuant found no compatible full-attention KV caches"
            )
        print(
            "local-llm: TurboQuant converted "
            f"{converted}/{len(caches)} caches "
            f"(bits={bits}, qjl={qjl}); recurrent caches preserved",
            flush=True,
        )
        return caches

    return make


def fixed_turbo4_cache_factory(
    original: Callable[[Any], list[Any]],
    *,
    max_size: int,
    group_size: int,
) -> Callable[[Any], list[Any]]:
    """Preallocate one hybrid-safe rotated 4-bit cache slot."""
    from mlx_turboquant.kv_cache import TurboQuantKVCache

    class FixedTurbo4KVCache(TurboQuantKVCache):
        def __init__(self) -> None:
            super().__init__(group_size=group_size, bits=4, qjl=False)
            self.max_size = max_size

        reserve = FixedQuantizedKVCache.reserve
        empty = FixedQuantizedKVCache.empty

        @property
        def merge(self):
            # A fixed arena is one physical cache slot.  Exposing the inherited
            # QuantizedKVCache.merge attribute makes MLX-LM classify the model
            # as batchable and bypass its sequential capacity/session-reuse
            # path, even when decode/prompt concurrency are both one.
            raise AttributeError(
                "FixedTurbo4KVCache is single-slot; not batchable"
            )

        @property
        def nbytes(self) -> int:
            return FixedQuantizedKVCache.nbytes.fget(self)

        def update_and_fetch(self, keys: Any, values: Any):
            return FixedQuantizedKVCache.update_and_fetch(
                self, self.rotate_key(keys), values
            )

    pools: dict[int, list[Any]] = {}

    def make(model: Any) -> list[Any]:
        key = id(model)
        if key in pools:
            caches = pools[key]
            for cache in caches:
                reset_cache(cache)
            return caches

        args = model_text_args(model)
        caches = original(model)
        converted = 0
        for index, cache in enumerate(caches):
            if isinstance(cache, KVCache):
                fixed = FixedTurbo4KVCache()
                fixed.reserve(
                    batch_size=1,
                    n_kv_heads=int(args.num_key_value_heads),
                    k_head_dim=int(args.head_dim),
                    v_head_dim=int(args.head_dim),
                    dtype=mx.bfloat16,
                )
                caches[index] = fixed
                converted += 1
        if not converted:
            raise RuntimeError("fixed Turbo4 found no full-attention caches")

        mx.eval(
            [
                part
                for cache in caches
                if isinstance(cache, FixedTurbo4KVCache)
                for pair in (cache.keys, cache.values)
                for part in pair
            ]
        )
        recurrent_reserved = reserve_recurrent_state(caches, args)
        pools[key] = caches
        reserved = sum(
            cache.nbytes
            for cache in caches
            if isinstance(cache, FixedTurbo4KVCache)
        )
        print(
            "local-llm: preallocated rotated Turbo4 "
            f"{converted}/{len(caches)} KV caches for {max_size:,} tokens "
            f"({reserved / (1024**3):.2f} GiB); "
            f"recurrent state {recurrent_reserved / (1024**2):.0f} MiB; "
            "prompt-cache retention disabled",
            flush=True,
        )
        return caches

    return make


def fused_turbo3_cache_factory(
    original: Callable[[Any], list[Any]],
) -> Callable[[Any], list[Any]]:
    """Build the fused 3-bit PolarQuant cache for full-attention layers only."""

    def make(model: Any) -> list[Any]:
        caches = original(model)
        converted = 0
        for index, cache in enumerate(caches):
            if isinstance(cache, KVCache):
                caches[index] = PrefillSafeFusedTurboQuantCache(
                    bits=3,
                    fused=True,
                )
                converted += 1
        if not converted:
            raise RuntimeError(
                "TurboQuant found no compatible full-attention KV caches"
            )
        print(
            "local-llm: fused PolarQuant converted "
            f"{converted}/{len(caches)} caches (bits=3); "
            "recurrent caches preserved",
            flush=True,
        )
        return caches

    return make


def install_turboquant(
    original: Callable[[Any], list[Any]],
    mode: str,
    group_size: int,
    preallocate_kv_size: int = 0,
) -> Callable[[Any], list[Any]]:
    """Install attention dispatch and return a hybrid-safe cache factory."""
    if mode not in {"turbo3", "turbo4"}:
        raise ValueError(f"unsupported TurboQuant mode: {mode}")
    if mode == "turbo3":
        if preallocate_kv_size:
            raise ValueError("fixed preallocation is not implemented for Turbo3")
        # The fused Lloyd-Max/PolarQuant implementation is materially faster
        # than the QJL prototype on this M1 Max and stores genuinely packed
        # 3-bit K/V.  Its attention patch includes custom Metal decode kernels.
        from turboquant_mlx.patch import apply_patch

        apply_patch()
        return fused_turbo3_cache_factory(original)

    patch_three_bit_allocation()
    from mlx_turboquant.patch import register

    register()
    if preallocate_kv_size:
        return fixed_turbo4_cache_factory(
            original,
            max_size=preallocate_kv_size,
            group_size=group_size,
        )
    return turboquant_cache_factory(
        original,
        bits=4,
        group_size=group_size,
        qjl=False,
    )
