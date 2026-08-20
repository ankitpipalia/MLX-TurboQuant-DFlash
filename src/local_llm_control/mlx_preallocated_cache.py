"""Fixed-capacity MLX KV caches for llama.cpp-style memory reservation.

MLX-LM's normal KV caches grow in 256-token chunks.  These caches allocate the
entire configured token capacity once, keep that storage for the lifetime of
the server's single cache slot, and reject requests that exceed the capacity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache
from mlx_turboquant.kv_cache import TurboQuantKVCache


class FixedQuantizedKVCache(QuantizedKVCache):
    """A quantized KV cache backed by one fixed-size Metal allocation."""

    def __init__(self, max_size: int, group_size: int = 64, bits: int = 4):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        super().__init__(group_size=group_size, bits=bits)
        self.max_size = max_size

    def reserve(
        self,
        *,
        batch_size: int,
        n_kv_heads: int,
        k_head_dim: int,
        v_head_dim: int,
        dtype: Any,
    ) -> None:
        """Allocate packed K/V, scales, and biases at full capacity."""

        def allocate(dim: int) -> tuple[Any, ...]:
            # Ask MLX for the exact packed shapes.  This is also correct for
            # non-divisor widths such as 3-bit quantization.
            template = mx.quantize(
                mx.zeros((batch_size, n_kv_heads, 1, dim), dtype=dtype),
                group_size=self.group_size,
                bits=self.bits,
            )
            return tuple(
                mx.zeros(
                    (*part.shape[:-2], self.max_size, part.shape[-1]),
                    dtype=part.dtype,
                )
                for part in template
            )

        self.keys = allocate(k_head_dim)
        self.values = allocate(v_head_dim)
        self.offset = 0

    def update_and_fetch(self, keys: Any, values: Any):
        if self.keys is None:
            self.reserve(
                batch_size=keys.shape[0],
                n_kv_heads=keys.shape[1],
                k_head_dim=keys.shape[-1],
                v_head_dim=values.shape[-1],
                dtype=keys.dtype,
            )

        previous = self.offset
        requested = previous + keys.shape[2]
        if requested > self.max_size:
            raise ValueError(
                "MLX fixed KV capacity exceeded: "
                f"requested {requested:,} tokens, reserved {self.max_size:,}"
            )

        q_keys = mx.quantize(
            keys, group_size=self.group_size, bits=self.bits
        )
        q_values = mx.quantize(
            values, group_size=self.group_size, bits=self.bits
        )
        self.offset = requested
        for index in range(len(self.keys)):
            self.keys[index][..., previous:requested, :] = q_keys[index]
            self.values[index][..., previous:requested, :] = q_values[index]
        return tree_map(
            lambda part: part[..., :requested, :],
            (self.keys, self.values),
        )

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def merge(self):
        # MLX-LM treats a cache as batchable when every entry exposes ``merge``.
        # Raising here makes ``hasattr(cache, "merge")`` false so the server
        # picks its single-sequence path — required because one fixed arena
        # cannot be shared by concurrent sequences, and so the capacity guard
        # and session reuse both run on the sequential code path.
        raise AttributeError("FixedQuantizedKVCache is single-slot; not batchable")

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return tree_reduce(
            lambda total, part: total + part.nbytes,
            (self.keys, self.values),
            0,
        )


class FixedTurbo4KVCache(TurboQuantKVCache):
    """A rotated TurboQuant cache backed by one fixed Metal allocation.

    Borrows the fixed-arena storage from :class:`FixedQuantizedKVCache` while
    keeping TurboQuant's key rotation, so stored keys stay in the frame the
    patched attention rotates queries into.
    """

    def __init__(
        self, max_size: int, group_size: int = 64, bits: int = 4
    ) -> None:
        super().__init__(group_size=group_size, bits=bits, qjl=False)
        if self.qjl:
            # ``update_and_fetch`` below delegates to the fixed-arena
            # implementation, which never builds the QJL sketch/rnorm side
            # buffers.  TurboQuant's attention wrapper then reads
            # ``sketch is None`` and quietly uses the plain estimator, so QJL
            # would look enabled while contributing nothing.  Fail loudly
            # instead of silently degrading.
            raise ValueError(
                "FixedTurbo4KVCache cannot serve QJL: the fixed arena "
                "reserves no sketch buffers"
            )
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size

    reserve = FixedQuantizedKVCache.reserve
    empty = FixedQuantizedKVCache.empty

    @property
    def nbytes(self) -> int:
        return FixedQuantizedKVCache.nbytes.fget(self)

    def update_and_fetch(self, keys: Any, values: Any):
        return FixedQuantizedKVCache.update_and_fetch(
            self, self.rotate_key(keys), values
        )


def model_text_args(model: Any) -> Any:
    language_model = getattr(model, "language_model", model)
    args = getattr(language_model, "args", None)
    if args is None:
        raise RuntimeError("cannot discover MLX text-model dimensions")
    return args


def reset_cache(cache: Any) -> None:
    if isinstance(cache, QuantizedKVCache) and hasattr(cache, "max_size"):
        cache.offset = 0
    elif isinstance(cache, ArraysCache):
        # Preserve the recurrent buffers allocated by the startup warm-up and
        # restore their true initial (zero) state in place.
        for index, value in enumerate(cache.cache):
            if value is not None:
                cache.cache[index] = mx.zeros_like(value)
        cache.left_padding = None
        cache.lengths = None


def reserve_recurrent_state(caches: list[Any], args: Any) -> int:
    """Preallocate Qwen3.5/3.6 GatedDeltaNet state without running the model."""
    required = (
        "linear_conv_kernel_dim",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_num_value_heads",
        "linear_value_head_dim",
    )
    if not all(hasattr(args, name) for name in required):
        return 0

    key_dim = int(args.linear_num_key_heads) * int(args.linear_key_head_dim)
    value_dim = int(args.linear_num_value_heads) * int(
        args.linear_value_head_dim
    )
    conv_dim = key_dim * 2 + value_dim
    arrays = [cache for cache in caches if isinstance(cache, ArraysCache)]
    for cache in arrays:
        cache.cache[0] = mx.zeros(
            (
                1,
                int(args.linear_conv_kernel_dim) - 1,
                conv_dim,
            ),
            dtype=mx.bfloat16,
        )
        cache.cache[1] = mx.zeros(
            (
                1,
                int(args.linear_num_value_heads),
                int(args.linear_value_head_dim),
                int(args.linear_key_head_dim),
            ),
            dtype=mx.float32,
        )
    mx.eval([cache.state for cache in arrays])
    return sum(cache.nbytes for cache in arrays)


def fixed_native_cache_factory(
    original: Callable[[Any], list[Any]],
    *,
    max_size: int,
    bits: int,
    group_size: int,
) -> Callable[[Any], list[Any]]:
    """Return a one-slot cache pool allocated during model loading.

    Fixed caches intentionally do not implement ``merge``.  MLX-LM therefore
    selects its sequential server path, which guarantees that this single slot
    cannot be shared by concurrent requests.
    """

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
                fixed = FixedQuantizedKVCache(
                    max_size=max_size,
                    group_size=group_size,
                    bits=bits,
                )
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
            raise RuntimeError("fixed KV mode found no full-attention caches")

        # Force lazy zeros to become real Metal allocations before the server
        # reports the model as loaded.
        mx.eval(
            [
                part
                for cache in caches
                if isinstance(cache, FixedQuantizedKVCache)
                for pair in (cache.keys, cache.values)
                for part in pair
            ]
        )
        recurrent_reserved = reserve_recurrent_state(caches, args)
        pools[key] = caches
        reserved = sum(
            cache.nbytes
            for cache in caches
            if isinstance(cache, FixedQuantizedKVCache)
        )
        print(
            "local-llm: preallocated "
            f"{converted}/{len(caches)} KV caches for {max_size:,} tokens "
            f"({bits}-bit, {reserved / (1024**3):.2f} GiB); "
            f"recurrent state {recurrent_reserved / (1024**2):.0f} MiB; "
            "prompt-cache retention must be disabled",
            flush=True,
        )
        return caches

    return make
