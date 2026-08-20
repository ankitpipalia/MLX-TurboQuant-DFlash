"""Compatibility launcher for current official DFlash draft configs.

The June 2026 z-lab Qwen3.6 drafts moved ``block_size`` under
``dflash_config`` and ``rope_theta`` under ``rope_parameters``. dflash-mlx
0.1.10 still requires both values at the top level. Normalize the in-memory
configuration before delegating to the pinned upstream CLI; never modify the
downloaded Hugging Face snapshot.
"""

from __future__ import annotations

import os
from typing import Any


def normalize_draft_config(params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy compatible with dflash-mlx's draft argument parser."""
    data = dict(params)
    dflash_config = dict(data.get("dflash_config") or {})
    rope_parameters = dict(data.get("rope_parameters") or {})
    if data.get("block_size") is None and dflash_config.get("block_size") is not None:
        data["block_size"] = dflash_config["block_size"]
    if data.get("rope_theta") is None and rope_parameters.get("rope_theta") is not None:
        data["rope_theta"] = rope_parameters["rope_theta"]
    return data


def install_config_compatibility() -> None:
    """Patch the upstream parser once, before any model loading occurs."""
    from dflash_mlx.model import DFlashDraftModelArgs

    current = DFlashDraftModelArgs.from_dict
    if getattr(current, "_local_llm_schema_compat", False):
        return
    original = current.__func__

    def from_dict(
        cls: type[DFlashDraftModelArgs], params: dict[str, Any]
    ) -> DFlashDraftModelArgs:
        return original(cls, normalize_draft_config(params))

    from_dict._local_llm_schema_compat = True  # type: ignore[attr-defined]
    DFlashDraftModelArgs.from_dict = classmethod(from_dict)


def install_dflash_turboquant() -> None:
    """Use rotated Turbo4 for DFlash's growing target KV caches.

    DFlash owns its hybrid cache factory and therefore bypasses MLX-LM's
    normal ``make_prompt_cache`` seam.  Replace only its quantized
    full-attention entries; the recurrent rollback caches must remain native.
    TurboQuant's attention registration is equally important because stored
    keys are rotated and queries must be transformed into the same frame.
    """
    mode = os.getenv("LOCAL_LLM_DFLASH_TURBOQUANT", "").strip().lower()
    if not mode:
        return
    if mode != "turbo4":
        raise ValueError(
            "LOCAL_LLM_DFLASH_TURBOQUANT must be empty or 'turbo4'"
        )

    import mlx.core as mx
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
    from mlx_lm.models.cache import QuantizedKVCache
    from mlx_turboquant.kv_cache import TurboQuantKVCache
    from mlx_turboquant.patch import register

    from .mlx_preallocated_cache import FixedQuantizedKVCache

    max_size_raw = os.getenv("LOCAL_LLM_DFLASH_TURBOQUANT_MAX_SIZE", "0")
    try:
        max_size = int(max_size_raw)
    except ValueError:
        raise ValueError(
            "LOCAL_LLM_DFLASH_TURBOQUANT_MAX_SIZE must be an integer"
        ) from None
    if max_size < 0:
        raise ValueError(
            "LOCAL_LLM_DFLASH_TURBOQUANT_MAX_SIZE must be >= 0"
        )

    class DFlashFixedTurbo4KVCache(TurboQuantKVCache):
        def __init__(self) -> None:
            super().__init__(group_size=64, bits=4, qjl=False)
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

    current = QwenGdnTargetOps.make_cache
    if getattr(current, "_local_llm_dflash_turbo4", False):
        return

    fixed_pools: dict[int, dict[int, Any]] = {}

    def make_cache(self: Any, *args: Any, **kwargs: Any) -> list[Any]:
        caches = current(self, *args, **kwargs)
        converted = 0
        pool_key = id(args[0] if args else kwargs.get("target_model"))
        fixed_pool = fixed_pools.setdefault(pool_key, {})
        text_args = None
        if max_size:
            target_model = args[0] if args else kwargs.get("target_model")
            text_args = self.text_wrapper(target_model).args
        for index, cache in enumerate(caches):
            if isinstance(cache, QuantizedKVCache) and not isinstance(
                cache, TurboQuantKVCache
            ):
                if max_size:
                    fixed = fixed_pool.get(index)
                    if fixed is None:
                        fixed = DFlashFixedTurbo4KVCache()
                        fixed.reserve(
                            batch_size=1,
                            n_kv_heads=int(text_args.num_key_value_heads),
                            k_head_dim=int(text_args.head_dim),
                            v_head_dim=int(text_args.head_dim),
                            dtype=mx.bfloat16,
                        )
                        fixed_pool[index] = fixed
                    else:
                        fixed.offset = 0
                    caches[index] = fixed
                else:
                    caches[index] = TurboQuantKVCache(
                        group_size=64,
                        bits=4,
                        qjl=False,
                    )
                converted += 1
        if not converted:
            raise RuntimeError(
                "DFlash Turbo4 requires --quantize-kv-cache and found no "
                "full-attention QuantizedKVCache entries"
            )
        if max_size and any(cache.keys is not None for cache in fixed_pool.values()):
            mx.eval(
                [
                    part
                    for cache in fixed_pool.values()
                    for pair in (cache.keys, cache.values)
                    for part in pair
                ]
            )
        reserved = sum(cache.nbytes for cache in fixed_pool.values())
        suffix = (
            f"; fixed {max_size:,}-token arena "
            f"{reserved / (1024**3):.2f} GiB"
            if max_size
            else ""
        )
        print(
            "local-llm: DFlash Turbo4 converted "
            f"{converted}/{len(caches)} target caches; recurrent rollback "
            f"state preserved{suffix}",
            flush=True,
        )
        return caches

    make_cache._local_llm_dflash_turbo4 = True  # type: ignore[attr-defined]
    make_cache._local_llm_original = current  # type: ignore[attr-defined]
    QwenGdnTargetOps.make_cache = make_cache

    # This wraps MLX-LM attention before the target model module is imported.
    # Its quantized SDPA stays intact; only the matching query rotation is
    # injected for TurboQuant cache instances.
    register()


def main() -> None:
    install_config_compatibility()
    install_dflash_turboquant()
    from dflash_mlx.cli import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()
