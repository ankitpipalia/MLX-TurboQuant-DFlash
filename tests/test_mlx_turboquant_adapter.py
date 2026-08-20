import copy

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache

from local_llm_control.mlx_turboquant_adapter import (
    fused_turbo3_cache_factory,
    patch_three_bit_allocation,
    turboquant_cache_factory,
)


def test_factory_preserves_recurrent_caches() -> None:
    recurrent = ArraysCache(size=2)
    make = turboquant_cache_factory(
        lambda _model: [recurrent, KVCache(), recurrent, KVCache()],
        bits=4,
        group_size=64,
        qjl=False,
    )

    caches = make(object())

    assert caches[0] is recurrent
    assert caches[2] is recurrent
    assert type(caches[1]).__name__ == "TurboQuantKVCache"
    assert type(caches[3]).__name__ == "TurboQuantKVCache"
    assert caches[1].bits == 4
    assert caches[1].qjl is False


def test_three_bit_cache_allocation_matches_mlx_packing() -> None:
    patch_three_bit_allocation()
    cache = QuantizedKVCache(group_size=64, bits=3)
    keys = mx.random.normal((1, 2, 19, 256))
    values = mx.random.normal((1, 2, 19, 256))

    quantized_keys, quantized_values = cache.update_and_fetch(keys, values)
    mx.eval(quantized_keys, quantized_values)

    expected = mx.quantize(keys, group_size=64, bits=3)
    assert quantized_keys[0].shape[-1] == expected[0].shape[-1] == 24
    assert cache.offset == 19


def test_fused_turbo3_factory_preserves_recurrent_caches() -> None:
    recurrent = ArraysCache(size=2)
    make = fused_turbo3_cache_factory(
        lambda _model: [recurrent, KVCache(), recurrent]
    )

    caches = make(object())

    assert caches[0] is recurrent
    assert caches[2] is recurrent
    assert type(caches[1]).__name__ == "PrefillSafeFusedTurboQuantCache"
    assert caches[1].quant_bits == 3
    assert caches[1].fused is True


def test_fused_turbo3_cache_can_be_deepcopied_after_initialization() -> None:
    from local_llm_control.mlx_turboquant_adapter import (
        PrefillSafeFusedTurboQuantCache,
    )

    cache = PrefillSafeFusedTurboQuantCache(bits=3, fused=True)
    keys = mx.random.normal((1, 2, 8, 256)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)

    clone = copy.deepcopy(cache)

    assert clone is not cache
    assert clone.offset == cache.offset == 8
    assert clone._dtype is cache._dtype
    assert clone.k_packed is not cache.k_packed


def test_fused_turbo3_releases_prefill_dequant_buffers_on_decode() -> None:
    from local_llm_control.mlx_turboquant_adapter import (
        PrefillSafeFusedTurboQuantCache,
    )

    cache = PrefillSafeFusedTurboQuantCache(bits=3, fused=True)
    prefill = mx.random.normal((1, 2, 16, 256)).astype(mx.bfloat16)
    cache.update_and_fetch(prefill, prefill)
    assert cache._k_deq_buf is not None
    assert cache._v_deq_buf is not None

    decode = mx.random.normal((1, 2, 1, 256)).astype(mx.bfloat16)
    cache.update_and_fetch(decode, decode)

    assert cache._k_deq_buf is None
    assert cache._v_deq_buf is None
