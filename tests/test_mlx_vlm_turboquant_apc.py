import mlx.core as mx

from local_llm_control.mlx_vlm_text_server import (
    install_compressed_turboquant_apc_bridge,
)


def test_turboquant_apc_snapshot_stays_compressed() -> None:
    from mlx_vlm import apc_adapters
    from mlx_vlm.turboquant import BatchTurboQuantKVCache, TurboQuantKVCache

    install_compressed_turboquant_apc_bridge()
    cache = TurboQuantKVCache(bits=4)
    keys = mx.random.normal((1, 2, 16, 32))
    values = mx.random.normal((1, 2, 16, 32))
    cache.update_and_fetch(keys, values)

    targets = []
    snapshot = apc_adapters.clone_cache_entry(
        cache, min_capacity_tokens=64, eval_targets=targets
    )
    mx.eval(targets)

    assert isinstance(snapshot, TurboQuantKVCache)
    assert snapshot.offset == 16
    assert snapshot.nbytes < keys.nbytes + values.nbytes

    warm = apc_adapters.merge_cache_entries([snapshot], [16])
    assert isinstance(warm, BatchTurboQuantKVCache)
    assert warm._idx == 16
    assert warm.offset.tolist() == [16]
