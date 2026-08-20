"""TurboQuant stores rotated keys, so attention must rotate the query too.

DFlash's custom Qwen attention helpers capture ``scaled_dot_product_attention``
by value at import time, and TurboQuant's ``patch_attention`` only repoints
modules under ``mlx_lm.models.*`` -- so ``dflash_mlx.engine.gqa_sdpa`` keeps an
unrotating reference forever. That reference is real, and it is unreachable:
``_install_full_attention_gqa_hook`` refuses to route any ``QuantizedKVCache``
into it and defers to the model's own attention, which TurboQuant *does* patch.

The safety of the whole Turbo4-under-DFlash path rests on that one guard, so
assert it by execution rather than by reading upstream source. If an upstream
bump ever lets a quantized cache reach ``grouped_gqa_sdpa``, queries would be
matched unrotated against rotated keys -- silently wrong output, no crash.
"""

from __future__ import annotations

import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen3_5
import pytest


def _tiny_hybrid_model():
    """A 4-layer hybrid Qwen3.5: 3 GatedDeltaNet layers plus 1 full attention."""
    args = qwen3_5.TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,  # Qwen3.8's real head_dim; exercises the same branches
        vocab_size=64,
        full_attention_interval=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        max_position_embeddings=4096,
    )
    model = qwen3_5.Model(
        qwen3_5.ModelArgs(model_type="qwen3_5", text_config=vars(args))
    )
    mx.eval(model.parameters())
    return model


def test_turboquant_queries_are_rotated_on_dflash_attention_path(monkeypatch):
    import dflash_mlx.engine.gqa_sdpa as dflash_gqa
    import dflash_mlx.engine.target_qwen_gdn as target_qwen_gdn
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
    from mlx_turboquant.kv_cache import TurboQuantKVCache

    from local_llm_control.dflash_compat import install_dflash_turboquant

    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT", "turbo4")
    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT_MAX_SIZE", "0")

    # Reinstall from the pristine factory so this test is order-independent:
    # the bridge patches the class in place and refuses to double-wrap.
    installed = QwenGdnTargetOps.make_cache
    monkeypatch.setattr(
        QwenGdnTargetOps,
        "make_cache",
        getattr(installed, "_local_llm_original", installed),
    )

    model = _tiny_hybrid_model()
    ops = QwenGdnTargetOps()
    assert ops.family(model) == "hybrid_gdn"

    # Install the real DFlash attention hook, exactly as the server does.
    ops.install_speculative_hooks(model)
    attention = [
        layer.self_attn
        for layer in ops.text_model(model).layers
        if hasattr(layer, "self_attn")
    ]
    assert len(attention) == 1
    assert (
        type(attention[0]).__call__.__qualname__
        == "_install_full_attention_gqa_hook.<locals>.attention_call"
    ), "DFlash's custom attention path must be active for this test to mean anything"

    install_dflash_turboquant()
    caches = ops.make_cache(
        model, enable_speculative_linear_cache=True, quantize_kv_cache=True
    )
    turbo = [cache for cache in caches if isinstance(cache, TurboQuantKVCache)]
    assert len(turbo) == 1, "exactly the one full-attention layer is converted"

    # The stale, unrotating reference genuinely exists -- that is the premise.
    assert not getattr(
        dflash_gqa.scaled_dot_product_attention, "_turboquant_wrapped", False
    ), "premise changed: dflash's SDPA is now patched, revisit this test"

    counts = {"rotate_query": 0, "quantized_into_stale_path": 0}

    original_rotate = TurboQuantKVCache.rotate_query

    def counting_rotate(self, queries):
        counts["rotate_query"] += 1
        return original_rotate(self, queries)

    monkeypatch.setattr(TurboQuantKVCache, "rotate_query", counting_rotate)

    original_grouped = dflash_gqa.grouped_gqa_sdpa

    def counting_grouped(queries, keys, values, cache=None, **kwargs):
        if cache is not None and hasattr(cache, "bits"):
            counts["quantized_into_stale_path"] += 1
        return original_grouped(queries, keys, values, cache=cache, **kwargs)

    monkeypatch.setattr(dflash_gqa, "grouped_gqa_sdpa", counting_grouped)
    monkeypatch.setattr(target_qwen_gdn, "grouped_gqa_sdpa", counting_grouped)

    mx.eval(model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=caches))
    mx.eval(model(mx.array([[9]]), cache=caches))

    assert turbo[0].offset == 9, "the Turbo4 cache must actually be in use"
    assert counts["quantized_into_stale_path"] == 0, (
        "a quantized cache reached grouped_gqa_sdpa, whose captured SDPA never "
        "rotates the query: attention would compare Q against rotated keys"
    )
    assert counts["rotate_query"] == 2, (
        "expected one query rotation per forward pass through the single "
        f"full-attention layer, saw {counts['rotate_query']}"
    )
