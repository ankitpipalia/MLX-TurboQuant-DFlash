"""Turbo4 prefix snapshots must own their data and reuse the pooled arena.

Two invariants here are worth more than the rest of the file combined. A
snapshot must survive the arena being rewound and overwritten -- upstream
publishes end-of-request snapshots with ``adopt_cache_arrays=True``, which
stores live references and skips ``mx.eval``, and a fixed arena writes in place
rather than growing. And a restore must fill the pooled arena instead of
building a fresh cache, or every cache hit abandons a multi-GiB allocation.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from local_llm_control.dflash_turbo4_snapshot import (
    capture_turbo4_state,
    restore_turbo4_state,
    turbo4_format_version,
)
from local_llm_control.mlx_preallocated_cache import FixedTurbo4KVCache


def _arena(max_size: int = 64):
    """A fixed Turbo4-shaped arena: rotated keys are irrelevant to the codec."""
    cache = FixedTurbo4KVCache(max_size=max_size, group_size=64, bits=4)
    cache.reserve(
        batch_size=1,
        n_kv_heads=2,
        k_head_dim=64,
        v_head_dim=64,
        dtype=mx.bfloat16,
    )
    return cache


def _fill(cache, tokens: int, value: float):
    keys = mx.full((1, 2, tokens, 64), value, dtype=mx.bfloat16)
    values = mx.full((1, 2, tokens, 64), -value, dtype=mx.bfloat16)
    cache.update_and_fetch(keys, values)
    mx.eval([*cache.keys, *cache.values])


def test_snapshot_survives_the_arena_being_rewound_and_overwritten():
    """The adoption hazard: a captured prefix must not alias the arena."""
    cache = _arena()
    _fill(cache, 8, 1.0)
    state = capture_turbo4_state(cache)
    captured = [mx.array(part) for part in state[0]]
    mx.eval(captured)

    # Exactly what the next request does to a pooled arena.
    cache.offset = 0
    _fill(cache, 8, -7.0)

    for before, after in zip(captured, state[0]):
        assert mx.array_equal(before, after), (
            "captured prefix changed when the arena was overwritten: the "
            "snapshot is aliasing live arena storage"
        )
    assert state[2] == 8


def test_restore_fills_the_pooled_arena_instead_of_allocating():
    cache = _arena()
    _fill(cache, 8, 1.0)
    state = capture_turbo4_state(cache)
    identity = [id(part) for part in cache.keys]

    cache.offset = 0
    _fill(cache, 8, -7.0)

    restored = restore_turbo4_state(cache, state)

    assert restored is cache, "restore must return the pooled arena"
    assert [id(part) for part in cache.keys] == identity, (
        "restore reallocated the arena instead of writing into it"
    )
    assert cache.offset == 8
    # And the arena now holds the snapshot's bytes, not the overwrite's.
    for part, expected in zip(cache.keys, state[0]):
        assert mx.array_equal(part[..., :8, :], expected)


def test_round_trip_then_append_continues_from_the_restored_offset():
    cache = _arena()
    _fill(cache, 8, 1.0)
    state = capture_turbo4_state(cache)

    fresh = _arena()
    restore_turbo4_state(fresh, state)
    assert fresh.offset == 8
    _fill(fresh, 3, 2.0)
    assert fresh.offset == 11, "append must continue from the restored prefix"


def test_capture_copies_only_the_live_prefix_not_the_whole_arena():
    cache = _arena(max_size=4096)
    _fill(cache, 16, 1.0)
    state = capture_turbo4_state(cache)

    for part in (*state[0], *state[1]):
        assert int(part.shape[2]) == 16, (
            "snapshot must hold the live prefix only, not the reserved arena"
        )
    snapshot_bytes = sum(int(p.nbytes) for p in (*state[0], *state[1]))
    assert snapshot_bytes < int(cache.nbytes) / 100


def test_restore_rejects_mismatched_geometry():
    small = _arena()
    _fill(small, 8, 1.0)
    state = capture_turbo4_state(small)

    wide = FixedTurbo4KVCache(max_size=64, group_size=64, bits=4)
    wide.reserve(
        batch_size=1,
        n_kv_heads=4,  # different head count
        k_head_dim=64,
        v_head_dim=64,
        dtype=mx.bfloat16,
    )
    with pytest.raises(ValueError, match="incompatible with arena shape"):
        restore_turbo4_state(wide, state)


def test_restore_rejects_a_prefix_longer_than_the_arena():
    cache = _arena(max_size=64)
    _fill(cache, 32, 1.0)
    state = capture_turbo4_state(cache)

    tiny = _arena(max_size=16)
    with pytest.raises(ValueError, match="exceeds the reserved"):
        restore_turbo4_state(tiny, state)


def test_format_version_separates_every_meaningful_cache_format():
    """A rotated Turbo4 prefix restored under another format is silently wrong,
    so each of these must land in a different prefix-cache namespace."""
    base = dict(bits=4, group_size=64, seed=0xC0FFEE, qjl=False)
    versions = {
        turbo4_format_version(**base),
        turbo4_format_version(**{**base, "bits": 8}),
        turbo4_format_version(**{**base, "group_size": 32}),
        turbo4_format_version(**{**base, "seed": 0xBADBAD}),
        turbo4_format_version(**{**base, "qjl": True}),
    }
    assert len(versions) == 5, "format identities collided"

    # Stable across calls, and clear of upstream's small integers (currently 3).
    assert turbo4_format_version(**base) == turbo4_format_version(**base)
    assert turbo4_format_version(**base) > 1000


def _install_isolated(monkeypatch, format_version: int = 0x54ABCDEF):
    """Install the codec patches with automatic teardown."""
    from dflash_mlx.cache import codecs, prefix_l2
    from dflash_mlx.server import prefix_cache_manager

    import local_llm_control.dflash_turbo4_snapshot as snap

    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT_SNAPSHOTS", "1")
    # Snapshot every attribute the installer rebinds so pytest restores them.
    monkeypatch.setattr(codecs, "serialize_target_cache", codecs.serialize_target_cache)
    monkeypatch.setattr(codecs, "hydrate_target_cache", codecs.hydrate_target_cache)
    monkeypatch.setattr(
        prefix_cache_manager, "build_prefix_key", prefix_cache_manager.build_prefix_key
    )
    monkeypatch.setattr(prefix_l2, "_serialize", prefix_l2._serialize)
    monkeypatch.setattr(snap, "_format_version_override", format_version)

    snap.install_turbo4_prefix_snapshots()
    return codecs, prefix_l2, prefix_cache_manager


def _mixed_cache(arena):
    """A hybrid layer list: recurrent linear layers plus one Turbo4 FA layer."""
    from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

    recurrent = RecurrentRollbackCache(size=2, conv_kernel_size=4)
    recurrent.cache = [mx.zeros((1, 2, 4)), mx.zeros((1, 2, 4))]
    return [recurrent, arena]


def test_installed_codec_serializes_and_hydrates_a_hybrid_cache(monkeypatch):
    codecs, _, _ = _install_isolated(monkeypatch)

    arena = _arena()
    _fill(arena, 8, 1.0)
    caches = _mixed_cache(arena)

    fa_states, gdn_states = codecs.serialize_target_cache(caches)
    assert fa_states[0] is None and gdn_states[0] is not None, "linear layer"
    assert fa_states[1] is not None and gdn_states[1] is None, "attention layer"
    assert len(fa_states[1]) == 3, "arity 3 keeps upstream's offset checks valid"
    assert isinstance(fa_states[1][0], tuple), "packed triple, not one array"

    snapshot = type(
        "Snap", (), {"fa_states": fa_states, "gdn_states": gdn_states, "prefix_len": 8}
    )()

    arena.offset = 0
    _fill(arena, 8, -7.0)
    restored = codecs.hydrate_target_cache(snapshot, caches)

    assert restored[1] is arena, "hydrate must reuse the pooled arena"
    assert restored[1].offset == 8
    for part, expected in zip(restored[1].keys, fa_states[1][0]):
        assert mx.array_equal(part[..., :8, :], expected)


def test_installed_codec_ignores_the_adopt_flag_for_turbo4(monkeypatch):
    """clone=False is upstream's adoption path; a fixed arena cannot honour it."""
    codecs, _, _ = _install_isolated(monkeypatch)

    arena = _arena()
    _fill(arena, 8, 1.0)
    caches = _mixed_cache(arena)

    fa_states, _ = codecs.serialize_target_cache(caches, clone=False)
    captured = [mx.array(p) for p in fa_states[1][0]]
    mx.eval(captured)

    arena.offset = 0
    _fill(arena, 8, -7.0)

    for before, after in zip(captured, fa_states[1][0]):
        assert mx.array_equal(before, after), (
            "adopt_cache_arrays must not be honoured for a pooled arena"
        )


def test_installed_codec_delegates_untouched_cache_lists(monkeypatch):
    """Native-only cache lists must keep upstream's exact behaviour."""
    from mlx_lm.models.cache import KVCache

    codecs, _, _ = _install_isolated(monkeypatch)

    native = KVCache()
    native.update_and_fetch(
        mx.zeros((1, 2, 4, 64), dtype=mx.bfloat16),
        mx.zeros((1, 2, 4, 64), dtype=mx.bfloat16),
    )
    fa_states, _ = codecs.serialize_target_cache([native])
    assert fa_states[0] is not None
    assert not isinstance(fa_states[0][0], tuple), "delegated to upstream shape"


def test_installed_codec_rejects_unsupported_entries(monkeypatch):
    codecs, _, _ = _install_isolated(monkeypatch)
    arena = _arena()
    _fill(arena, 4, 1.0)

    with pytest.raises(TypeError, match="not supported alongside Turbo4"):
        codecs.serialize_target_cache([object(), arena])


def test_prefix_key_is_stamped_with_the_cache_format(monkeypatch):
    """Without this a native snapshot and a Turbo4 snapshot share a namespace."""
    from dflash_mlx.cache import codecs, prefix_l2
    from dflash_mlx.cache.fingerprints import DFlashPrefixKey
    from dflash_mlx.server import prefix_cache_manager

    import local_llm_control.dflash_turbo4_snapshot as snap

    plain = DFlashPrefixKey(
        target_model_id="t",
        draft_model_id="d",
        capture_layer_ids=(1,),
        draft_sink_size=64,
        draft_window_size=2048,
        template_hash="h",
        prompt_policy_hash="p",
    )
    assert plain.format_version == 3, "upstream default changed; revisit"

    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT_SNAPSHOTS", "1")
    monkeypatch.setattr(codecs, "serialize_target_cache", codecs.serialize_target_cache)
    monkeypatch.setattr(codecs, "hydrate_target_cache", codecs.hydrate_target_cache)
    monkeypatch.setattr(prefix_l2, "_serialize", prefix_l2._serialize)
    monkeypatch.setattr(snap, "_format_version_override", 0x54123456)
    # Stub the original *before* installing, so the wrapper wraps the stub.
    monkeypatch.setattr(
        prefix_cache_manager, "build_prefix_key", lambda *a, **k: plain
    )

    snap.install_turbo4_prefix_snapshots()

    stamped = prefix_cache_manager.build_prefix_key("ctx", "draft")
    assert stamped.format_version == 0x54123456
    assert stamped != plain, "a Turbo4 key must not collide with a native key"
    # Every other field is preserved so genuine cache hits still match.
    assert stamped.target_model_id == plain.target_model_id
    assert stamped.capture_layer_ids == plain.capture_layer_ids


def test_l2_write_fails_closed_for_packed_snapshots(monkeypatch):
    _, prefix_l2, _ = _install_isolated(monkeypatch)

    arena = _arena()
    _fill(arena, 8, 1.0)
    state = capture_turbo4_state(arena)
    snapshot = type("Snap", (), {"fa_states": (state,), "gdn_states": (None,)})()

    with pytest.raises(NotImplementedError, match="no-prefix-cache-l2"):
        prefix_l2._serialize(snapshot)


def test_snapshots_require_the_turbo4_cache_bridge(monkeypatch):
    """Misconfiguration must fail at startup, not on the first request."""
    import local_llm_control.dflash_turbo4_snapshot as snap

    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT_SNAPSHOTS", "1")
    monkeypatch.setattr(snap, "_format_version_override", None)

    with pytest.raises(ValueError, match="LOCAL_LLM_DFLASH_TURBOQUANT=turbo4"):
        snap.install_turbo4_prefix_snapshots()


def test_launcher_wiring_registers_the_format_and_patches_the_codec(monkeypatch):
    """Mirrors dflash_compat.main(): bridge first, then snapshot codec."""
    from dflash_mlx.cache import codecs, prefix_l2
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
    from dflash_mlx.server import prefix_cache_manager

    import local_llm_control.dflash_turbo4_snapshot as snap
    from local_llm_control.dflash_compat import install_dflash_turboquant

    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT", "turbo4")
    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT_MAX_SIZE", "0")
    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT_SNAPSHOTS", "1")
    monkeypatch.setattr(snap, "_format_version_override", None)

    installed = QwenGdnTargetOps.make_cache
    monkeypatch.setattr(
        QwenGdnTargetOps,
        "make_cache",
        getattr(installed, "_local_llm_original", installed),
    )
    monkeypatch.setattr(codecs, "serialize_target_cache", codecs.serialize_target_cache)
    monkeypatch.setattr(codecs, "hydrate_target_cache", codecs.hydrate_target_cache)
    monkeypatch.setattr(
        prefix_cache_manager, "build_prefix_key", prefix_cache_manager.build_prefix_key
    )
    monkeypatch.setattr(prefix_l2, "_serialize", prefix_l2._serialize)

    install_dflash_turboquant()
    assert snap._format_version_override is not None, (
        "the cache bridge must register the format identity"
    )
    snap.install_turbo4_prefix_snapshots()

    assert getattr(
        codecs.serialize_target_cache, "_local_llm_turbo4_snapshots", False
    ), "codec was not patched"


def test_sidecar_becomes_eligible_only_for_the_supported_turbo4_layout(monkeypatch):
    """Without this, reuse stops at the cold-prompt frontier every turn."""
    from dflash_mlx.cache import codecs
    from mlx_lm.models.cache import QuantizedKVCache

    codecs_mod, _, _ = _install_isolated(monkeypatch)
    arena = _arena()
    _fill(arena, 8, 1.0)

    assert codecs_mod.sidecar_eligible(_mixed_cache(arena)) is True

    # Any other quantized layout stays ineligible: we can only slice ours.
    assert codecs_mod.sidecar_eligible(
        [QuantizedKVCache(group_size=64, bits=8)]
    ) is False
    # And a Turbo4 cache mixed with an unsupported entry is rejected wholesale.
    assert codecs_mod.sidecar_eligible([object(), arena]) is False
    assert codecs is codecs_mod


def _packed_snapshot(arena, tokens: int, boundary: int):
    from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
    from dflash_mlx.cache.fingerprints import DFlashPrefixKey

    _fill(arena, tokens, 1.0)
    state = capture_turbo4_state(arena)
    key = DFlashPrefixKey(
        target_model_id="t",
        draft_model_id="d",
        capture_layer_ids=(1,),
        draft_sink_size=64,
        draft_window_size=2048,
        template_hash="h",
        prompt_policy_hash="p",
    )
    return DFlashPrefixSnapshot(
        token_ids=tuple(range(tokens)),
        fa_states=(None, state),
        gdn_states=(( mx.zeros((1, 2, 4)),), None),
        target_hidden_chunks=(mx.zeros((1, tokens, 8)),),
        target_hidden_chunk_spans=((0, tokens),),
        target_hidden_total_len=tokens,
        last_logits=mx.zeros((1, 16)),
        key=key,
        kind="prefill",
        sidecar_boundary=boundary,
        sidecar_gdn_states=((mx.ones((1, 2, 4)),), None),
        sidecar_last_logits=mx.ones((1, 16)),
    )


def test_sidecar_slicing_trims_every_packed_component(monkeypatch):
    codecs_mod, _, _ = _install_isolated(monkeypatch)
    arena = _arena()
    snapshot = _packed_snapshot(arena, tokens=32, boundary=16)

    sliced = codecs_mod.slice_snapshot_at_sidecar_boundary(snapshot)

    assert sliced.prefix_len == 16
    assert len(sliced.token_ids) == 16
    state = sliced.fa_states[1]
    assert state[2] == 16
    for part in (*state[0], *state[1]):
        assert int(part.shape[2]) == 16, (
            "every packed component must be trimmed to the boundary"
        )
    # The recurrent state is replaced by the sidecar capture, not sliced.
    assert sliced.gdn_states is snapshot.sidecar_gdn_states


def test_sidecar_slicing_rejects_an_out_of_range_boundary(monkeypatch):
    codecs_mod, _, _ = _install_isolated(monkeypatch)
    arena = _arena()
    snapshot = _packed_snapshot(arena, tokens=32, boundary=64)

    with pytest.raises(ValueError, match="outside"):
        codecs_mod.slice_snapshot_at_sidecar_boundary(snapshot)


def test_sidecar_slicing_delegates_for_native_snapshots(monkeypatch):
    """Native snapshots must keep upstream's exact slicing behaviour."""
    from types import SimpleNamespace

    from dflash_mlx.cache import codecs as codecs_module, prefix_l2
    from dflash_mlx.server import prefix_cache_manager

    import local_llm_control.dflash_turbo4_snapshot as snap

    seen = {}

    def stub_original(snapshot, *, require_full_coverage=False):
        seen["require_full_coverage"] = require_full_coverage
        return "delegated"

    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT_SNAPSHOTS", "1")
    monkeypatch.setattr(snap, "_format_version_override", 0x54ABCDEF)
    for module, name in (
        (codecs_module, "serialize_target_cache"),
        (codecs_module, "hydrate_target_cache"),
        (codecs_module, "sidecar_eligible"),
        (prefix_l2, "_serialize"),
        (prefix_cache_manager, "build_prefix_key"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))
    # Install over a stub so delegation is observable.
    monkeypatch.setattr(
        codecs_module, "slice_snapshot_at_sidecar_boundary", stub_original
    )

    snap.install_turbo4_prefix_snapshots()

    native = SimpleNamespace(
        fa_states=(
            None,
            (mx.zeros((1, 2, 4, 8)), mx.zeros((1, 2, 4, 8)), 4),
        )
    )
    result = codecs_module.slice_snapshot_at_sidecar_boundary(
        native, require_full_coverage=True
    )
    assert result == "delegated", "a native snapshot must reach upstream"
    assert seen["require_full_coverage"] is True, "kwargs must pass through"


def test_packed_snapshot_is_sizeable(monkeypatch):
    """The L1 cache sizes every snapshot it admits; upstream reads fa[0].nbytes,
    which raises on a packed triple, so admission itself would fail."""
    from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot

    _install_isolated(monkeypatch)
    arena = _arena()
    snapshot = _packed_snapshot(arena, tokens=16, boundary=8)

    breakdown = snapshot.nbytes_breakdown()
    assert set(breakdown) == {
        "fa_kv",
        "gdn_state",
        "draft_context",
        "last_logits",
        "sidecar",
    }, "keys must match upstream: callers read them individually"
    assert breakdown["fa_kv"] > 0
    assert snapshot.nbytes == sum(breakdown.values())

    # And a native snapshot still goes through upstream's own accounting.
    native = DFlashPrefixSnapshot(
        token_ids=(1, 2),
        fa_states=((mx.zeros((1, 2, 2, 8)), mx.zeros((1, 2, 2, 8)), 2),),
        gdn_states=(None,),
        target_hidden_chunks=(mx.zeros((1, 2, 8)),),
        target_hidden_chunk_spans=((0, 2),),
        target_hidden_total_len=2,
        last_logits=mx.zeros((1, 4)),
        key=snapshot.key,
        kind="prefill",
    )
    assert native.nbytes > 0


def test_packed_snapshot_survives_a_real_l1_cache_round_trip(monkeypatch):
    """The integration seam: insert -> stats -> lookup -> hydrate -> append."""
    from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache

    codecs_mod, _, _ = _install_isolated(monkeypatch)
    arena = _arena(max_size=256)
    snapshot = _packed_snapshot(arena, tokens=16, boundary=8)

    cache = DFlashPrefixCache(
        max_entries=1,
        max_bytes=64 * 1024 * 1024,
        max_snapshot_tokens=0,
    )
    assert cache.insert(snapshot) is True, "a packed snapshot must be admitted"

    stats = cache.stats()
    assert int(stats["current_bytes"]) > 0, (
        "byte accounting must see the packed state"
    )
    assert int(stats["current_entries"]) == 1
    assert int(stats["insertions"]) == 1
    assert int(stats["skipped_too_long"]) == 0

    hit_tokens, restored = cache.lookup(list(snapshot.token_ids), snapshot.key)
    assert restored is not None, "the snapshot must be findable again"
    assert hit_tokens > 0, "the lookup must report reusable prefix tokens"

    # Hydrating into a fresh arena must reproduce the prefix and allow an append.
    fresh = _arena(max_size=256)
    caches = _mixed_cache(fresh)
    rebuilt = codecs_mod.hydrate_target_cache(restored, caches)
    assert rebuilt[1] is fresh
    assert rebuilt[1].offset == 16
    _fill(fresh, 4, 3.0)
    assert fresh.offset == 20, "append must continue past the restored prefix"
