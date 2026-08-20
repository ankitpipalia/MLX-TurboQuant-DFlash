"""Session prefix-reuse cache logic, exercised with lightweight fakes.

These avoid importing mlx/mlx_lm so the reuse math is testable without the
heavy runtime; the cache module's isinstance checks are monkeypatched onto the
fakes.
"""

import sys
import types

import pytest


@pytest.fixture
def session_module(monkeypatch):
    """Import mlx_session_cache with mx/mlx_lm replaced by fakes."""
    # Minimal mx stub: array() copies a list, zeros_like() returns zeros, eval() no-op.
    mx = types.ModuleType("mlx.core")

    class _Arr(list):
        @property
        def nbytes(self):
            return len(self) * 4

    mx.array = lambda a: _Arr(a)
    mx.zeros_like = lambda a: _Arr([0] * len(a))
    mx.eval = lambda *a, **k: None
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx

    class ArraysCache:
        def __init__(self, size=2):
            self.cache = [None] * size
            self.left_padding = None
            self.lengths = None

    class QuantizedKVCache:
        def __init__(self):
            self.keys = None
            self.values = None
            self.offset = 0

    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.ArraysCache = ArraysCache
    cache_mod.QuantizedKVCache = QuantizedKVCache
    mlx_lm_pkg = types.ModuleType("mlx_lm")
    models_pkg = types.ModuleType("mlx_lm.models")

    for name, mod in {
        "mlx": mlx_pkg, "mlx.core": mx, "mlx_lm": mlx_lm_pkg,
        "mlx_lm.models": models_pkg, "mlx_lm.models.cache": cache_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules.pop("local_llm_control.mlx_session_cache", None)
    import local_llm_control.mlx_session_cache as m
    return m


def _fake_caches(module):
    """One recurrent layer + one KV layer, like a hybrid Qwen block pair."""
    rec = module.ArraysCache(size=2)
    rec.cache = [module.mx.array([1, 2, 3]), module.mx.array([4, 5])]
    kv = module.QuantizedKVCache()
    kv.offset = 0
    return [rec, kv], rec, kv


def test_stats_by_type_matches_server_contract(session_module):
    # The server's _log_cache_stats reads n_sequences / n_bytes per type.
    cache = session_module.SessionPromptCache()
    stats = cache.stats_by_type()
    for entry in stats.values():
        assert "n_sequences" in entry and "n_bytes" in entry


def test_first_request_has_no_cache(session_module):
    cache = session_module.SessionPromptCache()
    result, rest = cache.fetch_nearest_cache(None, [1, 2, 3, 4])
    assert result is None
    assert rest == [1, 2, 3, 4]
    assert len(cache) == 0


def test_cold_cache_factory_binds_prompt_boundary_owner(session_module):
    cache = session_module.SessionPromptCache(max_checkpoints=2)
    result, _rest = cache.fetch_nearest_cache(None, [1, 2, 3, 4])
    assert result is None

    caches, _rec, kv = _fake_caches(session_module)
    assert session_module.bind_new_prompt_cache(caches)
    kv.offset = 5
    assert session_module.checkpoint_prompt_boundary(caches, 90)
    assert [checkpoint.pos for checkpoint in cache._checkpoints] == [5]


def test_pure_append_reuses_full_prefix_without_replay(session_module):
    cache = session_module.SessionPromptCache()
    caches, _rec, kv = _fake_caches(session_module)
    kv.offset = 5
    cache.insert_cache(None, [10, 11, 12, 13, 14], caches)  # session of 5 tokens

    # Next turn appends 3 new tokens to the identical prefix.
    reused, rest = cache.fetch_nearest_cache(None, [10, 11, 12, 13, 14, 20, 21, 22])
    assert reused is caches
    assert cache.reused_last == 5          # whole prior sequence reused
    assert rest == [20, 21, 22]            # only new tokens re-prefilled
    assert kv.offset == 5                  # KV offset positioned at reuse point


def test_divergence_without_checkpoint_falls_back_to_full(session_module):
    cache = session_module.SessionPromptCache(max_checkpoints=2)
    caches, _rec, kv = _fake_caches(session_module)
    kv.offset = 5
    cache.insert_cache(None, [10, 11, 12, 13, 14], caches)

    # New prompt shares only tokens [10, 11] then diverges; the only checkpoint
    # is at position 5 (> common prefix 2), so it must re-prefill from zero.
    reused, rest = cache.fetch_nearest_cache(None, [10, 11, 99, 98])
    assert reused is caches
    assert cache.reused_last == 0
    assert rest == [10, 11, 99, 98]
    assert kv.offset == 0


def test_identical_prompt_cannot_rewind_recurrent_by_one(session_module):
    # An exact re-submission would need the recurrent state one token before the
    # end, which is not snapshotted; documented limitation → full reprefill.
    # (Agent clients always append a new turn, so this case does not arise.)
    cache = session_module.SessionPromptCache()
    caches, _rec, kv = _fake_caches(session_module)
    kv.offset = 4
    cache.insert_cache(None, [1, 2, 3, 4], caches)
    _reused, rest = cache.fetch_nearest_cache(None, [1, 2, 3, 4])
    assert cache.reused_last == 0
    assert rest == [1, 2, 3, 4]


def test_divergence_rewinds_to_checkpoint_when_present(session_module):
    # Two turn boundaries recorded (pos 3 and pos 6). A new prompt that shares
    # the first 4 tokens then diverges must rewind to the pos-3 checkpoint.
    cache = session_module.SessionPromptCache(max_checkpoints=4)
    c1, _r1, k1 = _fake_caches(session_module)
    cache.insert_cache(None, [1, 2, 3], c1)          # checkpoint at 3
    k1.offset = 6
    cache.insert_cache(None, [1, 2, 3, 4, 5, 6], c1)  # checkpoint at 6 (same slot)

    reused, rest = cache.fetch_nearest_cache(None, [1, 2, 3, 4, 99, 98])
    assert cache.reused_last == 3            # nearest checkpoint <= common prefix (4)
    assert rest == [4, 99, 98]
    assert k1.offset == 3


def test_checkpoint_ring_is_bounded(session_module):
    cache = session_module.SessionPromptCache(max_checkpoints=2)
    for turn in range(5):
        caches, _rec, _kv = _fake_caches(session_module)
        cache.insert_cache(None, list(range(turn + 1)), caches)
    assert len(cache._checkpoints) == 2


def test_prompt_boundary_checkpoint_avoids_full_tool_suffix_replay(session_module):
    cache = session_module.SessionPromptCache(max_checkpoints=2)
    caches, rec, kv = _fake_caches(session_module)

    # A prior assistant turn is live at token 3.
    kv.offset = 3
    cache.insert_cache(None, [1, 2, 3], caches)

    # The next request appends a tool result. At the first generated token MLX
    # has advanced the cache through prompt [1..5] plus token 90.
    reused, rest = cache.fetch_nearest_cache(None, [1, 2, 3, 4, 5])
    assert reused is caches
    assert rest == [4, 5]
    rec.cache = [
        session_module.mx.array([9, 9, 9]),
        session_module.mx.array([8, 8]),
    ]
    kv.offset = 6
    assert session_module.checkpoint_prompt_boundary(caches, 90)

    # Generation continues beyond the stable prompt boundary.
    rec.cache = [
        session_module.mx.array([7, 7, 7]),
        session_module.mx.array([6, 6]),
    ]
    kv.offset = 8
    cache.insert_cache(None, [1, 2, 3, 4, 5, 90, 91, 92], caches)

    # OpenCode re-renders the assistant suffix differently after token 90.
    # The boundary checkpoint at 6 must be restored instead of replaying all 9.
    reused, rest = cache.fetch_nearest_cache(
        None, [1, 2, 3, 4, 5, 90, 70, 71, 72]
    )
    assert reused is caches
    assert cache.reused_last == 6
    assert rest == [70, 71, 72]
    assert kv.offset == 6
    assert rec.cache[0] == [9, 9, 9]


def test_prompt_boundary_checkpoint_rejects_wrong_kv_position(session_module):
    cache = session_module.SessionPromptCache(max_checkpoints=2)
    caches, _rec, kv = _fake_caches(session_module)
    kv.offset = 2
    cache.insert_cache(None, [1, 2], caches)
    cache.fetch_nearest_cache(None, [1, 2, 3])

    # Expected offset is prompt length 3 + first generated token = 4.
    kv.offset = 5
    assert not session_module.checkpoint_prompt_boundary(caches, 90)


def test_stale_checkpoint_from_abandoned_branch_is_never_restored(session_module):
    cache = session_module.SessionPromptCache(max_checkpoints=4)
    caches, _rec, kv = _fake_caches(session_module)

    old_branch = [1, 2, 3, 4]
    cache.insert_cache(None, old_branch, caches)
    kv.offset = 8
    cache.insert_cache(None, old_branch + [5, 6, 7, 8], caches)

    # Switch to a new branch sharing only two tokens. Neither old checkpoint is
    # valid for its recurrent state, so this request starts from zero.
    new_branch = [1, 2, 90, 91, 92, 93, 94, 95, 96, 97]
    cache.fetch_nearest_cache(None, new_branch)
    assert cache.reused_last == 0
    cache.insert_cache(None, new_branch, caches)
    assert [checkpoint.pos for checkpoint in cache._checkpoints] == [10]

    # A later divergence at position six must not resurrect the old branch's
    # numeric position-four checkpoint.
    next_prompt = [1, 2, 90, 91, 92, 93, 80, 81]
    cache.fetch_nearest_cache(None, next_prompt)
    assert cache.reused_last == 0
    assert kv.offset == 0
