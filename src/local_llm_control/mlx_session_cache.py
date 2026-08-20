"""Cross-request context reuse for the single-slot MLX server.

MLX-LM's stock ``LRUPromptCache`` reuses a prefix only when the whole cache can
be trimmed, and ``can_trim_prompt_cache`` is false for hybrid Qwen3.5/3.6
checkpoints because their recurrent ``ArraysCache`` layers are not trimmable.
The practical effect is that every coding-agent turn re-prefills the entire
conversation — minutes of wasted work on a 100K-token history.

This module replaces the prompt cache with a session-aware, single-slot cache
that mirrors llama.cpp's context-checkpoint trick:

* The full-attention KV arena is a per-token store, so reuse of tokens
  ``[0:k]`` costs nothing but moving an integer offset — the quantized data is
  already sitting in the fixed arena.
* The recurrent GatedDeltaNet state is *not* rewindable, but it is small and
  fixed size, so we snapshot it at a few checkpoint positions and restore the
  nearest one at or before the reuse boundary, then replay only the tokens
  after that checkpoint.

For the common append-only case (turn N+1 = turn N tokens + new user message)
the reuse boundary equals the full previous length, so nothing is snapshotted
or replayed: we keep the live state in place and prefill only the new tokens.
"""

from __future__ import annotations

import logging
from threading import local
from typing import Any
from weakref import WeakValueDictionary

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, QuantizedKVCache

from .mlx_metrics import (
    record_cache_fetch,
    record_checkpoint,
    record_prompt_boundary_checkpoint,
)


_CACHE_OWNERS: WeakValueDictionary[int, "SessionPromptCache"] = (
    WeakValueDictionary()
)
_PENDING = local()


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _is_prefix(prefix: list[int], tokens: list[int]) -> bool:
    return len(prefix) <= len(tokens) and all(
        left == right for left, right in zip(prefix, tokens)
    )


def _snapshot_recurrent(caches: list[Any]) -> list[list[Any]]:
    """Deep-copy the recurrent arrays so later generation cannot mutate them."""
    snapshot: list[list[Any]] = []
    flat: list[Any] = []
    for cache in caches:
        if isinstance(cache, ArraysCache):
            copied = [None if a is None else mx.array(a) for a in cache.cache]
            flat.extend(a for a in copied if a is not None)
            snapshot.append(copied)
        else:
            snapshot.append([])
    # Force concrete Metal buffers so the snapshot cannot retain a lazy graph
    # that would balloon memory or be mutated by continued generation.
    if flat:
        mx.eval(flat)
    return snapshot


def _restore_recurrent(caches: list[Any], snapshot: list[list[Any]]) -> None:
    for cache, saved in zip(caches, snapshot):
        if isinstance(cache, ArraysCache):
            cache.cache = [None if a is None else mx.array(a) for a in saved]
            cache.left_padding = None
            cache.lengths = None


def _kv_offset(caches: list[Any]) -> int:
    for cache in caches:
        if isinstance(cache, QuantizedKVCache):
            return int(cache.offset)
    return 0


def _set_kv_offset(caches: list[Any], offset: int) -> None:
    for cache in caches:
        if isinstance(cache, QuantizedKVCache):
            cache.offset = offset


def _zero_recurrent(caches: list[Any]) -> None:
    for cache in caches:
        if isinstance(cache, ArraysCache):
            cache.cache = [None if a is None else mx.zeros_like(a) for a in cache.cache]
            cache.left_padding = None
            cache.lengths = None


class _Checkpoint:
    __slots__ = ("pos", "tokens", "recurrent")

    def __init__(self, tokens: list[int], recurrent: list[list[Any]]):
        self.tokens = list(tokens)
        self.pos = len(tokens)
        self.recurrent = recurrent


class SessionPromptCache:
    """A one-slot prompt cache that reuses shared prefixes across requests.

    Implements the subset of the ``LRUPromptCache`` interface that the MLX-LM
    server's sequential path uses: ``fetch_nearest_cache``, ``insert_cache``,
    ``trim_to``, ``__len__``, ``nbytes`` and ``stats_by_type``.
    """

    def __init__(self, max_checkpoints: int = 2):
        # llama.cpp defaults to two context checkpoints; keep the same trade-off
        # between rewind coverage and snapshot memory.  This model's recurrent
        # state is currently about 61 MiB per checkpoint.
        self.max_checkpoints = max(1, max_checkpoints)
        self._caches: list[Any] | None = None
        self._session_tokens: list[int] = []
        self._checkpoints: list[_Checkpoint] = []
        self._pending_prompt_tokens: list[int] | None = None
        self.reused_last = 0
        self.replayed_last = 0

    # ── LRU-compatible surface ───────────────────────────────────────────────
    def __len__(self) -> int:
        return 1 if self._caches is not None else 0

    @property
    def nbytes(self) -> int:
        total = 0
        if self._caches is not None:
            total += sum(getattr(c, "nbytes", 0) for c in self._caches)
        for ckpt in self._checkpoints:
            for arrays in ckpt.recurrent:
                total += sum(a.nbytes for a in arrays if a is not None)
        return total

    def stats_by_type(self) -> dict[str, dict[str, int]]:
        return {"session": {"n_sequences": len(self), "n_bytes": self.nbytes}}

    def trim_to(self, *, n_sequences: int | None = None, n_bytes: int | None = None) -> None:
        # The single live slot is never evicted; only bound the checkpoint ring.
        while len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints.pop(0)

    # ── Core reuse logic ─────────────────────────────────────────────────────
    def fetch_nearest_cache(self, model: Any, tokens: list[int]):
        self._pending_prompt_tokens = list(tokens)
        if self._caches is None:
            # ``ResponseGenerator`` creates the physical cache only after this
            # miss. Hand the owner across that synchronous factory call so the
            # first cold request can still save its prompt boundary.
            _PENDING.owner = self
            self.reused_last = 0
            self.replayed_last = 0
            record_cache_fetch(
                reused=0,
                replayed=len(tokens),
                had_session=False,
                restored_checkpoint=False,
            )
            return None, list(tokens)

        _CACHE_OWNERS[id(self._caches)] = self
        raw_prefix = _common_prefix_len(self._session_tokens, tokens)

        # Pure append: the new prompt strictly extends the whole session, so the
        # live recurrent state (valid only at its end position) is exactly right
        # and no snapshot or replay is needed — just continue from the end.
        if raw_prefix == len(self._session_tokens) and len(tokens) > raw_prefix:
            reuse = len(self._session_tokens)
            _set_kv_offset(self._caches, reuse)
            self.reused_last = reuse
            self.replayed_last = len(tokens) - reuse
            record_cache_fetch(
                reused=reuse,
                replayed=self.replayed_last,
                had_session=True,
                restored_checkpoint=False,
            )
            return self._caches, list(tokens[reuse:])

        # Otherwise we must land on a checkpoint at or before the common prefix
        # (the recurrent state cannot be rewound to an arbitrary token). Leave at
        # least one token for the generator to consume.
        prefix = min(raw_prefix, len(tokens) - 1)
        chosen: _Checkpoint | None = None
        for ckpt in self._checkpoints:
            if (
                ckpt.pos <= prefix
                and _is_prefix(ckpt.tokens, tokens)
                and (chosen is None or ckpt.pos > chosen.pos)
            ):
                chosen = ckpt
        if chosen is None:
            _zero_recurrent(self._caches)
            _set_kv_offset(self._caches, 0)
            self.reused_last = 0
            self.replayed_last = len(tokens)
            record_cache_fetch(
                reused=0,
                replayed=len(tokens),
                had_session=True,
                restored_checkpoint=False,
            )
            return self._caches, list(tokens)

        _restore_recurrent(self._caches, chosen.recurrent)
        _set_kv_offset(self._caches, chosen.pos)
        self.reused_last = chosen.pos
        self.replayed_last = len(tokens) - chosen.pos
        record_cache_fetch(
            reused=chosen.pos,
            replayed=self.replayed_last,
            had_session=True,
            restored_checkpoint=True,
        )
        return self._caches, list(tokens[chosen.pos:])

    def checkpoint_prompt_boundary(self, first_generated_token: int) -> bool:
        """Snapshot the hybrid state immediately after prompt prefill.

        MLX advances the live cache through the first generated token before
        ``stream_generate`` yields. That position is still a useful stable
        boundary: OpenCode may re-render or omit later assistant thinking/tool
        text, but the first generated token remains part of the common prefix.
        Keeping this snapshot prevents a divergent tool-history suffix from
        forcing a replay from token zero.
        """
        if self._caches is None or self._pending_prompt_tokens is None:
            return False
        tokens = [
            *self._pending_prompt_tokens,
            int(first_generated_token),
        ]
        if _kv_offset(self._caches) != len(tokens):
            # Do not retain a recurrent snapshot whose logical token position
            # is uncertain; restoring it would silently corrupt generation.
            logging.warning(
                "local-llm: skipped prompt-boundary checkpoint: KV offset "
                "%d != token position %d",
                _kv_offset(self._caches),
                len(tokens),
            )
            record_prompt_boundary_checkpoint(matched=False)
            return False
        self._checkpoints = [
            ckpt for ckpt in self._checkpoints
            if _is_prefix(ckpt.tokens, tokens)
        ]
        if not any(ckpt.tokens == tokens for ckpt in self._checkpoints):
            self._checkpoints.append(
                _Checkpoint(tokens, _snapshot_recurrent(self._caches))
            )
            record_checkpoint()
        self.trim_to()
        self._pending_prompt_tokens = None
        record_prompt_boundary_checkpoint(matched=True)
        return True

    def insert_cache(self, model: Any, tokens: list[int], prompt_cache: list[Any],
                     *, cache_type: str = "assistant") -> None:
        self._caches = prompt_cache
        _CACHE_OWNERS[id(prompt_cache)] = self
        self._session_tokens = list(tokens)
        self._pending_prompt_tokens = None
        # Checkpoints from an abandoned conversation branch can have a valid
        # numeric position but the wrong recurrent state. Keep only snapshots
        # whose complete token prefix belongs to the newly active branch.
        self._checkpoints = [
            ckpt for ckpt in self._checkpoints
            if _is_prefix(ckpt.tokens, tokens)
        ]
        # Record a checkpoint at this turn boundary so a later divergence that
        # lands before it can still rewind cheaply.
        self._checkpoints.append(
            _Checkpoint(tokens, _snapshot_recurrent(prompt_cache))
        )
        record_checkpoint()
        self.trim_to()


def checkpoint_prompt_boundary(
    prompt_cache: list[Any] | None, first_generated_token: int
) -> bool:
    """Checkpoint a live session cache from the stream-generation boundary."""
    if prompt_cache is None:
        return False
    owner = _CACHE_OWNERS.get(id(prompt_cache))
    if owner is None:
        return False
    return owner.checkpoint_prompt_boundary(first_generated_token)


def bind_new_prompt_cache(prompt_cache: list[Any]) -> bool:
    """Attach a cache created after a cold miss to its session owner."""
    owner = getattr(_PENDING, "owner", None)
    if owner is None:
        return False
    try:
        del _PENDING.owner
    except AttributeError:
        pass
    owner._caches = prompt_cache
    _CACHE_OWNERS[id(prompt_cache)] = owner
    return True
