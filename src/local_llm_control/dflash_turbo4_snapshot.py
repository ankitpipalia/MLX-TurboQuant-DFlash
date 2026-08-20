"""Prefix-snapshot support for DFlash's packed Turbo4 target KV caches.

DFlash models a full-attention layer as ``(K, V, offset)`` -- one array per side
-- and raises ``TypeError`` on anything else. A TurboQuant cache is six arrays
(packed data, scales and biases per side), so it lands in that reject branch,
which is why quantized target KV and prefix reuse cannot currently coexist.
Restore is additionally disabled outright whenever ``quantize_kv_cache`` is set,
while publish is *not* gated at all -- so the failure today is "reuse silently
off, and the first snapshot write crashes".

Three properties are load-bearing, and none is optional:

**Ownership on capture.** ``build_snapshot(adopt_cache_arrays=True)`` stores
references to the live cache arrays. Upstream is safe because an exact-length
``KVCache`` forces the growth path on its next update; a fixed arena never
grows -- it writes in place and rewinds ``offset`` to zero -- so an adopted
reference would be overwritten by the following request. Adoption also skips
``mx.eval`` while ``QuantizedKVCache.state`` returns *lazy* slices over the
arena, so such a snapshot would resolve after the overwrite. We therefore copy
and evaluate every component and ignore the adopt flag.

**Format identity.** ``DFlashPrefixKey`` describes model, draft, capture layers,
sink/window, template and prompt policy -- and nothing about the KV format, so
native, KV8 and Turbo4 snapshots share one namespace. Because stored keys are
Hadamard-rotated, restoring a Turbo4 prefix into a plain cache, or into a
Turbo4 cache built with a different seed, is silently wrong rather than an
error. ``meta_state`` carries only ``(offset, group_size, bits)`` and cannot
express the seed, so the format identity is folded into the key's
``format_version`` instead: it is already round-tripped and already hashed into
the L1 bucket, the L2 directory and the cache-manager singleton, so one field
re-namespaces all three.

**Ownership on restore.** Upstream hydrate builds a *new* cache per layer.
Doing that here would abandon the preallocated arena on every cache hit, so the
pooled template is filled in place and returned instead.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import replace
from typing import Any

_SNAPSHOT_ENV = "LOCAL_LLM_DFLASH_TURBOQUANT_SNAPSHOTS"

# Consumers that bind codec functions with ``from ... import name``. They are
# unloaded when the launcher runs, so patching the defining module is enough --
# but rebind any that are already present in case the import graph changes.
_CODEC_CONSUMERS = (
    "dflash_mlx.engine.spec_epoch",
    "dflash_mlx.cache.prefix_l1",
    "dflash_mlx.server.prefix_cache_flow",
)
_REBOUND_CODEC_NAMES = (
    "hydrate_target_cache",
    "build_prefix_key",
    "sidecar_eligible",
    "slice_snapshot_at_sidecar_boundary",
)


def is_packed_state(state: Any) -> bool:
    """True for a Turbo4 FA state, whose K/V slots hold packed triples."""
    return state is not None and isinstance(state[0], tuple)


def snapshots_enabled() -> bool:
    return os.getenv(_SNAPSHOT_ENV, "").strip() == "1"


def turbo4_format_version(
    *, bits: int, group_size: int, seed: int, qjl: bool
) -> int:
    """Derive a stable key discriminator from the full cache format identity.

    Everything that changes the meaning of the stored bytes is included, so a
    cache built with different bits, grouping, rotation seed or QJL setting can
    never collide with this one. Upstream uses small integers here (currently
    3), so the high byte keeps us clear of that space.
    """
    identity = (
        f"turboquant:k{int(bits)}:v{int(bits)}"
        f":g{int(group_size)}:seed{int(seed)}:qjl{int(bool(qjl))}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return 0x54000000 | (int(digest[:6], 16) & 0xFFFFFF)


def _clone_evaluated(parts: Any) -> tuple[Any, ...]:
    import mlx.core as mx

    copied = tuple(mx.array(part) for part in parts)
    mx.eval(list(copied))
    return copied


def capture_turbo4_state(cache: Any) -> tuple[Any, Any, int]:
    """Copy the live prefix out of a fixed arena into an owned snapshot state.

    Returns arity 3 so upstream's ``state[2]`` offset checks and its
    ``len(state) != 3`` sidecar guard keep working; slots 0 and 1 hold the
    packed ``(data, scales, biases)`` triples rather than single arrays.
    """
    keys, values = cache.state
    offset = int(cache.offset)
    return (_clone_evaluated(keys), _clone_evaluated(values), offset)


def _validate_geometry(template: Any, keys: Any, values: Any, offset: int) -> None:
    if template.keys is None or template.values is None:
        raise ValueError("Turbo4 template arena was never reserved")
    max_size = int(getattr(template, "max_size", 0) or 0)
    if max_size and offset > max_size:
        raise ValueError(
            f"Turbo4 snapshot prefix {offset:,} exceeds the reserved "
            f"{max_size:,}-token arena"
        )
    for label, parts, slots in (
        ("key", keys, template.keys),
        ("value", values, template.values),
    ):
        if len(parts) != len(slots):
            raise ValueError(
                f"Turbo4 snapshot has {len(parts)} packed {label} components, "
                f"arena expects {len(slots)}"
            )
        for index, (part, slot) in enumerate(zip(parts, slots)):
            if part.dtype != slot.dtype:
                # MLX casts silently on assignment, so an unchecked mismatch
                # would quietly degrade the stored scales.
                raise ValueError(
                    f"Turbo4 snapshot {label} component {index} dtype "
                    f"{part.dtype} != arena dtype {slot.dtype}"
                )
            if int(part.shape[2]) != int(offset):
                raise ValueError(
                    f"Turbo4 snapshot {label} component {index} holds "
                    f"{int(part.shape[2])} tokens, expected {int(offset)}"
                )
            if part.shape[:2] != slot.shape[:2] or part.shape[-1] != slot.shape[-1]:
                raise ValueError(
                    f"Turbo4 snapshot {label} component {index} shape "
                    f"{tuple(part.shape)} is incompatible with arena shape "
                    f"{tuple(slot.shape)}"
                )


def restore_turbo4_state(template: Any, state: tuple[Any, Any, int]) -> Any:
    """Fill the pooled arena in place and return it, never a fresh cache."""
    import mlx.core as mx

    keys, values, offset = state
    _validate_geometry(template, keys, values, int(offset))
    for index, part in enumerate(keys):
        template.keys[index][..., : int(offset), :] = part
    for index, part in enumerate(values):
        template.values[index][..., : int(offset), :] = part
    template.offset = int(offset)
    mx.eval([*template.keys, *template.values])
    return template


def install_turbo4_prefix_snapshots() -> None:
    """Teach DFlash's snapshot codec about the packed Turbo4 representation."""
    if not snapshots_enabled():
        return
    if _format_version_override is None:
        # Fail at startup rather than on the first request: without a registered
        # format identity the prefix key cannot be stamped, and an unstamped key
        # would share a namespace with native snapshots.
        raise ValueError(
            f"{_SNAPSHOT_ENV}=1 requires LOCAL_LLM_DFLASH_TURBOQUANT=turbo4 so "
            "the cache format identity is registered before keys are stamped"
        )

    from dflash_mlx.cache import codecs, prefix_l2
    from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache
    from dflash_mlx.server import prefix_cache_manager
    from mlx_turboquant.kv_cache import TurboQuantKVCache

    original_serialize = codecs.serialize_target_cache
    if getattr(original_serialize, "_local_llm_turbo4_snapshots", False):
        return
    original_hydrate = codecs.hydrate_target_cache
    original_build_key = prefix_cache_manager.build_prefix_key
    original_l2_serialize = prefix_l2._serialize
    original_sidecar_eligible = codecs.sidecar_eligible
    original_slice = codecs.slice_snapshot_at_sidecar_boundary

    def serialize_target_cache(target_cache: list[Any], *, clone: bool = True):
        if not any(isinstance(e, TurboQuantKVCache) for e in target_cache):
            return original_serialize(target_cache, clone=clone)
        fa: list[Any] = []
        gdn: list[Any] = []
        for index, entry in enumerate(target_cache):
            if isinstance(entry, TurboQuantKVCache):
                # ``clone`` is deliberately ignored -- see the module docstring:
                # adopting a fixed arena hands out a buffer the next request
                # overwrites.
                fa.append(capture_turbo4_state(entry))
                gdn.append(None)
            elif isinstance(entry, RecurrentRollbackCache):
                fa.append(None)
                gdn.append(_clone_evaluated(entry.cache))
            else:
                raise TypeError(
                    f"Cache entry type {type(entry).__name__} at layer {index} "
                    "is not supported alongside Turbo4 prefix snapshots"
                )
        return tuple(fa), tuple(gdn)

    def hydrate_target_cache(snapshot: Any, template_cache: list[Any]):
        if not any(isinstance(t, TurboQuantKVCache) for t in template_cache):
            return original_hydrate(snapshot, template_cache)
        if len(template_cache) != len(snapshot.fa_states):
            raise ValueError(
                f"Template cache length {len(template_cache)} != snapshot "
                f"layer count {len(snapshot.fa_states)}"
            )
        result: list[Any] = []
        for index, template in enumerate(template_cache):
            fa_state = snapshot.fa_states[index]
            gdn_state = snapshot.gdn_states[index]
            if isinstance(template, TurboQuantKVCache):
                if fa_state is None:
                    raise ValueError(f"Snapshot missing FA state at layer {index}")
                if int(fa_state[2]) != int(snapshot.prefix_len):
                    raise ValueError(
                        f"Snapshot FA offset {int(fa_state[2])} at layer {index} "
                        f"!= token prefix length {int(snapshot.prefix_len)}"
                    )
                result.append(restore_turbo4_state(template, fa_state))
            elif isinstance(template, RecurrentRollbackCache):
                if gdn_state is None:
                    raise ValueError(f"Snapshot missing GDN state at layer {index}")
                restored = RecurrentRollbackCache(
                    size=len(template.cache),
                    conv_kernel_size=template.conv_kernel_size,
                )
                restored.cache = list(gdn_state)
                result.append(restored)
            else:
                raise TypeError(
                    f"Cannot hydrate cache of type {type(template).__name__} "
                    f"at layer {index} alongside Turbo4 prefix snapshots"
                )
        return result

    def build_prefix_key(*args: Any, **kwargs: Any):
        key = original_build_key(*args, **kwargs)
        return replace(key, format_version=_active_format_version())

    def sidecar_eligible(target_cache: list[Any]) -> bool:
        """Admit the one quantized cache whose packed state we can slice.

        Upstream tests ``isinstance(entry, KVCache)``, and ``QuantizedKVCache``
        is a sibling rather than a subclass, so a Turbo4 cache silently
        disabled generation sidecars -- capping reuse at the cold-prompt
        frontier instead of following the conversation. Deliberately narrow:
        any other quantized layout stays ineligible.
        """
        if not any(isinstance(e, TurboQuantKVCache) for e in target_cache):
            return original_sidecar_eligible(target_cache)
        return all(
            isinstance(entry, (TurboQuantKVCache, RecurrentRollbackCache))
            for entry in target_cache
        )

    def slice_snapshot_at_sidecar_boundary(
        snapshot: Any, *, require_full_coverage: bool = False
    ):
        if not any(is_packed_state(s) for s in snapshot.fa_states):
            return original_slice(
                snapshot, require_full_coverage=require_full_coverage
            )

        from dflash_mlx.cache.codecs import snapshot_covers_prefix
        from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot

        boundary = int(snapshot.sidecar_boundary)
        if not 0 < boundary < snapshot.prefix_len:
            raise ValueError(
                f"Sidecar boundary {boundary} outside (0, {snapshot.prefix_len})"
            )
        if (
            snapshot.sidecar_gdn_states is None
            or snapshot.sidecar_last_logits is None
        ):
            raise ValueError("Snapshot has a sidecar boundary but no sidecar states")
        if require_full_coverage and not snapshot_covers_prefix(snapshot, boundary):
            raise ValueError(
                f"Snapshot feature spans do not cover sidecar boundary {boundary}"
            )

        fa: list[Any] = []
        for layer_idx, state in enumerate(snapshot.fa_states):
            if state is None:
                fa.append(None)
                continue
            if len(state) != 3:
                raise ValueError(
                    f"FA state at layer {layer_idx} is not boundary-sliceable"
                )
            keys, values, _offset = state
            # Scales and biases are grouped along the last axis, never the token
            # axis, so slicing axis 2 is valid for every packed component.
            fa.append(
                (
                    tuple(part[:, :, :boundary, :] for part in keys),
                    tuple(part[:, :, :boundary, :] for part in values),
                    boundary,
                )
            )

        chunks: list[Any] = []
        spans: list[tuple[int, int]] = []
        for chunk, (start, end) in zip(
            snapshot.target_hidden_chunks, snapshot.target_hidden_chunk_spans
        ):
            if start >= boundary:
                continue
            keep = min(end, boundary) - start
            chunks.append(chunk[:, :keep, :])
            spans.append((start, start + keep))

        return DFlashPrefixSnapshot(
            token_ids=snapshot.token_ids[:boundary],
            fa_states=tuple(fa),
            gdn_states=snapshot.sidecar_gdn_states,
            target_hidden_chunks=tuple(chunks),
            target_hidden_chunk_spans=tuple(spans),
            target_hidden_total_len=boundary,
            last_logits=snapshot.sidecar_last_logits,
            key=snapshot.key,
            kind="prefill",
            created_at=snapshot.created_at,
        )

    def _serialize(snapshot: Any):
        for state in snapshot.fa_states:
            if state is not None and isinstance(state[0], tuple):
                raise NotImplementedError(
                    "Turbo4 prefix snapshots cannot be written to the L2 disk "
                    "cache: schema v4 stores one array per side. Run with "
                    "--no-prefix-cache-l2."
                )
        return original_l2_serialize(snapshot)

    serialize_target_cache._local_llm_turbo4_snapshots = True  # type: ignore[attr-defined]
    serialize_target_cache._local_llm_original = original_serialize  # type: ignore[attr-defined]
    hydrate_target_cache._local_llm_original = original_hydrate  # type: ignore[attr-defined]
    build_prefix_key._local_llm_original = original_build_key  # type: ignore[attr-defined]
    _serialize._local_llm_original = original_l2_serialize  # type: ignore[attr-defined]

    codecs.serialize_target_cache = serialize_target_cache
    codecs.hydrate_target_cache = hydrate_target_cache
    codecs.sidecar_eligible = sidecar_eligible
    codecs.slice_snapshot_at_sidecar_boundary = slice_snapshot_at_sidecar_boundary
    prefix_cache_manager.build_prefix_key = build_prefix_key
    prefix_l2._serialize = _serialize

    replacements = {
        "hydrate_target_cache": (original_hydrate, hydrate_target_cache),
        "build_prefix_key": (original_build_key, build_prefix_key),
        "sidecar_eligible": (original_sidecar_eligible, sidecar_eligible),
        "slice_snapshot_at_sidecar_boundary": (
            original_slice,
            slice_snapshot_at_sidecar_boundary,
        ),
    }
    for module_name in _CODEC_CONSUMERS:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name in _REBOUND_CODEC_NAMES:
            previous, current = replacements[name]
            if getattr(module, name, None) is previous:
                setattr(module, name, current)


_format_version_override: int | None = None


def set_active_format_version(version: int | None) -> None:
    """Record the format identity of the caches this process actually builds."""
    global _format_version_override
    _format_version_override = version


def _active_format_version() -> int:
    if _format_version_override is None:
        raise RuntimeError(
            "Turbo4 prefix snapshots are enabled but no cache format identity "
            "was registered; the Turbo4 cache bridge must run first"
        )
    return _format_version_override
