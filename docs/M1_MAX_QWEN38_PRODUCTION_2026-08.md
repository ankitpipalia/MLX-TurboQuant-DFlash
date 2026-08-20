# Qwen3.8-27B on an M1 Max 32 GB — production configuration

Written 2026-08-20. Every number here was measured on this machine or read out
of the installed source; nothing is carried over from published benchmarks on
other hardware. Where a figure comes from a public report it says so.

## Hardware ground truth

`dflash doctor --json` (safe — weight loading is gated behind `--load-model`):

```
family = M1   tier = max   arch_gen = 13
bf16_native   = False    bf16_emulated = True     warning: old_apple_bf16
nax_capable   = False                             warning: nax_unavailable
```

Two consequences. Every published DFlash2/MTP figure from an M5 machine relies
on NAX verification kernels this GPU does not have, so those numbers do not
transfer — we run the fork's own `simdgroup_matrix<T,8,8>` fallback path. And
bf16 is *emulated* here, while all four local Qwen3.8 artifacts are bf16; an
fp16-based checkpoint may therefore be worth sourcing, but none exists locally
to A/B against (no `oQ4e`/`fp16` match under `~/Models`).

## Model architecture

Qwen3.8-27B (`model_type: qwen3_5`) is hybrid. Counted in both local
checkpoints, and independently confirmed by `mlx_lm`, which derives layer kind
from `(idx + 1) % full_attention_interval != 0` rather than reading
`layer_types`:

```
64 layers = 48 linear-attention (GatedDeltaNet)  +  16 full attention
            ^ fixed-size recurrent state            ^ grows with context
4 KV heads x 256 head_dim
```

The runtime agrees: the bridge logs `converted 16/64 target caches`.

Only those 16 layers cost memory per token. Affine quantization at
`group_size=64` adds `16 groups x 4 B = 64 B` of bf16 scales+biases per tensor
per token, independent of bit width — so 4-bit is really 4.5 effective bits and
metadata is 12.5% on top of payload, not negligible.

| KV format | per token | 32K | 128K | 262,144 |
|---|---|---|---|---|
| bf16 | 64 KiB | 2.00 GiB | 8.00 GiB | 16.00 GiB |
| 8-bit | 34 KiB | 1.06 GiB | 4.25 GiB | 8.50 GiB |
| **4-bit (Turbo4)** | **18 KiB** | 0.56 GiB | 2.25 GiB | **4.50 GiB** |

4.50 GiB at 262,144 is exact, not approximate — every factor is a power of two,
and the server logs precisely that figure when it reserves the arena.

Weights on disk: `EigenLabs--Qwen3.8-27B-4bit` 14.09 GiB (text-only),
`mlx-community--Qwen3.8-27B-4bit` 14.95 GiB (includes a vision tower),
DFlash2 draft 1.18 GiB, MTP head 0.22 GiB. Turbo4 at full 262K peaks around
25.5 GiB for the target alone — comfortable against the planner's 30.5 GiB
ceiling. KV8 at 262K is 29.5 GiB, which is not.

## Production configuration

Profile `mlx-vlm-qwen38-q4-turbo4-deep` (or `-turbo4` for a 2048 prefill):
serial decode, Turbo4 KV, compressed exact-prefix APC, single sequence.
Speculation is deliberately off — see below.

Client: `config/opencode.network-client-qwen38.json`, advertising 250,000 input
and 8,192 output against the 262,144 arena, with automatic compaction on.

### Measured

Serial Qwen3.8 Q4 + Turbo4 + compressed APC:

- short-prompt decode ~15.7 tok/s; ~4.0 tok/s at a 31K prefix
- 32K needle retrieval PASS
- 32K cached append: 4.55 s vs a cold prefill, 108.6x end-to-end
- 32K peak 22.04 GB, down from 30.87 GB before APC snapshots were kept
  compressed
- 4K case: 59.82 s cold -> 2.17 s warm (27.6x)

DFlash2 Q4 (experimental), from server logs:

- ~15.3-18.8 tok/s at short prompts, 60-74% draft acceptance
- 6.0-7.8 tok/s once the prompt reaches ~4K
- its own baseline AR control: 5.5 tok/s

That last line is the important one. DFlash2's "3x" is measured against
DFlash's own slow AR path; against the serial mlx-vlm path it is roughly
**parity**. On this hardware speculation has not yet beaten serial decode on a
coding workload, and prefix reuse is worth far more than either.

## Settings that look wrong and are not

**`--quantized-kv-start 1024`** — upstream defaults to 5000. Do not "fix" it.
Quantization is deferred while `prefill_length < quantized_kv_start`, so at
5000 every prompt under 5000 tokens would live in a plain float `BatchKVCache`.
Our compressed-APC bridge only has a compressed clone path for
`TurboQuantKVCache` and correctly delegates everything else upstream — which
means a **full float snapshot**, the exact blowup that produced the 30.87 GB
peak. It would also make the same prompt numerically different depending on
cache hit, and invalidates every on-disk APC snapshot, since the value is
folded into the cache namespace fingerprint.

**Symmetric K4/V4** — K is more quantization-sensitive than V, so K8/V4 is
normally preferable. It is not affordable at 250K on 32 GB, and it is not a
small change: `mlx_lm.models.base` passes a single `cache.bits` to *both* the
`Q·Kᵀ` and `P·V` matmuls, so asymmetric bits needs a replacement SDPA seam, not
just a new cache class. A cache class alone would dequantize V at K's width and
silently produce garbage. Worth testing at 128K, not here.

**MTP off** — measured on this machine: serial 15.69 tok/s vs MTP fixed-4 at
13.33 (implementation), 12.96 (debugging), 9.95 (tool call), adaptive 9.09. The
248,320-token vocabulary projection costs more per draft step than
verification saves, even at high acceptance. Neither local checkpoint even
ships MTP weights (0 `mtp`/`nextn` tensors despite `mtp_num_hidden_layers: 1`
in both configs); the head is a separate artifact.

**`--no-prefix-cache` on quantized profiles is forced, not chosen** —
`cache/codecs.py` serializes only `RecurrentRollbackCache`, `RotatingKVCache`
and `KVCache`. `QuantizedKVCache` is not a `KVCache` subclass, so it hits the
`else` and raises. Worse, publish is *not* gated on `quantize_kv_cache`
(restore is), and `validate_runtime_config` has no auto-disable rule, so
leaving the prefix cache on raises `TypeError` and fails the request. Observed
in `logs/mlx-dflash2-qwen38-q4.log`.

**Physical arena 262,208 vs logical context 262,144** — DFlash appends a whole
draft block to the target cache and trims only after verification, so peak
offset can exceed the context by up to `--verify-len-cap`. Equal sizes make
that a hard mid-stream `ValueError`. 64 slots cost ~1.1 MiB.

**`reasoning_effort` aliasing** — the template accepts only `xhigh`, `medium`,
`low` and calls `raise_exception` otherwise, so an OpenAI-standard `high` or
`minimal` fails the whole request. `LOCAL_LLM_QWEN_REASONING_ALIASES=1` maps
them. `medium` is the unsteered baseline (only `xhigh` and `low` inject
instruction text); the template's own default is `xhigh`, which spends most of
the output budget thinking here, so the profiles default to `medium`.

## The DFlash + TurboQuant query-rotation question

TurboQuant stores rotated keys, so attention must rotate the query too, and
that rotation lives *only* in a patched `scaled_dot_product_attention`.
DFlash's `engine/gqa_sdpa.py` captures that symbol by value at import time, and
TurboQuant's `patch_attention` repoints only modules under `mlx_lm.models.*` —
so DFlash's reference stays unrotating forever. All of that is true and
verified.

It is nonetheless **not a live bug**: a quantized cache never reaches the code
holding that stale reference. `_install_full_attention_gqa_hook` refuses to
route any `QuantizedKVCache` into it and defers to the model's own attention,
which *is* patched. Measured on a real hybrid model with a real Turbo4 cache:
`rotate_query` fires once per forward pass, `grouped_gqa_sdpa` is entered zero
times, and the cache offset advances normally.

The safety of the whole path rests on that single `isinstance` guard, so
`tests/test_dflash_turboquant_rotation.py` asserts it by execution — including
that the stale reference still exists, so the test fails loudly if the premise
changes rather than silently computing `Q·R(K)ᵀ`.

## Known limitations

- **262K is an allocation ceiling, not a validated depth.** Largest verified
  retrieval is 32K. A full cold 262K prefill has never completed here.
- **The baseline-fallback path can overrun the arena.** `engine/fallback.py`
  has no max-ctx guard and reuses the fixed pool. A prompt within
  `--verify-len-cap` of the arena that then generates will overflow mid-stream.
  The client-side 250,000 limit closes this for OpenCode traffic; a
  misconfigured client is still exposed.
- **DFlash2 + Turbo4 prefix reuse is implemented but unmeasured.**
  `mlx-dflash2-qwen38-q4-turbo4-cached` pairs compressed KV with exact prefix
  reuse at 128K, L1 only. It is correctness-tested but has **no throughput or
  retrieval numbers yet**, so serial + APC remains the production path until it
  is benchmarked against an append trace. L2 disk snapshots and generation
  sidecars are still unimplemented — the L2 schema stores one array per side and
  a write fails closed, and `sidecar_eligible` excludes quantized caches, which
  caps reuse at the cold-prompt frontier rather than following the conversation.
- `mlx-lm` is pinned `>=0.31.3` while we monkeypatch its internals. Pin it
  exactly and upgrade deliberately with a parity run.
- ~2.6 GiB of stale `.incomplete` downloads sit in
  `EigenLabs--Qwen3.8-27B-4bit/.cache`, reclaimable.

## Known-bad in the pinned DFlash revision

`serve.py` parses `repetition_penalty`, `presence_penalty`, `frequency_penalty`,
`xtc_probability`, `xtc_threshold`, `logit_bias`, `logprobs` and `top_logprobs`
from the request body, but `runtime.py` forwards only `temperature`, `top_p`,
`top_k` and `min_p`, and `stream_dflash_generate_impl` accepts only those four.
So on the DFlash path those parameters are **accepted and silently ignored** —
not rejected, not honoured. Only the experimental DFlash profiles are affected;
the serial mlx-vlm production path does not go through that engine. Any request
setting them should be routed to target-only AR before the DFlash server can be
called OpenCode-ready.

## Reproducing the environment

`uv.lock` is committed and pins `mlx-lm 0.31.3`, `mlx 0.32.0`,
`mlx-vlm 0.6.13`, `dflash-mlx 0.1.10+omlx.6`, `mlx-turboquant 0.0.2` and
`turboquant-mlx 0.3.0`. Install with `uv sync --frozen`. The `mlx-lm>=0.31.3`
declaration in `pyproject.toml` is looser than the lock, and this repo
monkeypatches mlx-lm internals, so an unconstrained rebuild can silently change
`QuantizedKVCache`, SDPA dispatch or cache layouts.

## Validating config changes

`opencode debug config` resolves a config against the installed binary and is
read-only. Use it before adopting configuration advice from anywhere: the
decoder ignores unknown keys silently, so a config using keys from a different
OpenCode generation resolves to **zero providers** while looking healthy.
