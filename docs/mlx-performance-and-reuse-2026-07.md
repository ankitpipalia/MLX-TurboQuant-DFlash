# MLX performance, context reuse, and agent-serving findings (July 2026)

> Historical native8-focused round. Turbo4 fixed-arena validation, deeper
> retrieval results, live metrics, and final OpenCode settings are in
> `MLX_PRODUCTION_M1_MAX_2026-07.md`.

This documents the second round of MLX work: cross-request context reuse,
KV-quantization quality, realistic context ceilings, and the settings that make
`mlx_lm.server` practical as a coding-agent backend for one opencode client over
the LAN. It builds on `mlx-fixed-kv-research-2026-07.md` (fixed KV arena).

## The headline problem and the fix

opencode resends the **entire conversation** every turn and has no client-side
compaction ([opencode #15298](https://github.com/anomalyco/opencode/issues/15298)),
and MLX has no early-streaming prefill. Time-to-first-token grows at least with
input length, and the full-attention layers make the effective prompt rate fall
at greater depth. Without prefix reuse, every turn re-pays that. Re-prefill
avoidance is the whole game.

Stock `mlx_lm.server` (0.31.2+) does checkpoint hybrid caches, but our
fixed-arena mode disabled the prompt cache, and stock reuse `deepcopy`s the whole
KV per request — untenable at multi-GiB caches. So we added
`local_llm_control.mlx_session_cache.SessionPromptCache`:

- **One slot, in place, no copy.** The full-attention KV arena is a per-token
  store, so reusing tokens `[0:k]` is just moving the integer offset — the
  quantized data already sits in the fixed arena.
- **Append-only fast path.** When turn N+1's prompt extends the whole of turn N
  (the normal agent pattern), the live recurrent state is already correct at its
  end position; nothing is snapshotted or replayed — only the new tokens prefill.
- **Checkpoint rewind on divergence.** The recurrent GatedDeltaNet state cannot
  be rewound to an arbitrary token, so a small ring of recurrent-state snapshots
  (default 2, like llama.cpp `--ctx-checkpoints`) is kept at turn boundaries; on
  divergence we restore the nearest checkpoint ≤ the common prefix and replay
  only from there.
- **Branch-safe checkpoints.** A checkpoint is restored only when its stored
  tokens are an exact prefix of the new request. Checkpoints from an abandoned
  conversation branch are removed, preventing a same-position snapshot with the
  wrong recurrent state from being restored.

Enabled with `--session-reuse` (default on for fixed-arena MLX profiles).

A latent bug was fixed alongside: `FixedQuantizedKVCache` inherited `merge` from
`QuantizedKVCache`, so the server treated it as batchable and routed around the
single-slot path (and the capacity guard). It now raises on `merge` so the
sequential path — required for a shared single arena — is actually selected.

## KV quantization: keep K at 8-bit

Community measurement is unambiguous: **native 4-bit K quantization collapses
greedy-decode quality** (perplexity 31 vs ~3 for fp16 on small Qwen), while V is
tolerant at 4-bit
([arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx),
[pythongiant/mlx_turboquant](https://github.com/pythongiant/mlx_turboquant)).
KV quantization also costs ~0% decode speed on Apple Silicon
([mlx #3134](https://github.com/ml-explore/mlx/discussions/3134)). The default is
therefore **native8** (K and V at 8-bit). These hybrids carry few full-attention
layers, so 8-bit KV stays cheap.

## Model choice and the real context ceiling

Per-token full-attention KV cost is set by the number of attention layers, not
model size:

| Model | Attention layers | 8-bit KV @262K | Fits 262K on 32 GiB? |
|---|---:|---:|---|
| **OptiQ 35B-A3B** (18.2 GB) | 10 | ~2.7 GiB | **Yes** — 18.2 + 2.7 ≈ 21 GiB, real workspace left |
| analogbox 40B (21 GB) | 24 | ~6.4 GiB @**131K** (measured) | No — ~28 GiB before workspace → Metal OOM (observed) |

The 40B was empirically confirmed to OOM: native8/131K reserves 6.38 GiB KV, so
21 + 6.4 + snapshots exceeds the 27 GiB wired ceiling before any prefill
workspace. **The OptiQ 35B is the correct model for large context on this
machine.**

Two hard caveats remain even on the 35B:

- **Upstream prefill OOM** at ~176K on this exact model class, driven by transient
  Metal command-buffer workspace, not KV size
  ([mlx-lm #1480](https://github.com/ml-explore/mlx-lm/issues/1480)). Allocating a
  262K arena is fine; *filling* it in one cold prefill may fail. Mitigation:
  smaller `--prefill-step-size` for very long prompts.
- A **reserved 262K arena is not proof of a successful 262K cold prefill**.
  The largest retrieval validation in this round is 19K, and the earlier 31K
  timing already showed substantial prompt-rate degradation. Do not quote an
  exact 260K cold-prefill time until a full fill completes. The practical win is
  a large arena that grows over append-only turns, plus reuse making follow-ups
  delta-only.

## Reproduced measurements

These results were rerun against the live server after the branch-safety fix:

| Test | Result |
|---|---:|
| Fixed native8 arena | 262,144 tokens, 2.66 GiB, 10 attention caches |
| 4,019-token cold prompt | 9.26 s end-to-end (~434 prompt tok/s) |
| 4,021-token exact-prefix follow-up | 0.36 s, 4,021 cached tokens, **25.7×** faster |
| Shallow 256-token decode | **52.45 tok/s** |
| Needle retrieval | **PASS**, exact key at 19,047 prompt tokens in 49.55 s |
| Test suite | **46 passed** |

The previously observed 33.9× reuse and 46 tok/s decode figures are credible,
but they are workload-dependent rather than fixed machine constants. Prompt
processing is similarly depth-dependent: the server log shows about 434 tok/s
end-to-end at 4K, about 393 tok/s at 19K, and only about 166 tok/s for an earlier
31K cold run.

## Recommended configuration

Machine: 32 GiB M1 Max, wired limit **27648 MiB**.
Model: OptiQ 35B-A3B (`cyberCyber99/…-OptiQ-4bit-MLX`).

```text
Engine: mlx-lm
KV mode: native8         # K at 8-bit — do not use native4 for agents
Fixed KV context: 262144 # ~2.7 GiB KV; allocation-safe on the 35B
Prefill step: 2048       # multiple of 32; drop to 512 if long prefills OOM
Session reuse: on        # delta-only follow-up prompts
Session checkpoints: 2
```

opencode client (`config/opencode.network-client.json`, provider `local-mlx`):

- `limit.context: 200000` leaves room below the reserved 262K capacity, but it
  does **not** guarantee that a cold 200K resend survives the reported ~176K
  transient-workspace failure. Build long sessions through exact-prefix reuse;
  after a server restart, a very large existing conversation may need
  compaction or a smaller `--prefill-step-size` before it can be restored.
- `timeout` and `chunkTimeout` set high (≥30 min) — opencode's 2-minute SSE
  default fires during long prefills ([opencode #17307](https://github.com/anomalyco/opencode/issues/17307)).
- **Do not switch agent/mode mid-session** (Plan↔Build rewrites the system prompt
  and forces a full re-prefill), and keep tool lists static. Prefix byte-stability
  is what makes reuse hit; the server already stabilises the Qwen no-thinking
  template for the same reason.

## Speculative decoding (evaluated, deferred)

`mlx_lm.server` supports `--draft-model`, and Qwen3.6 MTP drafters exist
(`mlx-community/Qwen3.6-35B-A3B-MTP-4bit`). But MoE targets with ~3B active params
see limited benefit (decode is already cheap per token), and rejected-token
rollback interacts badly with non-rewindable recurrent state. Left off by default;
revisit with a greedy-exactness A/B if decode becomes the bottleneck.

## Reliability fix

The manager now waits for the runtime port to be bindable before starting, so a
stop→start restart cannot race the MLX server's lingering socket (it does not set
SO_REUSEADDR) and fail with "Address already in use".
