# MLX production profiles on the 32 GiB M1 Max — July 2026

This is the current authoritative guide for the two installed MLX models. Older
benchmark notes in this repository remain useful historical evidence, but their
native4/native8 recommendations predate the fixed Turbo4 arena and the
cross-request session cache.

## Final model limits

| Model | Architecture | Production fixed context | Turbo4 arena | Prompt speed | Decode | Result |
|---|---|---:|---:|---:|---:|---|
| cyberCyber99 Qwen3.6 35B-A3B OptiQ-4bit | MoE, 10 full-attention + 30 recurrent layers | **262,144** (249,803 proven) | 1.41 GiB | 433 tok/s at 3.3K; ~66 tok/s for the final 19K delta at 250K depth | **61.7 tok/s shallow; ~9.5 tok/s at 230K** | Recommended server |
| analogbox Qwopus3.6 35B-A3B Coder MLX 5-bit | MoE, 10 full-attention + 30 recurrent layers | **262,144 headless** (259,412 proven; 98,304 normal-safe) | 1.41 GiB | **270 tok/s at 48K; 57.6 tok/s for the final 19K delta at 259K** | **61.4 tok/s shallow; 8.0 tok/s at 259K** | Coding specialist; extreme headless profile |
| analogbox Qwen3.6 40B 4bit | dense, 24 full-attention + 72 recurrent layers | **131,072** | 3.38 GiB | **44.6 tok/s** at 4K | **9.89 tok/s** | Quality model, slower and shorter |

The 35B passed exact retrieval at 20K, 62K, 230K, and 249,803 tokens. After
the 249,803-token retrieval, an exact append reused 249,813 cached tokens and
answered in 2.52 seconds. This is the practical production result: a cold
long-context fill is expensive, while byte-stable follow-up turns reuse the
resident state instead of prefilling the whole conversation again.

The 40B passed exact retrieval at 4K and 8K. With a 1,024-token prefill step,
the 4K test took 89.94 seconds and MLX reported a 28.06 GiB peak. A 1K
append-only cache test was 19.82 seconds cold and 1.43 seconds warm.

The 5-bit Qwopus coder passed exact retrieval at 19K, 86,509, 240,173, and
259,412 tokens. The original 1,024-token-chunk run toward 115K drove live Metal
use to about 30.2 GiB and available memory to 386 MiB, so it was deliberately
aborted. In a closed-lid loginwindow-only session, a 30,720 MiB Metal ceiling
plus 512/256/128 adaptive chunks kept long-fill Metal use near 26.2–26.9 GiB
with roughly 2.4–3.1 GiB available. The final request reused 240,183 tokens,
processed only a 19,229-token delta in 335.97 seconds, and returned the exact
needle. The normal/open-lid planner still recommends 98,304; 262K requires the
explicit **Extreme Headless Context** control and does not auto-start at boot.

## Why 260K fits the MoE but not the dense model

Only full-attention layers have a token-growing KV cache. Turbo4 group 64 uses
4.5 physical bits/value after scale and bias metadata.

```text
KV bytes = full_attention_layers × 2 × KV_heads × head_dim
           × context_tokens × effective_bits / 8
```

At 262,144 tokens:

- 35B: 10 × 2 × 2 × 256 → **1.41 GiB**
- 40B: 24 × 2 × 4 × 256 → **6.75 GiB**

The 35B planner includes its measured long-fill transient allowance and reports
a **27.84 GiB** peak estimate at 262K. This is deliberately higher than its
23.94 GiB shallow decode peak.

The calibrated 40B peak estimate is 20.475 GiB weights + 6.75 GiB KV +
216 MiB recurrent checkpoint + 5.8 GiB graph/workspace margin = **33.24 GiB**.
At 196K it is still 31.55 GiB. The manager now rejects those profiles before
they can pressure macOS into a freeze. Its measured-safe 128K estimate is
29.86 GiB.

`GET /api/mlx/memory-plan` returns this calculation for the selected model.

## What the local implementation changes

Stock MLX-LM 0.31.3's `--max-kv-size` is a rotating/sliding cache, not a full
llama.cpp-style reservation. Its current LRU prompt cache can checkpoint hybrid
models, but it deep-copies a cache when retrieving it. Deep-copying a
multi-GiB, fully preallocated 262K arena is exactly what this 32 GiB system must
avoid. The local `FixedQuantizedKVCache` and `FixedTurbo4KVCache` allocate every
packed K/V buffer during model load and reject prompt + output overflow before
inference.

Qwen3.5/3.6 is hybrid. Its recurrent `ArraysCache` cannot be trimmed like a
normal KV cache, which is why generic prefix-cache patches do not solve repeated
prefill. `SessionPromptCache` keeps one arena in place:

- exact append moves the KV offset and prefills only new tokens;
- two small recurrent-state checkpoints permit safe branch rewind;
- a checkpoint is captured at the prompt/first-generated-token boundary, before
  the assistant/tool suffix can be re-rendered differently by the client;
- checkpoint token prefixes are validated so an abandoned branch cannot restore
  stale recurrent state;
- only one session/model request may own the arena at a time, avoiding a
  multi-GiB deep copy.

The Qwen no-thinking chat template is patched so historical assistant messages
serialize byte-for-byte identically. Otherwise an unchanged OpenCode
conversation can miss the cache simply because an old assistant turn gains or
loses an empty `<think>` block.

OpenCode serializes historical tool-call `function.arguments` as a JSON object,
while MLX-LM 0.31.3 assumes it is always a JSON string and calls `json.loads`
unconditionally. Fixed-capacity validation also tokenizes before inference, so
the stock in-place normalization was run twice. The local server now accepts
either representation idempotently and tokenizes an isolated request copy for
the overflow check. Both ordinary and streaming tool-history requests are
covered by regression tests.

The response formatter also suppresses exact duplicate tool name/argument
pairs within one assistant response, including across streaming chunks. This
prevents a looping local checkpoint from making OpenCode execute the same
`Read` operation dozens of times; distinct paths and distinct tool arguments
remain untouched.

Cross-turn duplicates need a different guard because every tool result starts
a new HTTP request. Three identical consecutive tool turns now return an
immediate normal `stop` response without running inference. This permits one
legitimate retry while bounding an invalid `Edit /` loop.

The prompt-boundary checkpoint is essential for tool traffic. A July 24
incident showed a rendered tool-history suffix diverging after a 221K cached
prefix. With only the completed-turn checkpoint retained, the hybrid recurrent
state could not rewind and MLX replayed all 233,813 tokens. OpenCode's 40-minute
overall timeout expired at token 230,272 and the server observed a broken pipe.
The corrected three-turn test reused 1,692/1,756 tokens on the first tool
round-trip and 1,757/1,821 after deliberately changing historical tool
arguments; only 64 suffix tokens replayed. All boundary offsets are checked
against the KV position before a recurrent snapshot is accepted.

The server now exposes `GET /metrics` with prompt/generation speed, cached and
replayed tokens, peak MLX memory, checkpoint events, reprocesses, errors, and
request counters. The Python dashboard consumes it directly.

The Qwopus conversion exposed a Transformers warning for the inherited Mistral
tokenizer regex. The server and all context/cache probes now load Qwen
tokenizers with `fix_mistral_regex=True`; otherwise token counts and byte-stable
cache keys can be wrong. The manager also verifies every shard from
`model.safetensors.index.json`, hides partial downloads, reports only
materialized model bytes instead of double-counting Hugging Face cache
metadata, and previews the selected model/context memory plan before startup.

Long-fill testing found that one static prefill chunk is unsafe even when the
fixed KV arena fits. The server therefore adapts by total target depth:

- below 64K: use the configured step;
- 64K–100K: cap at 512;
- 100K–220K: cap at 256;
- above 220K: cap at 128.

The selected step and target depth are exported in metrics and the dashboard.
With the adaptive guard, the append-only probe completed stages at 57,644,
115,276, 172,908, and 230,541 total prompt tokens with zero reprocesses. It
then extended to 249,803 and retrieved the exact needle. MLX reported a
27.84 GiB peak. Measured growth:

| Total prompt depth | New-token prefill time | Result |
|---:|---:|---|
| 57,644 | 202.85 s | ACK |
| 115,276 | 397.06 s | ACK, 57,648 cached |
| 172,908 | 569.87 s | ACK, 115,280 cached |
| 230,541 | 744.01 s | Exact needle, 172,912 cached |
| 249,803 | 291.33 s | Exact needle, 230,574 cached |
| 249,833 | 2.52 s | Exact append, 249,813 cached |

The heavier Qwopus 5-bit headless run used a configured 512-token step and the
stricter adaptive caps above:

| Total prompt depth | New-token prefill time | Cached prefix | Result |
|---:|---:|---:|---|
| 48,044 | 177.99 s | 0 | ACK |
| 96,076 | 295.33 s | 48,048 | ACK |
| 144,108 | 435.18 s | 96,080 | ACK |
| 192,140 | 553.71 s | 144,112 | ACK |
| 240,173 | 724.57 s | 192,144 | Exact needle |
| 259,412 | 335.97 s | 240,183 | Exact needle |

Relevant upstream context:

- [MLX-LM README](https://github.com/ml-explore/mlx-lm) documents rotating
  `--max-kv-size`, prompt caches, and the Apple wired-limit recommendation.
- [MLX-LM hybrid-cache issue 980](https://github.com/ml-explore/mlx-lm/issues/980)
  explains why recurrent state cannot simply be trimmed.
- Merged [server-cache PR 911](https://github.com/ml-explore/mlx-lm/pull/911)
  and [batch-generation PR 1072](https://github.com/ml-explore/mlx-lm/pull/1072)
  add upstream checkpoints and hybrid reuse. The local single-arena fast path
  is retained because upstream retrieval still deep-copies its cache.
- [Qwen issue 1826](https://github.com/QwenLM/Qwen3/issues/1826) tracks the
  no-thinking historical-message instability.

## TurboQuant findings

Turbo4 here is the pinned
[`pythongiant/mlx_turboquant`](https://github.com/pythongiant/mlx_turboquant)
rotated-key cache. It preserves recurrent layers and uses MLX packed
quantized-matmul attention; it is not a fake full-cache dequantization path.
Rotation improves low-bit K quality but does not use fewer bytes than native4.

Group 128 reduced the 35B arena from 1.41 to 1.33 GiB and measured peak from
23.94 to 23.79 GiB. It passed a 20K needle, but was not faster: 49.94 versus
49.72 seconds, and 61.42 versus 61.73 decode tok/s. Group 64 therefore remains
the production quality default.

Turbo3's packed PolarQuant decode kernel works experimentally, but multi-token
prefill still needs a larger temporary path and fixed preallocation is not
implemented. It is not used for production long context.

The two upstream MLX TurboQuant proposals remain experimental:
[PR 1067](https://github.com/ml-explore/mlx-lm/pull/1067) has a fused 3-bit
path; [PR 1144](https://github.com/ml-explore/mlx-lm/pull/1144) dequantizes for
standard attention. Neither is an upstream hybrid-Qwen fixed server solution.
The community [TurboQuant MLX discussion](https://www.reddit.com/r/LocalLLaMA/comments/1s5vhf6/turboquant_on_mlx_46x_kv_cache_compression_with/)
was useful as a lead, but all production decisions above come from local
retrieval and memory tests.

## Optimizations deliberately not enabled

- MTP: [MLX-LM PR 990](https://github.com/ml-explore/mlx-lm/pull/990) reports
  useful dense-model speedups but only a few percent on this small-active MoE.
  Both installed conversions omit `mtp.*` weights, so enabling it is impossible
  without a compatible checkpoint.
- DFlash is now available as a separate experimental profile for the 35B MoE.
  It uses the matched `z-lab/Qwen3.6-35B-A3B-DFlash` draft and the pinned
  `bstnxbt/dflash-mlx` runtime, which includes hybrid-Qwen recurrent rollback
  and cross-turn prefix snapshots. It is deliberately not the default until
  load, acceptance, speed, memory, and long-context retrieval are measured on
  this M1 Max. See `docs/DFLASH_HAUHAU35_SETUP.md`.
- Turbo3 fixed 260K: decode compression does not remove its long-prefill
  temporary-memory risk.
- MLX main/nightly: installed MLX 0.32.0 and MLX-LM 0.31.3 match the latest
  relevant stable code; no current main-branch change fixes these hybrid-server
  constraints.

## OpenCode configuration that preserves reuse

Use `config/opencode.network-client.json` for the long-context 35B,
`config/opencode.network-client-qwopus35.json` for the 5-bit coder, or
`config/opencode.network-client-40b.json` for the 40B.

The limits reserve output inside the fixed arena:

- 35B: 253,952 input + 8,192 output = 262,144
- Qwopus coder headless: 253,952 input + 8,192 output = 262,144
- 40B: 122,880 input + 8,192 output = 131,072

The client profiles also:

- use long request and chunk timeouts because MLX does not stream during cold
  prefill; the Qwopus client profile uses 60 minutes as a cold-recovery
  fallback, while boundary reuse keeps normal tool turns far below it;
- disable automatic compaction and pruning, because rewriting old messages
  invalidates the byte-stable prefix;
- disable the background title agent, whose unrelated small-model request would
  replace the server's one live cache slot;
- deny parallel subagent requests for the same reason;
- keep normal MCP/tool calls enabled: they run on the OpenCode client, and their
  results become an ordinary appended conversation turn;
- select Build as the stable primary agent. Changing agent/system prompt during
  a session forces a full reprocess.

OpenCode's official [provider documentation](https://opencode.ai/docs/providers)
defines custom OpenAI-compatible models and separate input/output limits. Its
[configuration documentation](https://opencode.ai/docs/config/) documents
timeouts and compaction controls.

## Production settings

```text
35B MoE:
  Turbo4, group 64, fixed 262144, prefill 2048
  session reuse on, recurrent checkpoints 2, concurrency 1

Qwopus 35B coder 5-bit:
  Turbo4, group 64, fixed 262144, prefill 512, extreme headless enabled
  session reuse on, recurrent checkpoints 2, concurrency 1

40B dense:
  Turbo4, group 64, fixed 131072, prefill 1024
  session reuse on, recurrent checkpoints 2, concurrency 1

M1 Max:
  iogpu.wired_limit_mb=30720 for the Qwopus extreme headless profile
  manager + caffeinate enabled; trusted LAN only
```

The model API is `http://192.168.1.117:8098/v1`; dashboard and runtime control
are at `http://192.168.1.117:8090`.
