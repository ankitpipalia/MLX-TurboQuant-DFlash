# MLX optimization on the 32 GiB M1 Max

Verified environment: macOS, MLX 0.32.0, MLX-LM 0.31.3, one inference request
at a time, and a runtime-only 28 GiB Metal wired-memory ceiling.

## What caused multi-minute agent replies

Decode speed and response latency are different measurements. OpenCode sends a
large system prompt, tool schemas, MCP descriptions, conversation history, and
the new user message. A short visible message such as `hello` can therefore be
a large cold prefill.

Qwen3.x's published no-thinking template also rendered a live assistant prefix
with an empty `<think>` block, then removed that block when the same answer
became history. The changed token prefix prevented exact prompt-cache reuse and
could force the whole agent history to be processed again. The local MLX wrapper
now patches this template in memory at model load. Keep tool definitions,
ordering, system metadata, dates, and message IDs byte-stable for the same
reason.

## Current safe baseline

- Attention KV cache: 4-bit, group size 64. Recurrent/DeltaNet state is left in
  its native representation because it cannot be converted to an ordinary
  quantized attention cache.
- Decode concurrency: 1; prompt concurrency: 1. More concurrency increases the
  working set and is counterproductive for a single-user 32 GiB machine.
- Prefill step: 2048. Test 4096 for cold-prefill speed, but retain it only if
  the peak-memory probe passes; use 1024 when a model is close to the ceiling.
- In-memory prompt cache: one sequence, 2048 MiB. Raising it may retain a longer
  prefix but also duplicates active cache state and can cause an OOM.
- Network-client context: 65,536. The current Dawncr0w 35B model passed an 80K
  frontier probe, but that run used about 27.7 GiB of Metal memory and reached
  Serious thermal pressure. A model's 262,144 native position limit is not a
  memory, speed, or retrieval guarantee.
- Thinking: disabled for interactive agent work unless the task needs it.
- Metal wired-memory ceiling: 28 GiB at runtime only. Wired Metal memory cannot
  be rescued by swap, so leave several GiB for macOS and inference workspace.

## Benchmark protocol

For every model and prefill setting, record these separately:

1. Cold prompt tokens/second and time to first token.
2. Decode tokens/second after a warm-up request.
3. A second-turn prefix-cache request and its `cached_tokens` count.
4. Peak Metal memory, macOS available memory, swap, temperature, and thermal
   state.
5. Exact needle retrieval at progressively larger contexts.

Do not call an allocation-only start a successful context test. Stop the ladder
at the first retrieval failure, OOM, swap growth, or dangerously small Metal
headroom.

## Dawncr0w 35B OptiQ 6.12 bpw results

Model path:
`dawncr0w--Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-OptiQ-6bpw-MLX`.
These results use 4-bit attention KV, group size 64, prefill step 2048, and one
request at a time.

| Test | Result | Time / rate | Operational conclusion |
| --- | --- | --- | --- |
| Steady decode, 256 tokens | Pass | 49.34 tok/s | Excellent interactive decode for this MoE |
| 8K two-turn cache probe | Pass | 14.727 s cold, 0.366 s warm, 40.2x faster | 6,573 prefix tokens reused after the template fix |
| 32K needle (31,169 prompt tokens) | Pass | 89.278 s, about 349 prompt tok/s | Comfortable |
| 64K needle (62,291 prompt tokens) | Pass | 228.186 s, about 273 prompt tok/s | Recommended maximum for daily agent use |
| 80K needle (77,871 prompt tokens) | Pass | 316.545 s, about 246 prompt tok/s | Proven frontier only; about 27.7 GiB Metal and Serious thermal pressure |

The 4096 prefill-step A/B took 93.714 seconds for the identical 31,169-token
32K probe versus 89.278 seconds at 2048, and its sampled Metal use was roughly
1.6 GiB higher. The manager therefore keeps 2048 as this machine's default.

The model's mixed 4/8-bit weights occupy about 18.64 GiB on disk and about
19.1 GiB resident after load. Idle after loading, the machine retains roughly
8--10 GiB of available system memory. Unified memory is one physical pool, so
"system RAM" and "GPU RAM" are views of the same 32 GiB rather than additive
capacities.

## Engine choice

Stock MLX-LM is the controlled baseline and has upstream Qwen hybrid-cache fixes,
but its own documentation describes the HTTP server as non-production. oMLX
adds hot-RAM plus SSD prefix blocks; Rapid-MLX advertises radix caching and
DeltaNet snapshots. Those designs target divergent agent prefixes better than a
single in-memory LRU entry. Benchmark an alternative on the same model and
prompt before adopting it; do not compare vendor headline numbers from newer
M-series hardware to this M1 Max.

The local TurboQuant modes, source audit, patches, and measured limitations are
documented in [MLX_TURBOQUANT.md](MLX_TURBOQUANT.md).

For OpenCode specifically, keep one session's stable prefix intact, avoid
changing tool/MCP definitions during a session, and start a new/compacted
session before routinely approaching 64K. A cache cannot help the first cold
request, and intentional compaction or any early-prefix change necessarily
forces substantial re-prefill. Disable MCP servers that a session does not need:
their schemas consume prompt tokens even when the visible user message is only
`hello`.

## Primary references

- MLX-LM releases and cache fixes: <https://github.com/ml-explore/mlx-lm/releases>
- Hybrid prefix-cache analysis: <https://github.com/ml-explore/mlx-lm/issues/980>
- Qwen no-thinking template instability: <https://github.com/QwenLM/Qwen3/issues/1826>
- MLX-LM large-model and wired-memory guidance: <https://github.com/ml-explore/mlx-lm>
- MLX-LM HTTP server scope: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md>
- oMLX tiered cache: <https://github.com/jundot/omlx>
- Rapid-MLX cache design: <https://github.com/raullenchai/Rapid-MLX>
- OpenCode MCP context warning: <https://opencode.ai/docs/mcp-servers>
