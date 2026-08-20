# MLX fixed-KV research and M1 Max validation (July 2026)

## Result

Stock MLX-LM, MLX's Metal memory controls, and the current TurboQuant forks do
not reserve a llama.cpp-style full-context KV arena. They all allocate KV
storage lazily. `--max-kv-size` is a bounded rotating/sliding cache, not an
up-front allocation for the full prompt.

This workbench now adds `--preallocate-kv-size` to
`local_llm_control.mlx_quant_server`. It:

- allocates packed K/V, scales, and biases for every full-attention layer while
  the model loads;
- retains and resets one cache slot instead of allocating it again;
- disables the MLX LRU prompt cache so the fixed arena cannot be deep-copied;
- rejects `prompt tokens + requested output tokens` above the fixed capacity
  before inference;
- preallocates and resets Qwen's recurrent `ArraysCache` state;
- works with native 4/8-bit KV and the experimental rotated Turbo4 cache.

Turbo3 uses a different fused packed-cache implementation and remains dynamic.

## Recommended profile for this machine and model

Machine: 32 GiB Apple M1 Max
Model: `analogbox/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-4bit`

Use:

```text
KV mode: Turbo4
Fixed KV context: 131072
Prefill step: 2048
Decode concurrency: 1
Prompt concurrency: 1
Prompt-cache size/bytes: 0/0
Session reuse: on
Session checkpoints: 1
Metal wired ceiling: 29696 MiB
```

The Python manager stores this as the default MLX configuration, and the tested
configuration is saved as `mlx-40b-turbo4-fixed128k`. Select a different fixed
capacity in the MLX Engine panel if needed.

For this 96-layer hybrid Qwen checkpoint, only 24 layers have ordinary KV.
At 4-bit/group-size 64, their fixed arena costs exactly 27 KiB per token:

| Fixed capacity | Reserved full-attention KV | Assessment |
|---:|---:|---|
| 65,536 | 1.69 GiB | Comfortable |
| 131,072 | 3.38 GiB | Works at 29 GiB Metal ceiling; high memory pressure |
| 196,608 | 5.06 GiB | Risky; little workspace/system margin |
| 262,144 | 6.75 GiB | Likely Metal OOM on this 32 GiB machine |

The recurrent state adds approximately 220 MiB at startup. It is fixed-size
rather than proportional to context. Native 8-bit KV is approximately twice
the table above.

Turbo4 uses the same 4-bit packed size as native4. It does not create more
context capacity, but its rotation materially improved this checkpoint's
low-bit fidelity: fixed native4 failed a 4K retrieval once, while fixed Turbo4
passed both 4K and 8K probes. Turbo3 is smaller, but its current prefill path and
fixed-capacity behavior are not stable enough to make it the default.

## Tests performed

- Native fixed cache and upstream dynamic quantized cache produced identical
  logits on the real model (`max_abs_diff = 0.0`) for the same prompt.
- Unit validation showed the fixed cache byte count remained unchanged across
  multiple updates.
- A real 32K arena reserved 0.84 GiB at model load.
- A real 128K arena reserved 3.38 GiB at model load and completed generation
  without Metal OOM.
- The 72 recurrent layers reserved an additional 220 MiB at model load; two
  back-to-back requests verified that both KV offsets and recurrent state reset.
- Native4 fixed-128K: 128 output tokens at 9.25 tok/s overall.
- Turbo4 fixed-32K: 128 output tokens at 9.42 tok/s overall.
- Turbo4 fixed-128K with one recurrent checkpoint: a forced 128-token response
  completed at 7.21 tok/s end-to-end.
- Turbo4 fixed-128K session reuse: 835-token cold turn in 19.98 s; exact-prefix
  follow-up reused 837 tokens and completed in 1.48 s (**13.5x faster**).
- Turbo4 fixed-128K 4K needle: exact retrieval at 3,923 prompt tokens in
  90.16 s.
- Turbo4 fixed-128K 8K needle: exact retrieval at 7,818 prompt tokens in
  182.38 s. The same checkpoint had previously failed an 8K native4 probe.
- During the 8K run, allocation peaked near 29.87 GiB, macOS available memory
  bottomed near 768 MiB, swap stayed flat during the measured prefill, GPU power
  reached ~37 W, and temperature remained nominal at ~69 C.
- Turbo4 fixed-32K 4K needle: 3,923 prompt tokens, exact
  `BLUE-OTTER-7741` retrieval, 87.20 seconds total, approximately 45.9 prompt
  tok/s from server progress timestamps.
- Native4 with prefill step 4,096 reached approximately 71.2 prompt tok/s on
  the same 4K prompt, but produced degenerate output and failed retrieval.
  Keep 2,048 as the correctness-first default for this checkpoint.
- Capacity overflow: HTTP 404 JSON with the exact prompt/output/reserved token
  counts; the server remained healthy.
- Back-to-back Turbo4 requests returned independent answers after validating
  the fixed-slot reset path.
- Capacity overflow under fixed Turbo4 returns controlled JSON before inference.
- Repository test suite: 47 passed.

The 128K *allocation*, short generation, prefix reuse, and retrieval through 8K
were tested. A full 128K semantic needle test was not run: at roughly 43 prompt
tok/s it would take about 50 minutes before accounting for the falling rate of
full attention at depth. Allocation capacity is not proof that the checkpoint
uses every advertised token reliably.

## Upstream status and why this is local

- MLX-LM v0.31.3 and MLX v0.32.0 were the latest stable installed versions
  during this work.
- MLX-LM's documented `--max-kv-size` is a rotating fixed-size cache, which
  bounds retained history and can reduce quality. It is not whole-context
  reservation.
- The April 2026 batch-generator max-KV work bounds caches but still does not
  expose or implement llama.cpp-style full preallocation in the HTTP server.
- TurboQuant PRs 1067 and 1144 were still open. The tested community
  implementations expand storage in chunks.
- Hybrid prefix-cache reuse remains an open upstream problem. Locally,
  `SessionPromptCache` now keeps the fixed KV arena in place, reuses exact
  append-only prefixes, and checkpoints the small recurrent state. Checkpoints
  are token-prefix validated so abandoned conversation branches cannot restore
  stale recurrent state. This is a single-client/single-session optimization,
  not a general concurrent prompt cache.
- `mx.set_memory_limit`, `mx.set_cache_limit`, and `mx.set_wired_limit` set
  ceilings/retention policy. They do not reserve usable KV tensors.

References:

- <https://github.com/ml-explore/mlx-lm>
- <https://github.com/ml-explore/mlx-lm/pull/1106>
- <https://github.com/ml-explore/mlx-lm/issues/980>
- <https://github.com/ml-explore/mlx-lm/pull/1067>
- <https://github.com/ml-explore/mlx-lm/pull/1144>
- <https://github.com/arozanov/turboquant-mlx>
- <https://github.com/pythongiant/mlx_turboquant>
- <https://github.com/sharpner/turboquant-mlx>
- <https://www.reddit.com/r/LocalLLaMA/comments/1s5vhf6/>

## Operational notes

The fixed arena deliberately supports one active sequence. That matches this
machine's memory budget and the manager's concurrency settings. Exact append-only
turns reuse their full prefix; unrelated conversations replace the one cache
slot. Dynamic mode remains available when multiple retained prompts matter more
than a flat, known allocation.

The OpenCode MLX provider is configured with context 131,072 and output 8,192.
The server enforces prompt plus requested output against 131,072; clients should
keep input below approximately 122,880 tokens when requesting the full
8,192-token output.

This is not a comfortable 27 GiB normal-GUI profile. At the 27,648 MiB ceiling,
the first short request exhausted reported Metal headroom. The validated setup
uses the runtime-only 29,696 MiB ceiling and leaves very little system margin
during prefill. Do not run another memory-heavy application alongside it.
