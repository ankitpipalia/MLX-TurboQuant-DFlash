# M1 Max 32 GiB benchmark — 2026-07-14

## Test system

- MacBook Pro 18,2; Apple M1 Max, 24 GPU cores, 32 GiB unified memory
- macOS 26.5.2 in normal GUI mode, AC power / High Power mode
- Chrome and Maccy closed; model directory excluded from Spotlight indexing
- TurboQuant llama.cpp fork commit `4503343ffc05c09f6b50c309c8ecbabb49c66ea2`
- Model: `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`
- GGUF: Q4_K_M, 21,155,768,832 bytes, SHA-256
  `bbef58c37ce88820be9d98b6437f1cf4bac890c947bd55fc7b68e22098574231`

All llama.cpp trials use Metal, all GPU layers, no CPU MoE, Flash Attention,
one slot, four CPU threads, batch 2048, micro-batch 512, and `--no-host`.

## llama.cpp throughput

| KV configuration | 512-token prefill | 8K prefill | short decode |
|---|---:|---:|---:|
| q8_0 / q8_0 | 616.4 tok/s | 502.6 tok/s | 49.8 tok/s |
| q8_0 / turbo4 | 613.7 tok/s | 500.2 tok/s | 46.0 tok/s |
| turbo4 / turbo4, true symmetric | 600.7 tok/s | 479.7 tok/s | 43.3 tok/s |
| turbo3 / turbo3, true symmetric | 600.8 tok/s | 481.8 tok/s | 41.7 tok/s |

The current fork automatically upgrades a requested turbo K cache to q8_0 for
this model's 8:1 GQA ratio. True symmetric rows explicitly set
`TURBO_AUTO_ASYMMETRIC=0`; without that environment variable they are actually
q8_0/turbo configurations.

## Perplexity check

WikiText, 10 chunks at context 512:

| KV configuration | PPL | Relative to q8/q8 |
|---|---:|---:|
| q8_0 / q8_0 | 7.2097 | baseline |
| q8_0 / turbo4 | 7.2251 | +0.21% |
| turbo4 / turbo4, true symmetric | 7.2504 | +0.56% |
| turbo3 / turbo3, true symmetric | 7.2521 | +0.59% |

This small PPL run is a smoke test, not a full model-quality evaluation.

## Context capacity and retrieval

| Profile | Allocated context | Result | Process RSS | macOS available after load |
|---|---:|---|---:|---:|
| q8_0/turbo4 native | 262,144 | loaded and served | 21.84 GiB | 2.94 GiB |
| turbo3/turbo3 + YaRN | 350,208 | loaded and served | 21.26 GiB | 3.39 GiB |
| turbo3/turbo3 + YaRN | 395,264 | loaded and served | 21.42 GiB | 3.13 GiB |

No context-capacity trial used swap. The 350K and 395K profiles disable the
fork's automatic K upgrade so that they reproduce the old A5000 symmetric
Turbo3 strategy.

Measured q8_0/turbo4 probes:

- 31,167-token cold prefill: 335.8 tok/s; long-context decode 14.7 tok/s.
- Repeated 31K prompt: restored a 30,651-token recurrent checkpoint, processed
  only 518 tokens, recovered `BLUE-OTTER-7741`, and completed in 3.3 seconds.
- 62,291-token cold prefill: 223.8 tok/s; decode 9.37 tok/s; needle passed;
  total 279.5 seconds.

The model can allocate 395K, but that is not the same as proving useful 395K
quality. Its native training context is 262,144, and long-context prefill and
decode become the practical bottlenecks well before the allocation limit.

## MLX-LM comparison

MLX-LM 0.31.3 and MLX 0.32.0 were tested with
`mlx-community/Qwen3.6-35B-A3B-4bit`. The HauhauCS aggressive repository does
not publish safetensors or MLX weights, so these results compare runtimes on
the same model architecture but **not the same fine-tune**.

| Attention KV | Short decode | Peak memory reported by MLX |
|---|---:|---:|
| unquantized | 65.3 tok/s | 19.66 GB |
| 8-bit | 64.5 tok/s | 19.66 GB |
| 4-bit | 57.6 tok/s | 19.66 GB |

The 25-token prompt in this short generation test is too small for a useful
prefill comparison. The custom OpenAI-compatible MLX server was therefore
also tested with the same needle prompts used for llama.cpp and 4-bit
attention KV:

| Prompt tokens | End-to-end time | Retrieval | Swap |
|---:|---:|---|---:|
| 31,169 | 91.5 s | passed | 0 |
| 62,291 | 234.8 s | passed | 0 |

At 62K, MLX completed about 16% sooner than llama.cpp's 279.5-second run.
llama.cpp nevertheless remains the default for this deployment because it
runs the exact requested weights, exposes explicit context allocation, and
its recurrent checkpoints reduced a repeated 31K request to 3.3 seconds.
MLX's hybrid-model cross-request prompt-cache behavior should be treated as
experimental rather than assumed equivalent to those checkpoints.

## Configuration conclusion

Use q8_0/turbo4 at 262K as the quality-safe maximum native profile. For normal
agent work, configure the client around 64K–128K and let checkpoints preserve
reused prefixes. Use true symmetric Turbo3 with YaRN only when a request really
requires more than 262K; it is slower on M1 Max and has a slightly worse PPL
smoke-test result.

Use MLX when its faster decode and prefill are more important than matching the
HauhauCS fine-tune, and keep 4-bit KV for long-context headroom. For short
contexts, unquantized or 8-bit MLX KV retains nearly the full 65 tok/s decode
rate.
