# Qwen3.6 35B-A3B Q4_K_P — M1 Max 32 GB

Measured 2026-07-14 with the local TurboQuant+ llama.cpp build, macOS GUI
running, one server slot, all layers on Metal, and the runtime-only
`iogpu.wired_limit_mb=29696` ceiling.

The downloaded GGUF is 23.4 GB and its verified SHA-256 is
`8d344a4336d8ea7da0cbfc12792d1471e568be7abe8930c52260698bfd01d731`.

| Context | K / V cache | Scaling | Decode | Short prefill | Free pressure | Result |
|---:|---|---|---:|---:|---:|---|
| 262,144 | q8_0 / turbo4 | native | 44.84 tok/s | 132.86 tok/s | 17% | default quality profile |
| 524,288 | q8_0 / turbo2 | YaRN 2x | 45.20 tok/s | 85.32 tok/s | 12% | extended, preserves q8 K |
| 1,048,576 | turbo2 / turbo2 | YaRN 4x | 43.58 tok/s | 134.54 tok/s | 12% | experimental frontier |

At the 1M setting, a separate 9,250-token prompt prefetched at 506.15 tok/s,
returned the requested first word correctly, and left pressure at 12% with no
swap. The different short-prefill figures are dominated by startup/graph costs;
the 9,250-token run is the more useful sustained-prefill measurement.

The 262K profile is the default because it stays within the model's native
training context and retains q8 K. The 524K profile also retains q8 K, but uses
static YaRN and aggressive Turbo2 V. The 1M profile compresses K and therefore
proves allocation and short-prompt operation only—not million-token retrieval
quality. Use a long-context retrieval/perplexity suite before treating it as a
production agent profile.

All profiles use two context checkpoints with a 256-token minimum step. Neither
the model nor the Metal ceiling is activated automatically after reboot.

## July 15 runtime tuning

The daily 262K profile now uses 10 generation/batch threads, a 1024-token
micro-batch, and a bounded 1536 MiB host prompt cache. A live 7,818-token API
probe prefetched at 568.9 tok/s and decoded at 31.2 tok/s at that depth; the
identical second request reused 7,814 tokens and completed in 0.46 seconds.
See `LLAMA_CPP_M1_MAX_OPTIMIZATION_2026-07.md` for the controlled sweep and
rejected settings.
