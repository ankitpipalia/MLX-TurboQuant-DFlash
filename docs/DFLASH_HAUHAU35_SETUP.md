# HauhauCS 35B OptiQ + DFlash preparation

Prepared on 2026-07-28. The inference model was not loaded or started during
this preparation.

## Installed artifacts

- Target:
  `~/Models/mlx/cyberCyber99--Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-OptiQ-4bit-MLX`
- Draft:
  `~/Models/dflash/z-lab--Qwen3.6-35B-A3B-DFlash`
- Runtime: `dflash-mlx` 0.1.10, pinned in `pyproject.toml` and `uv.lock` to
  commit `9ca002898b48e14c9727dec17299f497e8467870`
- Launcher: `dflash-compat`, which maps the official June 2026 draft's nested
  `dflash_config.block_size` and `rope_parameters.rope_theta` fields to the
  top-level keys expected by runtime 0.1.10 without modifying the checkpoint
- Manager profile: `mlx-dflash-hauhau35-optiq4`
- OpenAI-compatible endpoint after an operator starts it:
  `http://192.168.1.117:8098/v1`

The target and draft have the same 2,048 hidden width, 40 target layers,
248,320-token vocabulary, and 262,144 native context ceiling. The draft has six
layers and explicitly targets layers 1, 6, 11, 16, 22, 27, 32, and 37.

## Prepared runtime policy

- 4-bit OptiQ target weights and a 4-bit DFlash draft
- adaptive DFlash verification with conservative prompt-copy speculation
- 2,048-token prompt chunks
- 262,144-token hard request cap
- native target KV. DFlash 0.1.10 cannot serialize MLX `QuantizedKVCache`
  entries into prefix snapshots, so combining its q8 KV option with prefix
  reuse causes request failure. Native KV preserves reuse and is faster on
  current MLX because q8 attention has no fused kernel.
- one in-memory prefix snapshot with a 3 GiB budget
- persistent SSD L2 prefix snapshots with a 20 GiB disk budget
- in-memory snapshot inserts capped at 131,072 tokens
- one inference server bound to `0.0.0.0:8098`
- thinking disabled and an 8,192-token default output ceiling

The 262,144 setting is a safety ceiling, not a fixed startup allocation. Cache
memory grows with the prompt. Prefix snapshots are intended to avoid repeating
unchanged OpenCode prefixes on later turns; exact reuse still requires a
byte-stable, append-only prefix.

DFlash does not currently use this repository's Turbo4 KV implementation. Its
supported memory-saving path is 8-bit target KV. Turbo4 remains available in
the separate llama.cpp and custom MLX server profiles.

## Static validation result

`dflash doctor` completed without loading weights and reported zero fatal
checks. It confirmed Metal, the runtime entry point, target/draft resolution,
the writable L2 directory, and every cache/verification setting. It produced
two expected M1 warnings:

- BF16 is emulated, so the quantized draft uses FP16 activations.
- M1 cannot use the newer NAX kernels and falls back to Steel kernels.

These warnings mean the actual speedup must be measured rather than assumed.
The first authorized test should compare target-only and DFlash generation,
then test cache reuse, memory growth, and needle retrieval at increasing
context depths before attempting the full 262K window.
