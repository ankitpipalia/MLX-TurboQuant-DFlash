# TurboQuant MLX on the 32 GiB M1 Max

> Historical benchmark note. The current production conclusions and the newer
> fixed 262K/128K profiles are in
> `MLX_PRODUCTION_M1_MAX_2026-07.md`.

Verified July 15, 2026 with MLX 0.32.0, MLX-LM 0.31.3, the Dawncr0w
Qwen3.6-35B-A3B OptiQ 6.12-bpw model, a 28 GiB runtime Metal ceiling, and one
request at a time.

## The architecture constraint

This Qwen model is not a conventional 40-layer attention model. It has 40
layers, but only every fourth layer uses full attention:

- 10 layers have a growing K/V cache.
- 30 layers use GatedDeltaNet recurrent `ArraysCache` state.

TurboQuant can compress only the 10 K/V caches. Replacing all 40 caches, as
several generic examples do, corrupts the model. The local integration starts
with MLX-LM's model-native cache list, replaces only actual `KVCache` objects,
and preserves all recurrent caches.

## Manager modes

The MLX panel exposes four explicit modes:

| Mode | Implementation | Intended use |
| --- | --- | --- |
| `native4` | MLX affine 4-bit | Proven daily default and 64K agent context |
| `native8` | MLX affine 8-bit | Fidelity-first, less context headroom |
| `turbo4` | Randomized Hadamard rotated 4-bit K/V | Experimental quality-oriented 4-bit cache |
| `turbo3` | Fused 3-bit PolarQuant/Lloyd-Max Metal cache | Experimental, tested only through 32K |

Turbo4 is pinned to `pythongiant/mlx_turboquant` commit
`0846d6aac9e3720a59de7a52782a9af63ad0146f`. Turbo3 is pinned to
`arozanov/turboquant-mlx` commit
`6e928d715595dee9f6b6cc3968baa44e1f408d28`.

The local adapter additionally fixes:

- hybrid-cache preservation;
- MLX-LM's invalid packed allocation for 3-bit widths that do not divide 32;
- fused decode placeholders being used incorrectly during multi-token prefill;
- non-pickleable `mlx.core.Dtype` breaking MLX-LM prefix-cache deepcopy;
- retained prefill dequant buffers duplicating packed cache memory.

## Measured results

All needle runs requested exact retrieval of `BLUE-OTTER-7741`.

| Cache | Decode, 256 tokens | 8K cold / warm | 32K needle | 64K needle | Observed conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| Native4 | 49.34 tok/s | 14.727 / 0.366 s | Pass, 89.278 s | Pass, 228.186 s | Recommended |
| Turbo4 | about 52.5 tok/s | 15.153 / 0.385 s | Pass, 89.569 s | Pass, 231.745 s | Works, but no memory win over another 4-bit cache |
| Turbo3 fused | about 52.5 tok/s | 14.341 / 0.540 s | Pass, 80.065 s | Stopped at about 49K processed | Fast, but multi-token prefill peak is too high for safe 64K |
| Turbo3 + QJL prototype | about 20.8 tok/s | 21.686 / 0.536 s | Pass, 204.787 s | Not attempted | Quality correction works but is too slow and memory-heavy |

Turbo4's 64K run peaked at 26,487 MiB Metal, left a minimum 2,552 MiB of
available system memory, reached 73 C, and used only 2 MiB swap. This is no
smaller than native4 because both store four physical bits per value; rotation
improves the quantization distribution, not its byte count.

Fused Turbo3's 32K run peaked at 24,902 MiB Metal and left 3,778 MiB available.
The 64K run was deliberately stopped around 49K processed when Metal reached
27,016 MiB and available memory fell to 2,316 MiB. The custom fused kernel is a
decode kernel; chunked prefill still materializes K/V temporarily. Continuing
would have approached the 28 GiB wired ceiling and was not worth risking a
machine reset.

## Source audit

- `pythongiant/mlx_turboquant` has the cleanest stock-MLX rotated cache and a
  custom QJL correction kernel, but its server helper assumes every layer is
  full attention and MLX-LM 0.31.3 cannot allocate its 3-bit cache correctly
  without the local fix. It powers only Turbo4 here.
- `arozanov/turboquant-mlx`, represented by MLX-LM PR 1067, has the fastest
  packed 3-bit Lloyd-Max decode kernels. It powers Turbo3 with local hybrid,
  prefill, deepcopy, and buffer-lifetime fixes.
- `flovflo/turboquant-mlx-qwen35-kv` correctly recognizes the 30/10 hybrid
  split, but describes itself as TurboQuant-inspired. Its "rotation" is a sign
  flip plus permutation rather than a Hadamard Gaussianizing transform, so it
  was used as architecture guidance rather than the production cache.
- `sharpner/turboquant-mlx` is a valuable proof of concept and kernel lab, but
  it is not an MLX-LM server integration and does not preserve hybrid caches.
- `helgklaizar/turboquant-mlx` repeatedly decompresses and concatenates the
  entire historical cache in `update_and_fetch`, has inconsistent package
  imports, and does not provide a credible long-context working-set reduction.

Upstream PR 1067 remains open and includes reports of garbage output on an MLA
model, confirming that cache compression must be validated per architecture.
The Qwen3.6 tests here therefore use actual generation, prefix reuse, and exact
needle retrieval rather than allocation success alone.

## Recommendation

Keep `native4` and a 65,536 client context as the default. Turbo4 is usable for
experimentation and passed 64K, but it does not extend context. Turbo3 is now a
working selectable server mode and is fast at short/32K contexts, but cap it at
32K until a fused multi-token prefill kernel removes its temporary K/V working
set and broader quality evaluation confirms 3-bit keys.

TurboQuant cannot turn this 32 GiB machine into a safe 262K host for this model:
only one quarter of the layers have compressible growing K/V, while weights,
recurrent state, Metal graphs, prompt-cache copies, and prefill workspace still
share the same unified memory.

## References

- TurboQuant paper: <https://arxiv.org/abs/2504.19874>
- Google Research overview: <https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/>
- MLX-LM TurboQuant PR: <https://github.com/ml-explore/mlx-lm/pull/1067>
- Reddit MLX implementation discussion: <https://www.reddit.com/r/LocalLLaMA/comments/1s5vhf6/turboquant_on_mlx_46x_kv_cache_compression_with/>
- Hybrid Qwen implementation: <https://huggingface.co/flovflo/turboquant-mlx-qwen35-kv>
- Pinned Turbo4 source: <https://github.com/pythongiant/mlx_turboquant>
- Pinned Turbo3 source: <https://github.com/arozanov/turboquant-mlx>
- Other audited implementations: <https://github.com/sharpner/turboquant-mlx>, <https://github.com/helgklaizar/turboquant-mlx>
