# Muse Glimmer 30B on the 32 GB M1 Max

Measured 2026-08-13 with the Blackfrost AWQ 4-bit target and the official
Meta Muse-Glimmer DFlash assistant. These results are specific to this M1 Max;
results from M4/M5 systems are not substituted for local measurements.

## Production recommendation

Use `mlx-vlm-muse-glimmer-30b-awq4-text` for OpenCode. It combines native
131,072-token support, 2,048-token prefill chunks, exact append-prefix reuse,
lazy vision loading, one request at a time, and the ATEM tool parser. Configure
OpenCode for 122,880 input tokens plus 8,192 output tokens so a request cannot
cross the model's total 131,072-token ceiling.

Do not make DFlash the general coding default on this machine. It helps very
predictable output, but all tested coding-agent categories decoded faster with
the plain target. Prefix reuse is the large practical win.

## Local results

| Runtime | Workload | Prefill tok/s | Decode tok/s | Peak MLX allocation |
|---|---:|---:|---:|---:|
| Plain AWQ4 | 256-token integer sequence | 62.0 | 15.0 | 16.62 GB |
| q4 DFlash adaptive | same sequence | 62.9 | 20.6 | 18.09 GB |
| q4 DFlash fixed 4 | same sequence | 59.4 | 14.4 | 18.29 GB |
| BF16 DFlash fixed 4 | same sequence | 46.6 | 10.7 | 21.97 GB |

The adaptive q4 assistant is 1.37x faster than plain only on the highly
predictable sequence. The official BF16 assistant is not suitable for this
32 GB machine: quantizing the assistant to affine 4-bit/group-64 saves about
3.7 GB at runtime and is substantially faster.

| Runtime | Implementation | Debugging | ATEM tool call |
|---|---:|---:|---:|
| Plain AWQ4 | 15.07 tok/s | 15.17 tok/s | 15.11 tok/s |
| q4 DFlash adaptive | 11.19 tok/s | 10.51 tok/s | 9.69 tok/s |
| q4 DFlash fixed 4 | 13.00 tok/s | 9.23 tok/s | 12.11 tok/s |
| q4 DFlash fixed 8 | 8.69 tok/s | 9.24 tok/s | 6.89 tok/s |

All coding outputs were valid, including the ATEM `read_file` tool call. Fixed
8 is rejected. Fixed 4 is retained only in the experimental cached-DFlash
profile. The direct DFlash comparison profile uses upstream adaptive sizing,
which starts at four and can grow toward the checkpoint's trained block of 16.

## Context and cache behavior

- Native model context: 131,072 tokens. This checkpoint is not a 262K model.
- A 31,698-token exact needle probe passed at 62.66 prompt tok/s and 18.16 GB
  peak allocation.
- Plain 8K cold/warm: 122.33 s / 1.18 s, 7,831 cached tokens, 104.11x faster.
- Cached-DFlash 8K cold/warm: 123.04 s / 1.18 s, 7,831 cached tokens, 104.01x
  faster, exact READY/WARM output.
- Cache reuse requires a byte-stable, append-only prefix. Changing the system
  prompt, prior assistant output, tools, or message serialization forces cold
  prefill.

The cache holds one exact snapshot to bound memory. OpenCode should keep one
append-only session rather than resending semantically equivalent but
byte-different history.

## Artifacts and implementation

- Target: `~/Models/mlx/Blackfrost-AI--Muse-Glimmer-30B-Abliterated-MLX-4bit-AWQ`
- Official BF16 assistant: `~/Models/dflash/meta-models--Muse-Glimmer-30B-assistant-bf16-hf`
- Local q4 assistant: `~/Models/dflash/meta-models--Muse-Glimmer-30B-assistant-4bit`
- Official assistant SHA-256:
  `fd88d337eb84f8d0e6ba33a7684d7efa6722d4460ba4d6badca9699418392a84`
- Raw benchmark JSON: `.artifacts/muse-glimmer-20260813/`
- Quantizer: `scripts/quantize_muse_dflash.py`
- Cached DFlash bridge: `src/local_llm_control/mlx_vlm_cached_dflash_server.py`

## Research conclusions

Upstream Muse DFlash already includes compiled transitions, pipelined context
projection, adaptive startup depth, a batch-one server path, and removal of
per-round allocator flushes. Local reimplementation of those features would be
duplicate work. Draft acceptance depends on workload, and community reports
also show large blocks can regress on Macs. The local A/B confirms that result.

MLX-VLM's chunked prefill still has synchronization overhead. Raising the chunk
from 512 to 2,048 is locally stable through 32K and modestly improves prefill;
larger unverified values are not used. Experimental decode proposals such as
mixed-bit recipes, fused sampling and FP8 KV remain upstream proposals rather
than production defaults here.

Sources:

- Muse DFlash implementation and benchmarks: https://github.com/Blaizzy/mlx-vlm/pull/1842
- Muse model correctness fixes: https://github.com/Blaizzy/mlx-vlm/pull/1838
- MLX-VLM prefill synchronization issue: https://github.com/Blaizzy/mlx-vlm/issues/945
- MLX decode optimization proposals: https://github.com/ml-explore/mlx-lm/issues/1450
- Community Muse DSpark/DFlash measurements: https://www.reddit.com/r/LocalLLaMA/comments/1vmo2sp/metas_muse_glimmer_30b_now_runs_up_to_33x_faster/
- Community DFlash acceptance/block discussion: https://www.reddit.com/r/LocalLLaMA/comments/1vlgkwh/muse_glimmer_30b_dflash_drafter_slower_than/
