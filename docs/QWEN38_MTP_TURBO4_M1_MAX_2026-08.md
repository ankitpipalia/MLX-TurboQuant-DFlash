# Qwen3.8 native MTP and Turbo4 on the 32 GB M1 Max

Measured 2026-08-17 with the pinned EigenLabs affine-Q4 target and
lowskillcoding native Q4 MTP head. Yukon published its winning result on an
M5 Max 128 GB using a Swift/Metal runner, eight simultaneous prompts, compact
vocabulary projection, and 512-token greedy decode. This document records the
different result from the local Python mlx-vlm path on an M1 Max 32 GB.

## Production choice

Use `mlx-vlm-qwen38-q4-turbo4`. It provides serial decoding, effective
TurboQuant 4-bit KV, one request at a time, a 262,144-token request ceiling,
and an append-only exact-prefix cache. Use
`mlx-vlm-qwen38-q4-turbo4-deep` only for a deliberately large cold prompt; its
512-token prefill chunks trade time for a smaller attention workspace.

The architecture ceiling is not a claim that 262K has passed retrieval on
this Mac. Exact retrieval and cached continuation are validated through 31,169
prompt tokens. Keep OpenCode append-only so an existing session grows in small
increments instead of repeatedly cold-prefilling the full transcript.

## Measured decode results

| Runtime | Predictable 512-token output | Implementation | Debugging | Tool call |
|---|---:|---:|---:|---:|
| Serial target | 15.69 tok/s | 15.76 | 15.94 | 16.28 |
| Fixed-4 native MTP | 15.78 tok/s | 13.33 | 12.96 | 9.95 |
| One-draft native MTP | 14.20 tok/s | not repeated | not repeated | not repeated |
| Yukon-style adaptive MTP | 9.09 tok/s | not repeated | not repeated | not repeated |

Fixed-4 accepted 381/390 proposed tokens on the predictable sequence, yet
only tied serial decoding. The one-draft test accepted 256/256 proposals and
still lost. Each Python draft step performs the 248,320-token vocabulary
projection that Yukon's compact-vocabulary Metal path avoids. Native MTP is
therefore retained as a benchmark option, not the coding default.

## Turbo4 and retained-prefix result

The first profile version used `--kv-key-scheme turboquant` and
`--kv-value-scheme turboquant`. mlx-vlm's continuous-batching selector ignored
those flags and logged `scheme=uniform`. The profiles now use the effective
`--kv-quant-scheme turboquant` switch and tests prevent regression.

The second issue was more serious: exact APC snapshots intentionally called
`dequantize_for_apc`, expanding packed TurboQuant cache state into float32. At
32K this raised the cached-append peak to 30.87 GB. The local text-server
wrapper now clones TurboQuant's packed NamedTuple state and reconstructs a
one-row `BatchTurboQuantKVCache` without dequantization.

| 32K Turbo4 test | Before compressed APC bridge | After bridge |
|---|---:|---:|
| Cold peak MLX allocation | 28.57 GB | 22.04 GB |
| Cached append peak | 30.87 GB | 22.04 GB |
| Cold elapsed | 495.37 s | 494.01 s |
| Append elapsed | 7.65 s | 4.55 s |
| Cached prompt tokens | 31,169 | 31,169 |
| End-to-end speed-up | 64.8x | 108.6x |
| Retrieval | pass | pass |

Cold prefill was 63.50 prompt tok/s. Decode at a 31K prefix was about 4.0
tok/s, compared with about 15.7 tok/s for short prompts. Turbo4 reduces cache
memory; it does not remove full-attention compute over a long prefix.

At 4K, the corrected path also passed exact retrieval, cached all 3,923 prompt
tokens, and reduced 59.82 seconds cold to 2.17 seconds append-only (27.6x).

## Reproduce

```bash
./scripts/download_qwen38_mtp.sh
.venv/bin/pytest -q
.venv/bin/python scripts/context_reuse_probe_mlx.py \
  --tokenizer ~/Models/mlx/EigenLabs--Qwen3.8-27B-4bit \
  --target-tokens 32768
```

The downloader pins both repository revisions and verifies every target shard
plus the MTP head by exact size and SHA-256 before creating the local draft
configuration. Raw results are under `.artifacts/qwen38-mtp/`.

## Sources

- [Yukon MLX Fast challenge result](https://www.yukon.org/mlxfast)
- [Yukon challenge implementation](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge)
- [Pinned EigenLabs Qwen3.8 target](https://huggingface.co/EigenLabs/Qwen3.8-27B-4bit)
- [Pinned native Q4 MTP head](https://huggingface.co/lowskillcoding/qwen38-mtp-head-4bit-g64)
- [Hugging Face Xet transfer controls](https://huggingface.co/docs/hub/xet/using-xet-storage)
