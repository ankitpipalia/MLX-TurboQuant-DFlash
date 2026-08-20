# Qwen3.8-27B NVFP4 MLX test — M1 Max (2026-08-15)

Model: [`Brooooooklyn/Qwen3.8-27B-NVFP4-mlx`](https://huggingface.co/Brooooooklyn/Qwen3.8-27B-NVFP4-mlx), revision `d2c6de4d861d0d06b04f442a2cedf85f8ec15678`.

## Download and integrity

- Five SafeTensors shards downloaded completely (23,417,338,336 bytes declared by the index).
- The index contains 1,600 tensor entries and every declared shard is present and opens successfully with `safetensors`.
- Tokenizer and processor assets were fetched separately because the initial download command selected only model files.

## Compatibility findings

The checkpoint is not a conventional Apple-MLX quantization. It stores 168 NVFP4 modules plus 233 raw E4M3 FP8 modules with BF16 per-output-channel scales. The model card targets `mlx-node` on Linux aarch64/DGX Spark and describes macOS as a fallback smoke-test path.

The existing Python `mlx-vlm` profile cannot load it: MLX 0.32 has no `fp8_e4m3` quantization mode and exits with `KeyError: 'fp8_e4m3'`. This is a format/runtime mismatch, not a corrupt download.

The intended `@mlx-node/lm` runtime (0.0.10) loaded the weights and reported:

```text
trainedWindowTokens: 262144
effectiveWindowTokens: 262144
```

Forcing the flat cache path (`MLX_QWEN35_FORCE_EAGER=1 MLX_QWEN35_PAGED_OVERRIDE=0`) avoided the loader's zero-budget paged-cache refusal. However, a real one-turn generation then exhausted the 32GB machine's memory budget: macOS reported roughly 47GB GPU allocation, near-zero free pages, and rapidly increasing swap. No token was returned in approximately 100 seconds, so the process was terminated safely.

## Result

The nominal 256K context is present in the checkpoint metadata, but it is **not a usable 256K configuration on this 32GB M1 Max**. The model's FP8 sidecars are reconstructed to dense runtime weights by `mlx-node`; together with NVFP4 weights and KV/workspace this exceeds the machine's practical unified-memory budget. No 256K needle or speed result is claimed.

The manager remains stopped, port 8098 is free, and memory has returned to approximately 27.9 GiB available. Keep the current Qwen3.8 profile marked experimental; do not start it through the generic `mlx-vlm` command. For this Mac, the existing Q4/Q5/Q6 MLX checkpoints remain the practical choices.
