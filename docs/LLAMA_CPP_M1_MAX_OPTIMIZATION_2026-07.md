# llama.cpp performance study — M1 Max 32 GiB

Verified July 15, 2026 on the 24-GPU-core M1 Max, macOS 26.5.2, normal GUI
mode, and TurboQuant+ commit `4503343ffc05c09f6b50c309c8ecbabb49c66ea2`.

## Result

The 35B MoE is the interactive model. The dense 27B Q6 is a quality option,
not a speed option. At long context, active context depth—not the server's
allocated context ceiling—is the dominant latency variable.

| Model and condition | Prompt processing | Generation |
| --- | ---: | ---: |
| 35B Q4_K_P, synthetic 512 / short decode | 614 tok/s | 46.3 tok/s |
| 35B Q4_K_P, 7,818-token live API prompt | 568.9 tok/s | 31.2 tok/s at 8K depth |
| 35B Q4_K_P, 62K historical needle | 223.8 tok/s | 9.37 tok/s at 62K depth |
| 35B Q4_K_P, real 177K OpenCode session | 37–60 tok/s for small suffixes | about 3.5 tok/s |
| 27B Q6_K_P, 512 / 4,096 synthetic prompt | 94.5 / 95.2 tok/s | 10.0 tok/s when cool |

The optimized 35B live probe took 14.10 seconds cold for 7,818 prompt tokens.
An identical second request reused 7,814 tokens and completed in 0.46 seconds.

## Applied settings

### System

- AC power mode is now High Power (`pmset` power mode 2). Battery remains
  unchanged. Automatic mode caused a measured 19% sustained-prefill drop as
  the thermal state reached Serious.
- The runtime remains one model and one server slot. Full Metal offload,
  Flash Attention, mmap, and `--no-host` remain enabled.

### 35B default

```text
--threads 10 --threads-batch 10
--batch-size 2048 --ubatch-size 1024
--ctx-size 262144 --parallel 1
--cache-type-k q8_0 --cache-type-v turbo4
--ctx-checkpoints 2 --checkpoint-min-step 256
--cache-ram 1536 --no-cont-batching
--flash-attn on --gpu-layers all --no-host
```

Changing the micro-batch from 512 to 1024 raised 8K prefill from 500.1 to
552.8 tok/s in `llama-bench`; the live server reached 568.9 tok/s. A 2048
micro-batch reached 555.1 tok/s, only 0.4% above 1024, with more workspace, so
1024 is the safer optimum. Ten generation threads improved short decode by
about 1.8% over four threads.

The host prompt cache was increased because real logs showed a 161,363-token
OpenCode state consuming 1,434.9 MiB while the old limit was only 512 MiB.
The same log contained 21 forced full re-prefills but only six successful
checkpoint restores.

### 27B default

Keep the existing safe full-context settings:

```text
--threads 4 --threads-batch 8
--batch-size 1024 --ubatch-size 256
--ctx-size 196608 --parallel 1
--cache-type-k q8_0 --cache-type-v turbo4
--ctx-checkpoints 2 --checkpoint-min-step 256
--cache-ram 512 --no-cont-batching
```

A 512 micro-batch improved a short 4K benchmark from 95.2 to 97.2 tok/s, but
the real 196K server then failed immediately with Metal
`kIOGPUCommandBufferCallbackErrorOutOfMemory`. It was rejected and reverted.

## Rejected or conditional tricks

- `--cache-reuse 256`: the real server reports that KV shifting is unsupported
  by this Qwen hybrid/recurrent context and disables it.
- Flash Attention off: 8K prefill was effectively unchanged (553.3 versus
  552.8 tok/s), while disabling it loses the long-context memory advantage.
- More than a 1024 micro-batch: negligible speed gain and worse memory margin.
- Unlimited `--cache-ram -1`: unsafe on a 32 GiB machine. A bounded 1536 MiB
  cache retains one observed large state without giving the cache permission
  to consume all remaining system memory.
- Symmetric Turbo K/V: saves capacity but compressing K is the known quality
  risk. Keep q8 K and compress V unless a model-specific retrieval/PPL test
  justifies otherwise.
- NextN/MTP speculative decoding: promising for the 35B MoE, but both local
  GGUF files contain zero NextN/MTP tensors. It cannot be enabled with these
  files. AtomicBot's MTP-aware GGUF plus its fork is a separate model/runtime
  experiment, not a command-line switch for the installed files.
- N-gram speculation: useful for repetitive code completion, but it does not
  improve ordinary generation and is output-dependent. It is not enabled in
  the general OpenAI server profile.

## OpenCode operating policy

Use the `local` provider with a 65,536-token client limit for daily work. The
server still allocates the full native 262,144 window, but OpenCode compacts
before decode falls into the 3–9 tok/s long-context range. The checked-in
network config also exposes `local-max` with a deliberate 250K limit.

Avoid switching Plan/Build modes in the middle of a long conversation: it
changes the early system prompt and invalidates the prefix. Avoid pruning or
reordering old tool results when possible. These changes cannot be repaired by
batch tuning because the transformer must recompute from the changed prefix.

## Research references

- Official llama-bench methodology and parameter sweeps:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md>
- Official server cache and runtime options:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- TurboQuant+ implementation and asymmetric K/V guidance:
  <https://github.com/TheTom/llama-cpp-turboquant>
- TurboQuant context-scaling investigation:
  <https://github.com/ggml-org/llama.cpp/discussions/20969>
- OpenCode/hybrid-cache reprocessing reports:
  <https://www.reddit.com/r/LocalLLaMA/comments/1td9stc/llamacpp_constantly_reprocessing_huge_prompts_with/>
- OpenCode prompt-processing discussion:
  <https://www.reddit.com/r/LocalLLaMA/comments/1ta0pws/why_is_opencode_so_slow_in_processing_the_prompt/>
- NextN requirements and measured MoE/dense differences:
  <https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant/blob/feature/turboquant-kv-cache/NEXTN.md>
