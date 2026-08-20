# Qwen3.6 27B Q6_K_P — M1 Max 32 GB

Measured 2026-07-14 with the local TurboQuant+ llama.cpp build, macOS GUI
running, one server slot, all model layers on Metal, and a runtime-only
`iogpu.wired_limit_mb=29696`. The model file SHA-256 is
`f17b88ee70d2fb6f93c220ac306be0c8de631f20b3ffc82854162bb5e1b14192`.

| Context | K / V cache | Scaling | Result | Decode | Free pressure after request |
|---:|---|---|---|---:|---:|
| 131,072 | q8_0 / turbo4 | native | loaded + short request | 10.08 tok/s | 17% |
| 196,608 | q8_0 / turbo4 | native | loaded + short request | 10.09 tok/s | 10% |
| 262,144 | turbo4 / turbo4 | native | loaded + short request | 9.93 tok/s | 12% |
| 395,264 | turbo3 / turbo3 | YaRN 1.5078125x | loaded + short request | 9.81 tok/s | 10% |
| 524,288 | turbo2 / turbo2 | YaRN 2x | loaded + short request | 10.00 tok/s | 10% |

The ~200K (196608-token) asymmetric profile is the default agent configuration. It keeps K
at q8 because TurboQuant+'s own guidance identifies K as the quality-sensitive
side of the cache. 262K and above require symmetric turbo K on this machine and
are therefore experimental even when their memory use is stable.

These figures prove server initialization, full KV reservation, and a 42-token
prompt plus 124–128-token decode. They do **not** prove retrieval accuracy near
the end of a 262K–524K prompt. A full 500K prefill at the measured prompt rate
would take hours, and YaRN plus low-bit K needs a dedicated needle/retrieval and
perplexity suite before production use.

The manager deliberately uses two context checkpoints with a 256-token minimum
step. This preserves the useful A5000 agent-chat reprocessing trick without the
larger memory cost of four checkpoints. The model and Metal ceiling are never
activated automatically at boot.
