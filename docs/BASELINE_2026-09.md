# Software baseline — September 2026

Refreshed 2026-09-14, after ~3½ weeks. No models are on disk, so **everything
below is verified by seam, signature and CLI checks, never by inference.**

## Versions

| Component | Was (Aug) | Now | Note |
|---|---|---|---|
| mlx / mlx-metal | 0.32.0 | **0.32.2** | patch |
| mlx-lm | 0.31.3 | **0.31.3** | unchanged — we monkeypatch its internals heavily |
| mlx-vlm | 0.6.13 | **0.7.0** | 311 commits; see below |
| dflash-mlx | 0.1.10+omlx.6 | **0.1.10+omlx.7** | one commit, directly relevant |
| mlx-dspark | 0.8.1 | **0.19.0** | optional group, one experimental profile |
| transformers | 5.15.0 | **5.17.0** | |
| llama.cpp fork | cd84d31 | **407f323** (b676) | rebuilt, Release + Metal |

`mlx-lm` standing still is the single most reassuring fact here — it is the
package this repo patches most.

## What actually matters

**mlx-vlm PR #2182 fixes a problem measured on this machine.** Shared-document
prompts that diverge at the end — `document + question A` then
`document + question B`, or an agent's turn 2 — previously got *zero* reuse and
prefilled cold every time. That is exactly the `cached_tokens=0` seen in August
when two OpenCode turns shared a ~7.5K system prefix. Upstream now matches at
the last shared complete block and keeps bounded intermediate checkpoints.

Consequently `APC_NUM_BLOCKS=1` / `APC_EXACT_CACHE_ENTRIES=1` had to go. Those
were a hand-rolled memory guard from when upstream enforced no budget at all.
With one block and one entry, shared-prefix matching has nothing to match
against, and any short auxiliary request — OpenCode's title call — evicts the
live conversation and forces a cold prefill on *every* turn. Upstream bounds
retention itself now, so the profiles state a budget instead of pinning to one.

**dflash-mlx now applies `repetition_penalty` inside the speculative loop**
(`engine/sampling.py`), with the same 0.0/1.0 no-op convention the guard
already used. The sampling guard no longer diverts those requests to exact AR —
doing so would sacrifice speculation for a feature the fast path now handles.
Still dropped, still routed to exact AR: presence and frequency penalties, XTC,
`logit_bias`, logprobs. Verified against the live signature of
`stream_dflash_generate_impl`, not release notes.

## New knobs worth knowing

APC gained a large surface: `APC_CHECKPOINT_ENTRIES`,
`APC_CHECKPOINT_INTERVAL_TOKENS`, `APC_CHECKPOINT_GUARD_TOKENS`,
`APC_MEMORY_MAX_GB`, `APC_MEMORY_RESERVE_GB`, and a full `APC_DISK_*` family.
Defaults: 2 exact entries, 2048 blocks of 16, an automatic memory budget (10%
of Metal's recommended working set, capped at 8 GiB), and **disk persistence on
by default**, 20 GB per model namespace under `~/.cache/mlx-vlm/apc`. The
profiles state 2 entries, a 4 GB memory budget and an 8 GB disk cap rather than
inheriting them.

Disk-backed APC is the interesting one: it can survive a restart and spare the
cold first-turn prefill that dominated the August measurements.

In the llama.cpp fork: `TURBO_LAYER_ADAPTIVE` auto-enables boundary-V for
turbo2-V (opt out with `=0`); `--cache-ram -1` now means *half of free host
memory at startup*; there is a new `--cache-idle-slots`; and the server bounds
an "unlimited" prompt cache by free host memory. Several GDN replay correctness
fixes landed (conv-state rollback), which matter for hybrid Qwen models. The
WHT sign bit-packing and warp-float4 work is CUDA-only — no benefit on Metal.

## Caveats

Nothing here has been run against a model. The upgrade is validated by: all 145
tests passing; every monkeypatched mlx-vlm seam still present *with matching
signatures*; the installers applying cleanly at runtime; `dflash-compat serve`
starting with every installer active; and all eleven llama.cpp profile flags
still recognised by the rebuilt binary, with turbo2/3/4 intact.

The APC reconfiguration in particular rests on upstream's described behaviour.
It should be confirmed with a two-turn reuse check (`cached_tokens > 0` on turn
2) as soon as a model is downloaded again.
