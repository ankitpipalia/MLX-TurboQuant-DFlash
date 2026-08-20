# Local LLM control for M1 Max

This workspace runs one large model at a time on a 32 GiB M1 Max. It contains
profiles for TheTom's TurboQuant llama.cpp fork and MLX-LM, plus a small Python
control API. Models are intentionally stored in `~/Models`, not in this repo.

## Setup

```bash
uv sync
source .venv/bin/activate
llm-control profiles
```

## Run

Start the control API:

```bash
uv run llm-control serve --port 8090
```

The low-memory dashboard launch service listens on the trusted LAN at
`http://<mac-ip>:8090/` (`0.0.0.0:8090`). The CLI keeps its safer localhost
default unless `--host 0.0.0.0` is passed explicitly.
The launchd job starts only this controller. It never starts llama.cpp or MLX
automatically, and it does not change the Metal memory limit at boot.

From the dashboard, use **LLM Mode — Metal Memory Ceiling** to choose a
runtime-only wired-memory ceiling, then select a model/profile and start it.
Server mode attaches an independent `caffeinate` assertion to the controller,
so stopping or restarting a model does not drop a lid-closed SSH session. The
display remains free to sleep. **Open Lid, Then Allow Sleep** stops inference,
removes the guard, and resets the Metal limit; the API refuses this operation
while the lid is closed. All changes disappear naturally at reboot.
The control and inference APIs do not require authentication, so keep them on
localhost. An SSH tunnel remains available for remote administration:

```bash
ssh -L 8090:127.0.0.1:8090 -L 8097:127.0.0.1:8097 \
  ankitpipalia@192.168.1.117
```

Or manage runtimes directly:

```bash
uv run llm-control start llama-turbo4-safe
uv run llm-control status
uv run llm-control stop
```

The inference endpoints are OpenAI-compatible:

- llama.cpp: `http://127.0.0.1:8097/v1`
- MLX-LM comparison: `http://127.0.0.1:8098/v1`

GGUF and MLX models are selected independently in the dashboard; do not assume
that similarly named conversions contain the same fine-tune. The local MLX
server wrapper enables 4-bit attention KV cache while leaving a hybrid model's
recurrent-state cache intact; this control is not exposed by the stock MLX-LM
server CLI.

Stock MLX allocates context dynamically, while this workspace can optionally
materialize one fixed KV arena at startup with `--preallocate-kv-size`. A
262,144-token architecture limit or a successful reservation is not proof that
the model retrieves correctly at that depth. Use **Benchmark → MLX Long-Context
Needle Test** to validate increasing depths for the exact model and KV mode.
These probes run as cancellable, caffeinated manager tasks and save JSON under
`benchmark-results/`.

See `docs/MLX_PRODUCTION_M1_MAX_2026-07.md` for the current 35B/40B production
profiles, the Qwopus 35B coder comparison, fixed Turbo4 cache, session-reuse
design, memory limits, benchmarks, OpenCode settings, and upstream research.
Older MLX notes are retained as historical experiments and may contain
superseded defaults.

See `docs/LLAMA_CPP_M1_MAX_OPTIMIZATION_2026-07.md` for the measured Metal,
thermal, micro-batch, prompt-cache, and OpenCode long-context tuning.

See `config/profiles.toml` for the exact performance and context settings.

The pinned Qwen3.8 27B target, native MTP experiments, effective Turbo4
configuration, compressed prefix-cache bridge, and local benchmarks are
documented in `docs/QWEN38_MTP_TURBO4_M1_MAX_2026-08.md`.

## Qwen3.8 DFlash2 + TurboQuant4

The `mlx-dflash2-qwen38-q4-turbo4-262k` profile connects DFlash2 to the
repository's real rotated TurboQuant4 KV implementation. It converts only the
16 context-growing full-attention caches and preserves all 48 recurrent
rollback caches. A single 262,144-token, 4.50 GiB KV arena is reserved and
reused instead of repeatedly growing and copying cache allocations.

This is an experimental maximum-context profile for a 32 GiB Mac. It uses
96-token cold-prefill chunks to bound long-attention workspace, disables
incompatible prefix snapshots, limits concurrency to the server's single
request path, and leaves model startup under manual dashboard control.

```bash
uv run llm-control start mlx-dflash2-qwen38-q4-turbo4-262k
```

## HisabClub finance VLM stack

The quality-first finance runtime uses `mlx-vlm==0.6.13` at the pinned upstream
revision recorded in `pyproject.toml` on localhost port
`8098`. It keeps the OCR/document models at BF16 and uses 6-bit only for the
larger Qwen3.5 9B semantic model so it stays within unified-memory limits. The
KV cache remains unquantized. Model repositories, immutable revisions,
precision, and local paths are recorded in `config/finance-models.toml`.

```bash
./shared-local-llm.sh start finance
./shared-local-llm.sh status finance
./shared-local-llm.sh stop finance
```

The default profile preloads OvisOCR2 BF16. HisabClub may request the pinned
PaddleOCR-VL 1.6 BF16, GLM-OCR BF16, Qwen3.5 9B 6-bit, or NuExtract3 BF16
paths through the same OpenAI-compatible endpoint; mlx-vlm serializes the
heavy generation runtime and changes the resident model as needed. The server
is explicitly bound to `127.0.0.1`.

Reproduce and verify the pinned downloads with:

```bash
.venv/bin/python scripts/download_finance_models.py
.venv/bin/python scripts/verify_finance_models.py
```
The measured Qwen3.6 27B Q6 results and the distinction between quality-tested
and allocation-tested contexts are in `docs/QWEN27_Q6_BENCHMARK.md`.
The replacement Qwen3.6 35B-A3B Q4_K_P measurements and its native, extended,
and 1M frontier presets are in `docs/QWEN35_Q4KP_BENCHMARK.md`.
