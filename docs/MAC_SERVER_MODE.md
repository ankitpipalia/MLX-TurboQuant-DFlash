# macOS LLM server mode

The Mac always boots with Apple's normal sleep settings, the default Metal
wired-memory policy, and no inference process. The launch daemon starts only
the low-memory Python control panel.

## Enter server mode

1. Open `http://192.168.1.117:8090/`.
2. In **Power**, select a Metal ceiling. `29696 MiB` is the highest setting
   pressure-tested on this 32 GiB M1 Max; `30720 MiB` is intentionally marked
   extreme.
3. Select a model and its K/V cache types in **Runtime**.
4. Start llama.cpp manually.

Enabling server mode, or selecting a nonzero ceiling, creates a small
`run/server-mode.enabled` flag and starts a controller-bound
`caffeinate -ims` assertion. It prevents clamshell/system/disk sleep but does
not keep the display awake. The manager recreates the assertion after its own
reload while the flag exists. Model downloads launched by the manager receive
their own assertion.

## Return to normal mode

Use **Stop Model & Normal macOS**. This stops the inference process, releases
the server-mode sleep assertion, and executes:

```text
iogpu.wired_limit_mb=0
```

The Metal setting remains runtime-only. Server mode survives a controller
reload but is removed by **Stop Model & Normal macOS**; a model is never started
automatically.

## Safety boundaries

The root-owned `/usr/local/sbin/local-llm-memory-control` helper accepts only
`0` or a value from `20480` through `30720` MiB. The web process cannot execute
arbitrary privileged commands. Wired Metal memory cannot be swapped; watch
wired memory, compressed memory, swap, and available RAM in the dashboard when
testing a new context profile.
