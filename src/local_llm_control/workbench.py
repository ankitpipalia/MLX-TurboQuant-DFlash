"""Mac-native backend for the A5000 workbench user interface."""

from __future__ import annotations

import asyncio
import json
import os
import plistlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import httpx
import psutil
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import Profile
from .state import manager, refresh_manager_settings, settings


router = APIRouter()
ROOT = settings.root
STATIC_DIR = Path(__file__).with_name("static")
MODEL_ROOT = Path.home() / "Models"
GGUF_ROOT = MODEL_ROOT / "gguf"
MLX_ROOT = MODEL_ROOT / "mlx"
RESULTS_DIR = ROOT / "benchmark-results"
CUSTOM_PATH = ROOT / "run" / "workbench-config.json"
EVENTS_PATH = ROOT / "run" / "workbench-events.jsonl"
LLAMA_BIN = ROOT / "vendor/llama-cpp-turboquant/build/bin/llama-server"
LLAMA_SOURCE = ROOT / "vendor/llama-cpp-turboquant"
MEMORY_HELPER = Path("/usr/local/sbin/local-llm-memory-control")
MACTOP_BIN = Path(shutil.which("mactop") or "/opt/homebrew/bin/mactop")
MACTOP_METRICS_URL = "http://127.0.0.1:9101/metrics"
MANAGER_LABEL = "com.ankitpipalia.local-llm-control"
MANAGER_PORT = 8090
TOTAL_MIB = 32768
STARTED_AT = time.time()

KV_TYPES = {
    "turbo4": ["TurboQuant 4-bit-equivalent cache", 4.25],
    "turbo3": ["TurboQuant long-context cache", 3.25],
    "turbo2": ["TurboQuant maximum-context cache", 2.50],
    "q8_0": ["Standard q8 quality baseline", 8.50],
    "q4_0": ["Standard q4 cache", 4.50],
    "f16": ["Uncompressed KV", 16.0],
}

# Measured after each default profile had fully loaded at the runtime-only
# 29696 MiB Metal ceiling. These are planning baselines, not reservations;
# macOS free/available pages vary with filesystem cache and background work.
PROFILE_MEMORY_BASELINES = [
    {
        "profile": "35B Q4_K_P default", "context": 262144,
        "kv": "q8_0 / turbo4", "system_free_mib": 57,
        "unified_available_mib": 3940, "gpu_pool_free_mib": 7018,
        "metal_headroom_mib": 3946, "metal_used_mib": 25750,
    },
    {
        "profile": "27B Q6_K_P default", "context": 196608,
        "kv": "q8_0 / turbo4", "system_free_mib": 57,
        "unified_available_mib": 1710, "gpu_pool_free_mib": 3935,
        "metal_headroom_mib": 863, "metal_used_mib": 28833,
    },
]

HISTORY: deque[dict[str, Any]] = deque(maxlen=17280)  # 24 h at 5 seconds
EVENTS: deque[dict[str, Any]] = deque(maxlen=2000)
HISTORY_LOCK = threading.Lock()
EVENTS_LOCK = threading.Lock()
SAMPLE_SECONDS = 5
SAMPLER_STARTED = False
SERVER_GUARD: subprocess.Popen[bytes] | None = None
SERVER_MODE_FLAG = ROOT / "run/server-mode.enabled"
MACTOP_PROCESS: subprocess.Popen[bytes] | None = None
MACTOP_LAST_ATTEMPT = 0.0
MACTOP_RETRY_SECONDS = 60.0
TASKS: dict[str, dict[str, Any]] = {}
TASK_FINISHED_KEEP = 40
TASK_PROCESSES: dict[str, subprocess.Popen[str]] = {}
TASK_LOCK = threading.Lock()
METRICS_LOCK = threading.Lock()
METRICS_CACHE: dict[str, float] = {}
METRICS_CACHE_AT = 0.0


def _run(command: list[str], timeout: float = 10) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _lan_host() -> str:
    """Return the Mac's current LAN address without relying on a stale DHCP lease."""
    result = _run(["/usr/sbin/ipconfig", "getifaddr", "en0"], timeout=2)
    return result.stdout.strip() or "127.0.0.1"


def _json_url(url: str, timeout: float = 3) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _text_url(url: str, timeout: float = 3) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _mlx_vlm_draft_log_stats() -> tuple[float, float]:
    """Return the latest DFlash (acceptance ratio, tokens/cycle) log sample."""
    state = manager._saved_state()
    try:
        text = Path(str(state.get("log") or "")).read_text(errors="replace")
    except OSError:
        return 0.0, 0.0
    matches = re.findall(
        r"Speculative decode:.*?accept=([0-9.]+).*?rounds=(\d+).*?drafted=(\d+)",
        text,
    )
    if not matches:
        return 0.0, 0.0
    tokens_per_cycle, rounds, drafted = matches[-1]
    cycle = float(tokens_per_cycle)
    rounds_n = int(rounds)
    drafted_n = int(drafted)
    accepted_n = max(0.0, cycle * rounds_n - rounds_n)
    return (
        accepted_n / drafted_n if drafted_n else 0.0,
        cycle,
    )


def _event(kind: str, message: str) -> None:
    row = {"ts": int(time.time()), "kind": kind, "message": message}
    with EVENTS_LOCK:
        EVENTS.append(row)
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_PATH.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


def _load_events() -> None:
    try:
        for line in EVENTS_PATH.read_text().splitlines()[-2000:]:
            EVENTS.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def _parse_mactop_metrics(text: str) -> dict[str, Any]:
    """Reduce mactop's Prometheus output to the dashboard telemetry fields."""
    values: dict[tuple[str, str], float] = {}
    gpu_temps: list[float] = []
    fan_rpms: list[float] = []
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        sample, raw_value = line.rsplit(" ", 1)
        try:
            value = float(raw_value)
        except ValueError:
            continue
        metric, _, labels = sample.partition("{")
        values[(metric, labels)] = value
        if metric == "mactop_temp_sensor_celsius" and 'name="GPU' in labels:
            # Some M1 SMC aliases return 0 or 9.2 C. mactop marks these as
            # filtered; only physical silicon readings are useful here.
            if 10.0 <= value <= 130.0:
                gpu_temps.append(value)
        elif metric == "mactop_fan_rpm":
            if value > 0:
                fan_rpms.append(value)

    def metric(name: str, label: str = "") -> float | None:
        for (candidate, labels), value in values.items():
            if candidate == name and (not label or label in labels):
                return value
        return None

    thermal_code = metric("mactop_thermal_state")
    thermal_states = {0: "Nominal", 1: "Fair", 2: "Serious", 3: "Critical"}
    return {
        "available": bool(values),
        "source": "mactop/IOReport+SMC",
        "gpu_power_w": metric("mactop_power_watts", 'component="gpu"'),
        "cpu_power_w": metric("mactop_power_watts", 'component="cpu"'),
        "system_power_w": metric("mactop_power_watts", 'component="system"'),
        "total_power_w": metric("mactop_power_watts", 'component="total"'),
        "gpu_temp_c": round(sum(gpu_temps) / len(gpu_temps), 1) if gpu_temps else None,
        "gpu_temp_max_c": round(max(gpu_temps), 1) if gpu_temps else None,
        "gpu_clock_mhz": metric("mactop_gpu_freq_mhz"),
        "gpu_util_pct": metric("mactop_gpu_usage_percent"),
        "fan_rpm": round(sum(fan_rpms) / len(fan_rpms)) if fan_rpms else None,
        "thermal_state": thermal_states.get(int(thermal_code)) if thermal_code is not None else None,
    }


def _apple_telemetry() -> dict[str, Any]:
    try:
        return _parse_mactop_metrics(_text_url(MACTOP_METRICS_URL, 1.0))
    except Exception:
        return {"available": False, "source": "mactop unavailable"}


def _mactop_pid() -> int | None:
    """Return the collector PID even when it outlived a manager reload."""
    if MACTOP_PROCESS and MACTOP_PROCESS.poll() is None:
        return MACTOP_PROCESS.pid
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info.get("cmdline") or []
            if (command and Path(command[0]).name == "mactop" and
                    "--prometheus" in command and "9101" in command):
                return int(process.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError):
            continue
    return None


def _ensure_mactop(telemetry_available: bool | None = None) -> None:
    """Keep one native telemetry sampler alive; restart it if it dies.

    Called at startup and from every sampler tick, so a mactop process that
    exits (or was started externally and later stopped) is revived within one
    retry window instead of staying down until the next controller restart.
    """
    global MACTOP_PROCESS, MACTOP_LAST_ATTEMPT
    if telemetry_available is None:
        telemetry_available = _apple_telemetry().get("available", False)
    if telemetry_available or not MACTOP_BIN.exists():
        return
    if MACTOP_PROCESS and MACTOP_PROCESS.poll() is None:
        # Recently launched and still warming up its metrics endpoint.
        return
    now = time.monotonic()
    if now - MACTOP_LAST_ATTEMPT < MACTOP_RETRY_SECONDS:
        return
    MACTOP_LAST_ATTEMPT = now
    log_path = ROOT / "logs" / "mactop.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    try:
        MACTOP_PROCESS = subprocess.Popen(
            [str(MACTOP_BIN), "--prometheus", "9101", "--headless",
             "--interval", str(SAMPLE_SECONDS * 1000)],
            stdout=subprocess.DEVNULL, stderr=log, start_new_session=True,
        )
        _event("telemetry", f"started mactop sampler pid {MACTOP_PROCESS.pid}")
    except OSError as exc:
        _event("error", f"mactop sampler: {exc}")
    finally:
        log.close()


def _restart_mactop() -> None:
    """Force a fresh mactop so a new sampling interval takes effect."""
    global MACTOP_PROCESS, MACTOP_LAST_ATTEMPT
    pid = _mactop_pid()
    if pid:
        try:
            process = psutil.Process(pid)
            process.terminate()
            try:
                process.wait(timeout=3)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        except psutil.NoSuchProcess:
            pass
    MACTOP_PROCESS = None
    MACTOP_LAST_ATTEMPT = 0.0
    _ensure_mactop(False)


def _gpu_state(telemetry: dict[str, Any] | None = None) -> dict[str, Any]:
    telemetry = telemetry if telemetry is not None else _apple_telemetry()
    result = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator", "-a"])
    try:
        devices = plistlib.loads(result.stdout.encode())
        device = devices[0]
        perf = device.get("PerformanceStatistics", {})
        active = int(perf.get("In use system memory", 0)) // 2**20
        allocated = int(perf.get("Alloc system memory", 0)) // 2**20
        # Apple reports model buffers as allocated system memory even while the GPU is idle.
        used = max(active, allocated)
        util = float(perf.get("Device Utilization %", 0))
        return {
            "name": device.get("model", "Apple M1 Max"),
            "used_mib": used,
            "app_vram": used,
            "allocated_mib": allocated,
            "active_mib": active,
            "free_mib": max(0, TOTAL_MIB - used),
            "total_mib": TOTAL_MIB,
            "util_pct": telemetry.get("gpu_util_pct") if telemetry.get("gpu_util_pct") is not None else util,
            "mem_util_pct": round(used / TOTAL_MIB * 100, 1),
            "temp_c": telemetry.get("gpu_temp_c"),
            "power_w": telemetry.get("gpu_power_w"),
            "clock_mhz": telemetry.get("gpu_clock_mhz"),
            "fan_rpm": telemetry.get("fan_rpm"),
            "thermal_state": telemetry.get("thermal_state"),
        }
    except (IndexError, KeyError, plistlib.InvalidFileException, ValueError):
        return {
            "name": "Apple M1 Max",
            "used_mib": 0,
            "app_vram": 0,
            "free_mib": TOTAL_MIB,
            "total_mib": TOTAL_MIB,
            "util_pct": 0,
            "mem_util_pct": 0,
            "temp_c": telemetry.get("gpu_temp_c"),
            "power_w": telemetry.get("gpu_power_w"),
            "clock_mhz": telemetry.get("gpu_clock_mhz"),
            "fan_rpm": telemetry.get("fan_rpm"),
            "thermal_state": telemetry.get("thermal_state"),
        }


def _llama_metrics() -> dict[str, float]:
    """Read inference metrics once per sample window from the active engine."""
    global METRICS_CACHE, METRICS_CACHE_AT
    now = time.monotonic()
    with METRICS_LOCK:
        if now - METRICS_CACHE_AT < SAMPLE_SECONDS:
            return dict(METRICS_CACHE)
        port = 8098 if _engine_is_mlx() else 8097
        if _engine_is_dspark():
            try:
                payload = _json_url(
                    f"http://127.0.0.1:{port}/metrics", 1.0
                )
                prefix = payload.get("prefix_cache") or {}
                values = {
                    "local_llm_requests_total": float(
                        payload.get("requests") or 0
                    ),
                    "local_llm_checkpoints_created_total": float(
                        len(prefix.get("slots") or [])
                    ),
                    "local_llm_checkpoints_restored_total": float(
                        prefix.get("hits") or 0
                    ),
                    "local_llm_forced_reprocess_total": 0.0,
                    "local_llm_last_cached_tokens": float(
                        prefix.get("cached_tokens") or 0
                    ),
                    "local_llm_last_replayed_tokens": 0.0,
                    "local_llm_last_generation_tps": float(
                        payload.get("mean_tokens_per_sec") or 0
                    ),
                    # mlx-dspark currently exposes cumulative prompt tokens,
                    # but no physical prefill time/rate.
                    "local_llm_last_prompt_tps": 0.0,
                    "local_llm_prompt_tokens_total": float(
                        payload.get("prompt_tokens") or 0
                    ),
                    "local_llm_tokens_generated_total": float(
                        payload.get("completion_tokens") or 0
                    ),
                    "local_llm_dflash_acceptance_ratio": 0.0,
                    "local_llm_dflash_tokens_per_cycle": float(
                        payload.get("mean_accept_len") or 0
                    ),
                }
            except Exception:
                return (
                    dict(METRICS_CACHE)
                    if now - METRICS_CACHE_AT < 30
                    else {}
                )
            METRICS_CACHE = values
            METRICS_CACHE_AT = now
            return dict(values)
        if _engine_is_dflash():
            try:
                payload = _json_url(
                    f"http://127.0.0.1:{port}/metrics", 1.0
                )
                totals = payload.get("totals") or {}
                rates = payload.get("rates") or {}
                last = payload.get("last_request") or {}
                values = {
                    "local_llm_requests_total": float(
                        totals.get("requests") or 0
                    ),
                    "local_llm_checkpoints_created_total": float(
                        (totals.get("cache_hits") or 0)
                        + (totals.get("cache_misses") or 0)
                    ),
                    "local_llm_checkpoints_restored_total": float(
                        totals.get("cache_hits") or 0
                    ),
                    "local_llm_forced_reprocess_total": 0.0,
                    "local_llm_last_cached_tokens": float(
                        last.get("cache_hit_tokens") or 0
                    ),
                    "local_llm_last_replayed_tokens": 0.0,
                    "local_llm_last_generation_tps": float(
                        last.get("decode_tok_s")
                        or rates.get("average_decode_tok_s")
                        or 0
                    ),
                    "local_llm_last_prompt_tps": float(
                        last.get("prefill_tok_s_physical")
                        or last.get("prefill_tok_s_apparent")
                        or 0
                    ),
                    "local_llm_dflash_acceptance_ratio": float(
                        last.get("acceptance_rate") or 0
                    ),
                    "local_llm_dflash_tokens_per_cycle": float(
                        last.get("tokens_per_cycle") or 0
                    ),
                }
            except Exception:
                return (
                    dict(METRICS_CACHE)
                    if now - METRICS_CACHE_AT < 30
                    else {}
                )
            METRICS_CACHE = values
            METRICS_CACHE_AT = now
            return dict(values)
        if _engine_is_mlx():
            try:
                payload = _json_url(
                    f"http://127.0.0.1:{port}/metrics", 1.0
                )
                latest = payload.get("latest") or {}
                summary = payload.get("summary") or {}
                server = payload.get("server") or {}
                apc = server.get("apc") or {}
                # mlx-vlm reports peak memory in decimal GB.  The manager's
                # existing public metric is GiB, so preserve that contract.
                peak_gib = float(latest.get("peak_memory_gb") or 0) / (
                    (2**30) / 1e9
                )
                draft_acceptance, draft_cycle = _mlx_vlm_draft_log_stats()
                values = {
                    "local_llm_requests_total": float(
                        summary.get("requests_completed") or 0
                    ),
                    "local_llm_errors_total": float(
                        summary.get("requests_failed") or 0
                    ),
                    "local_llm_checkpoints_created_total": float(
                        (apc.get("stores") or 0)
                        + (apc.get("exact_stores") or 0)
                    ),
                    "local_llm_checkpoints_restored_total": float(
                        apc.get("lookups_hit") or 0
                    ),
                    "local_llm_forced_reprocess_total": float(
                        apc.get("lookups_miss") or 0
                    ),
                    "local_llm_last_cached_tokens": float(
                        latest.get("cached_tokens") or 0
                    ),
                    "local_llm_last_replayed_tokens": 0.0,
                    "local_llm_last_prompt_tokens": float(
                        latest.get("prompt_tokens") or 0
                    ),
                    "local_llm_last_generation_tokens": float(
                        latest.get("generated_tokens") or 0
                    ),
                    "local_llm_last_generation_tps": float(
                        latest.get("decode_tok_s")
                        or summary.get("avg_decode_tok_s")
                        or 0
                    ),
                    "local_llm_last_prompt_tps": float(
                        latest.get("prefill_tok_s") or 0
                    ),
                    "local_llm_prompt_tokens_total": float(
                        summary.get("prompt_tokens_total") or 0
                    ),
                    "local_llm_tokens_generated_total": float(
                        summary.get("generated_tokens_total") or 0
                    ),
                    "local_llm_peak_memory_gib": peak_gib,
                    "local_llm_dflash_acceptance_ratio": draft_acceptance,
                    "local_llm_dflash_tokens_per_cycle": draft_cycle,
                }
            except Exception:
                # The project's patched mlx-lm server still exposes
                # Prometheus text.  Fall through to that parser below.
                pass
            else:
                METRICS_CACHE = values
                METRICS_CACHE_AT = now
                return dict(values)
        try:
            text = _text_url(f"http://127.0.0.1:{port}/metrics", 1.0)
        except Exception:
            return dict(METRICS_CACHE) if now - METRICS_CACHE_AT < 30 else {}
        values: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("#") or " " not in line:
                continue
            key, value = line.rsplit(" ", 1)
            try:
                values[key] = float(value)
            except ValueError:
                pass
        METRICS_CACHE = values
        METRICS_CACHE_AT = now
        return dict(values)


def _command_arg(command: list[str] | tuple[str, ...], flag: str, default: str = "") -> str:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError):
        return default


def _saved_command() -> list[str]:
    state = manager._saved_state()
    return [str(x) for x in state.get("command", [])]


def _default_config() -> dict[str, Any]:
    profile = manager.settings.profiles["llama-turbo4-safe"]
    command = list(profile.command)
    return {
        "engine": "llama.cpp",
        "mlx_model_path": "",
        # Turbo4 is the capacity/quality default.  The 40B analogbox checkpoint
        # passed 4K and 8K retrieval with rotated Turbo4 where its fixed native4
        # cache failed a 4K probe.  Quality-first 35B profiles can still select
        # native8 in the Runtime panel.
        "mlx_kv_bits": 4,
        "mlx_kv_mode": "turbo4",
        "mlx_kv_group_size": 64,
        "mlx_preallocate_kv_size": 131072,
        "mlx_prefill_step_size": 2048,
        "mlx_adaptive_prefill": True,
        "mlx_experimental_context": False,
        "mlx_prompt_cache_mib": 2048,
        "mlx_session_reuse": True,
        "mlx_session_checkpoints": 2,
        "model": Path(_command_arg(command, "-m")).name,
        "model_path": _command_arg(command, "-m"),
        "ctx": int(_command_arg(command, "--ctx-size", "262144")),
        "kv_k": _command_arg(command, "--cache-type-k", "q8_0"),
        "kv_v": _command_arg(command, "--cache-type-v", "turbo4"),
        "batch": _command_arg(command, "--batch-size", "2048"),
        "ubatch": _command_arg(command, "--ubatch-size", "512"),
        "ctx_chk": _command_arg(command, "--ctx-checkpoints", "4"),
        "checkpoint_every": _command_arg(command, "--checkpoint-min-step", "32768"),
        "cache_ram": _command_arg(command, "--cache-ram", "512"),
        "rope_scale": "",
        "yarn_orig_ctx": "262144",
        "profile": "llama-turbo4-safe",
        "mmproj_path": "",
        "image_min_tokens": "",
        "image_max_tokens": "",
        "graph_cap": "",
    }


def _config() -> dict[str, Any]:
    config = _default_config()
    try:
        config.update(json.loads(CUSTOM_PATH.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    # The runtime state file intentionally retains the last command after a
    # clean stop.  Only merge command-line values while that runtime is still
    # alive; otherwise a stale command can silently override a newly saved UI
    # configuration on the next launch.
    runtime = manager.status()
    command = _saved_command() if runtime.get("running") else []
    if runtime.get("engine"):
        config["engine"] = str(runtime["engine"])
    # A running MLX command contains Python's ``-m module``. Never parse that
    # as llama.cpp's ``-m model.gguf``; the saved config keeps both engine
    # selections so switching back to GGUF remains lossless.
    if command and not str(config.get("engine", "")).lower().startswith("mlx"):
        config.update({
            "model": Path(_command_arg(command, "-m", config["model_path"])).name,
            "model_path": _command_arg(command, "-m", config["model_path"]),
            "ctx": int(_command_arg(command, "--ctx-size", str(config["ctx"]))),
            "kv_k": _command_arg(command, "--cache-type-k", config["kv_k"]),
            "kv_v": _command_arg(command, "--cache-type-v", config["kv_v"]),
            "batch": _command_arg(command, "--batch-size", str(config["batch"])),
            "ubatch": _command_arg(command, "--ubatch-size", str(config["ubatch"])),
            "ctx_chk": _command_arg(command, "--ctx-checkpoints", str(config["ctx_chk"])),
            "checkpoint_every": _command_arg(command, "--checkpoint-min-step", str(config["checkpoint_every"])),
        })
    config["kv_types"] = KV_TYPES
    return config


def _save_config(update: dict[str, Any]) -> dict[str, Any]:
    config = _config()
    config.pop("kv_types", None)
    mapping = {
        "LOCAL_LLM_ENGINE": "engine",
        "LOCAL_LLM_MLX_MODEL_PATH": "mlx_model_path",
        "LOCAL_LLM_MLX_KV_BITS": "mlx_kv_bits",
        "LOCAL_LLM_MLX_KV_MODE": "mlx_kv_mode",
        "LOCAL_LLM_MLX_KV_GROUP_SIZE": "mlx_kv_group_size",
        "LOCAL_LLM_MLX_PREALLOCATE_KV_SIZE": "mlx_preallocate_kv_size",
        "LOCAL_LLM_MLX_PREFILL_STEP_SIZE": "mlx_prefill_step_size",
        "LOCAL_LLM_MLX_ADAPTIVE_PREFILL": "mlx_adaptive_prefill",
        "LOCAL_LLM_MLX_EXPERIMENTAL_CONTEXT": "mlx_experimental_context",
        "LOCAL_LLM_MLX_PROMPT_CACHE_MIB": "mlx_prompt_cache_mib",
        "LOCAL_LLM_MLX_SESSION_REUSE": "mlx_session_reuse",
        "LOCAL_LLM_MLX_SESSION_CHECKPOINTS": "mlx_session_checkpoints",
        "LOCAL_LLM_QWEN_MODEL": "model",
        "LOCAL_LLM_QWEN_MODEL_PATH": "model_path",
        "LOCAL_LLM_QWEN_CTX": "ctx",
        "LOCAL_LLM_QWEN_CACHE_TYPE_K": "kv_k",
        "LOCAL_LLM_QWEN_CACHE_TYPE_V": "kv_v",
        "LOCAL_LLM_QWEN_BATCH": "batch",
        "LOCAL_LLM_QWEN_UBATCH": "ubatch",
        "LOCAL_LLM_QWEN_CTX_CHECKPOINTS": "ctx_chk",
        "LOCAL_LLM_QWEN_CHECKPOINT_EVERY_NT": "checkpoint_every",
        "LOCAL_LLM_QWEN_CACHE_RAM": "cache_ram",
        "LOCAL_LLM_QWEN_ROPE_SCALE": "rope_scale",
        "LOCAL_LLM_QWEN_YARN_ORIG_CTX": "yarn_orig_ctx",
        "LOCAL_LLM_QWEN_MMPROJ_PATH": "mmproj_path",
        "LOCAL_LLM_QWEN_IMAGE_MIN_TOKENS": "image_min_tokens",
        "LOCAL_LLM_QWEN_IMAGE_MAX_TOKENS": "image_max_tokens",
        "GGML_CUDA_GRAPH_CACHE_MAX": "graph_cap",
    }
    # The browser preserves compatibility with the old environment-variable
    # field names, while API users naturally send the keys returned by
    # GET /api/config.  Accept both representations so PUT is symmetric.
    for target in mapping.values():
        if target in update and update[target] is not None:
            config[target] = update[target]
    for source, target in mapping.items():
        if source in update and update[source] is not None:
            config[target] = update[source]
    for numeric in (
        "ctx", "batch", "ubatch", "ctx_chk", "checkpoint_every", "cache_ram",
        "mlx_kv_bits", "mlx_kv_group_size", "mlx_preallocate_kv_size", "mlx_prefill_step_size",
        "mlx_prompt_cache_mib", "mlx_session_checkpoints",
    ):
        try:
            config[numeric] = int(config[numeric])
        except (TypeError, ValueError):
            pass
    for boolean in (
        "mlx_session_reuse",
        "mlx_adaptive_prefill",
        "mlx_experimental_context",
    ):
        if boolean in config:
            config[boolean] = str(config[boolean]).lower() not in (
                "false", "0", "no", "",
            )
    CUSTOM_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_PATH.write_text(json.dumps(config, indent=2) + "\n")
    return config


def _replace_arg(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command.extend([flag, value])


def _remove_arg(command: list[str], flag: str) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + 2]


def _custom_profile(config: dict[str, Any]) -> Profile:
    command = list(manager.settings.profiles["llama-turbo4-safe"].command)
    _replace_arg(command, "-m", str(config["model_path"]))
    _replace_arg(command, "--ctx-size", str(config["ctx"]))
    _replace_arg(command, "--batch-size", str(config["batch"]))
    _replace_arg(command, "--ubatch-size", str(config["ubatch"]))
    _replace_arg(command, "--ctx-checkpoints", str(config["ctx_chk"]))
    _replace_arg(command, "--checkpoint-min-step", str(config["checkpoint_every"]))
    _replace_arg(command, "--cache-ram", str(config.get("cache_ram", 512)))
    k_type = str(config.get("kv_k") or "q8_0")
    v_type = str(config.get("kv_v") or "turbo4")
    if k_type not in KV_TYPES or v_type not in KV_TYPES:
        raise ValueError("unsupported KV cache type")
    _replace_arg(command, "--cache-type-k", k_type)
    _replace_arg(command, "--cache-type-v", v_type)
    # TurboQuant+ protects high-GQA models by silently promoting symmetric
    # turbo K to q8_0.  When the operator explicitly selects a symmetric pair,
    # disable that safeguard so the launched configuration and memory estimate
    # match what the UI reports.  Asymmetric selections retain the safeguard.
    if k_type == v_type and k_type.startswith("turbo"):
        command = ["/usr/bin/env", "TURBO_AUTO_ASYMMETRIC=0", *command]
    for flag in ("--rope-scaling", "--rope-scale", "--yarn-orig-ctx", "--rope-freq-base", "--override-kv"):
        _remove_arg(command, flag)
    ctx = int(config["ctx"])
    if ctx > 262144:
        scale = float(config.get("rope_scale") or ctx / 262144)
        override_key = "qwen35" if "27B" in str(config["model_path"]) else "qwen35moe"
        command.extend([
            "--rope-scaling", "yarn", "--rope-scale", f"{scale:.8f}",
            "--yarn-orig-ctx", str(config.get("yarn_orig_ctx") or 262144),
            "--rope-freq-base", "10000000",
            "--override-kv", f"{override_key}.context_length=int:{ctx}",
        ])
    name = re.sub(r"[^a-zA-Z0-9_.-]", "-", Path(str(config["model_path"])).stem)[:48]
    return Profile(
        name=f"custom-{name}", engine="llama.cpp",
        description="Custom profile from the workbench UI", port=8097,
        health_path="/health", command=tuple(command),
    )


def _mlx_profile(config: dict[str, Any]) -> Profile:
    """Build an MLX-LM runtime profile from the workbench configuration."""
    model = str(config.get("mlx_model_path") or "")
    model_path = Path(model).expanduser()
    if not model or not model_path.is_dir():
        raise ValueError("select an MLX model directory first")
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map", {})
            expected_shards = sorted(set(weight_map.values()))
        except (OSError, json.JSONDecodeError):
            raise ValueError("MLX model shard index is unreadable") from None
        missing_shards = [
            shard
            for shard in expected_shards
            if not (model_path / shard).is_file()
            or (model_path / shard).stat().st_size == 0
        ]
        if missing_shards:
            raise ValueError(
                "MLX model download is incomplete: "
                f"{len(missing_shards)} of {len(expected_shards)} weight shards "
                "are missing"
            )
    elif not any(model_path.glob("*.safetensors")):
        raise ValueError("MLX model download is incomplete: no weight shards found")
    kv_bits = str(config.get("mlx_kv_bits") or "4")
    if kv_bits not in ("4", "8"):
        raise ValueError("MLX KV bits must be 4 or 8")
    kv_mode = str(config.get("mlx_kv_mode") or f"native{kv_bits}")
    if kv_mode not in ("native4", "native8", "turbo3", "turbo4"):
        raise ValueError("MLX KV mode must be native4, native8, turbo3, or turbo4")
    kv_bits = "8" if kv_mode == "native8" else "4"
    kv_group_size = int(config.get("mlx_kv_group_size") or 64)
    if kv_group_size not in (32, 64, 128):
        raise ValueError("MLX KV group size must be 32, 64, or 128")
    preallocate = max(
        0,
        min(int(config.get("mlx_preallocate_kv_size") or 0), 262144),
    )
    if preallocate and kv_mode == "turbo3":
        raise ValueError("MLX fixed preallocation is not supported by Turbo3")
    if preallocate:
        plan = _mlx_memory_plan(
            model_path,
            context=preallocate,
            kv_mode=kv_mode,
            group_size=kv_group_size,
            checkpoints=max(
                2, int(config.get("mlx_session_checkpoints") or 2)
            ),
        )
        experimental_context = bool(
            config.get("mlx_experimental_context", False)
        )
        if not plan["safe"] and not experimental_context:
            raise MemoryError(
                f"unsafe MLX fixed context for this 32 GiB Mac: estimated "
                f"{plan['peak_gib']:.2f} GiB peak exceeds the {plan['safe_peak_gib']:.1f} "
                f"GiB stability budget. Use at most {plan['recommended_context']:,} "
                f"tokens for this model/KV layout."
            )
        if not plan["safe"] and experimental_context:
            if not _clamshell_closed():
                raise MemoryError(
                    "experimental over-budget MLX context requires the lid "
                    "closed to minimize WindowServer/display allocations"
                )
            if _metal_limit() < 30720:
                raise MemoryError(
                    "experimental over-budget MLX context requires the "
                    "30,720 MiB Metal ceiling"
                )
    prefill_step = max(
        128,
        min(int(config.get("mlx_prefill_step_size") or 2048), 8192),
    )
    prompt_cache_mib = max(256, min(int(config.get("mlx_prompt_cache_mib") or 2048), 4096))
    command = [
        sys.executable, "-m", "local_llm_control.mlx_quant_server",
        "--kv-bits", kv_bits, "--kv-mode", kv_mode,
        "--kv-group-size", str(kv_group_size),
        "--model", str(model_path),
        "--host", "0.0.0.0", "--port", "8098",
        "--decode-concurrency", "1", "--prompt-concurrency", "1",
        "--prefill-step-size", str(prefill_step),
        "--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0.0",
        "--chat-template-args", '{"enable_thinking":false}',
    ]
    if not bool(config.get("mlx_adaptive_prefill", True)):
        command.append("--no-adaptive-prefill")
    session_reuse = bool(config.get("mlx_session_reuse", True))
    if preallocate:
        command.extend(["--preallocate-kv-size", str(preallocate)])
        if session_reuse:
            # Keep the fixed arena across turns and reuse the shared prefix so
            # follow-up coding prompts skip re-prefilling unchanged history.
            command.extend([
                "--session-reuse",
                "--session-checkpoints", str(max(2, int(config.get("mlx_session_checkpoints") or 2))),
            ])
        else:
            command.extend(["--prompt-cache-size", "0", "--prompt-cache-bytes", "0"])
    else:
        command.extend([
            "--prompt-cache-size", "1",
            "--prompt-cache-bytes", str(prompt_cache_mib * 1024 * 1024),
        ])
    name = re.sub(r"[^a-zA-Z0-9_.-]", "-", Path(model).name)[:44]
    return Profile(
        name=f"custom-mlx-{name}", engine="mlx-lm",
        description="Custom MLX profile from the workbench UI", port=8098,
        health_path="/v1/models", command=tuple(command),
    )


def _mlx_memory_plan(
    model_path: Path,
    *,
    context: int,
    kv_mode: str,
    group_size: int,
    checkpoints: int = 1,
) -> dict[str, Any]:
    """Estimate fixed-arena peak from model metadata and measured M1 Max margins."""
    try:
        raw = json.loads((model_path / "config.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}
    text_config = raw.get("text_config") or raw
    layer_types = text_config.get("layer_types") or []
    full_layers = sum(layer == "full_attention" for layer in layer_types)
    linear_layers = sum(layer == "linear_attention" for layer in layer_types)
    kv_heads = int(text_config.get("num_key_value_heads") or 0)
    head_dim = int(text_config.get("head_dim") or 0)
    weight_bytes = sum(
        path.stat().st_size for path in model_path.glob("*.safetensors")
    )
    bits = 8 if kv_mode == "native8" else 4
    effective_bits = bits + 32 / group_size
    kv_bytes = (
        full_layers
        * 2
        * kv_heads
        * head_dim
        * context
        * effective_bits
        / 8
    )
    is_dense = "moe" not in str(text_config.get("model_type", "")).lower()
    recurrent_mib = linear_layers * (3 if is_dense else 2)
    weight_gib = weight_bytes / 2**30
    heavy_moe = not is_dense and weight_gib > 21
    # Calibrated against the 40B fixed-128K startup/decode run and the 35B
    # adaptive long-fill run. MoE transient attention workspace grows with
    # retained depth: 4.3 GiB at shallow decode, rising to 8.2 GiB at the
    # measured 230K point. Modeling that curve avoids both claiming 260K for a
    # larger 5-bit model and needlessly limiting it to the worst-case margin at
    # 64K/128K.
    def workspace_for(target_context: int) -> float:
        if is_dense:
            return 5.8
        growth = 3.9 * max(0, target_context) / 230541
        # The 22.2 GiB Qwopus 5-bit run needed about another GiB of live
        # tokenizer/Metal pressure. At 115K it reached ~30.2 GiB and drove
        # available memory below 400 MiB, so retain that measured allowance.
        return 4.3 + min(3.9, growth) + (1.0 if heavy_moe else 0.0)

    workspace_gib = workspace_for(context)
    fixed_weight_gib = (
        weight_bytes / 2**30
        + recurrent_mib * max(1, checkpoints) / 1024
    )
    peak_gib = fixed_weight_gib + kv_bytes / 2**30 + workspace_gib
    # Leave at least ~1.5 GiB outside the inference estimate for macOS and
    # telemetry. Profiles near the old 31 GiB line were too sensitive to
    # ordinary transient allocations.
    safe_peak_gib = 29.8 if heavy_moe else 30.5
    if full_layers and kv_heads and head_dim:
        bytes_per_token = (
            full_layers * 2 * kv_heads * head_dim * effective_bits / 8
        )
        native_context = int(
            text_config.get("max_position_embeddings") or 262144
        )
        recommended = 0
        # Expose operator-friendly 16K steps. Re-evaluate workspace at every
        # depth because hybrid-MoE transient cost is not constant.
        for candidate in range(16384, native_context + 1, 16384):
            candidate_peak = (
                fixed_weight_gib
                + bytes_per_token * candidate / 2**30
                + workspace_for(candidate)
            )
            if candidate_peak <= safe_peak_gib:
                recommended = candidate
            else:
                break
        if is_dense and full_layers >= 20:
            recommended = min(recommended, 131072)
    else:
        recommended = context
    return {
        "model_weights_gib": round(weight_bytes / 2**30, 3),
        "full_attention_layers": full_layers,
        "linear_attention_layers": linear_layers,
        "effective_kv_bits": effective_bits,
        "kv_gib": round(kv_bytes / 2**30, 3),
        "recurrent_checkpoint_mib": recurrent_mib,
        "workspace_margin_gib": workspace_gib,
        "peak_gib": round(peak_gib, 3),
        "safe_peak_gib": safe_peak_gib,
        "safe": peak_gib <= safe_peak_gib,
        "recommended_context": recommended,
    }


def _engine_is_mlx(config: dict[str, Any] | None = None) -> bool:
    if config is None:
        runtime = manager.status()
        if runtime.get("running") and runtime.get("engine"):
            return str(runtime["engine"]).lower().startswith("mlx")
        config = _config()
    return str(config.get("engine", "")).lower().startswith("mlx")


def _engine_is_dflash(config: dict[str, Any] | None = None) -> bool:
    if config is None:
        runtime = manager.status()
        if runtime.get("running") and runtime.get("engine"):
            return str(runtime["engine"]).lower() == "mlx-dflash"
        config = _config()
    return str(config.get("engine", "")).lower() == "mlx-dflash"


def _engine_is_dspark(config: dict[str, Any] | None = None) -> bool:
    if config is None:
        runtime = manager.status()
        if runtime.get("running") and runtime.get("engine"):
            return str(runtime["engine"]).lower() == "mlx-dspark"
        config = _config()
    return str(config.get("engine", "")).lower() == "mlx-dspark"


def _active_port() -> int:
    """Port of the currently running (or configured) inference runtime."""
    port = _command_arg(_saved_command(), "--port", "")
    if port.isdigit():
        return int(port)
    return 8098 if _engine_is_mlx() else 8097


def _mlx_meta(path: Path) -> dict[str, Any]:
    try:
        cfg = json.loads((path / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    quant = cfg.get("quantization", {})
    bits = quant.get("bits")
    mode = quant.get("mode", "affine")
    quant_label = f"{bits}-bit {mode}" if bits else ""
    try:
        optiq = json.loads((path / "optiq_metadata.json").read_text())
        achieved = float(optiq["achieved_bpw"])
        candidates = "/".join(str(x) for x in optiq.get("candidate_bits", []))
        quant_label = f"OptiQ {achieved:.2f} bpw"
        if candidates:
            quant_label += f" ({candidates}-bit mixed)"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {
        "model_type": cfg.get("model_type", ""),
        "quant": quant_label,
    }


def _models() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(GGUF_ROOT.rglob("*.gguf")):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < 100 * 2**20:
            continue
        name = path.name
        quant = next((q for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "TQ4", "TQ3") if q.lower() in name.lower()), "")
        rows.append({
            "path": str(path), "name": name, "size_bytes": size,
            "size_mib": round(size / 2**20, 1), "repo_dir": path.parent.name,
            "engine": "llama.cpp",
            "arch": "qwen35moe" if "35B-A3B" in name else "qwen35" if "27B" in name else "",
            "quant": quant, "is_drafter": "draft" in name.lower(),
        })
    if MLX_ROOT.exists():
        for path in sorted(MLX_ROOT.iterdir()):
            if not path.is_dir() or not (path / "config.json").exists():
                continue
            shards = list(path.glob("*.safetensors"))
            if not shards:
                continue  # still downloading or not a weights directory
            index_path = path / "model.safetensors.index.json"
            if index_path.exists():
                try:
                    expected = set(
                        json.loads(index_path.read_text())
                        .get("weight_map", {})
                        .values()
                    )
                except (OSError, json.JSONDecodeError):
                    continue
                if any(
                    not (path / shard).is_file()
                    or (path / shard).stat().st_size == 0
                    for shard in expected
                ):
                    continue  # do not offer a partially downloaded model
            # Count the materialized model once. Hugging Face's local-dir
            # metadata lives under .cache and can contain partial/download
            # bookkeeping that otherwise makes the UI over-report model size.
            size = sum(p.stat().st_size for p in path.iterdir() if p.is_file())
            meta = _mlx_meta(path)
            rows.append({
                "path": str(path), "name": path.name, "size_bytes": size,
                "size_mib": round(size / 2**20, 1), "repo_dir": path.name,
                "engine": "mlx", "arch": meta.get("model_type", ""),
                "quant": meta.get("quant", ""), "is_drafter": False,
            })
    return rows


def _gguf_meta(path: Path) -> dict[str, Any]:
    # Reuse llama.cpp's model endpoint for the active model; keep offline parsing conservative.
    active = manager.status()
    if active["running"] and Path(_command_arg(_saved_command(), "-m", "")) == path:
        try:
            model = _json_url("http://127.0.0.1:8097/v1/models")["data"][0]
            return model.get("meta", {})
        except Exception:
            pass
    return {"size": path.stat().st_size, "general.name": path.stem}


def _session_stats() -> dict[str, Any]:
    if _engine_is_mlx():
        metrics = _llama_metrics()
        runtime_engine = str(manager.status().get("engine") or "mlx-lm")
        command = _saved_command()
        if "--draft-model" in command and _command_arg(
            command, "--draft-kind", ""
        ) == "dflash":
            runtime_engine = "mlx-vlm-dflash"
        return {
            "engine": (
                "mlx-dspark"
                if _engine_is_dspark()
                else "mlx-dflash" if _engine_is_dflash() else runtime_engine
            ),
            "requests": int(metrics.get("local_llm_requests_total", 0)),
            "checkpoints_created": int(
                metrics.get("local_llm_checkpoints_created_total", 0)
            ),
            "checkpoints_restored": int(
                metrics.get("local_llm_checkpoints_restored_total", 0)
            ),
            "forced_reprocess": int(
                metrics.get("local_llm_forced_reprocess_total", 0)
            ),
            "avg_prompt_ms": None,
            "p95_prompt_ms": None,
            "last_cached_tokens": int(
                metrics.get("local_llm_last_cached_tokens", 0)
            ),
            "last_replayed_tokens": int(
                metrics.get("local_llm_last_replayed_tokens", 0)
            ),
        }
    state = manager.status()
    log_path = state.get("log") or manager._saved_state().get("log")
    try:
        text = Path(log_path).read_text(errors="replace")
    except (FileNotFoundError, TypeError):
        text = ""
    prompt_ms = [float(x) for x in re.findall(r"prompt eval time\s*=\s*([0-9.]+) ms", text)]
    return {
        "engine": "mlx-lm" if _engine_is_mlx() else "llama.cpp",
        "requests": len(prompt_ms),
        "checkpoints_created": text.count("created context checkpoint"),
        "checkpoints_restored": text.count("restored context checkpoint"),
        "forced_reprocess": text.count("forcing full prompt"),
        "avg_prompt_ms": round(sum(prompt_ms) / len(prompt_ms), 1) if prompt_ms else None,
        "p95_prompt_ms": sorted(prompt_ms)[int((len(prompt_ms) - 1) * .95)] if prompt_ms else None,
    }


def _snapshot() -> dict[str, Any]:
    telemetry = _apple_telemetry()
    gpu = _gpu_state(telemetry)
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    metrics = _llama_metrics()
    runtime = manager.status()
    session = _session_stats()
    metal_ceiling = _metal_limit() or TOTAL_MIB
    return {
        "ts": int(time.time()), "gpu_util": gpu["util_pct"],
        "vram_used_mib": gpu["used_mib"], "vram_total_mib": TOTAL_MIB,
        "gpu_mem_util": gpu["mem_util_pct"], "gpu_power_w": gpu.get("power_w"),
        "gpu_temp": gpu.get("temp_c"), "gpu_fan": gpu.get("fan_rpm"),
        "gpu_pstate": gpu.get("thermal_state") or "Metal",
        "cpu_pct": psutil.cpu_percent(None), "mem_used_mib": vm.used // 2**20,
        "mem_free_mib": vm.free // 2**20,
        "mem_available_mib": vm.available // 2**20,
        "vram_free_mib": gpu["free_mib"],
        "metal_headroom_mib": max(0, metal_ceiling - gpu["used_mib"]),
        "metal_ceiling_mib": metal_ceiling,
        "mem_total_mib": vm.total // 2**20, "swap_used_mib": swap.used // 2**20,
        "load_1": os.getloadavg()[0], "load_5": os.getloadavg()[1], "load_15": os.getloadavg()[2],
        "eval_tok_s": metrics.get(
            "local_llm_last_generation_tps",
            metrics.get("llamacpp:predicted_tokens_seconds", 0),
        ),
        "prompt_tok_s": metrics.get(
            "local_llm_last_prompt_tps",
            metrics.get("llamacpp:prompt_tokens_seconds", 0),
        ),
        "telemetry_available": telemetry.get("available", False),
        "counters": {
            "requests_total": session["requests"],
            "errors_total": sum(1 for x in EVENTS if x.get("kind") == "error" and x.get("ts", 0) >= STARTED_AT),
            "forced_reprocess_total": session["forced_reprocess"],
            "checkpoints_created_total": session["checkpoints_created"],
            "checkpoints_restored_total": session["checkpoints_restored"],
            "session_start_ts": int(STARTED_AT),
            "prompt_tokens_total": int(metrics.get(
                "local_llm_prompt_tokens_total",
                metrics.get("llamacpp:prompt_tokens_total", 0),
            )),
            "tokens_predicted_total": int(metrics.get(
                "local_llm_tokens_generated_total",
                metrics.get("llamacpp:tokens_predicted_total", 0),
            )),
            "last_eval_tok_s": metrics.get(
                "local_llm_last_generation_tps",
                metrics.get("llamacpp:predicted_tokens_seconds", 0),
            ),
            "last_prompt_tok_s": metrics.get(
                "local_llm_last_prompt_tps",
                metrics.get("llamacpp:prompt_tokens_seconds", 0),
            ),
            "last_prefill_step": int(
                metrics.get("local_llm_last_prefill_step", 0)
            ),
            "last_target_depth": int(
                metrics.get("local_llm_last_target_depth", 0)
            ),
            "dflash_acceptance_ratio": metrics.get(
                "local_llm_dflash_acceptance_ratio", 0
            ),
            "dflash_tokens_per_cycle": metrics.get(
                "local_llm_dflash_tokens_per_cycle", 0
            ),
        },
        "services": {
            "local-llm-qwen.service": {
                "active": runtime["running"],
                "active_since": runtime.get("started_at"), "n_restarts": 0,
            }
        },
    }


def _sampler() -> None:
    while True:
        started = time.monotonic()
        try:
            snap = _snapshot()
            with HISTORY_LOCK:
                HISTORY.append(snap)
            _ensure_mactop(snap.get("telemetry_available"))
            manager.ensure_caffeinate()
            if SERVER_MODE_FLAG.exists():
                _enable_server_guard()
        except Exception as exc:
            _event("error", f"sampler: {exc}")
        time.sleep(max(1, SAMPLE_SECONDS - (time.monotonic() - started)))


def start_sampler() -> None:
    global SAMPLER_STARTED
    if SAMPLER_STARTED:
        return
    SAMPLER_STARTED = True
    _load_events()
    _ensure_mactop()
    manager.ensure_caffeinate()
    if _metal_limit() or SERVER_MODE_FLAG.exists():
        _enable_server_guard()
    threading.Thread(target=_sampler, name="mac-metrics", daemon=True).start()


class ConfigUpdate(BaseModel):
    model_config = {"extra": "allow"}


class ModelRequest(BaseModel):
    path: str


class ProfileRequest(BaseModel):
    name: str


class DownloadRequest(BaseModel):
    repo: str
    file: str | None = None


class TestRequest(BaseModel):
    tokens: int = 128
    gen: int = 64


class SweepRequest(BaseModel):
    contexts: list[int]
    tokens: int = 2048


class SamplingRequest(BaseModel):
    sample_rate_s: int


class MetalLimitRequest(BaseModel):
    limit_mib: int
    restart_runtime: bool = False


class NeedleProbeRequest(BaseModel):
    target_tokens: int = 32768


@router.get("/api/status")
def status() -> dict[str, Any]:
    runtime = manager.status()
    gpu = _gpu_state()
    gpu["power_w"] = None
    return {
        "service": {
            "active": runtime["running"], "healthy": runtime["running"],
            "pid": runtime.get("pid"), "api_base": f"http://{_lan_host()}:{_active_port()}/v1",
            "profile": runtime.get("profile"),
        },
        "gpu": gpu, "config": _config(), "system": _system_info(),
    }


@router.get("/api/ui/bootstrap")
def ui_bootstrap() -> dict[str, Any]:
    payload = status()
    payload.update({"models": _models(), "tasks": list(TASKS.values()), "auth_enabled": False, "bind_host": "0.0.0.0"})
    return payload


@router.get("/api/manager/health")
def manager_health() -> dict[str, Any]:
    return {"status": "healthy", "mode": "launchd", "active": True, "pid": os.getpid(), "uptime": time.time() - STARTED_AT, "drift": 0}


@router.get("/api/config")
def get_config() -> dict[str, Any]:
    return _config()


@router.put("/api/config")
def update_config(update: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, **_save_config(update)}


@router.get("/api/env-raw")
def env_raw() -> dict[str, str]:
    return {"content": json.dumps(_config(), indent=2, default=str)}


@router.put("/api/env-raw")
def put_env_raw(payload: dict[str, str]) -> dict[str, bool]:
    try:
        value = json.loads(payload.get("content", "{}"))
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Expected JSON configuration: {exc}") from exc
    _save_config(value)
    return {"ok": True}


START_ERRORS = (KeyError, ValueError, FileNotFoundError, MemoryError, RuntimeError, TimeoutError)


async def _start_custom() -> dict[str, Any]:
    """Start the configured custom profile, mapping failures to clean HTTP errors."""
    try:
        config = _config()
        profile = _mlx_profile(config) if _engine_is_mlx(config) else _custom_profile(config)
        result = await manager.start_profile(profile)
    except START_ERRORS as exc:
        _event("error", f"start failed: {exc}")
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _event("service_state", f"started {profile.name}")
    return result


@router.post("/api/service/start")
async def service_start() -> dict[str, Any]:
    return await _start_custom()


@router.post("/api/service/stop")
async def service_stop() -> dict[str, Any]:
    result = await manager.stop()
    _event("service_state", "stopped inference runtime")
    return result


@router.post("/api/service/restart")
@router.post("/api/service/restart-clean")
async def service_restart() -> dict[str, Any]:
    await manager.stop()
    return await _start_custom()


@router.post("/api/service/reset-stats")
def reset_stats() -> dict[str, Any]:
    return {"ok": True, "msg": "Counters are derived from the current llama log"}


@router.get("/api/session/stats")
def session_stats() -> dict[str, Any]:
    return _session_stats()


@router.get("/api/services/stack")
def services_stack() -> list[dict[str, Any]]:
    runtime = manager.status()
    controller = psutil.Process(os.getpid())
    if _engine_is_dflash():
        engine = "MLX DFlash speculative server"
    elif _engine_is_mlx():
        engine = "MLX-LM (KV-quant server)"
    else:
        engine = "TurboQuant Metal llama.cpp"
    return [
        {"key": "qwen", "desc": engine, "port": _active_port(),
         "active": "active" if runtime["running"] else "inactive", "healthy": runtime["running"],
         "pid": runtime.get("pid"), "since": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(runtime["started_at"])) if runtime.get("started_at") else None},
        {"key": "manager", "desc": "Mac workbench controller", "port": MANAGER_PORT,
         "active": "active", "healthy": True, "pid": controller.pid, "since": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(STARTED_AT))},
    ]


@router.post("/api/services/stack/{key}/{action}")
async def stack_action(key: str, action: str) -> dict[str, Any]:
    if key == "manager":
        if action == "restart":
            os.kill(os.getpid(), signal.SIGTERM)
            return {"ok": True}
        raise HTTPException(409, "The manager cannot stop itself from the dashboard")
    if key != "qwen":
        raise HTTPException(404, "unknown service")
    if action == "start":
        await _start_custom()
    elif action == "stop":
        await manager.stop()
    elif action == "restart":
        await manager.stop(); await _start_custom()
    else:
        raise HTTPException(400, "unknown action")
    return {"ok": True}


@router.get("/api/services/stack/{key}/logs")
def stack_logs(key: str, lines: int = 120) -> dict[str, str]:
    path = ROOT / "logs/control-service.log" if key == "manager" else Path(manager._saved_state().get("log", ROOT / "logs/llama-turbo4-safe.log"))
    try:
        content = "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        content = ""
    return {"content": content}


@router.get("/api/models")
def models() -> list[dict[str, Any]]:
    return _models()


def _resolve_model(path: str) -> tuple[Path, bool]:
    """Validate a model path inside either root; returns (path, is_mlx)."""
    target = Path(path).resolve()
    if target.is_relative_to(MLX_ROOT.resolve()) and target.is_dir():
        return target, True
    if target.is_relative_to(GGUF_ROOT.resolve()) and target.exists():
        return target, False
    raise HTTPException(404, "model not found")


@router.get("/api/models/info")
def model_info(path: str) -> dict[str, Any]:
    target, is_mlx = _resolve_model(path)
    if is_mlx:
        size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
        return {"engine": "mlx", "size": size, "general.name": target.name, **_mlx_meta(target)}
    return _gguf_meta(target)


@router.post("/api/models/use")
async def model_use(req: ModelRequest) -> dict[str, Any]:
    target, is_mlx = _resolve_model(req.path)
    if is_mlx:
        _save_config({"LOCAL_LLM_ENGINE": "mlx-lm", "LOCAL_LLM_MLX_MODEL_PATH": str(target)})
    else:
        _save_config({"LOCAL_LLM_ENGINE": "llama.cpp",
                      "LOCAL_LLM_QWEN_MODEL": target.name,
                      "LOCAL_LLM_QWEN_MODEL_PATH": str(target)})
    await manager.stop()
    return {"ok": True, "runtime": await _start_custom()}


@router.delete("/api/models")
async def model_delete(req: ModelRequest) -> dict[str, Any]:
    target, is_mlx = _resolve_model(req.path)
    config = _config()
    selected = {Path(config["model_path"]).expanduser().resolve()}
    if config.get("mlx_model_path"):
        selected.add(Path(config["mlx_model_path"]).expanduser().resolve())
    if target in selected:
        raise HTTPException(409, "cannot delete the selected model")
    if is_mlx:
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"ok": True}


@router.get("/api/models/active")
def models_active() -> dict[str, Any]:
    config = _config()
    return {"models": [_active_model_name(config)] if manager.status()["running"] else []}


@router.get("/api/hf/search")
def hf_search(q: str) -> Any:
    try:
        return _json_url(f"https://huggingface.co/api/models?search={urllib.parse.quote(q)}&limit=15&filter=gguf", 15)
    except Exception as exc:
        raise HTTPException(502, f"Hugging Face search failed: {exc}") from exc


@router.get("/api/hf/files")
def hf_files(repo: str) -> Any:
    try:
        return _json_url(f"https://huggingface.co/api/models/{urllib.parse.quote(repo)}/tree/main?recursive=true", 15)
    except Exception as exc:
        raise HTTPException(502, f"Hugging Face file listing failed: {exc}") from exc


def _new_task(task_id: str, extra: dict[str, Any] | None = None) -> None:
    """Register a task and prune old finished ones so TASKS cannot grow forever."""
    with TASK_LOCK:
        finished = [tid for tid, t in TASKS.items()
                    if t.get("status") in ("done", "failed", "cancelled")]
        for tid in finished[:-TASK_FINISHED_KEEP] if len(finished) > TASK_FINISHED_KEEP else []:
            TASKS.pop(tid, None)
            TASK_PROCESSES.pop(tid, None)
        TASKS[task_id] = {"id": task_id, "status": "running", "progress": 0,
                          "log": [], "result": None, "created_at": int(time.time()),
                          **(extra or {})}


def _download_worker(task_id: str, repo: str, filename: str | None) -> None:
    slug = repo.replace("/", "--")
    target = (GGUF_ROOT if filename and filename.lower().endswith(".gguf") else MLX_ROOT) / slug
    target.mkdir(parents=True, exist_ok=True)
    aria = shutil.which("aria2c")
    if filename and aria:
        url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(filename)}"
        command = [aria, "--continue=true", "--max-connection-per-server=16", "--split=16",
                   "--min-split-size=16M", "--file-allocation=none", "--summary-interval=5",
                   f"--dir={target}", f"--out={Path(filename).name}", url]
    else:
        command = [shutil.which("hf") or "/opt/homebrew/bin/hf", "download", repo]
        if filename:
            command.append(filename)
        command += ["--local-dir", str(target)]
    command = ["/usr/bin/caffeinate", "-dims", *command]
    with TASK_LOCK:
        TASKS[task_id]["log"].append(" ".join(command[:-2]))
    download_env = os.environ.copy()
    # Xet opened hundreds of CloudFront sockets and repeatedly stalled on this
    # host. Keep normal resumable HTTP transfers as the reliable default;
    # callers may explicitly override either variable.
    download_env.setdefault("HF_XET_HIGH_PERFORMANCE", "0")
    download_env.setdefault("HF_HUB_DISABLE_XET", "1")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True, env=download_env)
    TASK_PROCESSES[task_id] = proc
    for line in proc.stdout or []:
        with TASK_LOCK:
            TASKS[task_id]["log"] = (TASKS[task_id]["log"] + [line.rstrip()])[-200:]
    code = proc.wait()
    with TASK_LOCK:
        TASKS[task_id].update({"status": "done" if code == 0 else "failed", "progress": 100 if code == 0 else 0, "result": {"path": str(target), "returncode": code}})


@router.post("/api/hf/download")
def hf_download(req: DownloadRequest, background: BackgroundTasks) -> dict[str, str]:
    task_id = f"download-{uuid.uuid4().hex[:10]}"
    _new_task(task_id, {"repo": req.repo, "file": req.file})
    background.add_task(_download_worker, task_id, req.repo, req.file)
    return {"task_id": task_id, "task": task_id}


@router.get("/api/tasks")
def tasks() -> list[dict[str, Any]]:
    return list(TASKS.values())


@router.get("/api/tasks/{task_id}")
def task(task_id: str) -> dict[str, Any]:
    if task_id not in TASKS:
        raise HTTPException(404, "task not found")
    return TASKS[task_id]


@router.post("/api/tasks/{task_id}/cancel")
def task_cancel(task_id: str) -> dict[str, bool]:
    proc = TASK_PROCESSES.get(task_id)
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
    if task_id in TASKS:
        TASKS[task_id]["status"] = "cancelled"
    return {"ok": True}


@router.get("/api/hf/cli-status")
def hf_cli_status() -> dict[str, Any]:
    cli = shutil.which("hf")
    token = Path.home() / ".cache/huggingface/token"
    version = _run([cli, "--version"]).stdout.strip() if cli else ""
    return {"cli_name": "hf" if cli else None, "cli_path": cli, "version": version, "token_present": token.exists(), "cache_path": str(Path.home() / ".cache/huggingface")}


@router.get("/api/profiles")
def profiles() -> list[dict[str, Any]]:
    current_settings = refresh_manager_settings()
    rows = []
    for profile in current_settings.profiles.values():
        command = list(profile.command)
        model_arg = _command_arg(
            command, "--model", _command_arg(command, "-m", "")
        )
        is_dflash = profile.engine == "mlx-dflash"
        is_dspark = profile.engine == "mlx-dspark"
        is_mlx = profile.engine.startswith("mlx")
        if is_dflash:
            context = _command_arg(command, "--dflash-max-ctx")
        elif is_dspark:
            context = _command_arg(command, "--context-window")
        elif is_mlx:
            # mlx-vlm enforces its request ceiling with --max-kv-size.  Show
            # that real limit in the manager instead of the ambiguous
            # "native" label (especially important for Muse's 131K ceiling).
            context = _command_arg(command, "--max-kv-size", "native")
        else:
            context = _command_arg(command, "--ctx-size")
        if is_dspark:
            kv = _command_arg(command, "--kv-bits", "native")
        elif profile.engine == "mlx-vlm":
            kv = _command_arg(command, "--kv-bits", "native")
        else:
            kv = ""
        rows.append({
            "name": profile.name,
            "engine": profile.engine,
            "model": Path(model_arg).name,
            "ctx": context,
            "kv_k": (
                (
                    "q8"
                    if "--quantize-kv-cache" in command
                    else "native"
                )
                if is_dflash
                else kv if is_mlx else _command_arg(command, "--cache-type-k")
            ),
            "kv_v": (
                (
                    "q8"
                    if "--quantize-kv-cache" in command
                    else "native"
                )
                if is_dflash
                else kv if is_mlx else _command_arg(command, "--cache-type-v")
            ),
            "draft": Path(
                _command_arg(
                    command,
                    "--draft-model",
                    _command_arg(command, "--drafter", _command_arg(command, "--draft")),
                )
            ).name,
            "draft_block": _command_arg(command, "--draft-block-size"),
            "description": profile.description,
            "built_in": True,
        })
    saved_dir = ROOT / "run/saved-profiles"
    for path in sorted(saved_dir.glob("*.json")):
        try:
            saved = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({"name": saved.get("saved_name", path.stem), "engine": saved.get("engine", ""),
                     "model": saved.get("model", ""),
                     "ctx": saved.get("ctx"), "kv_k": saved.get("kv_k"), "kv_v": saved.get("kv_v"),
                     "description": "Saved workbench profile", "modified": path.stat().st_mtime,
                     "built_in": False})
    return rows


@router.post("/api/profiles/{name}/load")
async def profile_load(name: str) -> dict[str, Any]:
    current_settings = refresh_manager_settings()
    if name in current_settings.profiles:
        await manager.stop()
        try:
            result = await manager.start(name)
        except START_ERRORS as exc:
            _event("error", f"start {name} failed: {exc}")
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _event("service_state", f"started profile {name}")
        return {"ok": True, "runtime": result}
    path = ROOT / "run/saved-profiles" / f"{re.sub(r'[^A-Za-z0-9_.-]', '-', name)}.json"
    if not path.exists():
        raise HTTPException(404, "profile not found")
    try:
        saved = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "invalid saved profile") from exc
    saved.pop("kv_types", None)
    CUSTOM_PATH.write_text(json.dumps(saved, indent=2) + "\n")
    await manager.stop()
    return {"ok": True, "runtime": await _start_custom()}


@router.post("/api/profiles")
def profile_save(req: ProfileRequest) -> dict[str, Any]:
    config = _config()
    config.pop("kv_types", None)
    config["saved_name"] = req.name
    path = ROOT / "run/saved-profiles" / f"{re.sub(r'[^A-Za-z0-9_.-]', '-', req.name)}.json"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(config, indent=2) + "\n")
    return {"ok": True, "name": req.name}


@router.delete("/api/profiles/{name}")
def profile_delete(name: str) -> dict[str, Any]:
    path = ROOT / "run/saved-profiles" / f"{re.sub(r'[^A-Za-z0-9_.-]', '-', name)}.json"
    path.unlink(missing_ok=True)
    return {"ok": True}


@router.get("/api/vram/calculate")
def vram_calc_get(path: str = "", kv_k: str = "q8_0", kv_v: str = "turbo4") -> dict[str, Any]:
    return vram_calc({"path": path or _config()["model_path"], "kv_k": kv_k, "kv_v": kv_v})


@router.get("/api/mlx/memory-plan")
def mlx_memory_plan(
    path: str = "",
    context: int = 0,
    kv_mode: str = "",
    group_size: int = 0,
    checkpoints: int = 0,
) -> dict[str, Any]:
    """Plan either the saved runtime or an unsaved selection from the UI."""
    config = _config()
    return _mlx_memory_plan(
        Path(path or str(config.get("mlx_model_path") or "")).expanduser(),
        context=max(
            0,
            min(
                context or int(config.get("mlx_preallocate_kv_size") or 0),
                262144,
            ),
        ),
        kv_mode=kv_mode or str(config.get("mlx_kv_mode") or "native4"),
        group_size=group_size or int(config.get("mlx_kv_group_size") or 64),
        checkpoints=max(
            2,
            checkpoints or int(config.get("mlx_session_checkpoints") or 2),
        ),
    )


@router.post("/api/vram-calc")
def vram_calc(payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(payload.get("path", ""))
    kv_k = str(payload.get("kv_k", payload.get("kv", "q8_0")))
    kv_v = str(payload.get("kv_v", payload.get("kv", "turbo4")))
    if path.is_dir():
        size_mib = sum(p.stat().st_size for p in path.glob("*.safetensors")) / 2**20
    else:
        size_mib = path.stat().st_size / 2**20 if path.exists() else 0
    k_bits = KV_TYPES.get(kv_k, KV_TYPES["q8_0"])[1]
    v_bits = KV_TYPES.get(kv_v, KV_TYPES["turbo4"])[1]
    metal_ceiling = _metal_limit() or 25559
    estimates = []
    for ctx in (65536, 131072, 196608, 262144, 350208, 395264, 524288, 786432, 1048576):
        if "27B" in path.name:
            # 16 gated-attention layers × 4 KV heads × 256 head dim.
            kv_mib = 16 * 4 * 256 * ctx * (k_bits + v_bits) / 8 / 2**20
        else:
            # Empirical fit for the installed 35B hybrid MoE build.
            kv_mib = 20.0 * (ctx / 1000) * (k_bits + v_bits) / 32.0
        total = size_mib + kv_mib + 900
        estimates.append({"ctx": ctx, "model": round(size_mib), "kv": round(kv_mib), "gpu_total": round(total), "free_idle": round(metal_ceiling-total), "free_peak": round(metal_ceiling-total-1024)})
    return {"estimates": estimates, "metal_ceiling_mib": metal_ceiling, "kv_k": kv_k, "kv_v": kv_v}


@router.get("/api/logs")
def logs(lines: int = 100) -> dict[str, str]:
    return stack_logs("qwen", lines)


@router.get("/api/logs/filter")
def logs_filter(level: str = "all", lines: int = 100) -> dict[str, Any]:
    content = stack_logs("qwen", lines * 10)["content"].splitlines()
    # llama.cpp log lines carry a single-letter severity after the timestamp.
    markers = {"error": " E ", "warn": " W ", "info": " I "}
    if level == "checkpoint":
        content = [x for x in content if "checkpoint" in x.lower()]
    elif level in markers:
        content = [x for x in content if markers[level] in x or level in x.lower()]
    elif level != "all":
        content = [x for x in content if level in x.lower()]
    return {"logs": content[-lines:]}


def _system_info() -> dict[str, Any]:
    vm = psutil.virtual_memory(); swap = psutil.swap_memory()
    wired_result = _run(["sysctl", "-n", "iogpu.wired_limit_mb"])
    try:
        wired_limit_mib = int(wired_result.stdout.strip())
    except ValueError:
        wired_limit_mib = 0
    vm_text = _run(["vm_stat"]).stdout
    page_match = re.search(r"page size of (\d+) bytes", vm_text)
    page_size = int(page_match.group(1)) if page_match else 16384
    vm_pages: dict[str, int] = {}
    for label, value in re.findall(r'^([^:]+):\s+([0-9]+)\.?$', vm_text, re.MULTILINE):
        vm_pages[label] = int(value)
    compressed_mib = vm_pages.get("Pages occupied by compressor", 0) * page_size // 2**20
    wired_mib = vm_pages.get("Pages wired down", 0) * page_size // 2**20
    runtime = manager.status()
    battery = psutil.sensors_battery()
    return {"cpu_pct": psutil.cpu_percent(None), "cpu_cores": psutil.cpu_count(), "cpu_temp_c": None,
            "ram_total_mib": vm.total // 2**20, "ram_used_mib": vm.used // 2**20,
            "ram_free_mib": vm.free // 2**20, "ram_available_mib": vm.available // 2**20,
            "ram_buffers_mib": 0, "ram_cached_mib": vm.inactive // 2**20,
            "swap_total_mib": swap.total // 2**20, "swap_free_mib": swap.free // 2**20,
            "compressed_mib": compressed_mib, "wired_mib": wired_mib,
            "wired_limit_mib": wired_limit_mib,
            "wired_limit_gib": round(wired_limit_mib / 1024, 1) if wired_limit_mib else None,
            "server_mode": runtime.get("server_mode", False) or _server_guard_active(),
            "power_source": "AC" if battery and battery.power_plugged else "battery" if battery else "unknown",
            "uptime_secs": int(time.time() - psutil.boot_time())}


@router.get("/api/system/info")
def system_info() -> dict[str, Any]: return _system_info()


@router.get("/api/memory/system")
def memory_system() -> dict[str, Any]: return _system_info()


def _metal_limit() -> int:
    result = _run(["sysctl", "-n", "iogpu.wired_limit_mb"])
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def _server_guard_active() -> bool:
    return bool(SERVER_GUARD and SERVER_GUARD.poll() is None)


def _clamshell_closed() -> bool:
    """Return true when a Mac notebook lid is currently closed."""
    result = _run(["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"])
    return bool(re.search(r'"AppleClamshellState"\s*=\s*Yes', result.stdout))


def _enable_server_guard(*, persist: bool = False) -> None:
    global SERVER_GUARD
    if persist:
        SERVER_MODE_FLAG.parent.mkdir(parents=True, exist_ok=True)
        SERVER_MODE_FLAG.touch()
    if _server_guard_active():
        return
    SERVER_GUARD = subprocess.Popen(
        # Prevent system, idle-system and disk sleep. The display is allowed to
        # sleep; lid-closed server mode does not need a display assertion.
        ["/usr/bin/caffeinate", "-ims", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _disable_server_guard() -> None:
    global SERVER_GUARD
    if SERVER_GUARD and SERVER_GUARD.poll() is None:
        SERVER_GUARD.terminate()
    SERVER_GUARD = None
    SERVER_MODE_FLAG.unlink(missing_ok=True)


@router.get("/api/memory/metal-limit")
def metal_limit() -> dict[str, Any]:
    check = _run(["sudo", "-n", str(MEMORY_HELPER), str(_metal_limit())], 3)
    return {
        "current_mib": _metal_limit(), "default_mib": 0,
        "min_mib": 20480, "max_mib": 30720, "step_mib": 128,
        "presets": [0, 27648, 28672, 29696, 30720],
        "sudo_ok": MEMORY_HELPER.exists() and check.returncode == 0,
        "persistent": False, "restart_required": manager.status()["running"],
    }


@router.post("/api/memory/metal-limit")
async def set_metal_limit(req: MetalLimitRequest) -> dict[str, Any]:
    limit = int(req.limit_mib)
    if limit != 0 and not (20480 <= limit <= 30720):
        raise HTTPException(400, "limit must be 0 or between 20480 and 30720 MiB")
    was_running = manager.status()["running"]
    if was_running and req.restart_runtime:
        await manager.stop()
    result = _run(["sudo", "-n", str(MEMORY_HELPER), str(limit)], 10)
    if result.returncode != 0:
        raise HTTPException(500, (result.stderr or result.stdout or "memory helper failed").strip())
    if limit:
        _enable_server_guard(persist=True)
    else:
        _disable_server_guard()
    runtime = None
    if was_running and req.restart_runtime:
        runtime = await _start_custom()
    _event("metal_limit", f"Metal wired-memory ceiling changed to {limit or 'macOS default'} MiB")
    return {"ok": True, "current_mib": _metal_limit(), "persistent": False,
            "runtime_restarted": bool(runtime), "runtime": runtime}


@router.get("/api/system/llm-mode")
def llm_mode() -> dict[str, Any]:
    runtime = manager.status()
    battery = psutil.sensors_battery()
    return {
        "enabled": runtime["running"] or _server_guard_active(), "runtime": runtime,
        "metal_limit_mib": _metal_limit(), "persistent": False,
        "caffeinate_active": runtime.get("server_mode", False) or _server_guard_active(),
        "clamshell_closed": _clamshell_closed(),
        "power_source": "AC" if battery and battery.power_plugged else "battery" if battery else "unknown",
        "thermal": (_run(["pmset", "-g", "therm"]).stdout or "unavailable").strip(),
    }


@router.post("/api/system/llm-mode/server")
def enable_llm_server_mode() -> dict[str, Any]:
    _enable_server_guard(persist=True)
    _event("system_mode", "enabled persistent lid-closed server sleep guard")
    return {
        "ok": True,
        "mode": "server",
        "caffeinate_active": _server_guard_active(),
        "clamshell_closed": _clamshell_closed(),
        "runtime": manager.status(),
    }


@router.post("/api/system/llm-mode/normal")
async def normal_macos_mode() -> dict[str, Any]:
    # Removing the guard under a closed lid causes immediate clamshell sleep,
    # taking SSH and the dashboard down before the response can even arrive.
    if _clamshell_closed():
        raise HTTPException(
            409,
            "Open the MacBook lid before returning to normal macOS mode; "
            "the server sleep guard was left active.",
        )
    await manager.stop()
    _disable_server_guard()
    result = _run(["sudo", "-n", str(MEMORY_HELPER), "0"], 10)
    if result.returncode != 0:
        raise HTTPException(500, (result.stderr or result.stdout or "memory helper failed").strip())
    _event("system_mode", "returned to normal macOS mode")
    return {"ok": True, "mode": "normal", "metal_limit_mib": _metal_limit(), "runtime": manager.status()}


@router.get("/api/gpu/detail")
def gpu_detail() -> dict[str, Any]:
    gpu = _gpu_state()
    return {"name": gpu["name"], "vram_total_mib": TOTAL_MIB, "vram_used_mib": gpu["used_mib"], "vram_free_mib": gpu["free_mib"], "util_pct": gpu["util_pct"], "memory_util_pct": gpu["mem_util_pct"], "temperature_c": gpu.get("temp_c"), "power_w": gpu.get("power_w"), "fan_speed_pct": None, "fan_rpm": gpu.get("fan_rpm"), "clock_mhz": gpu.get("clock_mhz"), "memory_clock_mhz": None, "thermal_state": gpu.get("thermal_state")}


@router.get("/api/gpu/thermal")
def gpu_thermal() -> dict[str, Any]:
    telemetry = _apple_telemetry()
    current = telemetry.get("gpu_temp_c")
    return {"current_c": current, "max_c": telemetry.get("gpu_temp_max_c"), "pct": min(100, current) if current is not None else None, "state": telemetry.get("thermal_state"), "available": current is not None, "source": telemetry.get("source")}


@router.get("/api/gpu/clocks")
def gpu_clocks() -> dict[str, Any]:
    clock = _apple_telemetry().get("gpu_clock_mhz")
    return {"graphics_mhz": clock, "memory_mhz": None, "sm_mhz": clock, "available": clock is not None}


@router.get("/api/gpu/power-limit")
def gpu_power_limit() -> dict[str, Any]: return {"current_w": None, "default_w": None, "min_w": None, "max_w": None, "sudo_ok": False, "reason": "Apple manages SoC power dynamically"}


@router.post("/api/gpu/power-limit")
def set_gpu_power_limit() -> dict[str, Any]:
    raise HTTPException(409, "Apple Silicon exposes no supported per-GPU power-limit control")


@router.get("/api/gpu/processes")
def gpu_processes() -> list[dict[str, Any]]:
    state = manager.status()
    return [{"pid": state["pid"], "name": state["profile"], "memory_mib": round(state.get("process_rss_gib", 0)*1024)}] if state["running"] else []


@router.get("/api/power/current")
@router.get("/api/energy/total")
def power_current() -> dict[str, Any]:
    battery = psutil.sensors_battery()
    telemetry = _apple_telemetry()
    return {"power_w": telemetry.get("gpu_power_w"), "gpu_power_w": telemetry.get("gpu_power_w"), "cpu_power_w": telemetry.get("cpu_power_w"), "system_power_w": telemetry.get("system_power_w"), "total_power_w": telemetry.get("total_power_w"), "system_power_available": telemetry.get("available", False), "gpu_limit_w": None, "gpu_util_pct": _gpu_state(telemetry)["util_pct"], "total_wh": 0, "uptime": telemetry.get("source", "mactop unavailable"), "thermal_state": telemetry.get("thermal_state"), "ac_power": bool(battery and battery.power_plugged)}


@router.post("/api/energy/reset")
def energy_reset() -> dict[str, Any]: return {"ok": True, "reset": True}


@router.get("/api/vram-history")
@router.get("/api/power/history")
def history(window: str = "15m") -> dict[str, Any]:
    seconds = {"15m": 900, "1h": 3600, "6h": 21600, "24h": 86400}.get(window, 900)
    cutoff = time.time() - seconds
    with HISTORY_LOCK:
        rows = [x for x in HISTORY if x["ts"] >= cutoff]
    data = [{"t": x["ts"], "used": x["vram_used_mib"], "app": x["vram_used_mib"], "total": TOTAL_MIB, "power_w": x.get("gpu_power_w")} for x in rows]
    powers = [x["power_w"] for x in data if x.get("power_w") is not None]
    return {"window": window, "count": len(data), "data": data, "avg_power_w": round(sum(powers) / len(powers), 2) if powers else None}


@router.get("/api/dashboard/snapshot")
def dashboard_snapshot() -> dict[str, Any]: return _snapshot()


@router.get("/api/dashboard/profile-memory-baselines")
def profile_memory_baselines() -> dict[str, Any]:
    return {
        "measured_metal_ceiling_mib": 29696,
        "profiles": PROFILE_MEMORY_BASELINES,
        "note": (
            "Apple Silicon uses one shared pool. System free, macOS available, "
            "GPU pool free, and Metal headroom overlap and must not be added."
        ),
    }


@router.get("/api/dashboard/timeseries")
def dashboard_timeseries(window: str = "15m", metrics: str = "") -> dict[str, Any]:
    seconds = {"15m": 900, "1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800, "30d": 2592000}.get(window, 900)
    cutoff = time.time() - seconds
    with HISTORY_LOCK: rows = [x for x in HISTORY if x["ts"] >= cutoff]
    keys = [x for x in metrics.split(",") if x]
    return {"timestamps": [x["ts"] for x in rows], "series": {key: [x.get(key) for x in rows] for key in keys}}


@router.get("/api/dashboard/events")
def dashboard_events(window: str = "1h", limit: int = 200) -> dict[str, Any]:
    seconds = {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800}.get(window, 3600)
    return {"events": [x for x in EVENTS if x["ts"] >= time.time()-seconds][-limit:]}


@router.get("/api/dashboard/counters")
def dashboard_counters() -> dict[str, Any]: return _snapshot()["counters"]


@router.get("/api/dashboard/sampling-rate")
def sampling_rate() -> dict[str, int]: return {"sample_rate_s": SAMPLE_SECONDS}


@router.post("/api/dashboard/sampling-rate")
def set_sampling_rate(req: SamplingRequest) -> dict[str, int]:
    global SAMPLE_SECONDS
    SAMPLE_SECONDS = max(2, min(req.sample_rate_s, 30))
    # Relaunch mactop so its native sampling interval matches the new rate.
    threading.Thread(target=_restart_mactop, daemon=True).start()
    return {"sample_rate_s": SAMPLE_SECONDS}


def _active_model_name(config: dict[str, Any] | None = None) -> str:
    config = config or _config()
    if _engine_is_mlx(config):
        command_model = _command_arg(_saved_command(), "--model", "")
        if command_model:
            return command_model
        # mlx_lm.server loads whatever repo/path the request names; passing the
        # exact --model path makes it reuse the already-loaded weights.
        return str(config.get("mlx_model_path") or "default_model")
    return config["model"]


def _test_request(prompt: str, max_tokens: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        response = httpx.post(f"http://127.0.0.1:{_active_port()}/v1/chat/completions", json={"model": _active_model_name(), "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}, timeout=600)
        response.raise_for_status(); data = response.json(); elapsed = time.monotonic()-started
        count = data.get("usage", {}).get("completion_tokens", 0)
        return {"ok": True, "seconds": round(elapsed, 2), "tokens": count, "tok_s": round(count/elapsed, 2) if elapsed else 0, "content": data["choices"][0]["message"]["content"]}
    except Exception as exc: return {"ok": False, "error": str(exc)}


@router.get("/api/benchmark/quick")
def benchmark_quick() -> dict[str, Any]:
    result = _test_request("Reply concisely: why is unified memory useful?", 50)
    return {"ok": result["ok"], "latency_ms": result.get("seconds", 0)*1000, "tokens": result.get("tokens", 0), "tok_per_sec": result.get("tok_s", 0), "error": result.get("error")}


async def _sweep_worker(task_id: str, contexts: list[int], tokens: int) -> None:
    original = _config()
    results: list[dict[str, Any]] = []
    try:
        for index, ctx in enumerate(contexts):
            ctx = max(4096, min(int(ctx), 524288))
            with TASK_LOCK:
                TASKS[task_id]["log"].append(f"[{index + 1}/{len(contexts)}] Loading ctx={ctx}")
                TASKS[task_id]["progress"] = round(index / max(1, len(contexts)) * 100)
            cfg = dict(original)
            cfg["ctx"] = ctx
            cfg.pop("kv_types", None)
            await manager.stop()
            await manager.start_profile(_custom_profile(cfg))
            gpu = _gpu_state()
            test = await asyncio.to_thread(_test_request, ("Metal context benchmark token " * max(1, tokens // 4))[:750000], 64)
            results.append({"ctx": ctx, "status": "ok" if test.get("ok") else "error",
                            "idle_vram": gpu["used_mib"], "peak_free": gpu["free_mib"],
                            "prompt_tps": _llama_metrics().get("llamacpp:prompt_tokens_seconds", 0),
                            "gen_tps": test.get("tok_s", 0), "error": test.get("error")})
        with TASK_LOCK:
            TASKS[task_id].update({"status": "done", "progress": 100, "result": results})
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{task_id}.json").write_text(json.dumps({
            "task_id": task_id, "created_at": time.time(),
            "model": original.get("model"), "kv_k": original.get("kv_k"),
            "kv_v": original.get("kv_v"), "metal_limit_mib": _metal_limit(),
            "results": results,
        }, indent=2) + "\n")
    except Exception as exc:
        with TASK_LOCK:
            TASKS[task_id].update({"status": "failed", "result": results + [{"status": "error", "error": str(exc)}]})
    finally:
        try:
            original.pop("kv_types", None)
            await manager.stop()
            await manager.start_profile(_custom_profile(original))
        except Exception as exc:
            _event("error", f"failed to restore profile after sweep: {exc}")


@router.post("/api/bench/sweep")
async def bench_sweep(req: SweepRequest) -> dict[str, str]:
    if not req.contexts or len(req.contexts) > 8:
        raise HTTPException(400, "choose between one and eight context sizes")
    if _engine_is_mlx():
        raise HTTPException(409, "the context sweep drives llama.cpp profiles; switch to a GGUF model first")
    task_id = f"sweep-{uuid.uuid4().hex[:10]}"
    _new_task(task_id)
    asyncio.create_task(_sweep_worker(task_id, req.contexts, min(req.tokens, 200000)))
    return {"task_id": task_id, "task": task_id}


@router.post("/api/test/gen")
def test_gen(req: TestRequest) -> dict[str, Any]: return _test_request("Write a compact Python LRU cache.", min(req.tokens, 2048))


@router.post("/api/test/prompt")
def test_prompt(req: TestRequest) -> dict[str, Any]: return _test_request(("alpha beta gamma delta " * max(1, req.tokens//4))[:500000], 8)


@router.post("/api/test/codebase")
def test_codebase(req: TestRequest) -> dict[str, Any]: return _test_request(("def function(): return 42\n" * max(1, req.tokens//8))[:1000000], min(req.gen, 512))


@router.get("/api/tokens/count")
def token_count(text: str = "") -> dict[str, Any]:
    # /tokenize is llama.cpp-only; MLX falls through to the character estimate.
    if manager.status()["running"] and not _engine_is_mlx():
        try:
            response = httpx.post("http://127.0.0.1:8097/tokenize", json={"content": text}, timeout=10)
            response.raise_for_status()
            tokens = response.json().get("tokens", [])
            return {"count": len(tokens), "tokens": len(tokens), "model": _config()["model"], "method": "llama.cpp tokenizer"}
        except Exception:
            pass
    # ~4 characters per token is a far better offline estimate than word count,
    # especially for code and long-context payloads.
    estimate = max(len(text.split()), round(len(text) / 4)) if text else 0
    return {"count": estimate, "tokens": estimate, "model": _config()["model"], "method": "character estimate"}


@router.get("/api/service/version")
def service_version() -> dict[str, Any]:
    config = _config()
    if _engine_is_dflash(config):
        try:
            from importlib.metadata import version

            build = f"dflash-mlx {version('dflash-mlx')}"
        except Exception:
            build = "dflash-mlx version unavailable"
        command = _saved_command()
        cap = int(_command_arg(command, "--dflash-max-ctx", "262144"))
        draft = _command_arg(command, "--draft", "")
        draft_name = Path(draft).name if draft else "unknown"
        return {
            "binary": command[0] if command else str(ROOT / ".venv/bin/dflash"),
            "build": build,
            "port": _active_port(),
            "context": (
                f"dynamic target KV; cap {cap:,}; "
                f"{'q8' if '--quantize-kv-cache' in command else 'native'} "
                "target KV; w4 draft; prefix snapshots enabled"
            ),
            "model": _active_model_name(config),
            "draft": draft_name,
        }
    if _engine_is_mlx(config):
        try:
            import mlx_lm
            build = f"mlx-lm {getattr(mlx_lm, '__version__', 'unknown')}"
        except ImportError:
            build = "mlx-lm not installed"
        fixed_context = int(config.get("mlx_preallocate_kv_size") or 0)
        if fixed_context:
            context = (
                f"fixed {fixed_context:,}; {config.get('mlx_kv_mode', 'native4')} "
                f"group {int(config.get('mlx_kv_group_size') or 64)}; "
                f"adaptive prefill {'on' if config.get('mlx_adaptive_prefill', True) else 'off'}"
            )
        else:
            context = "dynamic KV; model-native ceiling applies"
        return {"binary": f"{sys.executable} -m local_llm_control.mlx_quant_server",
                "build": build, "port": 8098,
                "context": context,
                "model": _active_model_name(config)}
    output = _run([str(LLAMA_BIN), "--version"]).stdout.strip()
    return {"binary": str(LLAMA_BIN), "build": output.splitlines()[0] if output else "unknown", "port": 8097, "context": config["ctx"], "model": config["model"]}


@router.get("/api/health/deep")
@router.get("/api/service/health-detail")
def deep_health() -> dict[str, Any]:
    started = time.monotonic(); runtime = manager.status()
    port = _active_port()
    try:
        if _engine_is_mlx():
            ok = bool(_json_url(f"http://127.0.0.1:{port}/v1/models", 2).get("data"))
        else:
            ok = _json_url(f"http://127.0.0.1:{port}/health", 2).get("status") == "ok"
    except Exception: ok = False
    gpu = _gpu_state()
    return {"status": "healthy" if ok else "stopped" if not runtime["running"] else "unhealthy", "service": ok, "model": ok, "pid": runtime.get("pid"), "latency_ms": round((time.monotonic()-started)*1000, 1), "vram_mib": gpu["used_mib"], "gpu_util_pct": gpu["util_pct"], "temperature_c": gpu.get("temp_c")}


@router.get("/api/build/status")
def build_status() -> dict[str, Any]:
    return {"source": str(LLAMA_SOURCE), "binary": str(LLAMA_BIN), "exists": LLAMA_BIN.exists(), "branch": _run(["git", "-C", str(LLAMA_SOURCE), "branch", "--show-current"]).stdout.strip(), "commit": _run(["git", "-C", str(LLAMA_SOURCE), "rev-parse", "--short", "HEAD"]).stdout.strip()}


def _command_task_worker(
    task_id: str, command: list[str], cwd: Path, result_path: Path | None = None
) -> None:
    try:
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        TASK_PROCESSES[task_id] = proc
        for line in proc.stdout or []:
            with TASK_LOCK:
                TASKS[task_id]["log"] = (TASKS[task_id]["log"] + [line.rstrip()])[-500:]
        code = proc.wait()
        result: dict[str, Any] = {"returncode": code}
        if result_path and result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                result["path"] = str(result_path)
                result["returncode"] = code
            except (OSError, json.JSONDecodeError) as exc:
                result["result_error"] = str(exc)
        with TASK_LOCK:
            if TASKS[task_id].get("status") != "cancelled":
                TASKS[task_id].update({"status": "done" if code == 0 else "failed",
                                       "progress": 100 if code == 0 else 0,
                                       "result": result})
    except Exception as exc:
        with TASK_LOCK:
            TASKS[task_id].update({"status": "failed", "result": {"error": str(exc)}})
    finally:
        TASK_PROCESSES.pop(task_id, None)


@router.post("/api/bench/mlx-needle")
def bench_mlx_needle(req: NeedleProbeRequest, background: BackgroundTasks) -> dict[str, str]:
    config = _config()
    if not _engine_is_mlx(config):
        raise HTTPException(409, "the MLX needle probe requires a running MLX model")
    if not manager.status().get("running"):
        raise HTTPException(409, "start the MLX model before running a needle probe")
    target = int(req.target_tokens)
    if target < 1024 or target > 262144:
        raise HTTPException(400, "target_tokens must be between 1024 and 262144")
    if any(t.get("status") == "running" and str(t.get("id", "")).startswith("mlx-needle-")
           for t in TASKS.values()):
        raise HTTPException(409, "an MLX needle probe is already running")
    model_path = Path(_active_model_name(config)).expanduser()
    if not model_path.is_dir():
        raise HTTPException(409, "the selected MLX model directory is missing")
    task_id = f"mlx-needle-{target}-{uuid.uuid4().hex[:8]}"
    result_path = RESULTS_DIR / f"{task_id}.json"
    command = [
        "/usr/bin/caffeinate", "-dims", sys.executable,
        str(ROOT / "scripts/context_probe_mlx.py"),
        "--base-url", "http://127.0.0.1:8098",
        "--tokenizer", str(model_path),
        "--target-tokens", str(target),
        "--output", str(result_path),
    ]
    _new_task(task_id, {"kind": "mlx-needle", "target_tokens": target,
                        "result_path": str(result_path)})
    background.add_task(_command_task_worker, task_id, command, ROOT, result_path)
    return {"task_id": task_id, "task": task_id}


@router.post("/api/build/rebuild")
def build_rebuild(background: BackgroundTasks) -> dict[str, str]:
    task_id = f"rebuild-{uuid.uuid4().hex[:10]}"
    _new_task(task_id)
    command = ["cmake", "--build", str(LLAMA_SOURCE / "build"), "--target", "llama-server", "-j", str(psutil.cpu_count() or 4)]
    background.add_task(_command_task_worker, task_id, command, LLAMA_SOURCE)
    return {"task_id": task_id, "task": task_id}


@router.get("/api/disk/usage")
def disk_usage() -> dict[str, Any]:
    usage = shutil.disk_usage(MODEL_ROOT)
    return {"mount": str(MODEL_ROOT), "total": usage.total, "used": usage.used, "free": usage.free,
            "avail": usage.free, "pct": round(usage.used / usage.total * 100, 1),
            "models_bytes": sum(p.stat().st_size for p in MODEL_ROOT.rglob("*") if p.is_file())}


@router.get("/api/network/clients")
def network_clients() -> dict[str, Any]:
    rows = []
    try:
        connections = psutil.net_connections("tcp")
    except (psutil.AccessDenied, PermissionError):
        # macOS hides other users' process file descriptors from an unprivileged daemon.
        connections = []
    for conn in connections:
        if conn.laddr and conn.laddr.port in (8090, 8097, 8098):
            rows.append({"local": f"{conn.laddr.ip}:{conn.laddr.port}", "remote": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "", "state": conn.status})
    return {"connections": rows}


@router.get("/api/processes/top")
def processes_top() -> dict[str, Any]:
    rows=[]
    for proc in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
        try:
            memory_info = proc.info.get("memory_info")
            if memory_info is None:
                continue
            rows.append({"pid": proc.info["pid"], "name": proc.info["name"], "rss_mib": round(memory_info.rss/2**20, 1), "cpu_pct": proc.info["cpu_percent"]})
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    return {"processes": sorted(rows, key=lambda x: x["rss_mib"], reverse=True)[:25]}


@router.get("/api/systemd/status")
def launchd_status() -> dict[str, Any]:
    result = _run(["launchctl", "print", f"system/{MANAGER_LABEL}"])
    return {"status": result.stdout or result.stderr, "active": "state = running" in result.stdout, "service_manager": "launchd"}


@router.post("/api/systemd/install")
def launchd_install() -> dict[str, Any]:
    return {"ok": True, "message": "launchd daemon is already installed and enabled; restart it over SSH with sudo launchctl kickstart -k"}


@router.post("/api/systemd/uninstall")
def launchd_uninstall() -> dict[str, Any]:
    raise HTTPException(409, "Disabling the controller from inside its own dashboard is intentionally blocked")


@router.get("/api/system/power-status")
def system_power_status() -> dict[str, Any]:
    return {"supported": False, "pending": None, "reason": "Use SSH with sudo shutdown/reboot for macOS power control"}


@router.post("/api/system/{action}")
def system_power_action(action: str) -> dict[str, Any]:
    if action not in {"shutdown", "reboot", "cancel-shutdown"}:
        raise HTTPException(404, "unknown system action")
    raise HTTPException(409, "Remote shutdown/reboot is disabled in the web process; use SSH and sudo")


@router.post("/api/config/export")
def config_export() -> dict[str, Any]:
    return {"format": "local-llm-mac-v1", "config": _config()}


@router.post("/api/config/import")
def config_import(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise HTTPException(400, "configuration must be an object")
    config.pop("kv_types", None)
    CUSTOM_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_PATH.write_text(json.dumps(config, indent=2) + "\n")
    return {"ok": True, "config": _config()}


@router.put("/api/hf/token")
def hf_token(payload: dict[str, str]) -> dict[str, Any]:
    token = payload.get("token", "")
    if not token.startswith("hf_"):
        raise HTTPException(400, "invalid Hugging Face token")
    target = Path.home() / ".cache/huggingface/token"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(token)
    target.chmod(0o600)
    os.environ["HF_TOKEN"] = token
    return {"ok": True}


@router.get("/api/results/recent")
def recent_results() -> list[dict[str, Any]]:
    rows=[]
    for path in sorted(RESULTS_DIR.glob("*.json"), key=lambda p:p.stat().st_mtime, reverse=True)[:30]:
        rows.append({"name": path.name, "size": path.stat().st_size, "mtime": path.stat().st_mtime})
    return rows


@router.get("/api/results/{name}")
def result_file(name: str) -> Any:
    path = RESULTS_DIR / Path(name).name
    if not path.exists(): raise HTTPException(404, "result not found")
    return json.loads(path.read_text())


@router.delete("/api/cache/{kind}")
def clear_cache(kind: str) -> dict[str, Any]:
    if kind == "logs":
        for path in (ROOT / "logs").glob("*.log"): path.write_text("")
    elif kind == "run":
        for path in (ROOT / "run").glob("*.json"): path.unlink(missing_ok=True)
    elif kind in ("os", "all"):
        return {"ok": False, "msg": "macOS manages its unified page cache; forced purge is intentionally disabled"}
    return {"ok": True}


@router.get("/api/metrics/export")
def metrics_export() -> dict[str, Any]:
    with HISTORY_LOCK: return {"generated_at": time.time(), "samples": list(HISTORY)}


@router.get("/api/vision-mode")
def vision_mode() -> dict[str, Any]:
    cfg=_config(); return {"enabled": False, "supported": False, "reason": "No matching mmproj is installed", "ctx": cfg["ctx"], "batch": cfg["batch"], "ubatch": cfg["ubatch"], "mmproj_path": ""}


@router.post("/api/vision-mode")
def set_vision_mode() -> dict[str, Any]: raise HTTPException(409, "No matching vision projector is installed for this model")


@router.post("/api/research")
def research_unavailable() -> dict[str, Any]: raise HTTPException(409, "optillm deep research has not been installed on this Mac")


@router.post("/api/optillm/test")
def optillm_unavailable() -> dict[str, Any]: raise HTTPException(409, "optillm is not installed on this Mac")


@router.get("/api/telemetry/health")
def telemetry_health() -> dict[str, Any]:
    telemetry = _apple_telemetry()
    return {
        "available": telemetry.get("available", False),
        "source": telemetry.get("source"),
        "mactop_installed": MACTOP_BIN.exists(),
        "mactop_pid": _mactop_pid(),
        "retry_seconds": MACTOP_RETRY_SECONDS,
    }


@router.post("/api/telemetry/restart")
def telemetry_restart() -> dict[str, Any]:
    _restart_mactop()
    return {"ok": True, **telemetry_health()}


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = True


@router.post("/api/chat")
async def chat(req: ChatRequest) -> Any:
    """Playground proxy to the running model so the dashboard needs no CORS or second origin."""
    if not manager.status()["running"]:
        raise HTTPException(409, "no model is running")
    payload = {
        "model": _active_model_name(), "messages": req.messages,
        "max_tokens": max(1, min(req.max_tokens, 16384)),
        "temperature": req.temperature, "stream": req.stream,
    }
    url = f"http://127.0.0.1:{_active_port()}/v1/chat/completions"
    if not req.stream:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    yield f"data: {json.dumps({'error': body.decode('utf-8', 'replace')})}\n\n".encode()
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(relay(), media_type="text/event-stream")


@router.get("/api/logs/stream")
async def logs_stream() -> StreamingResponse:
    """Server-sent events that follow the active runtime log across restarts."""

    async def follow():
        current, position, idle = "", 0, 0
        while True:
            path = manager._saved_state().get("log") or ""
            if path != current:
                current, position, idle = path, 0, 0
                lines: list[str] = []
                if path:
                    try:
                        data = Path(path).read_bytes()
                        position = len(data)
                        lines = data.decode("utf-8", "replace").splitlines()[-200:]
                    except FileNotFoundError:
                        pass
                yield f"data: {json.dumps({'reset': True, 'lines': lines})}\n\n"
            elif path:
                try:
                    with Path(path).open("rb") as handle:
                        handle.seek(position)
                        chunk = handle.read()
                except FileNotFoundError:
                    current = ""
                    continue
                cut = chunk.rfind(b"\n")
                if cut >= 0:
                    position += cut + 1
                    lines = chunk[:cut].decode("utf-8", "replace").splitlines()
                    if lines:
                        idle = 0
                        yield f"data: {json.dumps({'lines': lines})}\n\n"
            idle += 1
            if idle >= 15:
                idle = 0
                yield ": keepalive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(follow(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
