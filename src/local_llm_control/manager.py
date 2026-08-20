from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

from .config import Profile, Settings


class RuntimeManager:
    """Own exactly one memory-heavy inference runtime at a time."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        self.settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.settings.runtime_dir / "state.json"
        self._lock = asyncio.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._caffeinate: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None

    def update_settings(self, settings: Settings) -> None:
        """Apply a freshly parsed profile file without restarting the manager.

        The web manager is intentionally lightweight and long lived.  Keeping
        the original Settings object for its entire lifetime meant edits to
        profiles.toml were visible on disk and in tests, but a subsequent UI
        launch still used the old command.  Refresh the immutable settings
        snapshot at the API boundary while preserving the active process.
        """
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.state_path = settings.runtime_dir / "state.json"

    def _saved_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _pid_is_ours(self, pid: int) -> bool:
        try:
            cmdline = psutil.Process(pid).cmdline()
        except (psutil.Error, OSError):
            return False
        owned_runtimes = (
            "llama-server",
            "mlx_lm.server",
            "mlx_vlm.server",
            "local_llm_control.mlx_vlm_text_server",
            "local_llm_control.mlx_vlm_cached_dflash_server",
            "local_llm_control.mlx_vlm_adaptive_mtp_server",
            "local_llm_control.mlx_dspark_text_server",
            "local_llm_control.mlx_quant_server",
            "dflash",
            "mlx-dspark",
        )
        return any(
            marker in arg for arg in cmdline for marker in owned_runtimes
        )

    def status(self) -> dict[str, Any]:
        state = self._saved_state()
        pid = int(state.get("pid", 0) or 0)
        running = pid > 0 and self._pid_is_ours(pid)
        profile_name = str(state.get("profile") or "")
        saved_engine = state.get("engine")
        if not saved_engine and profile_name in self.settings.profiles:
            saved_engine = self.settings.profiles[profile_name].engine
        memory = psutil.virtual_memory()
        result: dict[str, Any] = {
            "running": running,
            "profile": state.get("profile") if running else None,
            "engine": saved_engine if running else None,
            "pid": pid if running else None,
            "started_at": state.get("started_at") if running else None,
            "log": state.get("log") if running else None,
            "memory": {
                "available_gib": round(memory.available / 2**30, 2),
                "used_percent": memory.percent,
            },
        }
        if running:
            try:
                proc = psutil.Process(pid)
                result["process_rss_gib"] = round(proc.memory_info().rss / 2**30, 2)
            except psutil.Error:
                pass
        result["server_mode"] = running and self._caffeinate_running(pid)
        return result

    def _caffeinate_running(self, runtime_pid: int) -> bool:
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                command = proc.info.get("cmdline") or []
                if proc.info.get("name") == "caffeinate" and "-w" in command and str(runtime_pid) in command:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def ensure_caffeinate(self) -> None:
        """Keep the Mac and display awake while an inference runtime is alive."""
        state = self.status()
        pid = int(state.get("pid") or 0)
        if not pid or self._caffeinate_running(pid):
            return
        self._caffeinate = subprocess.Popen(
            ["/usr/bin/caffeinate", "-dims", "-w", str(pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def profiles(self) -> list[dict[str, Any]]:
        return [
            {
                "name": profile.name,
                "engine": profile.engine,
                "description": profile.description,
                "port": profile.port,
            }
            for profile in self.settings.profiles.values()
        ]

    @staticmethod
    def _validate_model_artifact(artifact: Path, kind: str) -> None:
        """Reject local model directories whose declared weights are incomplete."""
        if not artifact.is_dir():
            return
        index = artifact / "model.safetensors.index.json"
        if index.exists():
            try:
                payload = json.loads(index.read_text())
                shards = set((payload.get("weight_map") or {}).values())
            except (OSError, json.JSONDecodeError, AttributeError) as exc:
                raise FileNotFoundError(
                    f"{kind} has an invalid weight index: {index}"
                ) from exc
            missing = sorted(
                name
                for name in shards
                if not (artifact / name).is_file()
                or (artifact / name).stat().st_size == 0
            )
            if missing:
                preview = ", ".join(missing[:3])
                suffix = "…" if len(missing) > 3 else ""
                raise FileNotFoundError(
                    f"{kind} download incomplete: missing {preview}{suffix}"
                )
            return
        weights = artifact / "model.safetensors"
        if weights.exists() and weights.stat().st_size == 0:
            raise FileNotFoundError(
                f"{kind} download incomplete: empty {weights.name}"
            )

    def _validate(self, profile: Profile) -> None:
        executable = Path(profile.command[0])
        if not executable.exists():
            raise FileNotFoundError(f"runtime executable not found: {executable}")
        for index, arg in enumerate(profile.command[:-1]):
            if arg in {
                "-m", "--model", "--draft", "--draft-model", "--drafter",
            }:
                artifact = Path(profile.command[index + 1])
                if artifact.is_absolute() and not artifact.exists():
                    kind = "draft model" if "draft" in arg else "model"
                    raise FileNotFoundError(f"{kind} not found: {artifact}")
                if artifact.is_absolute():
                    kind = "draft model" if "draft" in arg else "model"
                    self._validate_model_artifact(artifact, kind)
        available = psutil.virtual_memory().available / 2**30
        if available < self.settings.minimum_available_gib:
            raise MemoryError(
                f"only {available:.2f} GiB available; require at least "
                f"{self.settings.minimum_available_gib:.2f} GiB before loading"
            )

    async def _wait_until_ready(self, profile: Profile) -> None:
        deadline = time.monotonic() + self.settings.startup_timeout_seconds
        url = f"http://127.0.0.1:{profile.port}{profile.health_path}"
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if self._process and self._process.poll() is not None:
                    raise RuntimeError(
                        f"runtime exited with code {self._process.returncode}; check log"
                    )
                try:
                    response = await client.get(url)
                    if response.status_code < 500:
                        if profile.engine != "mlx-lm":
                            return
                        state = self._saved_state()
                        log_path = Path(str(state.get("log") or ""))
                        try:
                            with log_path.open("rb") as handle:
                                handle.seek(
                                    int(state.get("log_start_offset", 0) or 0)
                                )
                                current_run_log = handle.read().decode(
                                    errors="replace"
                                )
                            loaded = (
                                "local-llm: model load complete"
                                in current_run_log
                            )
                        except OSError:
                            loaded = False
                        if loaded:
                            return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
        raise TimeoutError(
            "runtime did not become ready within "
            f"{self.settings.startup_timeout_seconds:g} seconds"
        )

    async def _await_port_free(self, port: int, timeout: float = 15.0) -> None:
        """Wait for the runtime port to be bindable again.

        A restart stops the old process and starts the new one back to back; the
        listening socket can linger for a moment after the process dies, and the
        MLX server does not set SO_REUSEADDR, so binding too soon fails with
        "Address already in use". Poll until the port is free (or give up and let
        the normal startup error surface).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("0.0.0.0", port))
                    return
                except OSError:
                    pass
            await asyncio.sleep(0.5)

    async def start(self, name: str) -> dict[str, Any]:
        async with self._lock:
            try:
                profile = self.settings.profiles[name]
            except KeyError as exc:
                raise KeyError(f"unknown profile: {name}") from exc
            return await self._start_unlocked(profile)

    async def start_profile(self, profile: Profile) -> dict[str, Any]:
        """Start a validated runtime profile supplied by the web configurator."""
        async with self._lock:
            return await self._start_unlocked(profile)

    async def _start_unlocked(self, profile: Profile) -> dict[str, Any]:
        current = self.status()
        if current["running"]:
            if current["profile"] == profile.name:
                return current
            raise RuntimeError(
                f"profile {current['profile']} is already running; stop it first"
            )
        self._validate(profile)
        await self._await_port_free(profile.port)

        log_path = self.settings.log_dir / f"{profile.name}.log"
        try:
            log_start_offset = log_path.stat().st_size
        except OSError:
            log_start_offset = 0
        self._log_handle = log_path.open("ab", buffering=0)
        env = os.environ.copy()
        env.setdefault("GGML_METAL_LOG_LEVEL", "1")
        env.update(dict(profile.environment))
        self._process = subprocess.Popen(
            profile.command,
            cwd=self.settings.root,
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state = {
            "profile": profile.name,
            "engine": profile.engine,
            "pid": self._process.pid,
            "started_at": time.time(),
            "log": str(log_path),
            "log_start_offset": log_start_offset,
            "command": list(profile.command),
        }
        self.state_path.write_text(json.dumps(state, indent=2) + "\n")
        self.ensure_caffeinate()
        try:
            await self._wait_until_ready(profile)
        except Exception:
            await self._stop_unlocked()
            raise
        return self.status()

    async def _stop_unlocked(self) -> dict[str, Any]:
        state = self._saved_state()
        pid = int(state.get("pid", 0) or 0)
        if pid and self._pid_is_ours(pid):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + self.settings.shutdown_timeout_seconds
            while self._pid_is_ours(pid) and time.monotonic() < deadline:
                await asyncio.sleep(0.25)
            if self._pid_is_ours(pid):
                os.killpg(pid, signal.SIGKILL)
        self.state_path.unlink(missing_ok=True)
        if self._log_handle:
            self._log_handle.close()
        self._log_handle = None
        self._process = None
        if self._caffeinate and self._caffeinate.poll() is None:
            self._caffeinate.terminate()
        self._caffeinate = None
        return self.status()

    async def stop(self) -> dict[str, Any]:
        async with self._lock:
            return await self._stop_unlocked()

    def tail(self, lines: int = 100) -> list[str]:
        state = self._saved_state()
        path = state.get("log")
        if not path:
            return []
        try:
            return Path(path).read_text(errors="replace").splitlines()[-lines:]
        except FileNotFoundError:
            return []
