from fastapi.testclient import TestClient
from types import SimpleNamespace

import local_llm_control.workbench as wb
from local_llm_control.api import app


def test_snapshot_exposes_throughput_series_and_telemetry_flag() -> None:
    snapshot = TestClient(app).get("/api/dashboard/snapshot").json()
    assert {"eval_tok_s", "prompt_tok_s", "telemetry_available"} <= snapshot.keys()
    counters = snapshot["counters"]
    assert {"requests_total", "errors_total", "prompt_tokens_total"} <= counters.keys()


def test_telemetry_health_reports_sampler_state() -> None:
    health = TestClient(app).get("/api/telemetry/health").json()
    assert {"available", "source", "mactop_installed", "mactop_pid"} <= health.keys()


def test_mactop_pid_discovers_collector_left_by_manager_reload(monkeypatch) -> None:
    class Process:
        info = {
            "pid": 4321,
            "cmdline": ["/opt/homebrew/bin/mactop", "--prometheus", "9101", "--headless"],
        }

    monkeypatch.setattr(wb, "MACTOP_PROCESS", None)
    monkeypatch.setattr(wb.psutil, "process_iter", lambda _attrs: [Process()])
    assert wb._mactop_pid() == 4321


def test_vram_calculate_get_alias_matches_post_route() -> None:
    client = TestClient(app)
    via_get = client.get("/api/vram/calculate", params={"kv_k": "q8_0", "kv_v": "turbo4"})
    assert via_get.status_code == 200
    assert via_get.json()["kv_k"] == "q8_0"
    assert [row["ctx"] for row in via_get.json()["estimates"]]


def test_start_failure_maps_to_conflict_not_500(monkeypatch) -> None:
    async def refuse(profile):
        raise MemoryError("only 2.07 GiB available")

    monkeypatch.setattr(
        wb, "_mlx_profile", lambda _config: SimpleNamespace(name="test")
    )
    monkeypatch.setattr(wb.manager, "start_profile", refuse)
    response = TestClient(app).post("/api/service/start")
    assert response.status_code == 409
    assert "2.07 GiB" in response.json()["detail"]


def test_new_task_prunes_finished_tasks() -> None:
    original = dict(wb.TASKS)
    try:
        wb.TASKS.clear()
        for index in range(wb.TASK_FINISHED_KEEP + 10):
            task_id = f"old-{index}"
            wb.TASKS[task_id] = {"id": task_id, "status": "done"}
        wb._new_task("fresh-1")
        finished = [t for t in wb.TASKS.values() if t["status"] == "done"]
        assert len(finished) == wb.TASK_FINISHED_KEEP
        assert wb.TASKS["fresh-1"]["status"] == "running"
    finally:
        wb.TASKS.clear()
        wb.TASKS.update(original)


def test_logs_filter_matches_llama_severity_markers(monkeypatch) -> None:
    lines = "\n".join([
        "0.1 I srv log: fine",
        "0.2 W slot update_slots: erased invalidated context checkpoint",
        "0.3 E srv log: broken",
    ])
    monkeypatch.setattr(wb, "stack_logs", lambda key, _lines=120: {"content": lines})
    assert wb.logs_filter("error")["logs"] == ["0.3 E srv log: broken"]
    assert wb.logs_filter("checkpoint")["logs"] == [
        "0.2 W slot update_slots: erased invalidated context checkpoint"
    ]


def test_dflash_metrics_are_mapped_into_dashboard_counters(monkeypatch) -> None:
    monkeypatch.setattr(wb, "_engine_is_mlx", lambda *_args: True)
    monkeypatch.setattr(wb, "_engine_is_dflash", lambda *_args: True)
    monkeypatch.setattr(
        wb,
        "_json_url",
        lambda *_args: {
            "totals": {
                "requests": 4,
                "cache_hits": 3,
                "cache_misses": 1,
            },
            "rates": {"average_decode_tok_s": 21.5},
            "last_request": {
                "cache_hit_tokens": 8192,
                "prefill_tok_s_physical": 300.0,
                "acceptance_rate": 0.72,
                "tokens_per_cycle": 5.4,
            },
        },
    )
    monkeypatch.setattr(wb, "METRICS_CACHE_AT", 0.0)

    metrics = wb._llama_metrics()

    assert metrics["local_llm_requests_total"] == 4
    assert metrics["local_llm_checkpoints_restored_total"] == 3
    assert metrics["local_llm_last_cached_tokens"] == 8192
    assert metrics["local_llm_last_generation_tps"] == 21.5
    assert metrics["local_llm_last_prompt_tps"] == 300.0
    assert metrics["local_llm_dflash_acceptance_ratio"] == 0.72


def test_dflash_metrics_engine_detection_includes_dspark() -> None:
    assert wb._engine_is_dflash({"engine": "mlx-dflash"}) is True
    assert wb._engine_is_dflash({"engine": "mlx-dspark"}) is False
    assert wb._engine_is_dspark({"engine": "mlx-dspark"}) is True
    assert wb._engine_is_dflash({"engine": "mlx-vlm"}) is False


def test_builtin_muse_profiles_expose_real_context_and_drafter() -> None:
    profiles = {item["name"]: item for item in wb.profiles()}
    baseline = profiles["mlx-vlm-muse-glimmer-30b-awq4-text"]
    dflash = profiles["mlx-vlm-muse-glimmer-30b-awq4-dflash"]

    assert baseline["ctx"] == "131072"
    assert baseline["built_in"] is True
    assert dflash["ctx"] == "131072"
    assert dflash["draft"] == "meta-models--Muse-Glimmer-30B-assistant-4bit"


def test_dspark_metrics_are_mapped_into_dashboard_counters(monkeypatch) -> None:
    monkeypatch.setattr(wb, "_engine_is_mlx", lambda *_args: True)
    monkeypatch.setattr(wb, "_engine_is_dspark", lambda *_args: True)
    monkeypatch.setattr(
        wb,
        "_json_url",
        lambda *_args: {
            "requests": 3,
            "prompt_tokens": 9000,
            "completion_tokens": 300,
            "mean_accept_len": 2.4,
            "mean_tokens_per_sec": 24.5,
            "prefix_cache": {
                "enabled": True,
                "cached_tokens": 8192,
                "hits": 2,
                "slots": [{"tokens": 8192}],
            },
        },
    )
    monkeypatch.setattr(wb, "METRICS_CACHE_AT", 0.0)

    metrics = wb._llama_metrics()

    assert metrics["local_llm_requests_total"] == 3
    assert metrics["local_llm_checkpoints_restored_total"] == 2
    assert metrics["local_llm_last_cached_tokens"] == 8192
    assert metrics["local_llm_last_generation_tps"] == 24.5
    assert metrics["local_llm_dflash_tokens_per_cycle"] == 2.4


def test_mlx_profile_builds_quant_server_command(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "SomeModel-4bit-MLX"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"x")
    profile = wb._mlx_profile({
        "mlx_model_path": str(model_dir), "mlx_kv_bits": 8,
        "mlx_kv_mode": "turbo4",
        "mlx_preallocate_kv_size": 131072,
        "mlx_prefill_step_size": 4096, "mlx_prompt_cache_mib": 1024,
    })
    assert profile.engine == "mlx-lm"
    assert profile.port == 8098
    assert profile.health_path == "/v1/models"
    command = list(profile.command)
    assert command[command.index("--kv-bits") + 1] == "4"
    assert command[command.index("--kv-mode") + 1] == "turbo4"
    assert command[command.index("--model") + 1] == str(model_dir)
    assert command[command.index("--prefill-step-size") + 1] == "4096"
    assert command[command.index("--preallocate-kv-size") + 1] == "131072"
    # Session reuse is on by default, so the fixed arena is kept across turns
    # rather than having the prompt cache disabled.
    assert "--session-reuse" in command
    assert "local_llm_control.mlx_quant_server" in command


def test_mlx_profile_session_reuse_can_be_disabled(tmp_path) -> None:
    model_dir = tmp_path / "M-4bit-MLX"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"x")
    profile = wb._mlx_profile({
        "mlx_model_path": str(model_dir), "mlx_kv_mode": "native8",
        "mlx_preallocate_kv_size": 65536, "mlx_session_reuse": False,
    })
    command = list(profile.command)
    assert "--session-reuse" not in command
    assert command[command.index("--prompt-cache-bytes") + 1] == "0"


def test_mlx_profile_adaptive_prefill_can_be_disabled(tmp_path) -> None:
    model_dir = tmp_path / "M-4bit-MLX"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"x")
    profile = wb._mlx_profile({
        "mlx_model_path": str(model_dir),
        "mlx_kv_mode": "turbo4",
        "mlx_preallocate_kv_size": 65536,
        "mlx_adaptive_prefill": False,
    })

    assert "--no-adaptive-prefill" in profile.command


def test_mlx_memory_plan_matches_qwen_dense_turbo4_arena(tmp_path) -> None:
    import json

    model_dir = tmp_path / "QwenDense"
    model_dir.mkdir()
    layers = [
        "full_attention" if index % 4 == 3 else "linear_attention"
        for index in range(96)
    ]
    (model_dir / "config.json").write_text(json.dumps({
        "text_config": {
            "model_type": "qwen3_5_text",
            "layer_types": layers,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
        }
    }))

    plan = wb._mlx_memory_plan(
        model_dir,
        context=131072,
        kv_mode="turbo4",
        group_size=64,
    )

    assert plan["full_attention_layers"] == 24
    assert plan["kv_gib"] == 3.375
    assert plan["recommended_context"] <= 131072


def test_mlx_profile_rejects_missing_dir_and_bad_bits(tmp_path) -> None:
    import pytest
    with pytest.raises(ValueError):
        wb._mlx_profile({"mlx_model_path": str(tmp_path / "nope"), "mlx_kv_bits": 4})
    real = tmp_path / "m"
    real.mkdir()
    (real / "weights.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError):
        wb._mlx_profile({"mlx_model_path": str(real), "mlx_kv_bits": 6})
    with pytest.raises(ValueError):
        wb._mlx_profile({
            "mlx_model_path": str(real), "mlx_kv_bits": 4,
            "mlx_kv_mode": "turbo2",
        })


def test_mlx_profile_rejects_incomplete_sharded_download(tmp_path) -> None:
    import json
    import pytest

    model = tmp_path / "partial"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }))
    (model / "model-00001-of-00002.safetensors").write_bytes(b"x")

    with pytest.raises(ValueError, match="1 of 2 weight shards"):
        wb._mlx_profile({
            "mlx_model_path": str(model),
            "mlx_kv_bits": 4,
            "mlx_kv_mode": "turbo4",
        })


def test_save_config_accepts_public_config_keys(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "workbench.json"
    monkeypatch.setattr(wb, "CUSTOM_PATH", config_path)
    monkeypatch.setattr(wb, "_saved_command", lambda: [])

    saved = wb._save_config({
        "mlx_prefill_step_size": "4096",
        "mlx_prompt_cache_mib": "1024",
        "mlx_experimental_context": "true",
    })

    assert saved["mlx_prefill_step_size"] == 4096
    assert saved["mlx_prompt_cache_mib"] == 1024
    assert saved["mlx_experimental_context"] is True


def test_models_lists_mlx_directories(tmp_path, monkeypatch) -> None:
    mlx_root = tmp_path / "mlx"
    model = mlx_root / "Test-4bit-MLX"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        '{"model_type": "qwen3_5_moe", "quantization": {"bits": 4, "mode": "affine"}}')
    (model / "model-00001.safetensors").write_bytes(b"x" * 1024)
    incomplete = mlx_root / "still-downloading"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}")
    monkeypatch.setattr(wb, "MLX_ROOT", mlx_root)
    monkeypatch.setattr(wb, "GGUF_ROOT", tmp_path / "gguf-none")
    rows = wb._models()
    assert [r["name"] for r in rows] == ["Test-4bit-MLX"]
    assert rows[0]["engine"] == "mlx"
    assert rows[0]["quant"] == "4-bit affine"
    assert rows[0]["arch"] == "qwen3_5_moe"


def test_mlx_meta_reports_mixed_optiq_bpw(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen3_5_moe","quantization":{"bits":4,"mode":"affine"}}'
    )
    (tmp_path / "optiq_metadata.json").write_text(
        '{"achieved_bpw":6.1249,"candidate_bits":[4,8]}'
    )
    assert wb._mlx_meta(tmp_path)["quant"] == "OptiQ 6.12 bpw (4/8-bit mixed)"


def test_engine_detection_and_port_routing(monkeypatch) -> None:
    monkeypatch.setattr(wb.manager, "status", lambda: {"running": False})
    monkeypatch.setattr(wb, "_config", lambda: {"engine": "mlx-lm"})
    monkeypatch.setattr(wb, "_saved_command", lambda: [])
    assert wb._engine_is_mlx() is True
    assert wb._active_port() == 8098
    monkeypatch.setattr(wb, "_config", lambda: {"engine": "llama.cpp"})
    assert wb._active_port() == 8097
    monkeypatch.setattr(wb, "_saved_command", lambda: ["x", "--port", "8098"])
    assert wb._active_port() == 8098


def test_mlx_python_module_flag_is_not_mistaken_for_gguf_model(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "workbench.json"
    config_path.write_text(
        '{"engine":"mlx-lm","model":"saved.gguf","model_path":"/models/saved.gguf",'
        '"mlx_model_path":"/models/mlx"}'
    )
    monkeypatch.setattr(wb, "CUSTOM_PATH", config_path)
    monkeypatch.setattr(
        wb,
        "_saved_command",
        lambda: ["python", "-m", "local_llm_control.mlx_quant_server", "--model", "/models/mlx"],
    )

    config = wb._config()

    assert config["model"] == "saved.gguf"
    assert config["model_path"] == "/models/saved.gguf"
    assert config["mlx_model_path"] == "/models/mlx"


def test_mlx_vlm_metrics_maps_current_json_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(wb, "_engine_is_mlx", lambda *_: True)
    monkeypatch.setattr(wb, "_engine_is_dflash", lambda *_: False)
    monkeypatch.setattr(wb, "METRICS_CACHE", {})
    monkeypatch.setattr(wb, "METRICS_CACHE_AT", 0)
    monkeypatch.setattr(
        wb,
        "_json_url",
        lambda url, *_args, **_kwargs: {
            "latest": {
                "prompt_tokens": 8192,
                "generated_tokens": 64,
                "prefill_tok_s": 311.0,
                "decode_tok_s": 42.5,
                "peak_memory_gb": 24.0,
            },
            "summary": {
                "requests_completed": 3,
                "requests_failed": 1,
                "prompt_tokens_total": 9000,
                "generated_tokens_total": 100,
            },
            "server": {
                "apc": {
                    "enabled": True,
                    "lookups_hit": 2,
                    "lookups_miss": 1,
                    "exact_stores": 2,
                }
            },
        }
        if url == "http://127.0.0.1:8098/metrics"
        else (_ for _ in ()).throw(AssertionError(url)),
    )

    metrics = wb._llama_metrics()
    assert metrics["local_llm_last_generation_tps"] == 42.5
    assert metrics["local_llm_last_prompt_tps"] == 311.0
    assert metrics["local_llm_requests_total"] == 3
    assert metrics["local_llm_errors_total"] == 1
    assert metrics["local_llm_checkpoints_restored_total"] == 2
    assert metrics["local_llm_peak_memory_gib"] == 24.0 / ((2**30) / 1e9)


def test_mlx_metrics_falls_back_to_prometheus_text(monkeypatch) -> None:
    monkeypatch.setattr(wb, "_engine_is_mlx", lambda *_: True)
    monkeypatch.setattr(wb, "_engine_is_dflash", lambda *_: False)
    monkeypatch.setattr(wb, "METRICS_CACHE", {})
    monkeypatch.setattr(wb, "METRICS_CACHE_AT", 0)
    monkeypatch.setattr(
        wb,
        "_json_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("not json")),
    )
    monkeypatch.setattr(
        wb,
        "_text_url",
        lambda url, *_args, **_kwargs: (
            "local_llm_last_generation_tps 42.5\n"
            if url == "http://127.0.0.1:8098/metrics"
            else (_ for _ in ()).throw(AssertionError(url))
        ),
    )

    assert wb._llama_metrics()["local_llm_last_generation_tps"] == 42.5


def test_active_models_reports_mlx_model_path(monkeypatch) -> None:
    monkeypatch.setattr(wb, "_config", lambda: {"engine": "mlx-lm", "mlx_model_path": "/models/mlx"})
    monkeypatch.setattr(wb.manager, "status", lambda: {"running": True})
    monkeypatch.setattr(wb, "_saved_command", lambda: [])

    assert wb.models_active() == {"models": ["/models/mlx"]}


def test_command_task_worker_loads_saved_probe_result(tmp_path) -> None:
    import json
    import sys

    task_id = "mlx-needle-test"
    result_path = tmp_path / "probe.json"
    result_path.write_text(json.dumps({"status": "passed", "needle_found": True}))
    wb._new_task(task_id)
    try:
        wb._command_task_worker(
            task_id,
            [sys.executable, "-c", "print('probe complete')"],
            tmp_path,
            result_path,
        )
        task = wb.TASKS[task_id]
        assert task["status"] == "done"
        assert task["result"]["needle_found"] is True
        assert task["result"]["path"] == str(result_path)
        assert task["log"] == ["probe complete"]
    finally:
        wb.TASKS.pop(task_id, None)
        wb.TASK_PROCESSES.pop(task_id, None)


def test_mlx_vlm_draft_log_stats_reads_latest_sample(
    tmp_path, monkeypatch
) -> None:
    log = tmp_path / "runtime.log"
    log.write_text(
        "Speculative decode: kind=dflash batch=1 tokens=20 "
        "accept=2.50 rounds=10 drafted=30\n"
        "Speculative decode: kind=dflash batch=1 tokens=40 "
        "accept=3.00 rounds=20 drafted=50\n"
    )
    monkeypatch.setattr(
        wb.manager, "_saved_state", lambda: {"log": str(log)}
    )

    acceptance, tokens_per_cycle = wb._mlx_vlm_draft_log_stats()

    assert acceptance == 0.8
    assert tokens_per_cycle == 3.0
