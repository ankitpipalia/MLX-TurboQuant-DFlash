import json

import local_llm_control.workbench as workbench_module
from local_llm_control.workbench import _config as load_workbench_config
from local_llm_control.workbench import _custom_profile, _default_config


def _config(*, kv_k: str, kv_v: str) -> dict:
    config = _default_config()
    config.update({"kv_k": kv_k, "kv_v": kv_v})
    return config


def test_symmetric_turbo_selection_disables_automatic_k_promotion() -> None:
    command = _custom_profile(_config(kv_k="turbo4", kv_v="turbo4")).command

    assert command[:2] == ("/usr/bin/env", "TURBO_AUTO_ASYMMETRIC=0")


def test_asymmetric_turbo_selection_keeps_quality_safeguard() -> None:
    command = _custom_profile(_config(kv_k="q8_0", kv_v="turbo4")).command

    assert "TURBO_AUTO_ASYMMETRIC=0" not in command


def test_custom_profile_applies_prompt_cache_ram() -> None:
    config = _config(kv_k="q8_0", kv_v="turbo4")
    config["cache_ram"] = 512
    command = _custom_profile(config).command

    assert command[command.index("--cache-ram") + 1] == "512"


def test_stopped_runtime_command_does_not_override_saved_config(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "workbench-config.json"
    config_path.write_text(json.dumps({"ctx": 262144}))
    stale = list(workbench_module.settings.profiles["llama-turbo4-safe"].command)
    stale[stale.index("--ctx-size") + 1] = "400000"

    monkeypatch.setattr(workbench_module, "CUSTOM_PATH", config_path)
    monkeypatch.setattr(workbench_module, "_saved_command", lambda: stale)
    monkeypatch.setattr(
        workbench_module.manager, "status", lambda: {"running": False}
    )

    assert load_workbench_config()["ctx"] == 262144
