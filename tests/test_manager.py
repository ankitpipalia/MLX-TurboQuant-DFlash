import json
from dataclasses import replace

import pytest

import local_llm_control.manager as manager_module
from local_llm_control.config import Profile, load_settings
from local_llm_control.manager import RuntimeManager


def test_manager_lists_profiles_without_starting_models(tmp_path) -> None:
    settings = replace(load_settings(), runtime_dir=tmp_path / "run")
    manager = RuntimeManager(settings)

    names = {profile["name"] for profile in manager.profiles()}
    assert "llama-turbo4-safe" in names
    assert "llama-turbo4-max" in names
    assert "llama-a5000-350k" in names
    assert "mlx-4bit" in names
    assert "mlx-dflash-hauhau35-optiq4" in names
    assert "mlx-vlm-muse-glimmer-30b-awq4" in names
    assert "mlx-vlm-muse-glimmer-30b-awq4-dflash" in names
    assert "mlx-vlm-muse-glimmer-30b-awq4-dflash-cached" in names
    assert "mlx-vlm-muse-glimmer-30b-awq4-dflash-bf16" in names
    assert "mlx-dspark-muse-glimmer-30b-awq4" in names
    assert "mlx-vlm-qwen38-nvfp4" in names
    assert "mlx-dflash2-qwen38-q4" in names
    assert "mlx-dflash2-qwen38-q4-cached" in names
    assert manager.status()["running"] is False


def test_manager_can_refresh_profile_snapshot(tmp_path) -> None:
    original = replace(load_settings(), runtime_dir=tmp_path / "old-run")
    refreshed = replace(load_settings(), runtime_dir=tmp_path / "new-run")
    manager = RuntimeManager(original)

    manager.update_settings(refreshed)

    assert manager.settings is refreshed
    assert manager.state_path == tmp_path / "new-run" / "state.json"
    assert manager.state_path.parent.is_dir()


def test_manager_recognizes_mlx_vlm_server_process(tmp_path, monkeypatch) -> None:
    manager = RuntimeManager(load_settings())

    class Process:
        def cmdline(self) -> list[str]:
            return ["python", "-m", "mlx_vlm.server", "--port", "8098"]

    monkeypatch.setattr("local_llm_control.manager.psutil.Process", lambda _pid: Process())

    assert manager._pid_is_ours(12345)


def test_manager_recognizes_quantized_mlx_wrapper(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, pid: int) -> None:
            assert pid == 1234

        def cmdline(self) -> list[str]:
            return [
                "python",
                "-m",
                "local_llm_control.mlx_quant_server",
                "--port",
                "8098",
            ]

    monkeypatch.setattr(manager_module.psutil, "Process", FakeProcess)
    manager = RuntimeManager(load_settings())
    assert manager._pid_is_ours(1234) is True


def test_manager_recognizes_dflash_runtime(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, pid: int) -> None:
            assert pid == 4321

        def cmdline(self) -> list[str]:
            return [
                "/Users/example/local-llm/.venv/bin/python",
                "/Users/example/local-llm/.venv/bin/dflash",
                "serve",
            ]

    monkeypatch.setattr(manager_module.psutil, "Process", FakeProcess)
    manager = RuntimeManager(load_settings())
    assert manager._pid_is_ours(4321) is True


def test_manager_recognizes_dspark_runtime(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, pid: int) -> None:
            assert pid == 9876

        def cmdline(self) -> list[str]:
            return ["/Users/example/local-llm/.venv/bin/mlx-dspark", "serve"]

    monkeypatch.setattr(manager_module.psutil, "Process", FakeProcess)
    manager = RuntimeManager(load_settings())
    assert manager._pid_is_ours(9876) is True


def test_manager_reports_saved_profile_engine(monkeypatch, tmp_path) -> None:
    settings = replace(load_settings(), runtime_dir=tmp_path)
    manager = RuntimeManager(settings)
    manager.state_path.write_text(
        '{"profile":"mlx-dflash-hauhau35-optiq4","pid":9876}'
    )
    monkeypatch.setattr(manager, "_pid_is_ours", lambda pid: pid == 9876)

    status = manager.status()

    assert status["running"] is True
    assert status["engine"] == "mlx-dflash"


def test_manager_rejects_missing_dflash_draft_before_start(tmp_path) -> None:
    executable = tmp_path / "dflash"
    executable.touch()
    target = tmp_path / "target"
    target.mkdir()
    missing_draft = tmp_path / "missing-draft"
    profile = Profile(
        name="dflash-preflight",
        engine="mlx-dflash",
        description="test",
        port=8098,
        health_path="/v1/models",
        command=(
            str(executable),
            "serve",
            "--model",
            str(target),
            "--draft",
            str(missing_draft),
        ),
    )
    manager = RuntimeManager(
        replace(load_settings(), runtime_dir=tmp_path / "run")
    )

    with pytest.raises(FileNotFoundError, match="draft model not found"):
        manager._validate(profile)


def test_manager_rejects_missing_dspark_drafter_before_start(tmp_path) -> None:
    executable = tmp_path / "python"
    executable.touch()
    target = tmp_path / "target"
    target.mkdir()
    missing_drafter = tmp_path / "missing-drafter"
    profile = Profile(
        name="dspark-preflight",
        engine="mlx-dspark",
        description="test",
        port=8098,
        health_path="/health",
        command=(
            str(executable),
            "-m",
            "local_llm_control.mlx_dspark_text_server",
            "serve",
            "--model",
            str(target),
            "--drafter",
            str(missing_drafter),
        ),
    )
    manager = RuntimeManager(
        replace(load_settings(), runtime_dir=tmp_path / "run")
    )

    with pytest.raises(FileNotFoundError, match="draft model not found"):
        manager._validate(profile)


def test_manager_rejects_incomplete_sharded_model(tmp_path) -> None:
    executable = tmp_path / "python"
    executable.touch()
    target = tmp_path / "target"
    target.mkdir()
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (target / "model-00001-of-00002.safetensors").write_bytes(b"complete")
    profile = Profile(
        name="incomplete-model",
        engine="mlx-vlm",
        description="test",
        port=8098,
        health_path="/health",
        command=(str(executable), "--model", str(target)),
    )
    manager = RuntimeManager(
        replace(load_settings(), runtime_dir=tmp_path / "run")
    )

    with pytest.raises(FileNotFoundError, match="download incomplete"):
        manager._validate(profile)
