from types import SimpleNamespace

import mlx_vlm

import local_llm_control.mlx_dspark_text_server as dspark_server


def test_dspark_installs_text_first_mlx_vlm_loader(monkeypatch) -> None:
    calls = {}
    fake_model = SimpleNamespace(
        language_model=SimpleNamespace(parameters=lambda: ["language"])
    )

    def fake_load(*args, **kwargs):
        calls["kwargs"] = kwargs
        return fake_model, "processor"

    monkeypatch.setattr(mlx_vlm, "load", fake_load)
    monkeypatch.setattr(
        "local_llm_control.mlx_vlm_text_server.mx.eval",
        lambda values: calls.setdefault("evaluated", values),
    )

    dspark_server.install_text_first_loader()
    result = mlx_vlm.load("model")

    assert result == (fake_model, "processor")
    assert calls["kwargs"]["lazy"] is True
    assert calls["evaluated"] == ["language"]
