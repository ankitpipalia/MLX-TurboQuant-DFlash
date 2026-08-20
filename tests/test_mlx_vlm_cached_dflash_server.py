from mlx_vlm.server import generation

from local_llm_control.mlx_vlm_cached_dflash_server import (
    _BatchDFlashKind,
    install_cached_dflash_path,
)


def test_batch_dflash_kind_skips_only_dedicated_worker_branch() -> None:
    kind = _BatchDFlashKind()

    assert kind == "dflash"
    assert not (kind != "mtp")
    assert kind != "eagle3"
    assert str(kind) == "dflash"


def test_cached_dflash_patch_marks_initialized_dflash(monkeypatch) -> None:
    def fake_initialize(self) -> None:
        self.draft_kind = "dflash"

    monkeypatch.setattr(
        generation.ResponseGenerator,
        "_initialize_model",
        fake_initialize,
    )
    install_cached_dflash_path()
    worker = object.__new__(generation.ResponseGenerator)

    generation.ResponseGenerator._initialize_model(worker)

    assert isinstance(worker.draft_kind, _BatchDFlashKind)
    assert worker.draft_kind == "dflash"
