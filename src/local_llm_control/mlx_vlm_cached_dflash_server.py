"""Run Muse DFlash through mlx-vlm's APC-capable batch generator.

Upstream mlx-vlm routes every non-MTP drafter to a dedicated speculative
worker.  That worker is fast for an isolated request, but it constructs a new
target cache for every request and therefore bypasses Automatic Prefix
Caching.  The generic ``BatchGenerator`` already supports DFlash and APC
together.  This wrapper selects that path while retaining the text-first lazy
vision policy used by the coding profiles.

This remains an experimental compatibility shim.  It is deliberately kept in
one module and covered by tests so it can be removed as soon as upstream's
dedicated DFlash worker gains APC support.
"""

from __future__ import annotations

from typing import Any, Self

from .mlx_vlm_text_server import install_text_first_loader


class _BatchDFlashKind(str):
    """A DFlash kind that skips only the worker's non-MTP fast-path branch.

    ``ResponseGenerator._run`` currently branches on ``kind != "mtp"`` before
    entering the generic batch loop.  All downstream dispatch compares the
    value with ``"dflash"``, which retains normal string semantics.
    """

    def __new__(cls) -> Self:
        return super().__new__(cls, "dflash")

    def __ne__(self, other: object) -> bool:
        if other == "mtp":
            return False
        return super().__ne__(other)


def install_cached_dflash_path() -> None:
    """Route DFlash to the APC-capable generic batch loop once."""
    from mlx_vlm.server import generation

    response_generator = generation.ResponseGenerator
    original = response_generator._initialize_model
    if getattr(original, "_local_llm_cached_dflash", False):
        return

    def initialize_with_batch_dflash(self: Any, *args: Any, **kwargs: Any):
        result = original(self, *args, **kwargs)
        if getattr(self, "draft_kind", None) == "dflash":
            self.draft_kind = _BatchDFlashKind()
        return result

    initialize_with_batch_dflash._local_llm_cached_dflash = True  # type: ignore[attr-defined]
    initialize_with_batch_dflash._local_llm_original = original  # type: ignore[attr-defined]
    response_generator._initialize_model = initialize_with_batch_dflash


def main() -> None:
    install_text_first_loader()
    install_cached_dflash_path()
    from mlx_vlm.server.cli import main as server_main

    server_main()


if __name__ == "__main__":
    main()
