"""Run mlx-dspark with the same lazy-vision policy as the coding VLM server."""

from __future__ import annotations

from typing import Any

from .mlx_vlm_text_server import load_text_first


def install_text_first_loader() -> None:
    import mlx_vlm

    if getattr(mlx_vlm.load, "_local_llm_text_first", False):
        return
    original_load = mlx_vlm.load

    def patched_load(*args: Any, **kwargs: Any):
        return load_text_first(original_load, *args, **kwargs)

    patched_load._local_llm_text_first = True  # type: ignore[attr-defined]
    mlx_vlm.load = patched_load


def main() -> None:
    install_text_first_loader()
    from mlx_dspark.cli import main as dspark_main

    dspark_main()


if __name__ == "__main__":
    main()
