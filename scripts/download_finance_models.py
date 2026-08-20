#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config/finance-models.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned HisabClub quality models")
    parser.add_argument("aliases", nargs="*", help="Model aliases; defaults to every model")
    args = parser.parse_args()

    with REGISTRY.open("rb") as handle:
        models = tomllib.load(handle)["models"]
    aliases = args.aliases or list(models)

    for alias in aliases:
        model = models.get(alias)
        if model is None:
            raise SystemExit(f"Unknown model alias: {alias}")
        print(f"Downloading {alias}: {model['repository']}@{model['revision']}")
        snapshot_download(
            repo_id=model["repository"],
            revision=model["revision"],
            local_dir=Path(model["path"]),
        )


if __name__ == "__main__":
    main()
