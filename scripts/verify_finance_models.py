#!/usr/bin/env python3
from __future__ import annotations

import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config/finance-models.toml"


def verify_model(alias: str, model: dict[str, str]) -> dict[str, object]:
    path = Path(model["path"])
    problems: list[str] = []
    if not path.is_dir():
        problems.append("model directory missing")
        return {"alias": alias, "ok": False, "path": str(path), "problems": problems}

    config_path = path / "config.json"
    if not config_path.is_file():
        problems.append("config.json missing")
    weight_files = sorted(path.glob("*.safetensors"))
    if not weight_files:
        problems.append("safetensors weights missing")
    if any(weight.stat().st_size == 0 for weight in weight_files):
        problems.append("empty safetensors weight file")
    if list(path.rglob("*.incomplete")):
        problems.append("incomplete downloads present")

    metadata_dir = path / ".cache/huggingface/download"
    observed_revisions = {
        metadata.read_text(encoding="utf-8").splitlines()[0]
        for metadata in metadata_dir.glob("*.metadata")
        if metadata.stat().st_size > 0
    }
    if model["revision"] not in observed_revisions:
        problems.append("pinned revision not found in download metadata")

    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config.get("quantization") or config.get("quantization_config")
        if model["precision"] == "bf16" and quantization:
            problems.append("BF16 registry entry contains quantization configuration")

    size_bytes = sum(weight.stat().st_size for weight in weight_files)
    return {
        "alias": alias,
        "ok": not problems,
        "path": str(path),
        "precision": model["precision"],
        "revision": model["revision"],
        "weight_bytes": size_bytes,
        "problems": problems,
    }


def main() -> None:
    with REGISTRY.open("rb") as handle:
        models = tomllib.load(handle)["models"]
    results = [verify_model(alias, model) for alias, model in models.items()]
    print(json.dumps(results, indent=2))
    if not all(result["ok"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
