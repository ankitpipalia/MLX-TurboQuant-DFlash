#!/usr/bin/env python3
"""Integrity-check and quantize Meta's Muse Glimmer DFlash assistant for MLX."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx_vlm.quant_utils import quantize_model
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.utils import save_config, save_weights


OFFICIAL_SHA256 = (
    "fd88d337eb84f8d0e6ba33a7684d7efa6722d4460ba4d6badca9699418392a84"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--expected-sha256", default=OFFICIAL_SHA256)
    args = parser.parse_args()

    weights = args.source / "model.safetensors"
    actual_sha = sha256(weights)
    if args.expected_sha256 and actual_sha != args.expected_sha256:
        raise ValueError(
            f"assistant SHA-256 mismatch: expected {args.expected_sha256}, "
            f"got {actual_sha}"
        )

    config = json.loads((args.source / "config.json").read_text())
    model, kind = load_drafter(str(args.source), kind="dflash", lazy=True)
    if kind != "dflash":
        raise ValueError(f"expected a DFlash assistant, got {kind!r}")
    model, config = quantize_model(
        model,
        config,
        group_size=args.group_size,
        bits=args.bits,
        mode="affine",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save_weights(args.output, model, donate_weights=True)
    save_config(config, config_path=args.output / "config.json")
    for name in ("README.md", "LICENSE", "USAGE_POLICY.md"):
        source = args.source / name
        if source.exists():
            shutil.copy2(source, args.output / name)

    mx.clear_cache()
    print(
        json.dumps(
            {
                "source_sha256": actual_sha,
                "output": str(args.output),
                "bits": args.bits,
                "group_size": args.group_size,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
