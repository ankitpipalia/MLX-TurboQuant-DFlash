#!/usr/bin/env python3
"""Add the MLX-VLM drafter metadata omitted by the benchmark-only MTP repo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HEAD_SHA256 = "0e267a482e74c2664ce41dc4c4326f480020d015372fc9f7654ea3a136d62815"
HEAD_BYTES = 238_934_093
TARGET_FILES = {
    "README.md": (81, "748c964f9b7e5f2c3770ce013bbc0153be7c54311d9692c343b6188eefe77ac6"),
    "chat_template.jinja": (8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "config.json": (4094, "558cd605a6f1c16c73f4918534d122a943e12e754d38567b3b704acc93596965"),
    "generation_config.json": (202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"),
    "model-00001-of-00003.safetensors": (5_328_325_648, "075eac5fbba3951bc4870c1ac65d684c32e0abb24284201bd27f49df56735963"),
    "model-00002-of-00003.safetensors": (5_354_185_130, "6c99c446987a432beb1f6ef7d6fde6db02682f1ebe1a913760953140e0fa4e47"),
    "model-00003-of-00003.safetensors": (4_450_532_735, "0e267246064a1e635077dd05e22181f5eda44a1fa8a67dcf46c903f858bbe35b"),
    "model.safetensors.index.json": (189_789, "a2161f6a6c9f7c434145f97a4a4262a74eb3e16a95088e713754eb108b198511"),
    "tokenizer.json": (19_989_325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    "tokenizer_config.json": (1161, "95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(target: Path, head: Path) -> None:
    for name, (expected_bytes, expected_sha256) in TARGET_FILES.items():
        artifact = target / name
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if artifact.stat().st_size != expected_bytes or sha256(artifact) != expected_sha256:
            raise ValueError(f"Target artifact does not match pinned revision: {name}")
    target_config = json.loads((target / "config.json").read_text())
    text_config = target_config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"{target} has no text_config")
    weights = head / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if weights.stat().st_size != HEAD_BYTES or sha256(weights) != HEAD_SHA256:
        raise ValueError(
            "MTP head does not match Yukon's winning pinned revision "
            "0966ddaff972fd3ca2be08f3640603b47e9ce70a"
        )

    quantization = {"group_size": 64, "bits": 4, "mode": "affine"}
    draft_config = {
        "model_type": "qwen3_5_mtp",
        "text_config": text_config,
        # One primary token plus three MTP proposals. The runtime can override
        # this ceiling after acceptance/cost calibration.
        "block_size": 4,
        "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", False)),
        "quantization": quantization,
        "quantization_config": quantization,
        "source_revisions": {
            "target": "eda45ab47f465d08d6558f0353a2346e2eb9d5b3",
            "head": "0966ddaff972fd3ca2be08f3640603b47e9ce70a",
        },
    }
    (head / "config.json").write_text(
        json.dumps(draft_config, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.target.expanduser(), args.head.expanduser())


if __name__ == "__main__":
    main()
