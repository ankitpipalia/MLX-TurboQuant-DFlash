from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "profiles.toml"


@dataclass(frozen=True)
class Profile:
    name: str
    engine: str
    description: str
    port: int
    health_path: str
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Settings:
    root: Path
    log_dir: Path
    runtime_dir: Path
    startup_timeout_seconds: float
    shutdown_timeout_seconds: float
    minimum_available_gib: float
    profiles: dict[str, Profile]


def _expand(value: str, env: dict[str, str]) -> str:
    for key, replacement in env.items():
        value = value.replace("{" + key + "}", replacement)
    return os.path.expanduser(value)


def load_settings(path: Path | None = None) -> Settings:
    config_path = path or Path(os.getenv("LOCAL_LLM_CONFIG", DEFAULT_CONFIG))
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    root = config_path.resolve().parents[1]
    general = raw["settings"]
    env = {
        "LLAMA_BIN": os.getenv(
            "LLAMA_BIN",
            str(root / "vendor/llama-cpp-turboquant/build/bin/llama-server"),
        ),
        "GGUF_MODEL": os.getenv(
            "GGUF_MODEL",
            "~/Models/gguf/Qwen3.6-35B-A3B-Uncensored-Aggressive/"
            "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
        ),
        "QWEN27_Q6_MODEL": os.getenv(
            "QWEN27_Q6_MODEL",
            "~/Models/gguf/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive/"
            "Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf",
        ),
        "MLX_MODEL": os.getenv(
            "MLX_MODEL", "~/Models/mlx/Qwen3.6-35B-A3B-4bit"
        ),
        "OVIS_OCR_BF16_MODEL": os.getenv(
            "OVIS_OCR_BF16_MODEL", "~/Models/mlx/OvisOCR2-MLX-BF16"
        ),
        "PADDLE_OCR_16_BF16_MODEL": os.getenv(
            "PADDLE_OCR_16_BF16_MODEL",
            "~/Models/mlx/PaddleOCR-VL-1.6-bf16",
        ),
        "GLM_OCR_BF16_MODEL": os.getenv(
            "GLM_OCR_BF16_MODEL", "~/Models/mlx/GLM-OCR-bf16"
        ),
        "QWEN_SEMANTIC_6BIT_MODEL": os.getenv(
            "QWEN_SEMANTIC_6BIT_MODEL", "~/Models/mlx/Qwen3.5-9B-6bit"
        ),
        "NUEXTRACT_BF16_MODEL": os.getenv(
            "NUEXTRACT_BF16_MODEL", "~/Models/mlx/NuExtract3-bf16"
        ),
        "MUSE_GLIMMER_AWQ_MODEL": os.getenv(
            "MUSE_GLIMMER_AWQ_MODEL",
            "~/Models/mlx/"
            "Blackfrost-AI--Muse-Glimmer-30B-Abliterated-MLX-4bit-AWQ",
        ),
        "MUSE_GLIMMER_DFLASH_MODEL": os.getenv(
            "MUSE_GLIMMER_DFLASH_MODEL",
            "~/Models/dflash/meta-models--Muse-Glimmer-30B-assistant-4bit",
        ),
        "MUSE_GLIMMER_DFLASH_BF16_MODEL": os.getenv(
            "MUSE_GLIMMER_DFLASH_BF16_MODEL",
            "~/Models/dflash/meta-models--Muse-Glimmer-30B-assistant-bf16-hf",
        ),
        "MUSE_GLIMMER_DSPARK_MODEL": os.getenv(
            "MUSE_GLIMMER_DSPARK_MODEL",
            "~/Models/dflash/DaoCloud--Muse-Glimmer-30B-DSpark-bf16",
        ),
        "QWEN38_NVFP4_MODEL": os.getenv(
            "QWEN38_NVFP4_MODEL",
            "~/Models/mlx/Brooooooklyn--Qwen3.8-27B-NVFP4-mlx",
        ),
        "QWEN38_MTP_MODEL": os.getenv(
            "QWEN38_MTP_MODEL",
            "~/Models/mlx/EigenLabs--Qwen3.8-27B-4bit",
        ),
        "QWEN38_MTP_HEAD": os.getenv(
            "QWEN38_MTP_HEAD",
            "~/Models/mlx/lowskillcoding--qwen38-mtp-head-4bit-g64",
        ),
        "QWEN38_DFLASH2_MODEL": os.getenv(
            "QWEN38_DFLASH2_MODEL",
            "~/Models/mlx/mlx-community--Qwen3.8-27B-4bit",
        ),
        "QWEN38_DFLASH2_DRAFT": os.getenv(
            "QWEN38_DFLASH2_DRAFT",
            "~/Models/mlx/ProCreations--Qwen3.8-27B-DFlash2-MLXFast-Q4",
        ),
        "HAUHAU35_OPTIQ_MLX_MODEL": os.getenv(
            "HAUHAU35_OPTIQ_MLX_MODEL",
            "~/Models/mlx/"
            "cyberCyber99--Qwen3.6-35B-A3B-Uncensored-HauhauCS-"
            "Aggressive-OptiQ-4bit-MLX",
        ),
        "DFLASH_35B_DRAFT": os.getenv(
            "DFLASH_35B_DRAFT",
            "~/Models/dflash/z-lab--Qwen3.6-35B-A3B-DFlash",
        ),
        "DFLASH_BIN": os.getenv(
            "DFLASH_BIN", str(root / ".venv/bin/dflash-compat")
        ),
        "PYTHON_BIN": os.getenv("MLX_PYTHON_BIN", sys.executable),
    }

    profiles: dict[str, Profile] = {}
    for name, item in raw["profiles"].items():
        profiles[name] = Profile(
            name=name,
            engine=item["engine"],
            description=item["description"],
            port=int(item["port"]),
            health_path=item["health_path"],
            command=tuple(_expand(arg, env) for arg in item["command"]),
            environment=tuple(
                (str(key), _expand(str(value), env))
                for key, value in item.get("environment", {}).items()
            ),
        )

    def relative(name: str) -> Path:
        configured = Path(general[name]).expanduser()
        return configured if configured.is_absolute() else root / configured

    return Settings(
        root=root,
        log_dir=relative("log_dir"),
        runtime_dir=relative("runtime_dir"),
        startup_timeout_seconds=float(general["startup_timeout_seconds"]),
        shutdown_timeout_seconds=float(general["shutdown_timeout_seconds"]),
        minimum_available_gib=float(general["minimum_available_gib"]),
        profiles=profiles,
    )
