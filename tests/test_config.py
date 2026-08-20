from pathlib import Path

import mlx.core as mx
import pytest

from local_llm_control.config import load_settings
from local_llm_control.dflash_compat import (
    install_dflash_turboquant,
    normalize_draft_config,
)


def test_profiles_expand_runtime_and_models(monkeypatch, tmp_path: Path) -> None:
    llama = tmp_path / "llama-server"
    gguf = tmp_path / "model.gguf"
    mlx = tmp_path / "mlx-model"
    monkeypatch.setenv("LLAMA_BIN", str(llama))
    monkeypatch.setenv("GGUF_MODEL", str(gguf))
    monkeypatch.setenv("MLX_MODEL", str(mlx))

    settings = load_settings()

    safe = settings.profiles["llama-turbo4-safe"]
    assert safe.command[0] == str(llama)
    assert str(gguf) in safe.command
    assert "turbo4" in safe.command
    assert "--no-host" in safe.command
    assert "--checkpoint-min-step" in safe.command


def test_only_one_parallel_slot_in_large_llama_profiles() -> None:
    settings = load_settings()
    for profile in settings.profiles.values():
        if profile.engine != "llama.cpp":
            continue
        position = profile.command.index("--parallel")
        assert profile.command[position + 1] == "1"


def test_finance_vlm_profiles_are_quality_first_and_localhost_only() -> None:
    settings = load_settings()
    expected = {
        "mlx-vlm-finance-bf16",
        "mlx-vlm-paddle16-bf16",
        "mlx-vlm-glm-ocr-bf16",
        "mlx-vlm-semantic-6bit",
        "mlx-vlm-nuextract-bf16",
    }

    for name in expected:
        profile = settings.profiles[name]
        command = profile.command
        assert profile.engine == "mlx-vlm"
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert command[command.index("--port") + 1] == "8098"
        assert int(command[command.index("--vision-cache-size") + 1]) >= 1
        assert "--kv-bits" not in command
        assert "4bit" not in " ".join(command).lower()
        if name == "mlx-vlm-semantic-6bit":
            assert "qwen3.5-9b-6bit" in " ".join(command).lower()
        else:
            assert "6bit" not in " ".join(command).lower()


def test_muse_glimmer_profile_uses_supported_vlm_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    model = tmp_path / "muse-glimmer-awq4"
    monkeypatch.setenv("MUSE_GLIMMER_AWQ_MODEL", str(model))

    profile = load_settings().profiles["mlx-vlm-muse-glimmer-30b-awq4"]
    command = profile.command

    assert profile.engine == "mlx-vlm"
    assert profile.health_path == "/health"
    assert command[command.index("--model") + 1] == str(model)
    assert command[command.index("--host") + 1] == "0.0.0.0"
    assert command[command.index("--max-num-seqs") + 1] == "1"
    assert command[command.index("--prefill-step-size") + 1] == "512"
    assert command[command.index("--max-kv-size") + 1] == "131072"
    assert "--kv-bits" not in command

    text_profile = load_settings().profiles[
        "mlx-vlm-muse-glimmer-30b-awq4-text"
    ]
    assert "local_llm_control.mlx_vlm_text_server" in text_profile.command
    assert text_profile.command[
        text_profile.command.index("--max-kv-size") + 1
    ] == "131072"
    assert text_profile.command[
        text_profile.command.index("--vision-cache-size") + 1
    ] == "0"

    dflash_profile = load_settings().profiles[
        "mlx-vlm-muse-glimmer-30b-awq4-dflash"
    ]
    assert dflash_profile.command[
        dflash_profile.command.index("--draft-kind") + 1
    ] == "dflash"
    assert "local_llm_control.mlx_vlm_text_server" in dflash_profile.command
    assert dflash_profile.command[
        dflash_profile.command.index("--max-kv-size") + 1
    ] == "131072"
    assert "--draft-block-size" not in dflash_profile.command
    assert dflash_profile.command[
        dflash_profile.command.index("--prefill-step-size") + 1
    ] == "2048"

    cached_dflash_profile = load_settings().profiles[
        "mlx-vlm-muse-glimmer-30b-awq4-dflash-cached"
    ]
    assert "local_llm_control.mlx_vlm_cached_dflash_server" in (
        cached_dflash_profile.command
    )
    assert dict(cached_dflash_profile.environment)["APC_ENABLED"] == "1"
    assert cached_dflash_profile.command[
        cached_dflash_profile.command.index("--draft-block-size") + 1
    ] == "4"
    assert cached_dflash_profile.command[
        cached_dflash_profile.command.index("--prefill-step-size") + 1
    ] == "2048"

    dspark_profile = load_settings().profiles[
        "mlx-dspark-muse-glimmer-30b-awq4"
    ]
    assert dspark_profile.engine == "mlx-dspark"
    assert "local_llm_control.mlx_dspark_text_server" in dspark_profile.command
    assert "--wired-limit" not in dspark_profile.command
    assert dspark_profile.command[
        dspark_profile.command.index("--max-draft") + 1
    ] == "auto"

    qwen38 = load_settings().profiles["mlx-vlm-qwen38-nvfp4"]
    assert qwen38.engine == "mlx-vlm"
    assert qwen38.command[qwen38.command.index("--max-kv-size") + 1] == "262144"
    assert qwen38.command[qwen38.command.index("--vision-cache-size") + 1] == "0"


def test_qwen38_native_mtp_profiles_are_pinned_and_bounded(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "qwen38-q4"
    head = tmp_path / "qwen38-mtp-q4"
    monkeypatch.setenv("QWEN38_MTP_MODEL", str(target))
    monkeypatch.setenv("QWEN38_MTP_HEAD", str(head))

    settings = load_settings()
    serial = settings.profiles["mlx-vlm-qwen38-q4-serial"]
    serial_turbo4 = settings.profiles["mlx-vlm-qwen38-q4-turbo4"]
    deep_turbo4 = settings.profiles["mlx-vlm-qwen38-q4-turbo4-deep"]
    mtp = settings.profiles["mlx-vlm-qwen38-q4-mtp"]
    fixed2 = settings.profiles["mlx-vlm-qwen38-q4-mtp-fixed2"]
    turbo4 = settings.profiles["mlx-vlm-qwen38-q4-mtp-turbo4"]
    adaptive = settings.profiles["mlx-vlm-qwen38-q4-mtp-adaptive"]

    for profile in (
        serial,
        serial_turbo4,
        deep_turbo4,
        mtp,
        fixed2,
        adaptive,
        turbo4,
    ):
        command = profile.command
        assert profile.engine == "mlx-vlm"
        assert command[command.index("--model") + 1] == str(target)
        assert command[command.index("--host") + 1] == "0.0.0.0"
        assert command[command.index("--max-num-seqs") + 1] == "1"
        assert command[command.index("--max-kv-size") + 1] == "262144"
        assert command[command.index("--vision-cache-size") + 1] == "0"
        assert dict(profile.environment)["APC_EXACT_CACHE_ENTRIES"] == "1"
        assert dict(profile.environment)["LOCAL_LLM_ALLOW_MISSING_VISION"] == "1"

    assert "--draft-model" not in serial.command
    assert "--draft-model" not in serial_turbo4.command
    assert serial_turbo4.command[
        serial_turbo4.command.index("--prefill-step-size") + 1
    ] == "2048"
    assert serial_turbo4.command[
        serial_turbo4.command.index("--kv-quant-scheme") + 1
    ] == "turboquant"
    assert "--kv-key-scheme" not in serial_turbo4.command
    assert deep_turbo4.command[
        deep_turbo4.command.index("--prefill-step-size") + 1
    ] == "512"
    assert deep_turbo4.command[
        deep_turbo4.command.index("--max-tokens") + 1
    ] == "8192"
    assert mtp.command[mtp.command.index("--draft-model") + 1] == str(head)
    assert mtp.command[mtp.command.index("--draft-kind") + 1] == "mtp"
    assert mtp.command[mtp.command.index("--draft-block-size") + 1] == "4"
    assert fixed2.command[fixed2.command.index("--draft-block-size") + 1] == "2"
    assert "--kv-bits" not in mtp.command
    assert "local_llm_control.mlx_vlm_adaptive_mtp_server" in adaptive.command
    assert adaptive.command[adaptive.command.index("--draft-block-size") + 1] == "6"
    assert dict(adaptive.environment)["LOCAL_LLM_MTP_HEAD_COST_RATIO"] == "0.18"
    assert turbo4.command[turbo4.command.index("--kv-bits") + 1] == "4"
    assert turbo4.command[
        turbo4.command.index("--kv-quant-scheme") + 1
    ] == "turboquant"
    assert "--kv-key-scheme" not in turbo4.command


def test_profile_environment_is_expanded(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "profiles.toml"
    model = tmp_path / "model"
    monkeypatch.setenv("MLX_MODEL", str(model))
    config.write_text(
        """
[settings]
log_dir = "logs"
runtime_dir = "run"
startup_timeout_seconds = 1
shutdown_timeout_seconds = 1
minimum_available_gib = 0

[profiles.test]
engine = "mlx-vlm"
description = "test"
port = 8098
health_path = "/health"
environment = { APC_ENABLED = "1", CACHE_PATH = "{MLX_MODEL}/cache" }
command = ["/usr/bin/true"]
"""
    )

    profile = load_settings(config).profiles["test"]

    assert dict(profile.environment) == {
        "APC_ENABLED": "1",
        "CACHE_PATH": f"{model}/cache",
    }


def test_dflash_profile_uses_explicit_matched_draft(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "hauhau-optiq"
    draft = tmp_path / "qwen36-35b-dflash"
    binary = tmp_path / "dflash"
    monkeypatch.setenv("HAUHAU35_OPTIQ_MLX_MODEL", str(target))
    monkeypatch.setenv("DFLASH_35B_DRAFT", str(draft))
    monkeypatch.setenv("DFLASH_BIN", str(binary))

    profile = load_settings().profiles["mlx-dflash-hauhau35-optiq4"]

    assert profile.engine == "mlx-dflash"
    assert profile.health_path == "/v1/models"
    assert profile.command[:2] == (str(binary), "serve")
    assert profile.command[profile.command.index("--model") + 1] == str(target)
    assert profile.command[profile.command.index("--draft") + 1] == str(draft)
    assert profile.command[profile.command.index("--draft-quant") + 1] == "w4"
    assert profile.command[profile.command.index("--dflash-max-ctx") + 1] == "262144"
    assert "--quantize-kv-cache" not in profile.command


def test_qwen38_dflash2_profile_uses_quantized_matched_pair(
    monkeypatch, tmp_path
) -> None:
    target = tmp_path / "qwen38-q4"
    draft = tmp_path / "qwen38-dflash2-q4"
    monkeypatch.setenv("QWEN38_DFLASH2_MODEL", str(target))
    monkeypatch.setenv("QWEN38_DFLASH2_DRAFT", str(draft))

    profile = load_settings().profiles["mlx-dflash2-qwen38-q4"]

    assert profile.engine == "mlx-dflash"
    assert profile.command[profile.command.index("--model") + 1] == str(target)
    assert profile.command[profile.command.index("--draft") + 1] == str(draft)
    assert profile.command[profile.command.index("--draft-quant") + 1] == "none"
    assert profile.command[profile.command.index("--verify-mode") + 1] == "dflash"
    assert "--quantize-kv-cache" in profile.command
    assert "--no-prefix-cache" in profile.command
    assert profile.command[profile.command.index("--dflash-max-ctx") + 1] == "98304"

    cached = load_settings().profiles["mlx-dflash2-qwen38-q4-cached"]
    assert "--quantize-kv-cache" not in cached.command
    assert "--prefix-cache" in cached.command
    assert cached.command[cached.command.index("--dflash-max-ctx") + 1] == "65536"

    turbo4 = load_settings().profiles[
        "mlx-dflash2-qwen38-q4-turbo4-262k"
    ]
    assert dict(turbo4.environment)["LOCAL_LLM_DFLASH_TURBOQUANT"] == "turbo4"
    assert "--quantize-kv-cache" in turbo4.command
    assert "--no-prefix-cache" in turbo4.command

    # The physical arena must exceed the logical context: DFlash appends a whole
    # draft block before trimming, so peak offset can overshoot by up to the
    # verify cap. Equal sizes make that overshoot a hard mid-stream ValueError.
    arena = int(dict(turbo4.environment)["LOCAL_LLM_DFLASH_TURBOQUANT_MAX_SIZE"])
    logical = int(turbo4.command[turbo4.command.index("--dflash-max-ctx") + 1])
    verify_cap = int(turbo4.command[turbo4.command.index("--verify-len-cap") + 1])
    assert arena >= logical + verify_cap, (
        f"arena {arena} leaves no room for a {verify_cap}-token overshoot past "
        f"the {logical}-token context"
    )

    # 96 measured ~50 prompt tok/s (~82 min for a full cold 262K prefill).
    prefill = int(turbo4.command[turbo4.command.index("--prefill-step-size") + 1])
    assert prefill >= 256, f"prefill chunk {prefill} is pathologically small"


def test_dflash_turbo4_converts_only_quantized_attention_caches(
    monkeypatch,
) -> None:
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
    from mlx_lm.models.cache import QuantizedKVCache
    from mlx_turboquant.kv_cache import TurboQuantKVCache

    original = QwenGdnTargetOps.make_cache
    native_quantized = QuantizedKVCache(group_size=64, bits=8)
    recurrent = object()

    def fake_make_cache(self, *args, **kwargs):
        return [recurrent, native_quantized]

    monkeypatch.setattr(QwenGdnTargetOps, "make_cache", fake_make_cache)
    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT", "turbo4")
    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT_MAX_SIZE", "8")
    install_dflash_turboquant()

    class Args:
        num_key_value_heads = 1
        head_dim = 64

    class Target:
        args = Args()

    ops = QwenGdnTargetOps()
    monkeypatch.setattr(ops, "text_wrapper", lambda target: target)
    caches = ops.make_cache(Target())

    assert caches[0] is recurrent
    assert isinstance(caches[1], TurboQuantKVCache)
    assert caches[1].bits == 4
    assert caches[1].max_size == 8
    monkeypatch.setattr(QwenGdnTargetOps, "make_cache", original)


def _install_turbo4_bridge(monkeypatch, caches, max_size):
    """Patch DFlash's cache factory and install the Turbo4 bridge over it."""
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps

    monkeypatch.setattr(
        QwenGdnTargetOps, "make_cache", lambda self, *a, **k: list(caches)
    )
    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT", "turbo4")
    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT_MAX_SIZE", str(max_size))
    install_dflash_turboquant()
    return QwenGdnTargetOps()


class _Layer:
    def __init__(self, is_linear: bool) -> None:
        self.is_linear = is_linear


class _Target:
    """Stand-in for a loaded target model; supports weak references.

    Shaped so DFlash's real ``text_wrapper``/``text_model`` helpers resolve it:
    one full-attention layer, matching the single quantized cache the fakes
    below hand back.
    """

    class args:
        num_key_value_heads = 1
        head_dim = 64

    def __init__(self) -> None:
        self.model = type("TextModel", (), {"layers": [_Layer(False)]})()


def test_dflash_turbo4_rejects_partially_converted_cache(monkeypatch) -> None:
    """A layout change must fail loudly, not quantize a subset of layers."""
    from mlx_lm.models.cache import QuantizedKVCache

    ops = _install_turbo4_bridge(
        monkeypatch, [object(), QuantizedKVCache(group_size=64, bits=8)], 0
    )
    # Two full-attention layers, but DFlash handed back only one quantized cache.
    target = _Target()
    target.model.layers = [_Layer(True), _Layer(False), _Layer(False)]

    with pytest.raises(RuntimeError, match="partially converted"):
        ops.make_cache(target)


def test_dflash_turbo4_reuses_one_arena_for_the_same_model(monkeypatch) -> None:
    from mlx_lm.models.cache import QuantizedKVCache

    ops = _install_turbo4_bridge(
        monkeypatch, [object(), QuantizedKVCache(group_size=64, bits=8)], 8
    )
    target = _Target()

    first = ops.make_cache(target)[1]
    first.update_and_fetch(mx.zeros((1, 1, 2, 64)), mx.zeros((1, 1, 2, 64)))
    assert first.offset == 2

    second = ops.make_cache(target)[1]
    assert second is first, "the fixed arena must be reused, not reallocated"
    assert second.offset == 0, "a reused arena must restart at offset zero"


def test_dflash_turbo4_releases_the_arena_when_a_model_is_retired(
    monkeypatch,
) -> None:
    """The pool holds caches but not the model, so it must not outlive it."""
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
    from mlx_lm.models.cache import QuantizedKVCache

    ops = _install_turbo4_bridge(
        monkeypatch, [object(), QuantizedKVCache(group_size=64, bits=8)], 8
    )
    pools = QwenGdnTargetOps.make_cache._local_llm_fixed_pools

    retired = _Target()
    retired_cache = ops.make_cache(retired)[1]
    assert len(pools) == 1

    del retired
    fresh_cache = ops.make_cache(_Target())[1]

    # Asserted on object identity rather than on the dict key, because CPython
    # may legitimately recycle the freed address for the replacement model --
    # which is precisely the case a bare ``id`` key would get wrong.
    assert fresh_cache is not retired_cache, (
        "a recycled id must not inherit the retired model's arena"
    )
    assert len(pools) == 1, "a retired model's arena must not accumulate"


def test_current_dflash_nested_schema_is_normalized_without_mutation() -> None:
    original = {
        "dflash_config": {"block_size": 16, "mask_token_id": 248077},
        "rope_parameters": {"rope_theta": 10_000_000, "rope_type": "default"},
    }

    normalized = normalize_draft_config(original)

    assert normalized["block_size"] == 16
    assert normalized["rope_theta"] == 10_000_000
    assert "block_size" not in original
    assert "rope_theta" not in original
