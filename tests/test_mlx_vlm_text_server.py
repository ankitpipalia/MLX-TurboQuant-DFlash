from types import SimpleNamespace

import local_llm_control.mlx_vlm_text_server as text_server


def test_text_first_loader_forces_lazy_and_evaluates_language(monkeypatch) -> None:
    calls = {}
    language = SimpleNamespace(parameters=lambda: ["language-weights"])
    model = SimpleNamespace(language_model=language)

    def original(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return model, "processor"

    monkeypatch.setattr(
        text_server.mx,
        "eval",
        lambda parameters: calls.setdefault("evaluated", parameters),
    )

    loaded = text_server.load_text_first(original, "model", lazy=False)

    assert loaded == (model, "processor")
    assert calls["kwargs"]["lazy"] is True
    assert calls["evaluated"] == ["language-weights"]


def test_reasoning_effort_is_forwarded_to_onyx_template(monkeypatch) -> None:
    from mlx_vlm.server.generation import GenerationArguments

    original = GenerationArguments.to_template_kwargs
    monkeypatch.setattr(
        GenerationArguments,
        "to_template_kwargs",
        getattr(original, "_local_llm_original", original),
    )
    text_server.install_reasoning_strength_bridge()

    args = GenerationArguments(reasoning_effort="none")
    assert args.to_template_kwargs()["reasoning_strength"] == "none"


def test_onyx_direct_answer_prompt_honors_thinking_switch() -> None:
    original = (
        "prefix <atem:function_calls> suffix "
        + text_server._ONYX_GENERATION_PROMPT
    )
    tokenizer = SimpleNamespace(chat_template=original)
    processor = SimpleNamespace(tokenizer=tokenizer, chat_template=original)

    assert text_server.patch_onyx_generation_prompt(processor) is True
    assert text_server._ONYX_REASONING_AWARE_GENERATION_PROMPT in processor.chat_template
    assert text_server._ONYX_REASONING_AWARE_GENERATION_PROMPT in tokenizer.chat_template
    assert text_server.patch_onyx_generation_prompt(processor) is True


def test_qwen_reasoning_aliases_map_openai_vocabulary_to_qwens_set() -> None:
    """Qwen3.8's template raises on anything outside xhigh/medium/low."""
    normalize = text_server.normalize_qwen_reasoning_effort

    # OpenAI-standard values that would otherwise fail the whole request.
    assert normalize("high") == "xhigh"
    assert normalize("minimal") == "low"
    assert normalize("none") == "low"
    assert normalize("max") == "xhigh"

    # Values Qwen already accepts survive untouched, case-insensitively.
    assert normalize("xhigh") == "xhigh"
    assert normalize("medium") == "medium"
    assert normalize("  LOW  ") == "low"

    # Absent stays absent, and an unknown value is passed through rather than
    # silently reinterpreted, so a typo still surfaces as a template error.
    assert normalize(None) is None
    assert normalize("quick") == "quick"


def test_qwen_reasoning_normalizer_is_opt_in(monkeypatch) -> None:
    from mlx_vlm.server.generation import GenerationArguments

    monkeypatch.delenv("LOCAL_LLM_QWEN_REASONING_ALIASES", raising=False)
    original = GenerationArguments.to_template_kwargs
    monkeypatch.setattr(GenerationArguments, "to_template_kwargs", original)

    text_server.install_qwen_reasoning_normalizer()

    assert GenerationArguments.to_template_kwargs is original, (
        "the Muse/Onyx profiles share this class and must be left alone"
    )


def test_qwen_reasoning_normalizer_applies_alias_and_default(monkeypatch) -> None:
    from mlx_vlm.server.generation import GenerationArguments

    monkeypatch.setenv("LOCAL_LLM_QWEN_REASONING_ALIASES", "1")
    monkeypatch.setenv("LOCAL_LLM_QWEN_REASONING_DEFAULT", "medium")
    monkeypatch.setattr(
        GenerationArguments,
        "to_template_kwargs",
        lambda self: dict(self.kw),
    )

    text_server.install_qwen_reasoning_normalizer()
    render = GenerationArguments.to_template_kwargs

    assert render(SimpleNamespace(kw={"reasoning_effort": "high"}))[
        "reasoning_effort"
    ] == "xhigh"
    # No client value: the template's own default is xhigh, which spends most
    # of the output budget thinking on this hardware.
    assert render(SimpleNamespace(kw={}))["reasoning_effort"] == "medium"


def test_qwen_reasoning_default_must_be_a_level_qwen_accepts(monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("LOCAL_LLM_QWEN_REASONING_ALIASES", "1")
    monkeypatch.setenv("LOCAL_LLM_QWEN_REASONING_DEFAULT", "high")

    with pytest.raises(ValueError, match="must be one of"):
        text_server.install_qwen_reasoning_normalizer()


def test_apc_bridge_retires_when_upstream_keeps_snapshots_packed(monkeypatch):
    """Patching over a native implementation is worse than not patching.

    mlx-vlm 0.7.0 gave quantized caches an explicit snapshot contract that
    clone_cache_entry consults before the legacy float fallback. Installing the
    bridge anyway would intercept ahead of it and substitute our copy for the
    maintained one, which also handles fractional bit widths and feeds the
    disk-backed APC.
    """
    from mlx_vlm import apc_adapters

    original_clone = apc_adapters.clone_cache_entry
    monkeypatch.setattr(apc_adapters, "clone_cache_entry", original_clone)
    monkeypatch.setattr(apc_adapters, "merge_cache_entries", apc_adapters.merge_cache_entries)

    monkeypatch.setattr(text_server, "upstream_keeps_apc_packed", lambda: True)
    text_server.install_compressed_turboquant_apc_bridge()
    assert apc_adapters.clone_cache_entry is original_clone, (
        "bridge must not intercept when upstream already keeps snapshots packed"
    )

    monkeypatch.setattr(text_server, "upstream_keeps_apc_packed", lambda: False)
    text_server.install_compressed_turboquant_apc_bridge()
    assert getattr(
        apc_adapters.clone_cache_entry, "_local_llm_turboquant_compressed", False
    ), "bridge must still install on a release without the contract"


def test_installed_mlx_vlm_keeps_turboquant_snapshots_packed():
    """Guards the assumption the retirement rests on, against the real release."""
    import mlx.core as mx
    from mlx_vlm import apc_adapters
    from mlx_vlm.turboquant import TurboQuantKVCache

    assert text_server.upstream_keeps_apc_packed() is True

    cache = TurboQuantKVCache(bits=4)
    cache.update_and_fetch(
        mx.zeros((1, 2, 8, 64), dtype=mx.bfloat16),
        mx.zeros((1, 2, 8, 64), dtype=mx.bfloat16),
    )
    clone = apc_adapters.clone_cache_entry(
        cache, min_capacity_tokens=0, eval_targets=[]
    )
    assert isinstance(clone, TurboQuantKVCache), (
        "upstream dequantized to float; the bridge should not have retired"
    )
