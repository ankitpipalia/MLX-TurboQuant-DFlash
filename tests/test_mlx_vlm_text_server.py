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
