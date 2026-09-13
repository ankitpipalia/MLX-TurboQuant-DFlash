"""Sampling features the speculative path drops must reach exact AR decode.

DFlash parses penalties, logit_bias and logprobs off the request and then
forwards only temperature/top_p/top_k/min_p, so those features are accepted and
silently ignored. Requests using them are routed to plain autoregressive decode
on the target instead, via the same drop-the-draft-model mechanism upstream's
AR fast path uses.
"""

from __future__ import annotations

from types import SimpleNamespace

from local_llm_control.dflash_sampling_guard import (
    install_exact_sampling_guard,
    unsupported_sampling_reasons,
)


def _args(**overrides):
    defaults = dict(
        repetition_penalty=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        xtc_probability=0.0,
        logit_bias=None,
        logprobs=False,
        top_logprobs=-1,
        temp=0.7,
        top_p=0.8,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_ordinary_sampling_keeps_the_speculative_path():
    """temperature/top_p/top_k/min_p are forwarded, so they are supported."""
    assert unsupported_sampling_reasons(_args()) == []
    assert unsupported_sampling_reasons(_args(temp=1.0, top_p=0.95)) == []


def test_repetition_penalty_is_supported_upstream_now():
    """dflash-mlx 0.1.10+omlx.7 applies it inside the speculative loop, so
    diverting these requests to exact AR would waste speculation."""
    assert unsupported_sampling_reasons(_args(repetition_penalty=1.0)) == []
    assert unsupported_sampling_reasons(_args(repetition_penalty=1.1)) == []


def test_every_dropped_feature_is_detected():
    for field, value in (
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("xtc_probability", 0.3),
        ("logit_bias", {"123": 5}),
        ("logprobs", True),
        ("top_logprobs", 5),
    ):
        reasons = unsupported_sampling_reasons(_args(**{field: value}))
        assert reasons == [field], f"{field} was not detected: {reasons}"


def test_missing_attributes_are_tolerated():
    """An older or newer request object must not crash the guard."""
    assert unsupported_sampling_reasons(SimpleNamespace()) == []


def test_guard_routes_flagged_requests_to_the_parent_handler(monkeypatch):
    import mlx_lm.server as mlx_server
    from dflash_mlx import serve as dflash_serve

    calls: list[str] = []
    provider = SimpleNamespace(draft_model="the-draft")

    def fake_original(self, request):
        calls.append(f"dflash:draft={self.model_provider.draft_model}")

    def fake_parent(self, request):
        calls.append(f"parent:draft={self.model_provider.draft_model}")

    monkeypatch.setattr(
        dflash_serve.DFlashResponseGenerator, "_serve_single", fake_original
    )
    monkeypatch.setattr(
        mlx_server.ResponseGenerator, "_serve_single", fake_parent
    )
    monkeypatch.setenv("LOCAL_LLM_DFLASH_EXACT_SAMPLING", "1")

    install_exact_sampling_guard()
    handler = SimpleNamespace(
        model_provider=provider,
        _serve_single=dflash_serve.DFlashResponseGenerator._serve_single,
    )

    # Plain request: speculative path, draft model untouched.
    dflash_serve.DFlashResponseGenerator._serve_single(
        handler, (None, None, _args())
    )
    assert calls == ["dflash:draft=the-draft"]

    # Flagged request: parent handler, and the draft model is dropped for it.
    dflash_serve.DFlashResponseGenerator._serve_single(
        handler, (None, None, _args(logprobs=True))
    )
    assert calls[1] == "parent:draft=None"
    assert provider.draft_model == "the-draft", "draft model must be restored"


def test_guard_restores_the_draft_model_even_if_the_request_raises(monkeypatch):
    import mlx_lm.server as mlx_server
    from dflash_mlx import serve as dflash_serve

    provider = SimpleNamespace(draft_model="the-draft")

    def boom(self, request):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(
        dflash_serve.DFlashResponseGenerator,
        "_serve_single",
        lambda self, request: None,
    )
    monkeypatch.setattr(mlx_server.ResponseGenerator, "_serve_single", boom)
    monkeypatch.setenv("LOCAL_LLM_DFLASH_EXACT_SAMPLING", "1")

    install_exact_sampling_guard()
    handler = SimpleNamespace(model_provider=provider)

    try:
        dflash_serve.DFlashResponseGenerator._serve_single(
            handler, (None, None, _args(logprobs=True))
        )
    except RuntimeError:
        pass
    assert provider.draft_model == "the-draft"


def test_guard_can_be_disabled(monkeypatch):
    from dflash_mlx import serve as dflash_serve

    monkeypatch.setenv("LOCAL_LLM_DFLASH_EXACT_SAMPLING", "0")
    original = dflash_serve.DFlashResponseGenerator._serve_single
    monkeypatch.setattr(
        dflash_serve.DFlashResponseGenerator, "_serve_single", original
    )

    install_exact_sampling_guard()

    assert dflash_serve.DFlashResponseGenerator._serve_single is original


def _real_args(**overrides):
    """A genuine mlx-lm GenerationArguments, not a hand-shaped stub.

    The flat-attribute stubs above document intent, but mlx-lm nests penalties
    under ``logits`` and XTC under ``sampling``. A guard written against flat
    attributes silently detects nothing, and a flat stub would never reveal it.
    Every field is required, so neutral values are spelled out.
    """
    from mlx_lm.server import (
        GenerationArguments,
        LogitsProcessorArguments,
        SamplingArguments,
    )

    sampling = dict(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        xtc_probability=0.0,
        xtc_threshold=0.0,
    )
    logits = dict(
        logit_bias=None,
        repetition_penalty=1.0,
        repetition_context_size=20,
        presence_penalty=0.0,
        presence_context_size=20,
        frequency_penalty=0.0,
        frequency_context_size=20,
    )
    top = dict(
        model="m",
        stop_words=[],
        max_tokens=128,
        num_draft_tokens=0,
        logprobs=False,
        top_logprobs=-1,
        seed=None,
        chat_template_kwargs={},
    )
    for key, value in overrides.items():
        if key in sampling:
            sampling[key] = value
        elif key in logits:
            logits[key] = value
        elif key in top:
            top[key] = value
        else:
            raise AssertionError(f"unknown GenerationArguments field: {key}")
    return GenerationArguments(
        sampling=SamplingArguments(**sampling),
        logits=LogitsProcessorArguments(**logits),
        **top,
    )


def test_real_generation_arguments_plain_request_is_supported():
    assert unsupported_sampling_reasons(_real_args()) == []
    assert unsupported_sampling_reasons(
        _real_args(temperature=0.7, top_p=0.8)
    ) == [], "ordinary non-greedy sampling is forwarded, so it is supported"


def test_real_generation_arguments_nested_penalties_are_detected():
    """These live on args.logits; a flat read would miss every one of them."""
    assert unsupported_sampling_reasons(
        _real_args(presence_penalty=0.5)
    ) == ["presence_penalty"]
    assert unsupported_sampling_reasons(
        _real_args(frequency_penalty=0.5)
    ) == ["frequency_penalty"]
    assert unsupported_sampling_reasons(
        _real_args(repetition_penalty=1.2)
    ) == [], "now handled natively by the speculative runtime"
    assert unsupported_sampling_reasons(
        _real_args(logit_bias={"7": 3.0})
    ) == ["logit_bias"]


def test_real_generation_arguments_nested_xtc_is_detected():
    """xtc_probability lives on args.sampling, not alongside the penalties."""
    assert unsupported_sampling_reasons(
        _real_args(xtc_probability=0.3)
    ) == ["xtc_probability"]


def test_real_generation_arguments_neutral_repetition_penalty_is_ignored():
    assert unsupported_sampling_reasons(_real_args(repetition_penalty=1.0)) == []


def test_real_generation_arguments_top_level_logprobs_are_detected():
    assert unsupported_sampling_reasons(_real_args(logprobs=True)) == ["logprobs"]
    assert unsupported_sampling_reasons(_real_args(top_logprobs=5)) == [
        "top_logprobs"
    ]


def test_fixed_arena_refuses_instead_of_falling_back_to_native_kv(monkeypatch):
    """Exact AR builds its own cache, so at long context it would abandon the
    reserved arena for a full-precision one -- a different memory regime."""
    import mlx_lm.server as mlx_server
    from dflash_mlx import serve as dflash_serve

    from local_llm_control.dflash_sampling_guard import fixed_arena_active

    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT", "turbo4")
    monkeypatch.setenv("LOCAL_LLM_DFLASH_TURBOQUANT_MAX_SIZE", "131136")
    monkeypatch.setenv("LOCAL_LLM_DFLASH_EXACT_SAMPLING", "1")
    assert fixed_arena_active() is True

    parent_calls: list[str] = []
    monkeypatch.setattr(
        dflash_serve.DFlashResponseGenerator,
        "_serve_single",
        lambda self, request: None,
    )
    monkeypatch.setattr(
        mlx_server.ResponseGenerator,
        "_serve_single",
        lambda self, request: parent_calls.append("parent"),
    )
    install_exact_sampling_guard()

    posted: list[object] = []
    queue = SimpleNamespace(put=posted.append)
    handler = SimpleNamespace(model_provider=SimpleNamespace(draft_model="d"))

    dflash_serve.DFlashResponseGenerator._serve_single(
        handler, (queue, None, _real_args(presence_penalty=0.5))
    )

    assert parent_calls == [], "must not fall back to a native-KV AR path"
    assert len(posted) == 1 and isinstance(posted[0], ValueError)
    assert "fixed Turbo4" in str(posted[0])


def test_without_a_fixed_arena_exact_ar_is_still_used(monkeypatch):
    import mlx_lm.server as mlx_server
    from dflash_mlx import serve as dflash_serve

    monkeypatch.delenv("LOCAL_LLM_DFLASH_TURBOQUANT", raising=False)
    monkeypatch.setenv("LOCAL_LLM_DFLASH_EXACT_SAMPLING", "1")

    parent_calls: list[str] = []
    monkeypatch.setattr(
        dflash_serve.DFlashResponseGenerator,
        "_serve_single",
        lambda self, request: None,
    )
    monkeypatch.setattr(
        mlx_server.ResponseGenerator,
        "_serve_single",
        lambda self, request: parent_calls.append("parent"),
    )
    install_exact_sampling_guard()

    handler = SimpleNamespace(model_provider=SimpleNamespace(draft_model="d"))
    dflash_serve.DFlashResponseGenerator._serve_single(
        handler, (SimpleNamespace(put=lambda _: None), None, _real_args(logprobs=True))
    )
    assert parent_calls == ["parent"], "dynamic mode should honour them via AR"
