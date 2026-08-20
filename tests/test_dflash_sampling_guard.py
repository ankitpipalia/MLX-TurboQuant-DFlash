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


def test_neutral_repetition_penalty_is_not_treated_as_a_request():
    """1.0 is the conventional no-op; losing speculation for it is waste."""
    assert unsupported_sampling_reasons(_args(repetition_penalty=1.0)) == []
    assert unsupported_sampling_reasons(_args(repetition_penalty=1.1)) == [
        "repetition_penalty"
    ]


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
