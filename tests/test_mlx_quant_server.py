import json

from local_llm_control.mlx_quant_server import (
    DeduplicatingToolCallFormatter,
    adaptive_prefill_stream,
    consecutive_tool_loop,
    fixed_capacity_guard,
    process_message_content_idempotent,
    prompt_boundary_checkpoint_stream,
    quantized_cache_factory,
    stable_template_loader,
    stabilize_qwen_chat_template,
    tool_loop_safe_generate,
)
from local_llm_control.mlx_preallocated_cache import FixedQuantizedKVCache
from local_llm_control.mlx_turboquant_adapter import fixed_turbo4_cache_factory
from mlx_lm.models.cache import KVCache
import mlx.core as mx
import pytest
from types import SimpleNamespace


class Quantizable:
    def to_quantized(self, *, bits: int, group_size: int):
        return ("quantized", bits, group_size)


class RecurrentState:
    pass


def test_quantizes_attention_cache_but_preserves_recurrent_state() -> None:
    recurrent = RecurrentState()
    factory = quantized_cache_factory(
        lambda model: [Quantizable(), recurrent], bits=4, group_size=64
    )

    caches = factory(object())

    assert caches[0] == ("quantized", 4, 64)
    assert caches[1] is recurrent


def test_qwen_no_thinking_template_is_made_prefix_stable() -> None:
    unstable = """before
{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
after"""
    stable = stabilize_qwen_chat_template(unstable)
    assert stable is not None
    assert "if enable_thinking is defined and enable_thinking is false" in stable
    assert "<think>\\n\\n</think>\\n\\n' + content" in stable
    assert stabilize_qwen_chat_template(stable) == stable


def test_model_loader_emits_ready_marker(capsys) -> None:
    class Tokenizer:
        chat_template = None

    class Provider:
        tokenizer = Tokenizer()
        _tokenizer_config = {}

    provider = Provider()
    result = stable_template_loader(lambda provider: "loaded")(provider)
    assert result == "loaded"
    assert provider._tokenizer_config["fix_mistral_regex"] is True
    assert "local-llm: model load complete" in capsys.readouterr().err


def test_fixed_quantized_cache_allocates_once_and_enforces_capacity() -> None:
    cache = FixedQuantizedKVCache(max_size=8, group_size=64, bits=4)
    cache.reserve(
        batch_size=1,
        n_kv_heads=2,
        k_head_dim=64,
        v_head_dim=64,
        dtype=mx.float16,
    )
    reserved = cache.nbytes
    assert reserved > 0
    assert cache.empty()

    chunk = mx.zeros((1, 2, 3, 64), dtype=mx.float16)
    cache.update_and_fetch(chunk, chunk)
    mx.eval(cache.state)
    assert cache.offset == 3
    assert cache.nbytes == reserved
    assert not cache.empty()

    cache.update_and_fetch(chunk, chunk)
    mx.eval(cache.state)
    assert cache.offset == 6
    assert cache.nbytes == reserved

    with pytest.raises(ValueError, match="reserved 8"):
        cache.update_and_fetch(chunk, chunk)


def test_fixed_turbo4_cache_forces_single_sequence_server_path() -> None:
    model = SimpleNamespace(
        args=SimpleNamespace(
            num_key_value_heads=2,
            head_dim=64,
        )
    )
    factory = fixed_turbo4_cache_factory(
        lambda _model: [KVCache()],
        max_size=8,
        group_size=64,
    )

    cache = factory(model)[0]

    assert not hasattr(cache, "merge")
    assert cache.max_size == 8


def test_adaptive_prefill_reduces_step_at_deep_offsets() -> None:
    observed = {}

    def original(*args, **kwargs):
        observed.update(kwargs)
        yield "ok"

    wrapped = adaptive_prefill_stream(original)

    assert list(wrapped(
        prompt=[1] * 60_000,
        prompt_cache=[SimpleNamespace(offset=125_000)],
        prefill_step_size=2048,
    )) == ["ok"]
    assert observed["prefill_step_size"] == 256


def test_adaptive_prefill_uses_minimum_chunk_at_extreme_depth() -> None:
    observed = {}

    def original(*args, **kwargs):
        observed.update(kwargs)
        yield "ok"

    assert list(adaptive_prefill_stream(original)(
        prompt=[1] * 30_000,
        prompt_cache=[SimpleNamespace(offset=200_000)],
        prefill_step_size=2048,
    )) == ["ok"]
    assert observed["prefill_step_size"] == 128


def test_adaptive_prefill_keeps_configured_step_for_short_prompt() -> None:
    observed = {}

    def original(*args, **kwargs):
        observed.update(kwargs)
        yield "ok"

    list(adaptive_prefill_stream(original)(
        prompt=[1] * 4096,
        prompt_cache=[SimpleNamespace(offset=0)],
        prefill_step_size=2048,
    ))
    assert observed["prefill_step_size"] == 2048


def test_stream_saves_prompt_boundary_on_first_generated_token(
    monkeypatch,
) -> None:
    observed = []

    def checkpoint(cache, token):
        observed.append((cache, token))

    monkeypatch.setattr(
        "local_llm_control.mlx_session_cache.checkpoint_prompt_boundary",
        checkpoint,
    )
    cache = [object()]
    responses = [SimpleNamespace(token=90), SimpleNamespace(token=91)]

    wrapped = prompt_boundary_checkpoint_stream(
        lambda **_kwargs: iter(responses)
    )

    assert list(wrapped(prompt_cache=cache)) == responses
    assert observed == [(cache, 90)]


def test_message_content_accepts_object_tool_arguments_repeatedly() -> None:
    messages = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": {"path": "/tmp/test"},
            },
        }],
    }]

    process_message_content_idempotent(messages)
    process_message_content_idempotent(messages)

    assert messages[0]["content"] == ""
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "path": "/tmp/test",
    }


def test_message_content_decodes_string_tool_arguments_once() -> None:
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "read_file",
                "arguments": '{"path": "/tmp/test"}',
            },
        }],
    }]

    process_message_content_idempotent(messages)
    process_message_content_idempotent(messages)

    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "path": "/tmp/test",
    }


def test_fixed_capacity_guard_tokenizes_an_isolated_request() -> None:
    request = SimpleNamespace(messages=[{
        "tool_calls": [{
            "function": {
                "arguments": '{"path": "/tmp/test"}',
            },
        }],
    }])
    args = SimpleNamespace(max_tokens=1)
    observed = {}

    class Generator:
        model_provider = SimpleNamespace(tokenizer=object())

        @staticmethod
        def _tokenize(_tokenizer, copied_request, _args):
            copied_request.messages[0]["tool_calls"][0]["function"][
                "arguments"
            ] = {"path": "/tmp/test"}
            return [1], [], [], "normal"

    def original(_generator, item):
        observed["arguments"] = (
            item[1].messages[0]["tool_calls"][0]["function"]["arguments"]
        )

    fixed_capacity_guard(original, max_size=8)(
        Generator(), (object(), request, args)
    )

    assert observed["arguments"] == '{"path": "/tmp/test"}'


def test_tool_formatter_suppresses_exact_duplicates_across_stream_chunks() -> None:
    def parser(_text, _tools):
        return {
            "name": "Read",
            "arguments": {"filePath": "app/src/main/AndroidManifest.xml"},
        }

    formatter = DeduplicatingToolCallFormatter(
        parser, tools=[], streaming=True
    )

    first = formatter(["first", "duplicate"])
    second = formatter(["duplicate-again"])

    assert len(first) == 1
    assert second == []
    assert formatter.duplicates_suppressed == 2


def _assistant_tool_call(arguments) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "Edit", "arguments": arguments},
        }],
    }


def test_detects_identical_cross_turn_tool_loop() -> None:
    messages = [
        {"role": "user", "content": "fix it"},
        _assistant_tool_call('{"filePath": "/"}'),
        {"role": "tool", "content": "invalid path"},
        _assistant_tool_call({"filePath": "/"}),
        {"role": "tool", "content": "invalid path"},
        _assistant_tool_call({"filePath": "/"}),
        {"role": "tool", "content": "invalid path"},
    ]

    loop = consecutive_tool_loop(messages)

    assert loop is not None
    fingerprint, repeats = loop
    assert json.loads(fingerprint) == {
        "name": "Edit",
        "arguments": {"filePath": "/"},
    }
    assert repeats == 3


def test_cross_turn_tool_loop_resets_on_distinct_call() -> None:
    messages = [
        _assistant_tool_call({"filePath": "/a"}),
        {"role": "tool", "content": "failed"},
        _assistant_tool_call({"filePath": "/b"}),
        {"role": "tool", "content": "failed"},
        _assistant_tool_call({"filePath": "/b"}),
        {"role": "tool", "content": "failed"},
    ]

    assert consecutive_tool_loop(messages) is None


def test_tool_loop_guard_returns_normal_response_without_inference() -> None:
    called = False

    def original(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("inference must not run for a detected loop")

    messages = []
    for _ in range(3):
        messages.extend([
            _assistant_tool_call({"filePath": "/"}),
            {"role": "tool", "content": "invalid path"},
        ])
    request = SimpleNamespace(messages=messages)

    context, responses = tool_loop_safe_generate(original)(
        object(), request, SimpleNamespace()
    )
    response = list(responses)

    assert called is False
    assert context.has_tool_calling is False
    assert len(response) == 1
    assert response[0].finish_reason == "stop"
    assert "Tool loop stopped" in response[0].text


def test_tool_loop_guard_allows_one_retry() -> None:
    expected = object()

    def original(*_args, **_kwargs):
        return expected

    request = SimpleNamespace(messages=[
        _assistant_tool_call({"filePath": "/"}),
        {"role": "tool", "content": "invalid path"},
        _assistant_tool_call({"filePath": "/"}),
        {"role": "tool", "content": "invalid path"},
    ])

    assert tool_loop_safe_generate(original)(
        object(), request, SimpleNamespace()
    ) is expected
