from types import SimpleNamespace

from local_llm_control import mlx_metrics


def test_stream_wrapper_records_final_generation_response() -> None:
    before = mlx_metrics.snapshot()

    def generate():
        yield SimpleNamespace(
            prompt_tokens=12,
            prompt_tps=100.0,
            generation_tokens=1,
            generation_tps=20.0,
            peak_memory=3.0,
            finish_reason=None,
        )
        yield SimpleNamespace(
            prompt_tokens=12,
            prompt_tps=100.0,
            generation_tokens=2,
            generation_tps=21.0,
            peak_memory=3.5,
            finish_reason="stop",
        )

    assert len(list(mlx_metrics.instrument_stream_generate(generate)())) == 2
    after = mlx_metrics.snapshot()
    assert after["requests_total"] == before["requests_total"] + 1
    assert after["prompt_tokens_total"] == before["prompt_tokens_total"] + 12
    assert after["tokens_generated_total"] == before["tokens_generated_total"] + 2
    assert after["last_generation_tps"] == 21.0


def test_prometheus_output_contains_cache_and_speed_metrics() -> None:
    text = mlx_metrics.prometheus_text()

    assert "local_llm_cached_tokens_total " in text
    assert "local_llm_last_prompt_tps " in text
    assert "local_llm_last_generation_tps " in text
