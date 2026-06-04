from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vllm_nnrp_adapter import BenchmarkConfig
from vllm_nnrp_adapter.benchmark import (
    _failed_sample,
    _http_headers,
    _long_context_chat_request,
    _measure_path_matrix,
    _openai_chunk_text_deltas,
    _run_http_sse_request_sync,
    _successful_sample,
    estimate_token_count,
    run_benchmark,
    run_comparison_benchmark_file,
    run_comparison_benchmark_file_with_backend_spec,
    run_in_process_comparison_benchmark,
    synthetic_prompt,
)
from vllm_nnrp_adapter.cli import main
from vllm_nnrp_adapter.conformance import MockChatCompletionBackend
from vllm_nnrp_adapter.embedded import EmbeddedTcpServerConfig


async def make_async_mock_backend() -> MockChatCompletionBackend:
    return MockChatCompletionBackend()


@pytest.mark.asyncio
async def test_benchmark_reports_core_latency_scenarios() -> None:
    report = await run_benchmark(
        backend=MockChatCompletionBackend(),
        config=BenchmarkConfig(iterations=3, warmup=1),
    )

    assert report["profile"] == "openai-compatible"
    assert report["iterations"] == 3
    scenario_names = {scenario["name"] for scenario in report["scenarios"]}
    assert scenario_names == {
        "chat.non_streaming.roundtrip",
        "chat.streaming.event_latency",
        "chat.streaming.cancellation_latency",
    }
    for scenario in report["scenarios"]:
        assert scenario["p50_us"] >= 0
        assert scenario["p95_us"] >= 0
        assert scenario["event_count"] > 0


def test_cli_writes_benchmark_report_file(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"

    exit_code = main(
        [
            "run-benchmark",
            "--output",
            str(output),
            "--backend",
            "mock",
            "--iterations",
            "2",
            "--warmup",
            "0",
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["adapter"] == "vllm-nnrp-adapter"
    assert report["scenarios"][0]["name"] == "chat.non_streaming.roundtrip"


@pytest.mark.asyncio
async def test_comparison_benchmark_loads_async_backend_factory(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"

    report = await run_comparison_benchmark_file_with_backend_spec(
        output,
        backend_spec=f"{__name__}:make_async_mock_backend",
        config=BenchmarkConfig(iterations=1, warmup=0, prompt_tokens=(4,), concurrency=(1,)),
    )

    assert output.exists()
    assert report["scenarios"][0]["success_count"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["benchmark_kind"] == "in_process_comparison"


@pytest.mark.asyncio
async def test_comparison_benchmark_reports_long_context_matrix() -> None:
    report = await run_in_process_comparison_benchmark(
        backend=MockChatCompletionBackend(),
        config=BenchmarkConfig(iterations=2, warmup=0, model="mock-model", prompt_tokens=(4, 8), concurrency=(1, 2)),
    )

    assert report["benchmark_kind"] == "in_process_comparison"
    assert report["paths"] == ["nnrp.direct_profile_events"]
    assert len(report["scenarios"]) == 4
    for scenario in report["scenarios"]:
        assert scenario["path"] == "nnrp.direct_profile_events"
        assert scenario["success_count"] == 2
        assert scenario["error_count"] == 0
        assert scenario["ttft_p50_us"] is not None
        assert scenario["rtt_p95_us"] is not None
        assert scenario["output_tokens_per_sec"] >= 0


@pytest.mark.asyncio
async def test_comparison_benchmark_can_include_http_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_http_sample(config: BenchmarkConfig, prompt: str, cancel_after_first_event: bool) -> dict[str, object]:
        assert config.http_url == "http://127.0.0.1:8000/v1/chat/completions"
        assert prompt
        assert isinstance(cancel_after_first_event, bool)
        return {
            "ok": True,
            "ttft_us": 10.0,
            "tpot_us": 2.0,
            "rtt_us": 20.0,
            "output_tokens": 5,
        }

    monkeypatch.setattr("vllm_nnrp_adapter.benchmark._run_http_sse_request_sync", fake_http_sample)

    report = await run_in_process_comparison_benchmark(
        backend=MockChatCompletionBackend(),
        config=BenchmarkConfig(
            iterations=1,
            warmup=0,
            model="mock-model",
            prompt_tokens=(4,),
            concurrency=(1,),
            http_url="http://127.0.0.1:8000/v1/chat/completions",
        ),
    )

    assert report["paths"] == ["nnrp.direct_profile_events", "openai.http_sse"]
    assert {scenario["path"] for scenario in report["scenarios"]} == {"nnrp.direct_profile_events", "openai.http_sse"}


def test_cli_starts_embedded_tcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, EmbeddedTcpServerConfig]] = []

    def fake_run_embedded_tcp_server_sync(
        serving_factory_spec: str,
        *,
        config: EmbeddedTcpServerConfig,
    ) -> int:
        calls.append((serving_factory_spec, config))
        return 0

    monkeypatch.setattr("vllm_nnrp_adapter.cli.run_embedded_tcp_server_sync", fake_run_embedded_tcp_server_sync)

    assert main(
        [
            "serve-tcp",
            "--serving-factory",
            "pkg.module:factory",
            "--host",
            "0.0.0.0",
            "--port",
            "8899",
            "--active-model-name",
            "llama",
            "--session-id",
            "77",
            "--accept-timeout",
            "3.5",
            "--receive-timeout",
            "2.5",
            "--max-sessions",
            "4",
            "--max-requests-per-session",
            "5",
            "--idle-timeout",
            "6.5",
            "--no-no-delay",
        ]
    ) == 0

    assert calls == [
        (
            "pkg.module:factory",
            EmbeddedTcpServerConfig(
                host="0.0.0.0",
                port=8899,
                active_model_name="llama",
                session_id=77,
                accept_timeout=3.5,
                receive_timeout=2.5,
                max_sessions=4,
                max_requests_per_session=5,
                idle_timeout=6.5,
                no_delay=False,
            ),
        )
    ]


def test_cli_writes_comparison_benchmark_report_file(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"

    exit_code = main(
        [
            "run-benchmark",
            "--comparison",
            "--output",
            str(output),
            "--backend",
            "mock",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--prompt-tokens",
            "4,8",
            "--concurrency",
            "1",
            "--max-completion-tokens",
            "2",
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["benchmark_kind"] == "in_process_comparison"
    assert [scenario["prompt_tokens"] for scenario in report["scenarios"]] == [4, 8]


@pytest.mark.asyncio
async def test_comparison_benchmark_file_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"

    report = await run_comparison_benchmark_file(
        output,
        backend=MockChatCompletionBackend(),
        config=BenchmarkConfig(iterations=1, warmup=0, prompt_tokens=(4,), concurrency=(1,)),
    )

    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.asyncio
async def test_measure_path_matrix_reports_errors_and_cancellation() -> None:
    calls = 0

    async def request_runner() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _failed_sample(1, "boom")
        return {"ok": True, "ttft_us": 1.0, "tpot_us": None, "rtt_us": 2.0, "output_tokens": 1}

    async def cancellation_runner() -> dict[str, Any]:
        return {"ok": True, "ttft_us": 1.0, "tpot_us": None, "rtt_us": 3.0, "output_tokens": 1}

    scenario = await _measure_path_matrix(
        "test.path",
        request_runner,
        cancellation_runner=cancellation_runner,
        config=BenchmarkConfig(iterations=2, warmup=0),
        prompt_tokens=4,
        concurrency=2,
    )

    assert scenario["success_count"] == 1
    assert scenario["error_count"] == 1
    assert scenario["error_rate"] == 0.5
    assert scenario["cancellation_p50_us"] == 3.0
    assert scenario["errors"] == ["boom"]


def test_http_sse_sync_parses_stream_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def __iter__(self) -> object:
            return iter(
                [
                    b"\n",
                    b'data: {"choices":[{"delta":{"content":"hello world"}}]}\n',
                    b"data: [DONE]\n",
                ]
            )

    captured: dict[str, Any] = {}

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    config = BenchmarkConfig(
        iterations=1,
        warmup=0,
        http_url="http://127.0.0.1:8000/v1/chat/completions",
        http_api_key="secret",
    )

    sample = _run_http_sse_request_sync(config, "hello", False)

    assert sample["ok"] is True
    assert sample["output_tokens"] == 2
    assert captured["timeout"] == 300
    assert _http_headers(config)["Authorization"] == "Bearer secret"


def test_http_sse_sync_reports_missing_url_and_json_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_http_sse_request_sync(BenchmarkConfig(http_url=None), "hello", False)["ok"] is False

    class BadJsonResponse:
        def __enter__(self) -> BadJsonResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def __iter__(self) -> object:
            return iter([b"data: not-json\n"])

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: BadJsonResponse())

    sample = _run_http_sse_request_sync(BenchmarkConfig(http_url="http://example.test"), "hello", False)

    assert sample["ok"] is False
    assert "JSONDecodeError" in sample["error"]


def test_benchmark_helpers_build_expected_shapes() -> None:
    config = BenchmarkConfig(model="model", max_completion_tokens=7)
    request = _long_context_chat_request(config, "hello", cancel_after_first_event=True)

    assert request["nnrp"] == {"cancel_after_events": 1}
    assert request["body"]["max_tokens"] == 7
    assert _openai_chunk_text_deltas({"choices": [{"delta": {"content": "x"}}, {"delta": {}}]}) == ["x"]
    assert _openai_chunk_text_deltas({"choices": "bad"}) == []
    assert synthetic_prompt(3).split() == ["a", "a", "a"]
    assert estimate_token_count("") == 0
    assert estimate_token_count("one two") == 2
    assert _successful_sample(1, None, None, 0)["ok"] is True


@pytest.mark.asyncio
async def test_benchmark_rejects_invalid_iteration_count() -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        await run_benchmark(backend=MockChatCompletionBackend(), config=BenchmarkConfig(iterations=0))


@pytest.mark.asyncio
async def test_benchmark_rejects_invalid_warmup_count() -> None:
    with pytest.raises(ValueError, match="warmup must be non-negative"):
        await run_benchmark(backend=MockChatCompletionBackend(), config=BenchmarkConfig(warmup=-1))


@pytest.mark.asyncio
async def test_comparison_benchmark_rejects_invalid_matrix() -> None:
    with pytest.raises(ValueError, match="prompt_tokens values"):
        await run_in_process_comparison_benchmark(
            backend=MockChatCompletionBackend(),
            config=BenchmarkConfig(iterations=1, warmup=0, prompt_tokens=(0,)),
        )
    with pytest.raises(ValueError, match="warmup must be non-negative"):
        await run_in_process_comparison_benchmark(
            backend=MockChatCompletionBackend(),
            config=BenchmarkConfig(iterations=1, warmup=-1),
        )
    with pytest.raises(ValueError, match="max_completion_tokens"):
        await run_in_process_comparison_benchmark(
            backend=MockChatCompletionBackend(),
            config=BenchmarkConfig(iterations=1, warmup=0, max_completion_tokens=0),
        )
    with pytest.raises(ValueError, match="concurrency values"):
        await run_in_process_comparison_benchmark(
            backend=MockChatCompletionBackend(),
            config=BenchmarkConfig(iterations=1, warmup=0, concurrency=(0,)),
        )
