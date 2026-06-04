import json
from pathlib import Path

import pytest

from vllm_nnrp_adapter import BenchmarkConfig
from vllm_nnrp_adapter.benchmark import run_benchmark
from vllm_nnrp_adapter.cli import main
from vllm_nnrp_adapter.conformance import MockChatCompletionBackend
from vllm_nnrp_adapter.embedded import EmbeddedTcpServerConfig


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


@pytest.mark.asyncio
async def test_benchmark_rejects_invalid_iteration_count() -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        await run_benchmark(backend=MockChatCompletionBackend(), config=BenchmarkConfig(iterations=0))


@pytest.mark.asyncio
async def test_benchmark_rejects_invalid_warmup_count() -> None:
    with pytest.raises(ValueError, match="warmup must be non-negative"):
        await run_benchmark(backend=MockChatCompletionBackend(), config=BenchmarkConfig(warmup=-1))
