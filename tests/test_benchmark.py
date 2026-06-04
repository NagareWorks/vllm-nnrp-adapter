import json
from pathlib import Path

import pytest

from vllm_nnrp_adapter import BenchmarkConfig
from vllm_nnrp_adapter.benchmark import run_benchmark
from vllm_nnrp_adapter.cli import main
from vllm_nnrp_adapter.conformance import MockChatCompletionBackend


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
async def test_benchmark_rejects_invalid_iteration_count() -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        await run_benchmark(backend=MockChatCompletionBackend(), config=BenchmarkConfig(iterations=0))


@pytest.mark.asyncio
async def test_benchmark_rejects_invalid_warmup_count() -> None:
    with pytest.raises(ValueError, match="warmup must be non-negative"):
        await run_benchmark(backend=MockChatCompletionBackend(), config=BenchmarkConfig(warmup=-1))
