from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapter import ChatCompletionBackend, OpenAiNnrpAdapter
from .conformance import load_backend
from .profile import CHAT_COMPLETIONS_CREATE, OPENAI_COMPATIBLE_SCHEMA_VERSION


@dataclass(frozen=True)
class BenchmarkConfig:
    iterations: int = 200
    warmup: int = 20
    model: str = "mock-model"


def run_benchmark_sync(output_path: Path, backend_spec: str, config: BenchmarkConfig) -> dict[str, Any]:
    return asyncio.run(run_benchmark_file(output_path, backend=load_backend(backend_spec), config=config))


async def run_benchmark_file(
    output_path: Path,
    *,
    backend: ChatCompletionBackend,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    report = await run_benchmark(backend=backend, config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return report


async def run_benchmark(*, backend: ChatCompletionBackend, config: BenchmarkConfig) -> dict[str, Any]:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.warmup < 0:
        raise ValueError("warmup must be non-negative")

    adapter = OpenAiNnrpAdapter(backend)
    scenarios = [
        await _measure_non_streaming_roundtrip(adapter, config),
        await _measure_streaming_event_latency(adapter, config),
        await _measure_cancellation_latency(adapter, config),
    ]

    return {
        "profile": "openai-compatible",
        "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "adapter": "vllm-nnrp-adapter",
        "iterations": config.iterations,
        "warmup": config.warmup,
        "scenarios": scenarios,
    }


async def _measure_non_streaming_roundtrip(
    adapter: OpenAiNnrpAdapter,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    request = _chat_request(config.model, stream=False)
    samples = await _measure_request_samples(adapter, request, config)
    return _latency_scenario("chat.non_streaming.roundtrip", samples, event_count=len(samples))


async def _measure_streaming_event_latency(
    adapter: OpenAiNnrpAdapter,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    request = _chat_request(config.model, stream=True)
    await _warmup(adapter, request, config.warmup)

    samples: list[float] = []
    total_events = 0
    started = time.perf_counter_ns()
    for _ in range(config.iterations):
        previous = time.perf_counter_ns()
        async for _event in adapter.handle_request(request):
            now = time.perf_counter_ns()
            samples.append(_elapsed_us(previous, now))
            previous = now
            total_events += 1
    elapsed_s = (time.perf_counter_ns() - started) / 1_000_000_000

    scenario = _latency_scenario("chat.streaming.event_latency", samples, event_count=total_events)
    scenario["throughput_events_per_sec"] = total_events / elapsed_s if elapsed_s > 0 else 0.0
    return scenario


async def _measure_cancellation_latency(
    adapter: OpenAiNnrpAdapter,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    request = _chat_request(config.model, stream=True, nnrp={"cancel_after_events": 1})
    samples = await _measure_request_samples(adapter, request, config)
    return _latency_scenario("chat.streaming.cancellation_latency", samples, event_count=len(samples))


async def _measure_request_samples(
    adapter: OpenAiNnrpAdapter,
    request: Mapping[str, Any],
    config: BenchmarkConfig,
) -> list[float]:
    await _warmup(adapter, request, config.warmup)
    samples: list[float] = []
    for _ in range(config.iterations):
        started = time.perf_counter_ns()
        async for _event in adapter.handle_request(request):
            pass
        samples.append(_elapsed_us(started, time.perf_counter_ns()))
    return samples


async def _warmup(adapter: OpenAiNnrpAdapter, request: Mapping[str, Any], count: int) -> None:
    for _ in range(count):
        async for _event in adapter.handle_request(request):
            pass


def _chat_request(
    model: str,
    *,
    stream: bool,
    nnrp: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "operation": CHAT_COMPLETIONS_CREATE,
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "stream": stream,
        },
    }
    if nnrp is not None:
        request["nnrp"] = dict(nnrp)
    return request


def _latency_scenario(name: str, samples: list[float], *, event_count: int) -> dict[str, Any]:
    sorted_samples = sorted(samples)
    return {
        "name": name,
        "event_count": event_count,
        "p50_us": _percentile(sorted_samples, 0.50),
        "p95_us": _percentile(sorted_samples, 0.95),
        "min_us": sorted_samples[0],
        "max_us": sorted_samples[-1],
    }


def _percentile(sorted_samples: list[float], percentile: float) -> float:
    if not sorted_samples:
        raise ValueError("samples must not be empty")
    index = min(len(sorted_samples) - 1, max(0, int(round((len(sorted_samples) - 1) * percentile))))
    return sorted_samples[index]


def _elapsed_us(started_ns: int, finished_ns: int) -> float:
    return (finished_ns - started_ns) / 1000
