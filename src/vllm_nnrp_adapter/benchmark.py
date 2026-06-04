from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapter import ChatCompletionBackend, OpenAiNnrpAdapter
from .conformance import load_backend_async
from .profile import CHAT_COMPLETIONS_CREATE, OPENAI_COMPATIBLE_SCHEMA_VERSION


@dataclass(frozen=True)
class BenchmarkConfig:
    iterations: int = 200
    warmup: int = 20
    model: str = "mock-model"
    prompt_tokens: tuple[int, ...] = (4096, 8192, 16384, 20480)
    concurrency: tuple[int, ...] = (1, 2, 4)
    max_completion_tokens: int = 128
    http_url: str | None = None
    http_api_key: str | None = None


def run_benchmark_sync(output_path: Path, backend_spec: str, config: BenchmarkConfig) -> dict[str, Any]:
    return asyncio.run(run_benchmark_file_with_backend_spec(output_path, backend_spec=backend_spec, config=config))


def run_comparison_benchmark_sync(output_path: Path, backend_spec: str, config: BenchmarkConfig) -> dict[str, Any]:
    return asyncio.run(
        run_comparison_benchmark_file_with_backend_spec(output_path, backend_spec=backend_spec, config=config)
    )


async def run_benchmark_file_with_backend_spec(
    output_path: Path,
    *,
    backend_spec: str,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    backend = await load_backend_async(backend_spec)
    return await run_benchmark_file(output_path, backend=backend, config=config)


async def run_comparison_benchmark_file_with_backend_spec(
    output_path: Path,
    *,
    backend_spec: str,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    backend = await load_backend_async(backend_spec)
    return await run_comparison_benchmark_file(output_path, backend=backend, config=config)


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


async def run_comparison_benchmark_file(
    output_path: Path,
    *,
    backend: ChatCompletionBackend,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    report = await run_in_process_comparison_benchmark(backend=backend, config=config)
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


async def run_in_process_comparison_benchmark(
    *,
    backend: ChatCompletionBackend,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if config.max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be positive")
    if not config.prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    if not config.concurrency:
        raise ValueError("concurrency must not be empty")
    if any(value <= 0 for value in config.prompt_tokens):
        raise ValueError("prompt_tokens values must be positive")
    if any(value <= 0 for value in config.concurrency):
        raise ValueError("concurrency values must be positive")

    adapter = OpenAiNnrpAdapter(backend)
    scenarios: list[dict[str, Any]] = []
    for prompt_tokens in config.prompt_tokens:
        for concurrency in config.concurrency:
            prompt = synthetic_prompt(prompt_tokens)

            async def nnrp_runner(prompt: str = prompt) -> dict[str, Any]:
                return await _run_nnrp_direct_request(adapter, config, prompt)

            async def nnrp_cancel_runner(prompt: str = prompt) -> dict[str, Any]:
                return await _run_nnrp_direct_request(
                    adapter,
                    config,
                    prompt,
                    cancel_after_first_event=True,
                )

            scenarios.append(
                await _measure_path_matrix(
                    "nnrp.direct_profile_events",
                    nnrp_runner,
                    cancellation_runner=nnrp_cancel_runner,
                    config=config,
                    prompt_tokens=prompt_tokens,
                    concurrency=concurrency,
                )
            )
            if config.http_url:

                async def http_runner(prompt: str = prompt) -> dict[str, Any]:
                    return await _run_http_sse_request(config, prompt)

                async def http_cancel_runner(prompt: str = prompt) -> dict[str, Any]:
                    return await _run_http_sse_request(
                        config,
                        prompt,
                        cancel_after_first_event=True,
                    )

                scenarios.append(
                    await _measure_path_matrix(
                        "openai.http_sse",
                        http_runner,
                        cancellation_runner=http_cancel_runner,
                        config=config,
                        prompt_tokens=prompt_tokens,
                        concurrency=concurrency,
                    )
                )

    return {
        "profile": "openai-compatible",
        "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "adapter": "vllm-nnrp-adapter",
        "benchmark_kind": "in_process_comparison",
        "iterations": config.iterations,
        "warmup": config.warmup,
        "model": config.model,
        "max_completion_tokens": config.max_completion_tokens,
        "paths": (
            ["nnrp.direct_profile_events", "openai.http_sse"]
            if config.http_url
            else ["nnrp.direct_profile_events"]
        ),
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


async def _measure_path_matrix(
    path_name: str,
    request_runner: Callable[[], Awaitable[dict[str, Any]]],
    *,
    cancellation_runner: Callable[[], Awaitable[dict[str, Any]]],
    config: BenchmarkConfig,
    prompt_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    for _ in range(config.warmup):
        await request_runner()

    results: list[dict[str, Any]] = []
    started = time.perf_counter_ns()
    remaining = config.iterations
    while remaining > 0:
        batch_size = min(concurrency, remaining)
        results.extend(await asyncio.gather(*(request_runner() for _ in range(batch_size))))
        remaining -= batch_size
    elapsed_s = (time.perf_counter_ns() - started) / 1_000_000_000
    cancellation_results = await _measure_cancellation_matrix(
        cancellation_runner,
        config=config,
        concurrency=concurrency,
    )

    successes = [result for result in results if result["ok"]]
    errors = [result for result in results if not result["ok"]]
    cancellation_successes = [result for result in cancellation_results if result["ok"]]
    cancellation_samples = [float(result["rtt_us"]) for result in cancellation_successes]
    ttft_samples = [float(result["ttft_us"]) for result in successes if result.get("ttft_us") is not None]
    tpot_samples = [float(result["tpot_us"]) for result in successes if result.get("tpot_us") is not None]
    rtt_samples = [float(result["rtt_us"]) for result in successes]
    output_tokens = sum(int(result["output_tokens"]) for result in successes)
    return {
        "name": f"{path_name}.chat_stream.long_context",
        "path": path_name,
        "prompt_tokens": prompt_tokens,
        "concurrency": concurrency,
        "iterations": config.iterations,
        "success_count": len(successes),
        "error_count": len(errors),
        "error_rate": len(errors) / len(results) if results else 0.0,
        "ttft_p50_us": _percentile(sorted(ttft_samples), 0.50) if ttft_samples else None,
        "ttft_p95_us": _percentile(sorted(ttft_samples), 0.95) if ttft_samples else None,
        "tpot_p50_us": _percentile(sorted(tpot_samples), 0.50) if tpot_samples else None,
        "tpot_p95_us": _percentile(sorted(tpot_samples), 0.95) if tpot_samples else None,
        "rtt_p50_us": _percentile(sorted(rtt_samples), 0.50) if rtt_samples else None,
        "rtt_p95_us": _percentile(sorted(rtt_samples), 0.95) if rtt_samples else None,
        "cancellation_p50_us": _percentile(sorted(cancellation_samples), 0.50) if cancellation_samples else None,
        "cancellation_p95_us": _percentile(sorted(cancellation_samples), 0.95) if cancellation_samples else None,
        "output_tokens": output_tokens,
        "output_tokens_per_sec": output_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "requests_per_sec": len(successes) / elapsed_s if elapsed_s > 0 else 0.0,
        "elapsed_s": elapsed_s,
        "errors": [result["error"] for result in errors[:5]],
    }


async def _measure_cancellation_matrix(
    cancellation_runner: Callable[[], Awaitable[dict[str, Any]]],
    *,
    config: BenchmarkConfig,
    concurrency: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    remaining = config.iterations
    while remaining > 0:
        batch_size = min(concurrency, remaining)
        results.extend(await asyncio.gather(*(cancellation_runner() for _ in range(batch_size))))
        remaining -= batch_size
    return results


async def _run_nnrp_direct_request(
    adapter: OpenAiNnrpAdapter,
    config: BenchmarkConfig,
    prompt: str,
    *,
    cancel_after_first_event: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    output_tokens = 0
    try:
        async for event in adapter.handle_request(
            _long_context_chat_request(config, prompt, cancel_after_first_event=cancel_after_first_event)
        ):
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                now = time.perf_counter_ns()
                if first_token_ns is None:
                    first_token_ns = now
                last_token_ns = now
                output_tokens += max(1, estimate_token_count(str(event.get("delta", ""))))
            elif event_type == "response.error":
                return _failed_sample(started, event.get("error"))
            elif event_type == "response.cancelled":
                return _successful_sample(started, first_token_ns, last_token_ns, output_tokens)
        return _successful_sample(started, first_token_ns, last_token_ns, output_tokens)
    except Exception as error:
        return _failed_sample(started, f"{type(error).__name__}: {error}")


async def _run_http_sse_request(
    config: BenchmarkConfig,
    prompt: str,
    *,
    cancel_after_first_event: bool = False,
) -> dict[str, Any]:
    return await asyncio.to_thread(_run_http_sse_request_sync, config, prompt, cancel_after_first_event)


def _run_http_sse_request_sync(config: BenchmarkConfig, prompt: str, cancel_after_first_event: bool) -> dict[str, Any]:
    started = time.perf_counter_ns()
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    output_tokens = 0
    if config.http_url is None:
        return _failed_sample(started, "http_url is not configured")

    body = json.dumps(_openai_chat_body(config, prompt), separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        config.http_url,
        data=body,
        headers=_http_headers(config),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                for delta in _openai_chunk_text_deltas(chunk):
                    now = time.perf_counter_ns()
                    if first_token_ns is None:
                        first_token_ns = now
                    last_token_ns = now
                    output_tokens += max(1, estimate_token_count(delta))
                    if cancel_after_first_event:
                        return _successful_sample(started, first_token_ns, last_token_ns, output_tokens)
        return _successful_sample(started, first_token_ns, last_token_ns, output_tokens)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        return _failed_sample(started, f"{type(error).__name__}: {error}")


def _long_context_chat_request(
    config: BenchmarkConfig,
    prompt: str,
    *,
    cancel_after_first_event: bool = False,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "operation": CHAT_COMPLETIONS_CREATE,
        "body": _openai_chat_body(config, prompt),
    }
    if cancel_after_first_event:
        request["nnrp"] = {"cancel_after_events": 1}
    return request


def _openai_chat_body(config: BenchmarkConfig, prompt: str) -> dict[str, Any]:
    return {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": config.max_completion_tokens,
        "stream_options": {"include_usage": True},
    }


def _http_headers(config: BenchmarkConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.http_api_key:
        headers["Authorization"] = f"Bearer {config.http_api_key}"
    return headers


def _openai_chunk_text_deltas(chunk: Mapping[str, Any]) -> list[str]:
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return []
    deltas: list[str] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            deltas.append(content)
    return deltas


def synthetic_prompt(target_tokens: int) -> str:
    # Use a single-token word for common OpenAI-compatible LLM tokenizers so the
    # matrix labels stay close to the requested context length.
    unit = "a "
    return (unit * target_tokens).strip()


def estimate_token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped.split()))


def _successful_sample(
    started_ns: int,
    first_token_ns: int | None,
    last_token_ns: int | None,
    output_tokens: int,
) -> dict[str, Any]:
    finished_ns = time.perf_counter_ns()
    if first_token_ns is None:
        first_token_ns = finished_ns
    if last_token_ns is None:
        last_token_ns = first_token_ns
    tpot_us = None
    if output_tokens > 1:
        tpot_us = _elapsed_us(first_token_ns, last_token_ns) / (output_tokens - 1)
    return {
        "ok": True,
        "ttft_us": _elapsed_us(started_ns, first_token_ns),
        "tpot_us": tpot_us,
        "rtt_us": _elapsed_us(started_ns, finished_ns),
        "output_tokens": output_tokens,
    }


def _failed_sample(started_ns: int, error: object) -> dict[str, Any]:
    return {
        "ok": False,
        "rtt_us": _elapsed_us(started_ns, time.perf_counter_ns()),
        "output_tokens": 0,
        "error": str(error),
    }


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
