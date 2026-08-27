from __future__ import annotations

import asyncio
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nnrp.core import MessageType  # type: ignore[import-untyped]
from nnrp.runtime import (  # type: ignore[import-untyped]
    ControlRequestMetadata,
    NativeRuntimeEvent,
    PressureMetadata,
    RuntimeEventMetadata,
    RuntimeEventMetadataKind,
    RuntimeEventTail,
    RuntimeFrameHeader,
    RuntimeRole,
    SchedulingMetadata,
)

from .adapter import ChatCompletionBackend, OpenAiNnrpAdapter, map_openai_stream_chunk
from .conformance import load_backend_async
from .pressure import OutboundCreditController
from .profile import (
    CHAT_COMPLETIONS_CREATE,
    OPENAI_COMPATIBLE_SCHEMA_VERSION,
    OpenAiNnrpCapabilityDocument,
    validate_request,
)
from .runtime_control import (
    RuntimeControlDisposition,
    RuntimeControlRegistry,
    decode_deadline_update,
    decode_operation_control,
    decode_priority_update,
)


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


def run_comparison_benchmark_sync(
    output_path: Path,
    backend_spec: str,
    config: BenchmarkConfig,
    markdown_output_path: Path | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        run_comparison_benchmark_file_with_backend_spec(
            output_path,
            backend_spec=backend_spec,
            config=config,
            markdown_output_path=markdown_output_path,
        )
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
    markdown_output_path: Path | None = None,
) -> dict[str, Any]:
    backend = await load_backend_async(backend_spec)
    return await run_comparison_benchmark_file(
        output_path,
        backend=backend,
        config=config,
        markdown_output_path=markdown_output_path,
    )


async def run_benchmark_file(
    output_path: Path,
    *,
    backend: ChatCompletionBackend,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    report = await run_benchmark(backend=backend, config=config)
    validate_benchmark_evidence(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return report


async def run_comparison_benchmark_file(
    output_path: Path,
    *,
    backend: ChatCompletionBackend,
    config: BenchmarkConfig,
    markdown_output_path: Path | None = None,
) -> dict[str, Any]:
    report = await run_in_process_comparison_benchmark(backend=backend, config=config)
    validate_benchmark_evidence(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    if markdown_output_path is not None:
        markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_output_path.write_text(
            render_comparison_markdown(report, raw_report_name=output_path.name),
            encoding="utf-8",
        )
    return report


def render_comparison_markdown(report: Mapping[str, Any], *, raw_report_name: str) -> str:
    validate_benchmark_evidence(report)
    _validate_benchmark_evidence_text(raw_report_name, location="raw_report_name")
    if report.get("benchmark_kind") != "in_process_comparison":
        raise ValueError("comparison markdown requires an in_process_comparison report")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("comparison report must contain scenarios")

    lines = [
        "# OpenAI NNRP And HTTP/SSE Comparison",
        "",
        f"Raw evidence: `{raw_report_name}`",
        "",
        "| Prompt tokens | Concurrency | Path | Success / error | TTFT p50 | TPOT p50 | RTT p50 | "
        "Cancel p50 | Requests/s | Output tokens/s |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in sorted(scenarios, key=_comparison_scenario_sort_key):
        lines.append(
            "| {prompt} | {concurrency} | `{path}` | {success} / {error} | {ttft} | {tpot} | "
            "{rtt} | {cancel} | {requests} | {tokens} |".format(
                prompt=_required_int(scenario, "prompt_tokens"),
                concurrency=_required_int(scenario, "concurrency"),
                path=_required_str(scenario, "path"),
                success=_required_int(scenario, "success_count"),
                error=_required_int(scenario, "error_count"),
                ttft=_format_milliseconds(scenario.get("ttft_p50_us")),
                tpot=_format_milliseconds(scenario.get("tpot_p50_us")),
                rtt=_format_milliseconds(scenario.get("rtt_p50_us")),
                cancel=_format_milliseconds(scenario.get("cancellation_p50_us")),
                requests=_format_rate(scenario.get("requests_per_sec")),
                tokens=_format_rate(scenario.get("output_tokens_per_sec")),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Parity in model-dominated chat workloads validates NNRP compatibility with the selected vLLM binding. "
            "It is not the adapter's primary performance or adoption claim. Operational-control and heavy-payload "
            "experiments must be evaluated separately with their own raw evidence.",
            "",
        ]
    )
    return "\n".join(lines)


_SENSITIVE_EVIDENCE_PATTERNS = (
    ("endpoint URL", re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)),
    ("IPv4 address", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
    ("Windows user path", re.compile(r"\b[a-z]:\\Users\\[^\\\s]+", re.IGNORECASE)),
    ("Unix user path", re.compile(r"/(?:home|Users)/[^/\s]+")),
    ("Bearer token", re.compile(r"\bBearer\s+\S+", re.IGNORECASE)),
    ("API token", re.compile(r"\b(?:sk|ghp|npm)_[A-Za-z0-9_-]{8,}")),
    (
        "machine or request UUID",
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE),
    ),
)


def validate_benchmark_evidence(value: object) -> None:
    _validate_benchmark_evidence_value(value, location="report")


def _validate_benchmark_evidence_value(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_benchmark_evidence_value(item, location=f"{location}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_benchmark_evidence_value(item, location=f"{location}[{index}]")
        return
    if isinstance(value, str):
        _validate_benchmark_evidence_text(value, location=location)


def _validate_benchmark_evidence_text(value: str, *, location: str) -> None:
    for label, pattern in _SENSITIVE_EVIDENCE_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"benchmark evidence contains {label} at {location}")


async def run_benchmark(*, backend: ChatCompletionBackend, config: BenchmarkConfig) -> dict[str, Any]:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.warmup < 0:
        raise ValueError("warmup must be non-negative")

    adapter = OpenAiNnrpAdapter(backend)
    runtime_control_scenarios = await _measure_runtime_control_processing(config)
    runtime_pressure_scenarios = await _measure_runtime_pressure_processing(config)
    scenarios = [
        _measure_profile_validation(config),
        _measure_profile_event_mapping(config),
        *runtime_control_scenarios,
        *runtime_pressure_scenarios,
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
        "integration": _integration_metadata(backend, config.model),
        "scenarios": scenarios,
    }


async def _measure_runtime_pressure_processing(config: BenchmarkConfig) -> list[dict[str, Any]]:
    backpressure = PressureMetadata(
        scope_id=0,
        credit_window=1,
        pressure_level=1,
        pressure_reason=0,
        retry_after_ms=0,
        flags=0,
    )
    credit = PressureMetadata(
        scope_id=0,
        credit_window=1,
        pressure_level=0,
        pressure_reason=0,
        retry_after_ms=0,
        flags=0,
    )

    controller = OutboundCreditController()
    for _ in range(config.warmup):
        await controller.apply(MessageType.BACKPRESSURE, backpressure)
    apply_samples: list[float] = []
    for _ in range(config.iterations):
        started = time.perf_counter_ns()
        await controller.apply(MessageType.BACKPRESSURE, backpressure)
        apply_samples.append(_elapsed_us(started, time.perf_counter_ns()))

    reservation_samples: list[float] = []
    for _ in range(config.warmup):
        await controller.apply(MessageType.CREDIT_UPDATE, credit)
        await controller.reserve(1)
    for _ in range(config.iterations):
        await controller.apply(MessageType.CREDIT_UPDATE, credit)
        started = time.perf_counter_ns()
        await controller.reserve(1)
        reservation_samples.append(_elapsed_us(started, time.perf_counter_ns()))
    await controller.close()

    for _ in range(config.warmup):
        await _measure_credit_recovery_once()
    recovery_samples = [await _measure_credit_recovery_once() for _ in range(config.iterations)]

    return [
        _latency_scenario(
            "runtime_pressure.backpressure_apply_after_native_delivery",
            apply_samples,
            event_count=config.iterations,
        ),
        _latency_scenario(
            "runtime_pressure.credit_reservation",
            reservation_samples,
            event_count=config.iterations,
        ),
        _latency_scenario(
            "runtime_pressure.credit_recovery_after_native_delivery",
            recovery_samples,
            event_count=config.iterations,
        ),
    ]


async def _measure_credit_recovery_once() -> float:
    controller = OutboundCreditController()
    paused = PressureMetadata(
        scope_id=0,
        credit_window=0,
        pressure_level=0,
        pressure_reason=0,
        retry_after_ms=0,
        flags=0,
    )
    resumed = PressureMetadata(
        scope_id=0,
        credit_window=1,
        pressure_level=0,
        pressure_reason=0,
        retry_after_ms=0,
        flags=0,
    )
    await controller.apply(MessageType.CREDIT_UPDATE, paused)
    waiter = asyncio.create_task(controller.reserve(1))
    await asyncio.sleep(0)
    if waiter.done():
        raise RuntimeError("zero-credit benchmark reservation did not block")

    started = time.perf_counter_ns()
    await controller.apply(MessageType.CREDIT_UPDATE, resumed)
    await waiter
    elapsed_us = _elapsed_us(started, time.perf_counter_ns())
    await controller.close()
    return elapsed_us


async def _measure_runtime_control_processing(config: BenchmarkConfig) -> list[dict[str, Any]]:
    priority_events = [
        _priority_control_event(sequence=index + 1) for index in range(config.warmup + config.iterations)
    ]
    deadline_events = [
        _deadline_control_event(sequence=index + 1) for index in range(config.warmup + config.iterations)
    ]

    priority_registry = RuntimeControlRegistry()
    priority_registry.register(1)
    for event in priority_events[: config.warmup]:
        _apply_priority_control_event(priority_registry, event)
    priority_samples: list[float] = []
    for event in priority_events[config.warmup :]:
        started = time.perf_counter_ns()
        _apply_priority_control_event(priority_registry, event)
        priority_samples.append(_elapsed_us(started, time.perf_counter_ns()))
    await priority_registry.clear()

    deadline_registry = RuntimeControlRegistry()
    deadline_registry.register(1)
    for event in deadline_events[: config.warmup]:
        await _apply_deadline_control_event(deadline_registry, event)
    deadline_samples: list[float] = []
    for event in deadline_events[config.warmup :]:
        started = time.perf_counter_ns()
        await _apply_deadline_control_event(deadline_registry, event)
        deadline_samples.append(_elapsed_us(started, time.perf_counter_ns()))
    await deadline_registry.clear()

    for _ in range(config.warmup):
        await _measure_cancel_control_once()
    cancel_samples: list[float] = []
    cleanup_samples: list[float] = []
    for _ in range(config.iterations):
        cancel_us, cleanup_us = await _measure_cancel_control_once()
        cancel_samples.append(cancel_us)
        cleanup_samples.append(cleanup_us)

    return [
        _latency_scenario(
            "runtime_control.priority_after_native_delivery",
            priority_samples,
            event_count=config.iterations,
        ),
        _latency_scenario(
            "runtime_control.deadline_after_native_delivery",
            deadline_samples,
            event_count=config.iterations,
        ),
        _latency_scenario(
            "runtime_control.cancel_dispatch_after_native_delivery",
            cancel_samples,
            event_count=config.iterations,
        ),
        _latency_scenario(
            "runtime_control.cancelled_registry_cleanup",
            cleanup_samples,
            event_count=config.iterations,
        ),
    ]


def _apply_priority_control_event(registry: RuntimeControlRegistry, event: NativeRuntimeEvent) -> None:
    update = decode_priority_update(event)
    if update is None:
        raise RuntimeError("benchmark priority event did not decode")
    disposition = registry.apply_priority(update, terminal=False)
    if disposition is not RuntimeControlDisposition.APPLIED:
        raise RuntimeError(f"benchmark priority update was not applied: {disposition}")


async def _apply_deadline_control_event(registry: RuntimeControlRegistry, event: NativeRuntimeEvent) -> None:
    update = decode_deadline_update(event)
    if update is None:
        raise RuntimeError("benchmark deadline event did not decode")
    disposition = await registry.apply_deadline(update, terminal=False)
    if disposition is not RuntimeControlDisposition.APPLIED:
        raise RuntimeError(f"benchmark deadline update was not applied: {disposition}")


async def _measure_cancel_control_once() -> tuple[float, float]:
    registry = RuntimeControlRegistry()
    slot = registry.register(1)
    task = asyncio.create_task(_wait_for_cancel())
    slot.bind(task)
    await asyncio.sleep(0)

    event = _cancel_control_event()
    started = time.perf_counter_ns()
    request = decode_operation_control(event)
    if request is None:
        raise RuntimeError("benchmark cancel event did not decode")
    disposition = await registry.apply(request, terminal=False)
    cancel_us = _elapsed_us(started, time.perf_counter_ns())
    if disposition is not RuntimeControlDisposition.APPLIED:
        raise RuntimeError(f"benchmark cancel was not applied: {disposition}")

    cleanup_started = time.perf_counter_ns()
    await asyncio.gather(task, return_exceptions=True)
    await registry.clear()
    cleanup_us = _elapsed_us(cleanup_started, time.perf_counter_ns())
    return cancel_us, cleanup_us


async def _wait_for_cancel() -> None:
    await asyncio.Event().wait()


def _priority_control_event(*, sequence: int) -> NativeRuntimeEvent:
    metadata = SchedulingMetadata(
        operation_id=1,
        control_sequence=sequence,
        priority_class=2,
        priority_delta=-1,
        deadline_unix_ms=0,
        flags=0,
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.PRIORITY_UPDATE),
        RuntimeEventMetadata(RuntimeEventMetadataKind.SCHEDULING, metadata),
        RuntimeEventTail.none(),
    )


def _deadline_control_event(*, sequence: int) -> NativeRuntimeEvent:
    metadata = SchedulingMetadata(
        operation_id=1,
        control_sequence=sequence,
        priority_class=0,
        priority_delta=0,
        deadline_unix_ms=int(time.time() * 1_000) + 60_000,
        flags=0,
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.DEADLINE),
        RuntimeEventMetadata(RuntimeEventMetadataKind.SCHEDULING, metadata),
        RuntimeEventTail.none(),
    )


def _cancel_control_event() -> NativeRuntimeEvent:
    metadata = ControlRequestMetadata(
        operation_id=1,
        control_sequence=1,
        reason_code=0,
        source_role=RuntimeRole.CLIENT,
        flags=0,
        diagnostic_bytes=0,
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.CANCEL),
        RuntimeEventMetadata(RuntimeEventMetadataKind.CONTROL_REQUEST, metadata),
        RuntimeEventTail.none(),
    )


def _measure_profile_validation(config: BenchmarkConfig) -> dict[str, Any]:
    request = _chat_request(config.model, stream=True)
    capabilities = OpenAiNnrpCapabilityDocument.level1()
    for _ in range(config.warmup):
        validate_request(request, capabilities)

    samples: list[float] = []
    for _ in range(config.iterations):
        started = time.perf_counter_ns()
        validate_request(request, capabilities)
        samples.append(_elapsed_us(started, time.perf_counter_ns()))
    return _latency_scenario(
        "profile.validation",
        samples,
        event_count=config.iterations,
    )


def _measure_profile_event_mapping(config: BenchmarkConfig) -> dict[str, Any]:
    chunk = {
        "id": "chatcmpl-benchmark",
        "model": config.model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": "hello"},
                "finish_reason": None,
            }
        ],
    }
    for _ in range(config.warmup):
        map_openai_stream_chunk(chunk)

    samples: list[float] = []
    mapped_events = 0
    for _ in range(config.iterations):
        started = time.perf_counter_ns()
        mapped_events += len(map_openai_stream_chunk(chunk))
        samples.append(_elapsed_us(started, time.perf_counter_ns()))
    return _latency_scenario(
        "profile.event_mapping",
        samples,
        event_count=mapped_events,
    )


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

            paths: list[
                tuple[
                    str,
                    Callable[[], Awaitable[dict[str, Any]]],
                    Callable[[], Awaitable[dict[str, Any]]],
                ]
            ] = [("nnrp.direct_profile_events", nnrp_runner, nnrp_cancel_runner)]
            if config.http_url:

                async def http_runner(prompt: str = prompt) -> dict[str, Any]:
                    return await _run_http_sse_request(config, prompt)

                async def http_cancel_runner(prompt: str = prompt) -> dict[str, Any]:
                    return await _run_http_sse_request(
                        config,
                        prompt,
                        cancel_after_first_event=True,
                    )

                paths.append(("openai.http_sse", http_runner, http_cancel_runner))

            scenarios.extend(
                await _measure_interleaved_path_matrix(
                    paths,
                    config=config,
                    prompt_tokens=prompt_tokens,
                    concurrency=concurrency,
                )
            )

    path_names = (
        ["nnrp.direct_profile_events", "openai.http_sse"] if config.http_url else ["nnrp.direct_profile_events"]
    )
    schedule = "rotating-batch-interleave" if config.http_url else "single-path"
    return {
        "profile": "openai-compatible",
        "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "adapter": "vllm-nnrp-adapter",
        "benchmark_kind": "in_process_comparison",
        "iterations": config.iterations,
        "warmup": config.warmup,
        "model": config.model,
        "integration": _integration_metadata(backend, config.model),
        "max_completion_tokens": config.max_completion_tokens,
        "paths": path_names,
        "schedule": schedule,
        "input_manifest": {
            "model": config.model,
            "iterations": config.iterations,
            "warmup": config.warmup,
            "prompt_tokens": list(config.prompt_tokens),
            "concurrency": list(config.concurrency),
            "max_completion_tokens": config.max_completion_tokens,
            "paths": path_names,
            "schedule": schedule,
        },
        "scenarios": scenarios,
    }


def _integration_metadata(backend: ChatCompletionBackend, model: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "backend_family": type(backend).__name__,
        "vllm_version": "unknown",
        "compatibility_binding": "unknown",
        "model": model,
        "engine_configuration": {"status": "unknown"},
        "gpu": _detect_gpu_metadata(),
    }
    provider = getattr(backend, "benchmark_metadata", None)
    if callable(provider):
        supplied = provider()
        if isinstance(supplied, Mapping):
            for key in ("vllm_version", "compatibility_binding", "engine_configuration"):
                if key in supplied:
                    metadata[key] = supplied[key]
    return metadata


def _detect_gpu_metadata() -> Mapping[str, object]:
    try:
        import torch  # type: ignore[import-not-found]
    except (ImportError, OSError):
        return {"status": "unavailable"}

    try:
        if not torch.cuda.is_available():
            return {"status": "unavailable"}
        count = torch.cuda.device_count()
        return {
            "status": "available",
            "count": count,
            "devices": [torch.cuda.get_device_name(index) for index in range(count)],
        }
    except (AssertionError, RuntimeError):
        return {"status": "unavailable"}


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
    request = _chat_request(config.model, stream=True)
    samples = await _measure_request_samples(adapter, request, config, cancel_after_events=1)
    return _latency_scenario("chat.streaming.cancellation_latency", samples, event_count=len(samples))


async def _measure_request_samples(
    adapter: OpenAiNnrpAdapter,
    request: Mapping[str, Any],
    config: BenchmarkConfig,
    *,
    cancel_after_events: int | None = None,
) -> list[float]:
    await _warmup(adapter, request, config.warmup)
    samples: list[float] = []
    for _ in range(config.iterations):
        started = time.perf_counter_ns()
        stream = adapter.handle_request(request)
        try:
            emitted_events = 0
            async for _event in stream:
                emitted_events += 1
                if cancel_after_events is not None and emitted_events >= cancel_after_events:
                    await stream.aclose()
                    break
        finally:
            await stream.aclose()
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

    return _build_path_scenario(
        path_name,
        results,
        cancellation_results,
        prompt_tokens=prompt_tokens,
        concurrency=concurrency,
        elapsed_s=elapsed_s,
    )


async def _measure_interleaved_path_matrix(
    paths: list[
        tuple[
            str,
            Callable[[], Awaitable[dict[str, Any]]],
            Callable[[], Awaitable[dict[str, Any]]],
        ]
    ],
    *,
    config: BenchmarkConfig,
    prompt_tokens: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError("paths must not be empty")

    for warmup_index in range(config.warmup):
        for _name, request_runner, _cancellation_runner in _rotated(paths, warmup_index):
            await request_runner()

    request_results: dict[str, list[dict[str, Any]]] = {name: [] for name, _runner, _cancel in paths}
    cancellation_results: dict[str, list[dict[str, Any]]] = {name: [] for name, _runner, _cancel in paths}
    request_elapsed_ns = {name: 0 for name, _runner, _cancel in paths}
    remaining = config.iterations
    batch_index = 0
    while remaining > 0:
        batch_size = min(concurrency, remaining)
        for name, request_runner, _cancellation_runner in _rotated(paths, batch_index):
            started = time.perf_counter_ns()
            request_results[name].extend(await asyncio.gather(*(request_runner() for _ in range(batch_size))))
            request_elapsed_ns[name] += time.perf_counter_ns() - started
        remaining -= batch_size
        batch_index += 1

    remaining = config.iterations
    batch_index = 0
    while remaining > 0:
        batch_size = min(concurrency, remaining)
        for name, _request_runner, cancellation_runner in _rotated(paths, batch_index):
            cancellation_results[name].extend(await asyncio.gather(*(cancellation_runner() for _ in range(batch_size))))
        remaining -= batch_size
        batch_index += 1

    return [
        _build_path_scenario(
            name,
            request_results[name],
            cancellation_results[name],
            prompt_tokens=prompt_tokens,
            concurrency=concurrency,
            elapsed_s=request_elapsed_ns[name] / 1_000_000_000,
        )
        for name, _runner, _cancel in paths
    ]


def _rotated(
    paths: list[
        tuple[
            str,
            Callable[[], Awaitable[dict[str, Any]]],
            Callable[[], Awaitable[dict[str, Any]]],
        ]
    ],
    index: int,
) -> list[
    tuple[
        str,
        Callable[[], Awaitable[dict[str, Any]]],
        Callable[[], Awaitable[dict[str, Any]]],
    ]
]:
    offset = index % len(paths)
    return paths[offset:] + paths[:offset]


def _build_path_scenario(
    path_name: str,
    results: list[dict[str, Any]],
    cancellation_results: list[dict[str, Any]],
    *,
    prompt_tokens: int,
    concurrency: int,
    elapsed_s: float,
) -> dict[str, Any]:

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
        "iterations": len(results),
        "sample_count": len(results),
        "cancellation_sample_count": len(cancellation_results),
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
        "mean_90ci_us": {
            "ttft": _mean_confidence_interval(ttft_samples),
            "tpot": _mean_confidence_interval(tpot_samples),
            "rtt": _mean_confidence_interval(rtt_samples),
            "cancellation": _mean_confidence_interval(cancellation_samples),
        },
        "samples": results,
        "cancellation_samples": cancellation_results,
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
        stream = adapter.handle_request(_long_context_chat_request(config, prompt))
        try:
            async for event in stream:
                event_type = event.get("type")
                if event_type == "response.output_text.delta":
                    now = time.perf_counter_ns()
                    if first_token_ns is None:
                        first_token_ns = now
                    last_token_ns = now
                    output_tokens += max(1, estimate_token_count(str(event.get("delta", ""))))
                    if cancel_after_first_event:
                        await stream.aclose()
                        break
                elif event_type == "response.error":
                    return _failed_sample(started, event.get("error"))
        finally:
            await stream.aclose()
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
) -> dict[str, Any]:
    return {
        "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "operation": CHAT_COMPLETIONS_CREATE,
        "body": _openai_chat_body(config, prompt),
    }


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


def _mean_confidence_interval(samples: list[float]) -> dict[str, float | int] | None:
    if not samples:
        return None
    mean = statistics.fmean(samples)
    margin = 0.0 if len(samples) == 1 else 1.6448536269514722 * statistics.stdev(samples) / math.sqrt(len(samples))
    return {
        "sample_count": len(samples),
        "mean_us": mean,
        "lower_us": max(0.0, mean - margin),
        "upper_us": mean + margin,
    }


def _comparison_scenario_sort_key(scenario: object) -> tuple[int, int, str]:
    if not isinstance(scenario, Mapping):
        raise ValueError("comparison scenarios must be objects")
    return (
        _required_int(scenario, "prompt_tokens"),
        _required_int(scenario, "concurrency"),
        _required_str(scenario, "path"),
    )


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"comparison scenario {key} must be an integer")
    return item


def _required_str(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"comparison scenario {key} must be a non-empty string")
    return item


def _format_milliseconds(value: object) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "n/a"
    return f"{value / 1_000:.2f} ms"


def _format_rate(value: object) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "n/a"
    return f"{value:.2f}"


def _elapsed_us(started_ns: int, finished_ns: int) -> float:
    return (finished_ns - started_ns) / 1000
