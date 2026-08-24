from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from .adapter import ChatCompletionBackend, OpenAiNnrpAdapter
from .profile import OPENAI_COMPATIBLE_SCHEMA_VERSION, build_cancelled_event

Terminal = Literal["success", "error", "cancelled"]
Outcome = Literal["passed", "failed", "skipped"]
_BASELINE_IGNORED_EVENT_FIELDS = frozenset({"openai_chunk"})


class BackendFactory(Protocol):
    def __call__(self) -> ChatCompletionBackend:
        pass


async def run_conformance_plan_file(
    plan_path: Path,
    output_path: Path,
    *,
    backend: ChatCompletionBackend,
) -> dict[str, Any]:
    plan = _load_json_object(plan_path)
    report = await run_conformance_plan(plan, backend=backend)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    _write_case_evidence(plan, report)
    return report


async def run_conformance_plan(plan: Mapping[str, Any], *, backend: ChatCompletionBackend) -> dict[str, Any]:
    adapter = OpenAiNnrpAdapter(backend)
    schema_version = _as_str(plan.get("schema_version"), "schema_version")
    adapter_name = _as_str(plan.get("adapter"), "adapter")
    cases = _as_list(plan.get("cases"), "cases")

    results = []
    for case in cases:
        case_mapping = _as_mapping(case, "case")
        results.append(await _run_case(adapter, schema_version, case_mapping))

    return {
        "$schema": "https://github.com/NagareWorks/nnrp-conformance/schemas/api-profile-case-results.schema.json",
        "profile": _as_str(plan.get("profile"), "profile"),
        "schema_version": schema_version,
        "adapter": adapter_name,
        "results": results,
    }


def load_backend(spec: str) -> ChatCompletionBackend:
    if spec == "mock":
        return MockChatCompletionBackend()

    backend = _call_backend_factory(spec)
    if inspect.isawaitable(backend):
        raise TypeError("backend factory returned an awaitable; use load_backend_async from async callers")
    return cast(ChatCompletionBackend, backend)


async def load_backend_async(spec: str) -> ChatCompletionBackend:
    if spec == "mock":
        return MockChatCompletionBackend()

    backend = _call_backend_factory(spec)
    if inspect.isawaitable(backend):
        backend = await backend
    return cast(ChatCompletionBackend, backend)


def _call_backend_factory(spec: str) -> object:
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("backend spec must be 'mock' or 'module.path:factory_name'")

    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    if not callable(factory):
        raise TypeError(f"backend factory is not callable: {spec}")
    return factory()


class MockChatCompletionBackend:
    supports_tool_calls = True

    def create_chat_completion(self, body: Mapping[str, Any]) -> Mapping[str, Any] | AsyncIterator[Mapping[str, Any]]:
        model = str(body.get("model", "mock-model"))
        if model == "backend-error":
            raise RuntimeError("mock backend error")

        if bool(body.get("stream", False)):
            return self._stream_completion(body)

        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def _stream_completion(self, body: Mapping[str, Any]) -> AsyncIterator[Mapping[str, Any]]:
        yield {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": str(body.get("model", "mock-model")),
            "choices": [{"index": 0, "delta": {"content": "hello"}}],
        }

        tools = body.get("tools")
        if isinstance(tools, list) and tools:
            yield {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call-mock",
                                    "type": "function",
                                    "function": {"name": "mock_tool", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }

        yield {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


async def _run_case(
    adapter: OpenAiNnrpAdapter,
    schema_version: str,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    request = _case_request(schema_version, case)
    cancel_after_events = _case_cancel_after_events(case)
    events: list[dict[str, Any]] = []
    stream = adapter.handle_request(request)
    try:
        async for event in stream:
            events.append(event)
            if cancel_after_events is not None and len(events) >= cancel_after_events:
                await stream.aclose()
                events.append(build_cancelled_event("caller_cancelled"))
                break
    finally:
        await stream.aclose()
    terminal = _terminal_from_events(events)
    expect = _as_mapping(case.get("expect"), "expect")
    expected_terminal = _as_str(expect.get("terminal"), "expect.terminal")
    expected_events = _as_list(expect.get("events"), "expect.events")
    messages: list[str] = []
    if terminal != expected_terminal:
        messages.append(f"terminal mismatch: expected {expected_terminal}, got {terminal}")
    messages.extend(_event_expectation_failures(events, expected_events))
    outcome: Outcome = "passed" if not messages else "failed"

    result: dict[str, Any] = {
        "id": _as_str(case.get("id"), "case.id"),
        "outcome": outcome,
        "terminal": terminal,
        "events": events,
    }
    if messages:
        result["message"] = "; ".join(messages)
    return result


def _case_request(schema_version: str, case: Mapping[str, Any]) -> dict[str, Any]:
    request = _as_mapping(case.get("request"), "case.request")
    envelope = {
        "schema_version": schema_version or OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "operation": _as_str(case.get("operation"), "case.operation"),
        "body": dict(_as_mapping(request.get("body"), "case.request.body")),
    }
    nnrp = request.get("nnrp")
    if isinstance(nnrp, Mapping):
        policy = {name: nnrp[name] for name in ("timeout_ms", "diagnostics") if name in nnrp}
        if policy:
            envelope["nnrp"] = policy
    return envelope


def _case_cancel_after_events(case: Mapping[str, Any]) -> int | None:
    # Cancellation timing belongs to the suite execution plan, not the profile request envelope.
    request = _as_mapping(case.get("request"), "case.request")
    nnrp = request.get("nnrp")
    if not isinstance(nnrp, Mapping) or "cancel_after_events" not in nnrp:
        return None
    value = nnrp["cancel_after_events"]
    if type(value) is not int or value <= 0:
        raise TypeError("case.request.nnrp.cancel_after_events must be a positive integer")
    return value


def _terminal_from_events(events: list[dict[str, Any]]) -> Terminal:
    if any(event.get("type") == "response.cancelled" for event in events):
        return "cancelled"
    if any(event.get("type") == "response.error" for event in events):
        return "error"
    return "success"


def _event_expectation_failures(events: list[dict[str, Any]], expected_events: list[Any]) -> list[str]:
    failures: list[str] = []
    baseline_events = [_without_ignored_baseline_fields(event) for event in events]
    for expectation_value in expected_events:
        expectation = _as_mapping(expectation_value, "expect.events[]")
        event_type = _as_str(expectation.get("type"), "expect.events[].type")
        matching_events = [event for event in baseline_events if event.get("type") == event_type]
        min_count = _as_non_negative_int(expectation.get("min_count"), default=0 if expectation.get("optional") else 1)
        fields = expectation.get("fields")
        if isinstance(fields, Mapping):
            matching_events = [event for event in matching_events if _mapping_contains(event, fields)]

        if len(matching_events) < min_count:
            failures.append(f"{event_type} count mismatch: expected at least {min_count}, got {len(matching_events)}")

    return failures


def _without_ignored_baseline_fields(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key not in _BASELINE_IGNORED_EVENT_FIELDS}


def _mapping_contains(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False

        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping) or not _mapping_contains(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False

    return True


def _load_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _as_mapping(value, str(path))


def _write_case_evidence(plan: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    artifacts = _as_mapping(plan.get("artifacts"), "artifacts")
    evidence_directory = Path(_as_str(artifacts.get("evidence_dir"), "artifacts.evidence_dir"))
    evidence_directory.mkdir(parents=True, exist_ok=True)
    results = _as_list(report.get("results"), "results")
    for index, result_value in enumerate(results, start=1):
        result = _as_mapping(result_value, "results[]")
        case_id = _as_str(result.get("id"), "results[].id")
        slug = "".join(character if character.isalnum() or character in ".-_" else "_" for character in case_id)
        evidence = {
            "profile": _as_str(report.get("profile"), "profile"),
            "schema_version": _as_str(report.get("schema_version"), "schema_version"),
            "adapter": _as_str(report.get("adapter"), "adapter"),
            "case": dict(result),
        }
        evidence_path = evidence_directory / f"{index:03d}-{slug}.json"
        evidence_path.write_text(f"{json.dumps(evidence, indent=2, sort_keys=True)}\n", encoding="utf-8")


def _as_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _as_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array")
    return value


def _as_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _as_non_negative_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or value < 0:
        raise TypeError("min_count must be a non-negative integer")
    return value


def run_conformance_plan_sync(plan_path: Path, output_path: Path, backend_spec: str) -> dict[str, Any]:
    return asyncio.run(run_conformance_plan_file(plan_path, output_path, backend=load_backend(backend_spec)))
