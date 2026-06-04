from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from .adapter import ChatCompletionBackend, OpenAiNnrpAdapter
from .profile import OPENAI_COMPATIBLE_SCHEMA_VERSION

Terminal = Literal["success", "error", "cancelled"]
Outcome = Literal["passed", "failed", "skipped"]


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

    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("backend spec must be 'mock' or 'module.path:factory_name'")

    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    if not callable(factory):
        raise TypeError(f"backend factory is not callable: {spec}")

    backend = factory()
    if inspect.isawaitable(backend):
        raise TypeError("backend factory must return a backend synchronously")
    return cast(ChatCompletionBackend, backend)


class MockChatCompletionBackend:
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
    events = [event async for event in adapter.handle_request(request)]
    terminal = _terminal_from_events(events)
    expected_terminal = _as_str(_as_mapping(case.get("expect"), "expect").get("terminal"), "expect.terminal")
    outcome: Outcome = "passed" if terminal == expected_terminal else "failed"

    return {
        "id": _as_str(case.get("id"), "case.id"),
        "outcome": outcome,
        "terminal": terminal,
        "events": events,
    }


def _case_request(schema_version: str, case: Mapping[str, Any]) -> dict[str, Any]:
    request = _as_mapping(case.get("request"), "case.request")
    envelope = {
        "schema_version": schema_version or OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "operation": _as_str(case.get("operation"), "case.operation"),
        "body": dict(_as_mapping(request.get("body"), "case.request.body")),
    }
    nnrp = request.get("nnrp")
    if isinstance(nnrp, Mapping):
        envelope["nnrp"] = dict(nnrp)
    return envelope


def _terminal_from_events(events: list[dict[str, Any]]) -> Terminal:
    if any(event.get("type") == "response.error" for event in events):
        return "error"
    return "success"


def _load_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _as_mapping(value, str(path))


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


def run_conformance_plan_sync(plan_path: Path, output_path: Path, backend_spec: str) -> dict[str, Any]:
    return asyncio.run(run_conformance_plan_file(plan_path, output_path, backend=load_backend(backend_spec)))
