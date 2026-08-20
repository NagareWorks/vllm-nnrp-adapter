from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol, cast

from .profile import (
    CHAT_COMPLETIONS_CREATE,
    OpenAiNnrpCapabilityDocument,
    OpenAiNnrpError,
    build_cancelled_event,
    build_completed_event,
    build_diagnostics_event,
    build_error_event,
    build_text_delta_event,
    build_tool_call_delta_event,
    build_usage_event,
    validate_request,
)

ChatCompletionResult = Mapping[str, Any] | AsyncIterator[Mapping[str, Any]]


class ChatCompletionBackend(Protocol):
    def create_chat_completion(
        self,
        body: Mapping[str, Any],
    ) -> ChatCompletionResult | Awaitable[ChatCompletionResult]:
        pass


class OpenAiNnrpAdapter:
    def __init__(
        self,
        backend: ChatCompletionBackend,
        *,
        capabilities: OpenAiNnrpCapabilityDocument | None = None,
    ) -> None:
        self._backend = backend
        self.capabilities = capabilities or _default_capabilities(backend)

    def _backend_observation_identity(self) -> tuple[str, str | None, str | None]:
        return (
            type(self._backend).__name__,
            _optional_string_attribute(self._backend, "compatibility_binding"),
            _optional_string_attribute(self._backend, "vllm_version"),
        )

    async def handle_request(self, request: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        async for event in self._handle_native_request(request):
            yield event

    async def _handle_native_request(
        self,
        request: Mapping[str, Any],
        *,
        backend_abort_observer: Callable[[bool | None], None] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            envelope = validate_request(request, self.capabilities)
            if envelope["operation"] != CHAT_COMPLETIONS_CREATE:
                raise OpenAiNnrpError(
                    "invalid_request_error",
                    "unsupported_operation",
                    "Unsupported profile operation.",
                )

            policy = envelope.get("nnrp")
            timeout_s = _timeout_seconds(policy)
            if _diagnostics_enabled(policy):
                yield build_diagnostics_event(
                    {
                        "selected_model": envelope["body"].get("model"),
                        "operation": envelope["operation"],
                        "backend_family": type(self._backend).__name__,
                    }
                )

            backend_body = dict(envelope["body"])
            request_id = envelope.get("request_id")
            if request_id is not None:
                backend_body["request_id"] = request_id
            result = self._backend.create_chat_completion(backend_body)
            if inspect.isawaitable(result):
                result = await _await_with_timeout(result, timeout_s)

            if _is_async_iterator(result):
                emitted_events = 0
                cancel_after_events = _cancel_after_events(envelope.get("nnrp"))
                chunks = cast(AsyncIterator[Mapping[str, Any]], result)
                try:
                    async for event in self._map_streaming_chunks(chunks, timeout_s=timeout_s):
                        yield event
                        emitted_events += 1
                        if cancel_after_events is not None and emitted_events >= cancel_after_events:
                            await _close_async_iterator(chunks, observer=backend_abort_observer)
                            yield build_cancelled_event()
                            return
                except asyncio.CancelledError:
                    await _close_async_iterator(chunks, observer=backend_abort_observer)
                    raise
                except TimeoutError as error:
                    await _close_async_iterator(chunks, observer=backend_abort_observer)
                    raise error
                yield build_completed_event({"object": "chat.completion.stream", "status": "completed"})
                return

            yield build_completed_event(cast(Mapping[str, Any], result))
        except OpenAiNnrpError as error:
            yield error.to_event()
        except TimeoutError as error:
            yield build_error_event("timeout_error", "request_timeout", str(error))
        except Exception as error:
            error_type, code = classify_backend_error(error)
            yield build_error_event(
                error_type,
                code,
                str(error),
                diagnostics={"backend_error_family": type(error).__name__},
            )

    async def _map_streaming_chunks(
        self,
        chunks: AsyncIterator[Mapping[str, Any]],
        *,
        timeout_s: float | None,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in _iterate_with_timeout(chunks, timeout_s):
            yielded = False
            for event in map_openai_stream_chunk(chunk):
                yielded = True
                yield event

            if not yielded and chunk.get("usage") is not None:
                yield build_usage_event(_as_mapping(chunk["usage"]))


def map_openai_stream_chunk(chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    usage = chunk.get("usage")
    if isinstance(usage, Mapping):
        events.append(build_usage_event(usage))

    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return events

    for choice in choices:
        if not isinstance(choice, Mapping):
            continue

        index = int(choice.get("index", 0))
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            continue

        content = delta.get("content")
        if isinstance(content, str) and content:
            events.append(build_text_delta_event(content, index=index, openai_chunk=chunk))

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, Mapping):
                    events.append(build_tool_call_delta_event(tool_call, index=index, openai_chunk=chunk))

    return events


def _default_capabilities(backend: ChatCompletionBackend) -> OpenAiNnrpCapabilityDocument:
    document = OpenAiNnrpCapabilityDocument.level1()
    tool_calls = getattr(backend, "supports_tool_calls", False) is True
    operations = tuple({**operation, "tool_calls": tool_calls} for operation in document.operations)
    return OpenAiNnrpCapabilityDocument(
        compatibility_levels=document.compatibility_levels,
        operations=operations,
        models=document.models,
    )


def _is_async_iterator(value: object) -> bool:
    return hasattr(value, "__aiter__") and hasattr(value, "__anext__")


def _optional_string_attribute(value: object, name: str) -> str | None:
    attribute = getattr(value, name, None)
    return attribute if isinstance(attribute, str) else None


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {"value": value}


def _cancel_after_events(policy: object) -> int | None:
    if not isinstance(policy, Mapping):
        return None

    value = policy.get("cancel_after_events")
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _diagnostics_enabled(policy: object) -> bool:
    return isinstance(policy, Mapping) and policy.get("diagnostics") is True


def _timeout_seconds(policy: object) -> float | None:
    if not isinstance(policy, Mapping):
        return None

    value = policy.get("timeout_ms")
    if isinstance(value, int) and value > 0:
        return value / 1000
    return None


async def _await_with_timeout(
    awaitable: Awaitable[ChatCompletionResult],
    timeout_s: float | None,
) -> ChatCompletionResult:
    try:
        if timeout_s is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=timeout_s)
    except TimeoutError as error:
        raise TimeoutError("backend did not return before nnrp.timeout_ms") from error


async def _iterate_with_timeout(
    chunks: AsyncIterator[Mapping[str, Any]],
    timeout_s: float | None,
) -> AsyncIterator[Mapping[str, Any]]:
    iterator = chunks.__aiter__()
    while True:
        try:
            if timeout_s is None:
                yield await iterator.__anext__()
            else:
                yield await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)
        except StopAsyncIteration:
            return
        except TimeoutError as error:
            raise TimeoutError("backend stream did not yield before nnrp.timeout_ms") from error


async def _close_async_iterator(
    chunks: AsyncIterator[Mapping[str, Any]],
    *,
    observer: Callable[[bool | None], None] | None = None,
) -> bool | None:
    closer = getattr(chunks, "aclose", None)
    if closer is None:
        if observer is not None:
            observer(None)
        return None
    result = closer()
    if inspect.isawaitable(result):
        result = await result
    accepted = result if isinstance(result, bool) else None
    if observer is not None:
        observer(accepted)
    return accepted


def classify_backend_error(error: Exception) -> tuple[str, str]:
    class_name = type(error).__name__.lower()
    message = str(error).lower()
    haystack = f"{class_name} {message}"

    if "overload" in haystack or "rate limit" in haystack or "too many" in haystack:
        return "server_error", "backend_overload"
    if "scheduler" in haystack and ("reject" in haystack or "full" in haystack):
        return "server_error", "scheduler_rejected"
    if "cancel" in haystack or "abort" in haystack:
        return "server_error", "backend_cancelled"
    return "server_error", "backend_error"
