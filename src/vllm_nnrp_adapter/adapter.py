from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
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
    build_tool_call_completed_event,
    build_tool_call_delta_event,
    build_tool_call_error_event,
    build_tool_call_started_event,
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
                close_guard = _AsyncIteratorCloseGuard(chunks, observer=backend_abort_observer)
                try:
                    async for event in self._map_streaming_chunks(chunks, timeout_s=timeout_s):
                        yield event
                        emitted_events += 1
                        if cancel_after_events is not None and emitted_events >= cancel_after_events:
                            await close_guard.close()
                            yield build_cancelled_event()
                            return
                except asyncio.CancelledError:
                    await close_guard.close()
                    raise
                except TimeoutError as error:
                    await close_guard.close()
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
                diagnostics=_backend_error_diagnostics(error),
            )

    async def _map_streaming_chunks(
        self,
        chunks: AsyncIterator[Mapping[str, Any]],
        *,
        timeout_s: float | None,
    ) -> AsyncIterator[dict[str, Any]]:
        mapper = OpenAiStreamEventMapper()
        try:
            async for chunk in _iterate_with_timeout(chunks, timeout_s):
                for event in mapper.map_chunk(chunk):
                    yield event
        except (Exception, asyncio.CancelledError) as error:
            for event in mapper.fail(error):
                yield event
            raise
        for event in mapper.finish():
            yield event


def map_openai_stream_chunk(chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
    return OpenAiStreamEventMapper().map_chunk(chunk)


@dataclass
class _ToolCallState:
    choice_index: int
    tool_index: int
    item_id: str
    call_id: str
    name: str = ""
    argument_parts: list[str] = dataclass_field(default_factory=list)
    emitted_argument_parts: int = 0
    started: bool = False
    closed: bool = False


class OpenAiStreamEventMapper:
    def __init__(self) -> None:
        self._tool_calls: dict[tuple[int, int], _ToolCallState] = {}

    def map_chunk(self, chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
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

            choice_index = _non_negative_index(choice.get("index"), default=0)
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    events.append(build_text_delta_event(content, index=choice_index, openai_chunk=chunk))

                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for position, tool_call in enumerate(tool_calls):
                        if isinstance(tool_call, Mapping):
                            events.extend(
                                self._map_tool_call_delta(
                                    choice_index,
                                    position,
                                    tool_call,
                                    chunk,
                                )
                            )

            if choice.get("finish_reason") is not None:
                events.extend(self._complete_choice(choice_index, chunk))

        return events

    def finish(self) -> list[dict[str, Any]]:
        return self._complete_states(self._open_states(), openai_chunk=None)

    def fail(self, error: BaseException) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for state in self._open_states():
            state.closed = True
            events.append(
                build_tool_call_error_event(
                    index=state.tool_index,
                    item_id=state.item_id,
                    call_id=state.call_id,
                    error_type="server_error",
                    code="tool_call_stream_interrupted",
                    message=str(error) or type(error).__name__,
                )
            )
        return events

    def _map_tool_call_delta(
        self,
        choice_index: int,
        position: int,
        tool_call: Mapping[str, Any],
        chunk: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        tool_index = _non_negative_index(tool_call.get("index"), default=position)
        key = (choice_index, tool_index)
        state = self._tool_calls.get(key)
        call_id = tool_call.get("id")
        if state is None:
            stable_call_id = call_id if isinstance(call_id, str) and call_id else f"call-{choice_index}-{tool_index}"
            state = _ToolCallState(
                choice_index=choice_index,
                tool_index=tool_index,
                item_id=f"tool-call-{choice_index}-{tool_index}",
                call_id=stable_call_id,
            )
            self._tool_calls[key] = state
        elif not state.started and isinstance(call_id, str) and call_id:
            state.call_id = call_id

        function = tool_call.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if not state.name and isinstance(name, str) and name:
                state.name = name
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                state.argument_parts.append(arguments)

        return self._emit_ready_state(state, chunk)

    def _emit_ready_state(
        self,
        state: _ToolCallState,
        chunk: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if state.closed or not state.name:
            return []

        events: list[dict[str, Any]] = []
        if not state.started:
            state.started = True
            events.append(
                build_tool_call_started_event(
                    index=state.tool_index,
                    item_id=state.item_id,
                    call_id=state.call_id,
                    name=state.name,
                    openai_chunk=chunk,
                )
            )

        for arguments_delta in state.argument_parts[state.emitted_argument_parts :]:
            events.append(
                build_tool_call_delta_event(
                    arguments_delta,
                    index=state.tool_index,
                    item_id=state.item_id,
                    call_id=state.call_id,
                    openai_chunk=chunk,
                )
            )
        state.emitted_argument_parts = len(state.argument_parts)
        return events

    def _complete_choice(
        self,
        choice_index: int,
        openai_chunk: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        states = [state for state in self._open_states() if state.choice_index == choice_index]
        return self._complete_states(states, openai_chunk=openai_chunk)

    def _complete_states(
        self,
        states: list[_ToolCallState],
        *,
        openai_chunk: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for state in sorted(states, key=lambda item: (item.choice_index, item.tool_index)):
            if state.closed:
                continue
            state.closed = True
            if not state.started:
                events.append(
                    build_tool_call_error_event(
                        index=state.tool_index,
                        item_id=state.item_id,
                        call_id=state.call_id,
                        error_type="server_error",
                        code="invalid_tool_call_stream",
                        message="tool call stream ended before a function name was provided",
                        openai_chunk=openai_chunk,
                    )
                )
                continue
            events.append(
                build_tool_call_completed_event(
                    index=state.tool_index,
                    item_id=state.item_id,
                    call_id=state.call_id,
                    name=state.name,
                    arguments="".join(state.argument_parts),
                    openai_chunk=openai_chunk,
                )
            )
        return events

    def _open_states(self) -> list[_ToolCallState]:
        return [state for state in self._tool_calls.values() if not state.closed]


def _non_negative_index(value: object, *, default: int) -> int:
    if type(value) is int and value >= 0:
        return value
    return default


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


class _AsyncIteratorCloseGuard:
    def __init__(
        self,
        chunks: AsyncIterator[Mapping[str, Any]],
        *,
        observer: Callable[[bool | None], None] | None = None,
    ) -> None:
        self._chunks = chunks
        self._observer = observer
        self._closed = False
        self._result: bool | None = None

    async def close(self) -> bool | None:
        if self._closed:
            return self._result
        self._closed = True
        self._result = await _close_async_iterator(self._chunks, observer=self._observer)
        return self._result


def classify_backend_error(error: Exception) -> tuple[str, str]:
    class_name = type(error).__name__.lower()
    message = str(error).lower()
    backend_type = getattr(error, "vllm_error_type", None)
    backend_status = getattr(error, "vllm_status_code", None)
    haystack = f"{class_name} {message} {backend_type or ''}".lower()

    if backend_status == 404:
        return "invalid_request_error", "unsupported_model"
    if backend_status == 429 or "overload" in haystack or "rate limit" in haystack or "too many" in haystack:
        return "server_error", "backend_overload"
    if "scheduler" in haystack and ("reject" in haystack or "full" in haystack):
        return "server_error", "scheduler_rejected"
    if "cancel" in haystack or "abort" in haystack:
        return "server_error", "backend_cancelled"
    if backend_status in {400, 422}:
        return "invalid_request_error", "invalid_backend_request"
    if backend_status == 503:
        return "server_error", "backend_overload"
    return "server_error", "backend_error"


def _backend_error_diagnostics(error: Exception) -> dict[str, object]:
    diagnostics: dict[str, object] = {"backend_error_family": type(error).__name__}
    for attribute, field in (
        ("vllm_error_type", "vllm_error_type"),
        ("vllm_status_code", "vllm_status_code"),
        ("vllm_parameter", "vllm_parameter"),
    ):
        value = getattr(error, attribute, None)
        if value is not None:
            diagnostics[field] = value
    return diagnostics
