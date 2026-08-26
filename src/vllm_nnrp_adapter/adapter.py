from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Protocol, cast, runtime_checkable

from .capability_ledger import openai_profile_extensions, openai_profile_operation_capabilities
from .profile import (
    CHAT_COMPLETIONS_CREATE,
    VLLM_DIAGNOSTICS_EXTENSION,
    OpenAiNnrpCapabilityDocument,
    OpenAiNnrpError,
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


@runtime_checkable
class _RuntimeContextAwareChatCompletionBackend(Protocol):
    def create_chat_completion_with_context(
        self,
        body: Mapping[str, Any],
        *,
        trace_headers: Mapping[str, str] | None = None,
        priority: int | None = None,
    ) -> ChatCompletionResult | Awaitable[ChatCompletionResult]:
        pass


@runtime_checkable
class _LivePriorityChatCompletionBackend(Protocol):
    supports_live_runtime_priority: bool

    def update_runtime_priority(
        self,
        request_id: str,
        priority: int,
    ) -> bool | Awaitable[bool]:
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

    def _supports_runtime_priority(self) -> bool:
        return (
            getattr(self._backend, "supports_runtime_priority", False) is True
            and isinstance(self._backend, _RuntimeContextAwareChatCompletionBackend)
        )

    async def _update_runtime_priority(self, request_id: str, priority: int) -> bool:
        backend = self._backend
        if (
            not isinstance(backend, _LivePriorityChatCompletionBackend)
            or backend.supports_live_runtime_priority is not True
        ):
            return False
        result = backend.update_runtime_priority(request_id, priority)
        if inspect.isawaitable(result):
            result = await result
        return result is True

    async def handle_request(self, request: Mapping[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        stream = self._handle_native_request(request)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    async def _handle_native_request(
        self,
        request: Mapping[str, Any],
        *,
        backend_abort_observer: Callable[[bool | None], None] | None = None,
        backend_trace_headers_factory: Callable[[], Mapping[str, str] | None] | None = None,
        backend_priority_factory: Callable[[], int | None] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
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
            if _diagnostics_enabled(policy, self.capabilities):
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
            backend_trace_headers = (
                None if backend_trace_headers_factory is None else backend_trace_headers_factory()
            )
            backend_priority = None if backend_priority_factory is None else backend_priority_factory()
            if (
                backend_trace_headers is not None or backend_priority is not None
            ) and isinstance(self._backend, _RuntimeContextAwareChatCompletionBackend):
                context_backend = self._backend
                if backend_trace_headers is not None and backend_priority is not None:
                    result = context_backend.create_chat_completion_with_context(
                        backend_body,
                        trace_headers=backend_trace_headers,
                        priority=backend_priority,
                    )
                elif backend_trace_headers is not None:
                    result = context_backend.create_chat_completion_with_context(
                        backend_body,
                        trace_headers=backend_trace_headers,
                    )
                else:
                    result = context_backend.create_chat_completion_with_context(
                        backend_body,
                        priority=backend_priority,
                    )
            elif backend_priority is not None:
                raise OpenAiNnrpError(
                    "server_error",
                    "runtime_priority_unsupported",
                    "The selected backend cannot apply NNRP runtime priority before admission.",
                )
            else:
                result = self._backend.create_chat_completion(backend_body)
            if inspect.isawaitable(result):
                result = await _await_with_timeout(result, timeout_s)

            if _is_async_iterator(result):
                chunks = cast(AsyncIterator[Mapping[str, Any]], result)
                close_guard = _AsyncIteratorCloseGuard(chunks, observer=backend_abort_observer)
                try:
                    async for event in self._map_streaming_chunks(chunks, timeout_s=timeout_s):
                        yield event
                except (asyncio.CancelledError, GeneratorExit):
                    await close_guard.close()
                    raise
                except TimeoutError as error:
                    await close_guard.close()
                    raise error
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


@dataclass
class _ChoiceResponseState:
    index: int
    role: str = "assistant"
    content_parts: list[str] = dataclass_field(default_factory=list)
    finish_reason: object = None


class OpenAiStreamEventMapper:
    def __init__(self) -> None:
        self._tool_calls: dict[tuple[int, int], _ToolCallState] = {}
        self._choices: dict[int, _ChoiceResponseState] = {}
        self._response_id: str | None = None
        self._model: str | None = None
        self._created: int | None = None
        self._usage: Mapping[str, Any] | None = None

    def map_chunk(self, chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        response_id = chunk.get("id")
        if self._response_id is None and isinstance(response_id, str) and response_id:
            self._response_id = response_id
        model = chunk.get("model")
        if self._model is None and isinstance(model, str) and model:
            self._model = model
        created = chunk.get("created")
        if self._created is None and isinstance(created, int) and created >= 0:
            self._created = created

        usage = chunk.get("usage")
        if isinstance(usage, Mapping):
            self._usage = dict(usage)
            events.append(build_usage_event(usage))

        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return events

        for choice in choices:
            if not isinstance(choice, Mapping):
                continue

            choice_index = _non_negative_index(choice.get("index"), default=0)
            choice_state = self._choices.setdefault(choice_index, _ChoiceResponseState(choice_index))
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                role = delta.get("role")
                if isinstance(role, str) and role:
                    choice_state.role = role
                content = delta.get("content")
                if isinstance(content, str) and content:
                    choice_state.content_parts.append(content)
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
                choice_state.finish_reason = choice.get("finish_reason")
                events.extend(self._complete_choice(choice_index, chunk))

        return events

    def finish(self) -> list[dict[str, Any]]:
        events = self._complete_states(self._open_states(), openai_chunk=None)
        completed_body = self._completed_body()
        if completed_body is not None:
            events.append(build_completed_event(completed_body))
        return events

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

    def _completed_body(self) -> dict[str, Any] | None:
        if not self._choices:
            return None
        choices: list[dict[str, Any]] = []
        for choice_index, state in sorted(self._choices.items()):
            tool_calls = [
                {
                    "id": tool_state.call_id,
                    "type": "function",
                    "function": {
                        "name": tool_state.name,
                        "arguments": "".join(tool_state.argument_parts),
                    },
                }
                for (owner_choice, _tool_index), tool_state in sorted(self._tool_calls.items())
                if owner_choice == choice_index and tool_state.started
            ]
            message: dict[str, Any] = {
                "role": state.role,
                "content": "".join(state.content_parts) if state.content_parts else None,
            }
            if tool_calls:
                message["tool_calls"] = tool_calls
            choices.append(
                {
                    "index": choice_index,
                    "message": message,
                    "finish_reason": state.finish_reason,
                }
            )
        body: dict[str, Any] = {
            "object": "chat.completion",
            "choices": choices,
        }
        if self._response_id is not None:
            body["id"] = self._response_id
        if self._model is not None:
            body["model"] = self._model
        if self._created is not None:
            body["created"] = self._created
        if self._usage is not None:
            body["usage"] = dict(self._usage)
        return body


def _non_negative_index(value: object, *, default: int) -> int:
    if type(value) is int and value >= 0:
        return value
    return default


def _default_capabilities(backend: ChatCompletionBackend) -> OpenAiNnrpCapabilityDocument:
    document = OpenAiNnrpCapabilityDocument.level1()
    operation_capabilities = openai_profile_operation_capabilities(
        supports_tool_calls=getattr(backend, "supports_tool_calls", False) is True
    )
    operations = tuple({**operation, **operation_capabilities} for operation in document.operations)
    return OpenAiNnrpCapabilityDocument(
        compatibility_levels=document.compatibility_levels,
        operations=operations,
        models=document.models,
        extensions=openai_profile_extensions(),
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


def _diagnostics_enabled(
    policy: object,
    capabilities: OpenAiNnrpCapabilityDocument,
) -> bool:
    return (
        isinstance(policy, Mapping)
        and policy.get("diagnostics") is True
        and capabilities.supports_non_critical_extension(VLLM_DIAGNOSTICS_EXTENSION)
    )


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
