from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Mapping
from typing import Any, Protocol, cast

from .profile import (
    CHAT_COMPLETIONS_CREATE,
    OpenAiNnrpCapabilityDocument,
    OpenAiNnrpError,
    build_completed_event,
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
        self.capabilities = capabilities or OpenAiNnrpCapabilityDocument.level1()

    async def handle_request(self, request: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        try:
            envelope = validate_request(request, self.capabilities)
            if envelope["operation"] != CHAT_COMPLETIONS_CREATE:
                raise OpenAiNnrpError(
                    "invalid_request_error",
                    "unsupported_operation",
                    "Unsupported profile operation.",
                )

            result = self._backend.create_chat_completion(envelope["body"])
            if inspect.isawaitable(result):
                result = await result

            if _is_async_iterator(result):
                async for event in self._map_streaming_chunks(cast(AsyncIterator[Mapping[str, Any]], result)):
                    yield event
                return

            yield build_completed_event(cast(Mapping[str, Any], result))
        except OpenAiNnrpError as error:
            yield error.to_event()
        except Exception as error:
            yield build_error_event("server_error", "backend_error", str(error))

    async def _map_streaming_chunks(self, chunks: AsyncIterator[Mapping[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        async for chunk in chunks:
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


def _is_async_iterator(value: object) -> bool:
    return hasattr(value, "__aiter__") and hasattr(value, "__anext__")


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {"value": value}
