from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Mapping
from typing import Any, Protocol, cast


class RawHttpSseBackend(Protocol):
    def create_chat_completion(self, body: Mapping[str, Any]) -> object | Awaitable[object]:
        pass


class HttpSseSmokeBackend:
    """Explicit smoke-only adapter for comparing an HTTP/SSE-shaped backend."""

    supports_tool_calls = True

    def __init__(self, backend: RawHttpSseBackend) -> None:
        self._backend = backend

    async def create_chat_completion(
        self,
        body: Mapping[str, Any],
    ) -> Mapping[str, Any] | AsyncIterator[Mapping[str, Any]]:
        result = self._backend.create_chat_completion(body)
        if inspect.isawaitable(result):
            result = await result
        if _is_async_iterator(result):
            return _normalize_sse_stream(cast(AsyncIterator[object], result))
        normalized = _normalize_object(result)
        if isinstance(normalized, Mapping):
            return normalized
        raise TypeError(f"unsupported HTTP/SSE smoke result shape: {type(result).__name__}")


async def _normalize_sse_stream(chunks: AsyncIterator[object]) -> AsyncIterator[Mapping[str, Any]]:
    async for chunk in chunks:
        if not isinstance(chunk, str):
            normalized = _normalize_object(chunk)
            if not isinstance(normalized, Mapping):
                raise TypeError(f"unsupported HTTP/SSE smoke chunk shape: {type(chunk).__name__}")
            yield normalized
            continue
        for event in _parse_sse_chunk(chunk):
            yield event


def _parse_sse_chunk(chunk: str) -> tuple[Mapping[str, Any], ...]:
    parsed: list[Mapping[str, Any]] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        data = line[5:].strip() if line.startswith("data:") else line
        if not data or data == "[DONE]":
            continue
        value = json.loads(data)
        if not isinstance(value, Mapping):
            raise TypeError("HTTP/SSE smoke data must decode to a JSON object")
        parsed.append(value)
    return tuple(parsed)


def _normalize_object(value: object) -> object:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return dict_method()
    return value


def _is_async_iterator(value: object) -> bool:
    return hasattr(value, "__aiter__") and hasattr(value, "__anext__")
