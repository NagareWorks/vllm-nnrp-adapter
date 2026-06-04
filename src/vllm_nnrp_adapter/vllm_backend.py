from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

CHAT_METHOD_CANDIDATES = (
    "create_chat_completion",
    "create_chat_completion_raw",
    "chat_completion",
)


class VllmBackend:
    def __init__(self, serving_chat: object, *, request_factory: object | None = None) -> None:
        self._serving_chat = serving_chat
        self._chat_method_name = _resolve_chat_method(serving_chat)
        self._request_factory = request_factory

    async def create_chat_completion(self, body: Mapping[str, Any]) -> Any:
        method = getattr(self._serving_chat, self._chat_method_name)
        request = self._build_request(body)
        result = _call_chat_method(method, request)
        if inspect.isawaitable(result):
            result = await result
        return _normalize_chat_result(result)

    def _build_request(self, body: Mapping[str, Any]) -> object:
        if self._request_factory is None:
            return dict(body)
        if not callable(self._request_factory):
            raise TypeError("request_factory must be callable")
        return self._request_factory(dict(body))


def _resolve_chat_method(serving_chat: object) -> str:
    for name in CHAT_METHOD_CANDIDATES:
        candidate = getattr(serving_chat, name, None)
        if callable(candidate):
            return name

    joined = ", ".join(CHAT_METHOD_CANDIDATES)
    raise TypeError(f"vLLM serving object does not expose a supported chat completion method: {joined}.")


def _call_chat_method(method: object, request: object) -> object:
    if not callable(method):
        raise TypeError("resolved vLLM chat completion method is not callable")

    try:
        return method(request)
    except TypeError as positional_error:
        try:
            return method(request, raw_request=None)
        except TypeError:
            try:
                return method(request=request, raw_request=None)
            except TypeError:
                raise positional_error from None


def _normalize_chat_result(result: object) -> object:
    if _is_async_iterator(result):
        return _normalize_stream(result)
    return _normalize_object(result)


async def _normalize_stream(chunks: object) -> AsyncIterator[Mapping[str, Any]]:
    iterator = cast(AsyncIterator[object], chunks)
    async for chunk in iterator:
        for normalized in _normalize_stream_chunk(chunk):
            yield normalized


def _normalize_stream_chunk(chunk: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(chunk, str):
        return tuple(_parse_sse_chunk(chunk))
    normalized = _normalize_object(chunk)
    if isinstance(normalized, Mapping):
        return (normalized,)
    raise TypeError(f"unsupported vLLM stream chunk shape: {type(chunk).__name__}")


def _parse_sse_chunk(chunk: str) -> list[Mapping[str, Any]]:
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
            raise TypeError("vLLM SSE stream data must decode to a JSON object")
        parsed.append(value)
    return parsed


def _normalize_object(value: object) -> object:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        dumped = dict_method()
        if isinstance(dumped, Mapping):
            return dumped
    return value


def _is_async_iterator(value: object) -> bool:
    return hasattr(value, "__aiter__") and hasattr(value, "__anext__")
