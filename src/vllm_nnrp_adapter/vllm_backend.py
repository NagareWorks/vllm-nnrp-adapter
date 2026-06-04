from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

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
            return await result
        return result

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
