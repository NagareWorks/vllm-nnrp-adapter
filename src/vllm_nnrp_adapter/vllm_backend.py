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
    def __init__(self, serving_chat: object) -> None:
        self._serving_chat = serving_chat
        self._chat_method_name = _resolve_chat_method(serving_chat)

    async def create_chat_completion(self, body: Mapping[str, Any]) -> Any:
        method = getattr(self._serving_chat, self._chat_method_name)
        result = method(dict(body))
        if inspect.isawaitable(result):
            return await result
        return result


def _resolve_chat_method(serving_chat: object) -> str:
    for name in CHAT_METHOD_CANDIDATES:
        candidate = getattr(serving_chat, name, None)
        if callable(candidate):
            return name

    joined = ", ".join(CHAT_METHOD_CANDIDATES)
    raise TypeError(f"vLLM serving object does not expose a supported chat completion method: {joined}.")

