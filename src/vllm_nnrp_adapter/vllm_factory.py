from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol, cast

from .vllm_backend import VllmBackend

CHAT_COMPLETION_REQUEST_PATH = "vllm.entrypoints.openai.chat_completion.protocol:ChatCompletionRequest"


class RequestConstructor(Protocol):
    def __call__(self, **payload: Any) -> object:
        pass


def create_chat_completion_request(body: Mapping[str, Any]) -> object:
    request_type = _load_symbol(CHAT_COMPLETION_REQUEST_PATH)
    payload = dict(body)
    model_validate = getattr(request_type, "model_validate", None)
    if callable(model_validate):
        return model_validate(payload)
    constructor = cast(RequestConstructor, request_type)
    return constructor(**payload)


def create_vllm_backend(serving_chat: object) -> VllmBackend:
    return VllmBackend(serving_chat, request_factory=create_chat_completion_request)


def create_backend_from_serving_factory(factory_spec: str) -> VllmBackend:
    factory = _load_symbol(factory_spec)
    if not callable(factory):
        raise TypeError(f"vLLM serving factory is not callable: {factory_spec}")
    return create_vllm_backend(factory())


def _load_symbol(spec: str) -> object:
    module_name, separator, symbol_name = spec.partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError("symbol spec must use 'module.path:attribute'")

    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)
