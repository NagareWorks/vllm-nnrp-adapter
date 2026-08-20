from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from .nnrp_contract import validate_nnrp_runtime_contract
from .vllm_backend import VllmBackend
from .vllm_compat import VllmEngineDirectBinding, resolve_vllm_compatibility


class RequestConstructor(Protocol):
    def __call__(self, **payload: Any) -> object:
        pass


@dataclass(frozen=True, slots=True)
class _BoundRequestFactory:
    request_type: object
    compatibility_binding: str
    vllm_version: str
    engine_direct_binding: VllmEngineDirectBinding

    def __call__(self, body: Mapping[str, Any]) -> object:
        return _construct_request(self.request_type, body)


def create_chat_completion_request(body: Mapping[str, Any]) -> object:
    _detected_version, _binding, request_type = resolve_vllm_compatibility()
    return _construct_request(request_type, body)


def _construct_request(request_type: object, body: Mapping[str, Any]) -> object:
    payload = dict(body)
    model_validate = getattr(request_type, "model_validate", None)
    if callable(model_validate):
        return model_validate(payload)
    constructor = cast(RequestConstructor, request_type)
    return constructor(**payload)


def create_vllm_backend(serving_chat: object) -> VllmBackend:
    validate_nnrp_runtime_contract()
    detected_version, binding, request_type = resolve_vllm_compatibility(serving_chat)
    request_factory = _BoundRequestFactory(
        request_type=request_type,
        compatibility_binding=binding.name,
        vllm_version=detected_version,
        engine_direct_binding=binding.engine_direct,
    )
    return VllmBackend(serving_chat, request_factory=request_factory)


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
