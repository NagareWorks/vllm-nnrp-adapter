from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from vllm_nnrp_adapter.vllm_factory import (
    create_backend_from_serving_factory,
    create_chat_completion_request,
)


class FakeChatCompletionRequest:
    def __init__(self, **payload: Any) -> None:
        self.payload = payload

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> FakeChatCompletionRequest:
        return cls(**payload)


class FakeServingChat:
    def create_chat_completion(self, request: FakeChatCompletionRequest) -> dict[str, Any]:
        return {"model": request.payload["model"]}


def make_serving_chat() -> FakeServingChat:
    return FakeServingChat()


@pytest.fixture
def fake_vllm_protocol_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    module.ChatCompletionRequest = FakeChatCompletionRequest
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.chat_completion.protocol", module)


def test_create_chat_completion_request_uses_vllm_model_validate(fake_vllm_protocol_module: None) -> None:
    request = create_chat_completion_request({"model": "llama", "messages": []})

    assert isinstance(request, FakeChatCompletionRequest)
    assert request.payload["model"] == "llama"


def test_create_chat_completion_request_falls_back_to_legacy_vllm_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("vllm.entrypoints.openai.protocol")
    module.ChatCompletionRequest = FakeChatCompletionRequest
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.protocol", module)

    request = create_chat_completion_request({"model": "llama", "messages": []})

    assert isinstance(request, FakeChatCompletionRequest)
    assert request.payload["model"] == "llama"


@pytest.mark.asyncio
async def test_create_backend_from_serving_factory_wraps_real_request_factory(
    fake_vllm_protocol_module: None,
) -> None:
    backend = create_backend_from_serving_factory(f"{__name__}:make_serving_chat")

    assert await backend.create_chat_completion({"model": "llama", "messages": []}) == {"model": "llama"}


def test_create_backend_from_serving_factory_rejects_bad_spec() -> None:
    with pytest.raises(ValueError):
        create_backend_from_serving_factory("missing_separator")
