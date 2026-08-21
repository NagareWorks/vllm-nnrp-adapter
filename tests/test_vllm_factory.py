from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, cast

import pytest

from vllm_nnrp_adapter.vllm_compat import (
    VLLM_COMPATIBILITY_BINDINGS,
    VLLM_INSTALLATION_RANGE,
    VllmCompatibilityError,
    render_vllm_compatibility_table,
    resolve_vllm_compatibility,
)
from vllm_nnrp_adapter.vllm_factory import (
    create_backend_from_serving_factory,
    create_chat_completion_request,
    create_vllm_backend,
)


class FakeChatCompletionRequest:
    def __init__(self, **payload: Any) -> None:
        self.payload = payload

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> FakeChatCompletionRequest:
        return cls(**payload)


class FakeServingChat:
    model_config = type(
        "FakeModelConfig",
        (),
        {"max_model_len": 4096, "dtype": "float16", "quantization": None, "task": "generate"},
    )()

    def create_chat_completion(self, request: FakeChatCompletionRequest) -> dict[str, Any]:
        return {"model": request.payload["model"]}


def make_serving_chat() -> FakeServingChat:
    return FakeServingChat()


@pytest.fixture(autouse=True)
def fake_vllm_installation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vllm_nnrp_adapter.vllm_compat._installed_vllm_version", lambda: "0.26.0")
    module = ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    module.ChatCompletionRequest = FakeChatCompletionRequest
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.chat_completion.protocol", module)


def test_create_chat_completion_request_uses_vllm_model_validate() -> None:
    request = create_chat_completion_request({"model": "llama", "messages": []})

    assert isinstance(request, FakeChatCompletionRequest)
    assert request.payload["model"] == "llama"


@pytest.mark.parametrize(
    ("installed_version", "binding_name"),
    (("0.18.1", "legacy-0.18"), ("0.22.1", "transition-0.22"), ("0.26.0", "current-0.26")),
)
def test_compatibility_anchors_select_named_bindings(installed_version: str, binding_name: str) -> None:
    detected, binding, request_type = resolve_vllm_compatibility(
        FakeServingChat(),
        installed_version=installed_version,
    )

    assert detected == installed_version
    assert binding.name == binding_name
    assert request_type is FakeChatCompletionRequest


def test_backend_records_selected_binding_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vllm_nnrp_adapter.vllm_compat._installed_vllm_version", lambda: "0.22.1")

    backend = create_vllm_backend(FakeServingChat())

    assert backend.compatibility_binding == "transition-0.22"
    assert backend.vllm_version == "0.22.1"
    bound_factory = cast(Any, backend.__dict__["_request_factory"])
    assert bound_factory.engine_direct_binding is VLLM_COMPATIBILITY_BINDINGS[1].engine_direct
    assert backend.benchmark_metadata() == {
        "vllm_version": "0.22.1",
        "compatibility_binding": "transition-0.22",
        "engine_configuration": {"max_model_len": 4096, "dtype": "float16", "task": "generate"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("installed_version", "binding_name"),
    (("0.18.1", "legacy-0.18"), ("0.22.1", "transition-0.22"), ("0.26.0", "current-0.26")),
)
async def test_non_streaming_chat_uses_each_named_request_binding(
    monkeypatch: pytest.MonkeyPatch,
    installed_version: str,
    binding_name: str,
) -> None:
    monkeypatch.setattr(
        "vllm_nnrp_adapter.vllm_compat._installed_vllm_version",
        lambda: installed_version,
    )
    backend = create_vllm_backend(FakeServingChat())

    result = await backend.create_chat_completion({"model": "llama", "messages": [], "stream": False})

    assert result == {"model": "llama"}
    assert backend.compatibility_binding == binding_name


def test_compatibility_rejects_untested_minor_inside_installation_band() -> None:
    with pytest.raises(VllmCompatibilityError) as captured:
        resolve_vllm_compatibility(FakeServingChat(), installed_version="0.20.0")

    message = str(captured.value)
    assert "detected version 0.20.0" in message
    assert "missing feature: named compatibility binding" in message
    assert "tested anchors: 0.18.1, 0.22.1, 0.26.0" in message


def test_compatibility_rejects_missing_serving_feature() -> None:
    with pytest.raises(VllmCompatibilityError, match=r"missing feature: serving_chat\.create_chat_completion"):
        resolve_vllm_compatibility(object(), installed_version="0.26.0")


def test_compatibility_rejects_version_outside_installation_band() -> None:
    with pytest.raises(VllmCompatibilityError, match=VLLM_INSTALLATION_RANGE.replace(".", r"\.")):
        resolve_vllm_compatibility(FakeServingChat(), installed_version="0.27.0")


def test_generated_compatibility_table_comes_from_runtime_registry() -> None:
    table = render_vllm_compatibility_table()

    for binding in VLLM_COMPATIBILITY_BINDINGS:
        assert binding.name in table
        assert binding.anchor_version in table
        assert binding.version_range in table


@pytest.mark.asyncio
async def test_create_backend_from_serving_factory_wraps_real_request_factory(
) -> None:
    backend = create_backend_from_serving_factory(f"{__name__}:make_serving_chat")

    assert await backend.create_chat_completion({"model": "llama", "messages": []}) == {"model": "llama"}


def test_create_backend_from_serving_factory_rejects_bad_spec() -> None:
    with pytest.raises(ValueError):
        create_backend_from_serving_factory("missing_separator")
