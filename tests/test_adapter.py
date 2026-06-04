from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from vllm_nnrp_adapter import OpenAiNnrpAdapter
from vllm_nnrp_adapter.adapter import map_openai_stream_chunk
from vllm_nnrp_adapter.profile import CHAT_COMPLETIONS_CREATE, OPENAI_COMPATIBLE_SCHEMA_VERSION
from vllm_nnrp_adapter.vllm_backend import VllmBackend


class StreamingBackend:
    def create_chat_completion(self, body: Mapping[str, Any]) -> AsyncIterator[Mapping[str, Any]]:
        assert body["model"] == "llama"

        async def chunks() -> AsyncIterator[Mapping[str, Any]]:
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "hello"},
                    }
                ]
            }
            yield {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        return chunks()


class NonStreamingBackend:
    async def create_chat_completion(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
        }


class FailingBackend:
    def create_chat_completion(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        raise RuntimeError(f"backend failed for {body['model']}")


@pytest.mark.asyncio
async def test_adapter_maps_streaming_chat_chunks() -> None:
    adapter = OpenAiNnrpAdapter(StreamingBackend())

    events = [
        event
        async for event in adapter.handle_request(
            {
                "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
                "operation": CHAT_COMPLETIONS_CREATE,
                "body": {
                    "model": "llama",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            }
        )
    ]

    assert events[0]["type"] == "response.output_text.delta"
    assert events[0]["delta"] == "hello"
    assert events[1]["type"] == "response.usage"


@pytest.mark.asyncio
async def test_adapter_maps_streaming_cancellation_policy() -> None:
    adapter = OpenAiNnrpAdapter(StreamingBackend())

    events = [
        event
        async for event in adapter.handle_request(
            {
                "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
                "operation": CHAT_COMPLETIONS_CREATE,
                "body": {
                    "model": "llama",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
                "nnrp": {"cancel_after_events": 1},
            }
        )
    ]

    assert [event["type"] for event in events] == ["response.output_text.delta", "response.cancelled"]


@pytest.mark.asyncio
async def test_adapter_maps_non_streaming_chat_body() -> None:
    adapter = OpenAiNnrpAdapter(NonStreamingBackend())

    events = [
        event
        async for event in adapter.handle_request(
            {
                "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
                "operation": CHAT_COMPLETIONS_CREATE,
                "body": {
                    "model": "llama",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            }
        )
    ]

    assert events == [
        {
            "type": "response.completed",
            "body": {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
            },
        }
    ]


@pytest.mark.asyncio
async def test_adapter_emits_profile_error_for_bad_request() -> None:
    adapter = OpenAiNnrpAdapter(NonStreamingBackend())

    events = [
        event
        async for event in adapter.handle_request(
            {
                "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
                "operation": CHAT_COMPLETIONS_CREATE,
                "body": {"model": "llama"},
            }
        )
    ]

    assert events[0]["type"] == "response.error"
    assert events[0]["error"]["code"] == "missing_messages"


@pytest.mark.asyncio
async def test_adapter_maps_backend_error() -> None:
    adapter = OpenAiNnrpAdapter(FailingBackend())

    events = [
        event
        async for event in adapter.handle_request(
            {
                "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
                "operation": CHAT_COMPLETIONS_CREATE,
                "body": {
                    "model": "llama",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            }
        )
    ]

    assert events[0]["type"] == "response.error"
    assert events[0]["error"]["code"] == "backend_error"


def test_stream_chunk_mapper_preserves_tool_call_delta() -> None:
    events = map_openai_stream_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                        ]
                    },
                }
            ]
        }
    )

    assert events[0]["type"] == "response.tool_call.delta"
    assert events[0]["tool_call"]["id"] == "call-1"


def test_stream_chunk_mapper_ignores_unknown_choice_shape() -> None:
    assert map_openai_stream_chunk({"choices": [{"delta": None}]}) == []
    assert map_openai_stream_chunk({"choices": "bad"}) == []


@pytest.mark.asyncio
async def test_vllm_backend_probes_supported_method() -> None:
    class ServingChat:
        def create_chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
            return {"echo": body["model"]}

    backend = VllmBackend(ServingChat())

    assert await backend.create_chat_completion({"model": "llama"}) == {"echo": "llama"}


@pytest.mark.asyncio
async def test_vllm_backend_uses_request_factory_and_raw_request_fallback() -> None:
    class ServingChat:
        async def create_chat_completion(self, request: object, raw_request: object | None = None) -> dict[str, Any]:
            assert raw_request is None
            return {"request": request}

    backend = VllmBackend(ServingChat(), request_factory=lambda body: ("request", body["model"]))

    assert await backend.create_chat_completion({"model": "llama"}) == {"request": ("request", "llama")}


def test_vllm_backend_rejects_unknown_serving_object() -> None:
    with pytest.raises(TypeError):
        VllmBackend(object())
