import asyncio
import sys
from collections.abc import AsyncIterator, Mapping
from types import ModuleType
from typing import Any

import pytest

from vllm_nnrp_adapter import OpenAiNnrpAdapter
from vllm_nnrp_adapter.adapter import (
    _AsyncIteratorCloseGuard,
    _close_async_iterator,
    classify_backend_error,
    map_openai_stream_chunk,
)
from vllm_nnrp_adapter.http_sse_smoke import HttpSseSmokeBackend
from vllm_nnrp_adapter.profile import CHAT_COMPLETIONS_CREATE, OPENAI_COMPATIBLE_SCHEMA_VERSION
from vllm_nnrp_adapter.vllm_backend import EngineDirectChatStream, VllmBackend, VllmProductionBoundaryError
from vllm_nnrp_adapter.vllm_compat import VLLM_COMPATIBILITY_BINDINGS, VllmEngineDirectBinding


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


class SlowBackend:
    async def create_chat_completion(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        await asyncio.sleep(0.05)
        return {"id": "slow"}


class OverloadedBackend:
    def create_chat_completion(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        raise RuntimeError("backend overload: too many queued requests")


class PydanticLikeCompletion:
    def model_dump(self, *, mode: str = "python") -> Mapping[str, Any]:
        assert mode == "json"
        return {"id": "chatcmpl-model", "choices": [{"message": {"content": "hello"}}]}


class PydanticLikeErrorResponse:
    def model_dump(self, *, mode: str = "python") -> Mapping[str, Any]:
        assert mode == "json"
        return {
            "error": {
                "message": "scheduler full",
                "type": "SchedulerRejected",
                "param": "priority",
                "code": 503,
            }
        }


class FakeRequestOutput:
    def __init__(self, *, outputs: list[object], finished: bool = False) -> None:
        self.request_id = "chatcmpl-nnrp-test"
        self.prompt_token_ids = [1, 2, 3]
        self.encoder_prompt_token_ids = None
        self.outputs = outputs
        self.finished = finished


class FakeCompletionOutput:
    def __init__(
        self,
        *,
        text: str,
        token_ids: list[int],
        finish_reason: str | None = None,
    ) -> None:
        self.index = 0
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason
        self.stop_reason = None


class FakeSamplingRequest:
    stream = True
    use_beam_search = False
    request_id = "test"
    priority = 0
    max_completion_tokens = None
    max_tokens = 16
    truncate_prompt_tokens = 7
    include_reasoning = False

    def to_sampling_params(self, max_tokens: int, default_sampling_params: Mapping[str, Any]) -> dict[str, Any]:
        return {"max_tokens": max_tokens, **default_sampling_params}


class FakeModelConfig:
    max_model_len = 1024


class FakeModels:
    def model_name(self, lora_request: object) -> str:
        assert lora_request is None
        return "llama"


class FakeEngineClient:
    def __init__(self) -> None:
        self.aborted: list[str] = []
        self.last_generate_kwargs: dict[str, object] = {}

    def generate(
        self,
        engine_prompt: object,
        sampling_params: object,
        request_id: str,
        **kwargs: object,
    ) -> AsyncIterator[FakeRequestOutput]:
        assert engine_prompt == {"prompt_token_ids": [1, 2, 3]}
        assert sampling_params == {"max_tokens": 16, "temperature": 0.2}
        assert kwargs["priority"] == 0
        self.last_generate_kwargs = dict(kwargs)

        async def outputs() -> AsyncIterator[FakeRequestOutput]:
            yield FakeRequestOutput(outputs=[FakeCompletionOutput(text="hello", token_ids=[10])])
            yield FakeRequestOutput(
                outputs=[FakeCompletionOutput(text=" world", token_ids=[11], finish_reason="stop")],
                finished=True,
            )

        return outputs()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class FakeDirectServingChat:
    model_config = FakeModelConfig()
    default_sampling_params = {"temperature": 0.2}
    override_max_tokens = None
    reasoning_parser_cls = None
    parser_cls = None
    use_harmony = False
    models = FakeModels()

    def __init__(self) -> None:
        self.engine_client = FakeEngineClient()
        self.fallback_calls = 0

    async def render_chat_request(
        self,
        request: FakeSamplingRequest,
    ) -> tuple[list[dict[str, str]], list[dict[str, list[int]]]]:
        return ([{"role": "user", "content": "hello"}], [{"prompt_token_ids": [1, 2, 3]}])

    def _extract_prompt_components(self, engine_prompt: Mapping[str, list[int]]) -> object:
        return object()

    def _extract_prompt_len(self, engine_prompt: Mapping[str, list[int]]) -> int:
        return len(engine_prompt["prompt_token_ids"])

    def create_chat_completion(self, request: object, raw_request: object | None = None) -> Mapping[str, Any]:
        self.fallback_calls += 1
        return {"choices": [{"message": {"content": "fallback"}}]}


class FakeErrorServingChat(FakeDirectServingChat):
    async def render_chat_request(self, request: FakeSamplingRequest) -> object:
        error = type(
            "ErrorInfo",
            (),
            {
                "message": "The requested model is not available.",
                "type": "NotFoundError",
                "param": "model",
                "code": 404,
            },
        )()
        return type("ErrorResponse", (), {"error": error})()


class FakeBoundRequestFactory:
    def __init__(self, binding: VllmEngineDirectBinding | None = None) -> None:
        self.engine_direct_binding = binding or VLLM_COMPATIBILITY_BINDINGS[0].engine_direct

    def __call__(self, body: Mapping[str, Any]) -> FakeSamplingRequest:
        return FakeSamplingRequest()


class FakeNoUsageRequestOutput(FakeRequestOutput):
    def __init__(self) -> None:
        super().__init__(outputs=[FakeCompletionOutput(text="", token_ids=[])], finished=False)
        self.prompt_token_ids = None
        self.encoder_prompt_token_ids = [4, 5]


class ClosableStreamingBackend:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def create_chat_completion(self, body: Mapping[str, Any]) -> AsyncIterator[Mapping[str, Any]]:
        backend = self

        class Chunks:
            def __aiter__(self) -> "Chunks":
                return self

            async def __anext__(self) -> Mapping[str, Any]:
                if backend.closed:
                    raise StopAsyncIteration
                return {"choices": [{"index": 0, "delta": {"content": "hello"}}]}

            async def aclose(self) -> None:
                backend.close_calls += 1
                backend.closed = True

        return Chunks()


class SlowClosableStreamingBackend:
    def __init__(self) -> None:
        self.close_calls = 0

    def create_chat_completion(self, body: Mapping[str, Any]) -> AsyncIterator[Mapping[str, Any]]:
        backend = self

        class Chunks:
            def __aiter__(self) -> AsyncIterator[Mapping[str, Any]]:
                return self

            async def __anext__(self) -> Mapping[str, Any]:
                await asyncio.sleep(0.05)
                return {"choices": [{"index": 0, "delta": {"content": "late"}}]}

            async def aclose(self) -> bool:
                backend.close_calls += 1
                return True

        return Chunks()


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
    assert events[2]["type"] == "response.completed"


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
async def test_adapter_closes_stream_when_cancellation_policy_fires() -> None:
    backend = ClosableStreamingBackend()
    adapter = OpenAiNnrpAdapter(backend)

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

    assert backend.closed is True
    assert backend.close_calls == 1
    assert events[-1]["type"] == "response.cancelled"


@pytest.mark.asyncio
async def test_adapter_closes_timed_out_stream_once() -> None:
    backend = SlowClosableStreamingBackend()
    adapter = OpenAiNnrpAdapter(backend)

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
                "nnrp": {"timeout_ms": 1},
            }
        )
    ]

    assert backend.close_calls == 1
    assert events[-1]["type"] == "response.error"
    assert events[-1]["error"]["code"] == "request_timeout"


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
async def test_adapter_passes_profile_request_id_to_backend_request() -> None:
    class RequestIdBackend:
        def __init__(self) -> None:
            self.request_id: str | None = None

        def create_chat_completion(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
            self.request_id = str(body["request_id"])
            return {"choices": []}

    backend = RequestIdBackend()
    adapter = OpenAiNnrpAdapter(backend)

    _events = [
        event
        async for event in adapter.handle_request(
            {
                "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
                "operation": CHAT_COMPLETIONS_CREATE,
                "request_id": "request-profile-1",
                "body": {"model": "llama", "messages": [{"role": "user", "content": "hello"}]},
            }
        )
    ]

    assert backend.request_id == "request-profile-1"


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
    assert events[0]["diagnostics"]["backend_error_family"] == "RuntimeError"


@pytest.mark.asyncio
async def test_adapter_maps_backend_overload_family() -> None:
    adapter = OpenAiNnrpAdapter(OverloadedBackend())

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

    assert events[0]["error"]["code"] == "backend_overload"


@pytest.mark.asyncio
async def test_adapter_emits_request_diagnostics_when_requested() -> None:
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
                "nnrp": {"diagnostics": True},
            }
        )
    ]

    assert events[0]["type"] == "response.diagnostics"
    assert events[0]["diagnostics"]["selected_model"] == "llama"
    assert events[1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_adapter_maps_timeout_policy() -> None:
    adapter = OpenAiNnrpAdapter(SlowBackend())

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
                "nnrp": {"timeout_ms": 1},
            }
        )
    ]

    assert events[0]["type"] == "response.error"
    assert events[0]["error"]["code"] == "request_timeout"


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


def test_backend_error_classifier_maps_scheduler_rejections_and_cancellation() -> None:
    assert classify_backend_error(RuntimeError("scheduler full: reject request")) == (
        "server_error",
        "scheduler_rejected",
    )
    assert classify_backend_error(RuntimeError("request cancelled by backend")) == (
        "server_error",
        "backend_cancelled",
    )


@pytest.mark.parametrize(
    ("status", "backend_type", "expected"),
    [
        (400, "BadRequestError", ("invalid_request_error", "invalid_backend_request")),
        (422, "ValidationError", ("invalid_request_error", "invalid_backend_request")),
        (429, "RateLimitError", ("server_error", "backend_overload")),
        (503, "ServiceUnavailable", ("server_error", "backend_overload")),
    ],
)
def test_backend_error_classifier_maps_structured_status(
    status: int,
    backend_type: str,
    expected: tuple[str, str],
) -> None:
    error = RuntimeError("structured backend failure")
    error.vllm_status_code = status  # type: ignore[attr-defined]
    error.vllm_error_type = backend_type  # type: ignore[attr-defined]

    assert classify_backend_error(error) == expected


@pytest.mark.asyncio
async def test_vllm_backend_probes_supported_method() -> None:
    class ServingChat:
        def create_chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
            return {"echo": body["model"]}

    backend = VllmBackend(ServingChat())

    assert await backend.create_chat_completion({"model": "llama"}) == {"echo": "llama"}


@pytest.mark.asyncio
async def test_vllm_backend_normalizes_structured_non_streaming_error() -> None:
    class ServingChat:
        def create_chat_completion(self, body: dict[str, Any]) -> PydanticLikeErrorResponse:
            return PydanticLikeErrorResponse()

    events = [
        event
        async for event in OpenAiNnrpAdapter(VllmBackend(ServingChat())).handle_request(
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

    assert events[0]["error"] == {
        "type": "server_error",
        "code": "scheduler_rejected",
        "message": "scheduler full",
    }
    assert events[0]["diagnostics"] == {
        "backend_error_family": "_VllmBackendResponseError",
        "vllm_error_type": "SchedulerRejected",
        "vllm_status_code": 503,
        "vllm_parameter": "priority",
    }


@pytest.mark.asyncio
async def test_vllm_backend_uses_request_factory_and_raw_request_fallback() -> None:
    class ServingChat:
        async def create_chat_completion(self, request: object, raw_request: object | None = None) -> dict[str, Any]:
            assert raw_request is None
            return {"request": request}

    backend = VllmBackend(ServingChat(), request_factory=lambda body: ("request", body["model"]))

    assert await backend.create_chat_completion({"model": "llama"}) == {"request": ("request", "llama")}


@pytest.mark.asyncio
async def test_explicit_smoke_backend_normalizes_in_process_sse_stream() -> None:
    class ServingChat:
        async def create_chat_completion(
            self,
            body: Mapping[str, Any],
        ) -> AsyncIterator[str]:
            assert body["model"] == "llama"

            async def chunks() -> AsyncIterator[str]:
                yield 'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n'
                yield "data: [DONE]\n\n"

            return chunks()

    backend = HttpSseSmokeBackend(ServingChat())
    adapter = OpenAiNnrpAdapter(backend)

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

    assert [event["type"] for event in events] == ["response.output_text.delta", "response.completed"]
    assert events[0]["delta"] == "hello"
    assert adapter.capabilities.operations[0]["tool_calls"] is True


@pytest.mark.asyncio
async def test_production_backend_rejects_http_sse_stream() -> None:
    class ServingChat:
        async def create_chat_completion(self, request: object) -> AsyncIterator[str]:
            async def chunks() -> AsyncIterator[str]:
                yield 'data: {"choices":[]}\n\n'

            return chunks()

    backend = VllmBackend(ServingChat())

    with pytest.raises(VllmProductionBoundaryError, match="HTTP/SSE-shaped stream"):
        await backend.create_chat_completion({"model": "llama"})


@pytest.mark.asyncio
async def test_vllm_backend_normalizes_pydantic_like_completion() -> None:
    class ServingChat:
        async def create_chat_completion(
            self,
            request: object,
            raw_request: object | None = None,
        ) -> PydanticLikeCompletion:
            return PydanticLikeCompletion()

    backend = VllmBackend(ServingChat())

    assert await backend.create_chat_completion({"model": "llama"}) == {
        "id": "chatcmpl-model",
        "choices": [{"message": {"content": "hello"}}],
    }


@pytest.mark.asyncio
async def test_vllm_backend_prefers_engine_direct_stream_without_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.utils",
        _module_with_get_max_tokens(),
    )
    serving_chat = FakeDirectServingChat()
    backend = VllmBackend(serving_chat, request_factory=FakeBoundRequestFactory())
    adapter = OpenAiNnrpAdapter(backend)

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
                    "stream_options": {"include_usage": True},
                },
            }
        )
    ]

    assert serving_chat.fallback_calls == 0
    assert [event["type"] for event in events] == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.usage",
        "response.completed",
    ]
    assert events[0]["delta"] == "hello"
    assert events[1]["delta"] == " world"
    assert events[2]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


@pytest.mark.asyncio
async def test_vllm_backend_engine_direct_cancel_aborts_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.utils",
        _module_with_get_max_tokens(),
    )
    serving_chat = FakeDirectServingChat()
    backend = VllmBackend(serving_chat, request_factory=FakeBoundRequestFactory())
    adapter = OpenAiNnrpAdapter(backend)

    abort_observations: list[bool | None] = []
    events = [
        event
        async for event in adapter._handle_native_request(
            {
                "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
                "operation": CHAT_COMPLETIONS_CREATE,
                "nnrp": {"cancel_after_events": 1},
                "body": {
                    "model": "llama",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            },
            backend_abort_observer=abort_observations.append,
        )
    ]

    assert [event["type"] for event in events] == ["response.output_text.delta", "response.cancelled"]
    assert serving_chat.engine_client.aborted == ["chatcmpl-nnrp-test"]
    assert abort_observations == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("binding_index", "module_name", "expected_truncate", "expected_parser_kwargs", "expected_reasoning_ended"),
    (
        (0, "vllm.entrypoints.utils", None, False, None),
        (1, "vllm.entrypoints.utils", 7, True, True),
        (2, "vllm.entrypoints.serve.utils.api_utils", 7, True, True),
    ),
)
async def test_engine_direct_binding_matches_each_vllm_family(
    monkeypatch: pytest.MonkeyPatch,
    binding_index: int,
    module_name: str,
    expected_truncate: int | None,
    expected_parser_kwargs: bool,
    expected_reasoning_ended: bool | None,
) -> None:
    binding = VLLM_COMPATIBILITY_BINDINGS[binding_index].engine_direct
    helper_module, calls = _module_with_versioned_get_max_tokens(
        module_name,
        supports_truncate=binding.supports_truncate_prompt_tokens,
    )
    monkeypatch.setitem(sys.modules, module_name, helper_module)
    serving_chat = FakeDirectServingChat()
    backend = VllmBackend(serving_chat, request_factory=FakeBoundRequestFactory(binding))

    stream = await backend.create_chat_completion({"model": "llama", "stream": True})

    assert isinstance(stream, EngineDirectChatStream)
    assert calls == [expected_truncate]
    assert serving_chat.engine_client.last_generate_kwargs["reasoning_ended"] is expected_reasoning_ended
    assert (
        "reasoning_parser_kwargs" in serving_chat.engine_client.last_generate_kwargs
    ) is expected_parser_kwargs
    await stream.aclose()


@pytest.mark.asyncio
async def test_engine_direct_requires_named_compatibility_binding() -> None:
    serving_chat = FakeDirectServingChat()
    backend = VllmBackend(serving_chat, request_factory=lambda body: FakeSamplingRequest())

    with pytest.raises(VllmProductionBoundaryError, match="not supported by the engine-direct backend"):
        await backend.create_chat_completion({"model": "llama", "stream": True})

    assert serving_chat.fallback_calls == 0


@pytest.mark.asyncio
async def test_engine_direct_stream_abort_is_idempotent_and_stops_iteration() -> None:
    engine_client = FakeEngineClient()

    async def outputs() -> AsyncIterator[FakeRequestOutput]:
        yield FakeRequestOutput(outputs=[FakeCompletionOutput(text="late", token_ids=[10])])

    stream = EngineDirectChatStream(
        outputs(),
        engine_client=engine_client,
        request_id="chatcmpl-nnrp-idempotent",
        model_name="llama",
        include_usage=False,
    )

    assert await stream.aclose() is True
    assert await stream.aclose() is True
    assert engine_client.aborted == ["chatcmpl-nnrp-idempotent"]
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_engine_direct_stream_emits_usage_after_terminal_delta_without_recounting_prompt() -> None:
    first = FakeNoUsageRequestOutput()
    terminal = FakeNoUsageRequestOutput()
    terminal.finished = True

    async def outputs() -> AsyncIterator[FakeRequestOutput]:
        yield first
        yield terminal

    stream = EngineDirectChatStream(
        outputs(),
        engine_client=object(),
        request_id="chatcmpl-nnrp-usage",
        model_name="llama",
        include_usage=True,
    )

    assert "usage" not in await stream.__anext__()
    assert "usage" not in await stream.__anext__()
    usage_chunk = await stream.__anext__()
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2}
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_engine_direct_stream_reports_unsupported_abort() -> None:
    async def outputs() -> AsyncIterator[FakeRequestOutput]:
        yield FakeRequestOutput(outputs=[])

    stream = EngineDirectChatStream(
        outputs(),
        engine_client=object(),
        request_id="chatcmpl-nnrp-unsupported",
        model_name="llama",
        include_usage=False,
    )

    assert await stream.aclose() is False
    assert await stream.aclose() is False


@pytest.mark.asyncio
async def test_engine_direct_stream_reports_failed_abort() -> None:
    class FailingAbortClient:
        async def abort(self, request_id: str) -> None:
            raise RuntimeError(f"cannot abort {request_id}")

    async def outputs() -> AsyncIterator[FakeRequestOutput]:
        yield FakeRequestOutput(outputs=[])

    stream = EngineDirectChatStream(
        outputs(),
        engine_client=FailingAbortClient(),
        request_id="chatcmpl-nnrp-failed",
        model_name="llama",
        include_usage=False,
    )

    assert await stream.aclose() is False


@pytest.mark.asyncio
async def test_close_iterator_records_unknown_when_close_is_unavailable() -> None:
    class IteratorWithoutClose:
        def __aiter__(self) -> AsyncIterator[Mapping[str, Any]]:
            return self

        async def __anext__(self) -> Mapping[str, Any]:
            raise StopAsyncIteration

    observations: list[bool | None] = []

    assert await _close_async_iterator(IteratorWithoutClose(), observer=observations.append) is None
    assert observations == [None]


@pytest.mark.asyncio
async def test_close_guard_closes_and_observes_once() -> None:
    backend = SlowClosableStreamingBackend()
    chunks = backend.create_chat_completion({})
    observations: list[bool | None] = []
    guard = _AsyncIteratorCloseGuard(chunks, observer=observations.append)

    assert await guard.close() is True
    assert await guard.close() is True
    assert backend.close_calls == 1
    assert observations == [True]


@pytest.mark.asyncio
async def test_vllm_backend_direct_path_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.utils",
        _module_with_get_max_tokens(),
    )
    serving_chat = FakeDirectServingChat()
    backend = VllmBackend(
        serving_chat,
        request_factory=FakeBoundRequestFactory(),
        prefer_engine_direct=False,
    )

    with pytest.raises(VllmProductionBoundaryError, match="requires the engine-direct backend"):
        await backend.create_chat_completion({"model": "llama", "stream": True})
    assert serving_chat.fallback_calls == 0


@pytest.mark.asyncio
async def test_vllm_backend_rejects_direct_render_error_without_sse_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.utils",
        _module_with_get_max_tokens(),
    )
    serving_chat = FakeErrorServingChat()
    backend = VllmBackend(serving_chat, request_factory=FakeBoundRequestFactory())

    events = [
        event
        async for event in OpenAiNnrpAdapter(backend).handle_request(
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

    assert events[0]["error"] == {
        "type": "invalid_request_error",
        "code": "unsupported_model",
        "message": "The requested model is not available.",
    }
    assert events[0]["diagnostics"]["vllm_error_type"] == "NotFoundError"
    assert events[0]["diagnostics"]["vllm_status_code"] == 404
    assert events[0]["diagnostics"]["vllm_parameter"] == "model"
    assert serving_chat.fallback_calls == 0


@pytest.mark.asyncio
async def test_vllm_backend_rejects_unimplemented_complex_features_without_sse_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.utils",
        _module_with_get_max_tokens(),
    )
    serving_chat = FakeDirectServingChat()
    backend = VllmBackend(serving_chat, request_factory=FakeBoundRequestFactory())

    with pytest.raises(VllmProductionBoundaryError, match="not supported by the engine-direct backend"):
        await backend.create_chat_completion({"model": "llama", "stream": True, "tools": [{"type": "function"}]})
    assert serving_chat.fallback_calls == 0


def test_vllm_backend_rejects_unknown_serving_object() -> None:
    with pytest.raises(TypeError):
        VllmBackend(object())


def _module_with_get_max_tokens() -> ModuleType:
    module = ModuleType("vllm.entrypoints.utils")

    def get_max_tokens(
        max_model_len: int,
        request_max_tokens: int | None,
        prompt_len: int,
        default_sampling_params: Mapping[str, Any],
        override_max_tokens: int | None,
    ) -> int:
        assert max_model_len == 1024
        assert prompt_len == 3
        assert default_sampling_params == {"temperature": 0.2}
        assert override_max_tokens is None
        return request_max_tokens or 32

    module.get_max_tokens = get_max_tokens
    return module


def _module_with_versioned_get_max_tokens(
    module_name: str,
    *,
    supports_truncate: bool,
) -> tuple[ModuleType, list[int | None]]:
    module = ModuleType(module_name)
    calls: list[int | None] = []

    if supports_truncate:

        def get_max_tokens(
            max_model_len: int,
            request_max_tokens: int | None,
            prompt_len: int,
            default_sampling_params: Mapping[str, Any],
            override_max_tokens: int | None,
            *,
            truncate_prompt_tokens: int | None,
        ) -> int:
            calls.append(truncate_prompt_tokens)
            return request_max_tokens or 32

    else:

        def get_max_tokens(  # type: ignore[misc]
            max_model_len: int,
            request_max_tokens: int | None,
            prompt_len: int,
            default_sampling_params: Mapping[str, Any],
            override_max_tokens: int | None,
        ) -> int:
            calls.append(None)
            return request_max_tokens or 32

    module.get_max_tokens = get_max_tokens
    return module, calls
