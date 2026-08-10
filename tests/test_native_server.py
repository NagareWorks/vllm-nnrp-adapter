from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import pytest
from nnrp import NativeWouldBlockError
from nnrp.core import FrameSubmitMetadata, InputProfile, PayloadKind, ResultClass, ResultPushMetadata
from nnrp.native import FFI_STATUS_WOULD_BLOCK, NativeStatus
from nnrp.runtime import PartialResultMetadata
from nnrp.server import NativeServerAcceptOptions, NativeServerBootstrapOptions, NativeServerProviderRoute

from vllm_nnrp_adapter import NnrpServerConfig, OpenAiNnrpAdapter, serve


class StreamingBackend:
    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        async def events() -> AsyncIterator[Mapping[str, Any]]:
            yield {"choices": [{"index": 0, "delta": {"content": body["model"]}}]}
            yield {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        return events()


@dataclass
class FakeOperation:
    operation_id: int
    frame_id: int
    body: bytes
    metadata: FrameSubmitMetadata
    terminal_results: list[tuple[ResultPushMetadata, bytes]]

    def send_result(self, metadata: ResultPushMetadata, body: bytes = b"") -> None:
        self.terminal_results.append((metadata, body))


class FakeSession:
    def __init__(self, operation: FakeOperation, stop_event: asyncio.Event) -> None:
        self.active_transport_name = "ipc"
        self._operation = operation
        self._stop_event = stop_event
        self._delivered = False
        self.partial_results: list[tuple[PartialResultMetadata, bytes]] = []
        self.closed = False

    def receive_submit(self, *, timeout_ms: int = 0, max_events: int = 1) -> FakeOperation:
        assert timeout_ms == 10
        assert max_events == 1
        if not self._delivered:
            self._delivered = True
            return self._operation
        self._stop_event.set()
        raise NativeWouldBlockError(NativeStatus(FFI_STATUS_WOULD_BLOCK))

    def send_partial_result(self, metadata: PartialResultMetadata, body: bytes = b"") -> None:
        self.partial_results.append((metadata, body))

    def close(self) -> None:
        self.closed = True


class FakeServer:
    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self._accepted = False
        self.closed = False

    def accept(self, options: NativeServerAcceptOptions | None = None) -> FakeSession:
        assert options is not None
        assert options.timeout_ms == 10
        if self._accepted:
            raise NativeWouldBlockError(NativeStatus(FFI_STATUS_WOULD_BLOCK))
        self._accepted = True
        return self._session


class FakeServerContext:
    def __init__(self, server: FakeServer) -> None:
        self.server = server
        self.exited = False

    def __enter__(self) -> FakeServer:
        return self.server

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exited = True


@pytest.mark.asyncio
async def test_native_server_emits_ordered_partial_results_and_one_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    operation = FakeOperation(
        operation_id=71,
        frame_id=19,
        body=json.dumps(_chat_request()).encode("utf-8"),
        metadata=_submit_metadata(operation_id=71),
        terminal_results=[],
    )
    session = FakeSession(operation, stop_event)
    server_context = FakeServerContext(FakeServer(session))
    captured_options: list[NativeServerBootstrapOptions] = []

    def fake_listen(options: NativeServerBootstrapOptions) -> FakeServerContext:
        captured_options.append(options)
        return server_context

    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", fake_listen)
    config = NnrpServerConfig(
        endpoint="nnrp://runtime.local/vllm",
        provider_routes={"ipc": NativeServerProviderRoute(provider_endpoint="npipe://nnrp-vllm")},
        accept_timeout_ms=10,
        receive_timeout_ms=10,
        max_active_sessions=1,
        max_operations_per_session=1,
        native_worker_count=2,
    )

    statistics = await serve(OpenAiNnrpAdapter(StreamingBackend()), config=config, stop_event=stop_event)

    assert statistics.accepted_sessions == 1
    assert statistics.accepted_operations == 1
    assert statistics.partial_results == 2
    assert statistics.terminal_results == 1
    assert [metadata.result_sequence for metadata, _body in session.partial_results] == [1, 2]
    assert [json.loads(body)["type"] for _metadata, body in session.partial_results] == [
        "response.output_text.delta",
        "response.usage",
    ]
    assert len(operation.terminal_results) == 1
    terminal_metadata, terminal_body = operation.terminal_results[0]
    assert terminal_metadata.result_class is ResultClass.COMPLETE
    assert terminal_metadata.payload_kind_bitmap is PayloadKind.STRUCTURED_EVENT
    assert json.loads(terminal_body)["type"] == "response.completed"
    assert captured_options[0].endpoint.uri == "nnrp://runtime.local/vllm"
    assert captured_options[0].provider_routes["ipc"].provider_endpoint == "npipe://nnrp-vllm"
    assert session.closed is True
    assert server_context.exited is True


@pytest.mark.parametrize("endpoint", ["tcp://127.0.0.1:7766", "unix:///tmp/nnrp.sock", "ws://host/nnrp"])
def test_server_config_rejects_provider_locator_as_application_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError, match="nnrp:// or nnrps://"):
        NnrpServerConfig(endpoint=endpoint)


@pytest.mark.asyncio
async def test_invalid_submit_body_produces_one_terminal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stop_event = asyncio.Event()
    operation = FakeOperation(
        operation_id=72,
        frame_id=20,
        body=b"not-json",
        metadata=_submit_metadata(operation_id=72),
        terminal_results=[],
    )
    session = FakeSession(operation, stop_event)
    server_context = FakeServerContext(FakeServer(session))
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        lambda _options: server_context,
    )

    statistics = await serve(
        OpenAiNnrpAdapter(StreamingBackend()),
        config=NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            accept_timeout_ms=10,
            receive_timeout_ms=10,
            max_active_sessions=1,
            max_operations_per_session=1,
            native_worker_count=2,
        ),
        stop_event=stop_event,
    )

    assert statistics.terminal_results == 1
    assert len(operation.terminal_results) == 1
    metadata, body = operation.terminal_results[0]
    assert metadata.status_code == 500
    assert json.loads(body)["error"]["code"] == "invalid_submit_body"


def _chat_request() -> dict[str, Any]:
    return {
        "schema_version": "openai-compatible/1",
        "operation": "chat.completions.create",
        "body": {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    }


def _submit_metadata(*, operation_id: int) -> FrameSubmitMetadata:
    return FrameSubmitMetadata(
        src_width=0,
        src_height=0,
        tile_width=0,
        tile_height=0,
        tile_count=0,
        section_count=0,
        frame_class=0,
        input_profile=InputProfile.UNSPECIFIED,
        tile_index_mode=0,
        reserved0=0,
        latency_budget_ms=0,
        target_fps_x100=0,
        retry_of_frame=0,
        tile_base_id=0,
        camera_bytes=0,
        tile_index_bytes=0,
        operation_id=operation_id,
        payload_kind_bitmap=PayloadKind.STRUCTURED_EVENT,
        payload_frame_count=1,
    )
