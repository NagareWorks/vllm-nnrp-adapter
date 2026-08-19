from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any

import pytest
from nnrp import NativeWouldBlockError
from nnrp.core import FrameSubmitMetadata, InputProfile, MessageType, PayloadKind, ResultClass, ResultPushMetadata
from nnrp.native import FFI_STATUS_WOULD_BLOCK, NativeStatus
from nnrp.runtime import (
    ControlRequestMetadata,
    InFlightPolicy,
    NativeRuntimeEvent,
    PartialResultMetadata,
    ProgressMetadata,
    ResultDropReasonCode,
    ResultDropReasonMetadata,
    RuntimeEventMetadata,
    RuntimeEventMetadataKind,
    RuntimeEventTail,
    RuntimeFrameHeader,
    RuntimeRole,
    SchedulingMetadata,
    SessionCloseMetadata,
    SessionCloseReason,
    SupersedeMetadata,
)
from nnrp.server import NativeServerAcceptOptions, NativeServerBootstrapOptions, NativeServerProviderRoute

from vllm_nnrp_adapter import NnrpServerConfig, OpenAiNnrpAdapter, serve


class StreamingBackend:
    def __init__(self) -> None:
        self.requests: list[Mapping[str, Any]] = []

    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        self.requests.append(body)

        async def events() -> AsyncIterator[Mapping[str, Any]]:
            yield {"choices": [{"index": 0, "delta": {"content": body["model"]}}]}
            yield {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        return events()


class InterleavingBackend:
    def __init__(self, expected_operations: int) -> None:
        self._expected_operations = expected_operations
        self._all_started = asyncio.Event()
        self.started: list[str] = []

    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        async def events() -> AsyncIterator[Mapping[str, Any]]:
            model = str(body["model"])
            self.started.append(model)
            if len(self.started) == self._expected_operations:
                self._all_started.set()
            await asyncio.wait_for(self._all_started.wait(), timeout=1)
            yield {"choices": [{"index": 0, "delta": {"content": f"{model}:1"}}]}
            await asyncio.sleep(0)
            yield {"choices": [{"index": 0, "delta": {"content": f"{model}:2"}}]}

        return events()


class CancellableBackend:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.closed = threading.Event()
        self._release = asyncio.Event()

    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        async def events() -> AsyncIterator[Mapping[str, Any]]:
            self.started.set()
            try:
                yield {"choices": [{"index": 0, "delta": {"content": "first"}}]}
                await self._release.wait()
                yield {"choices": [{"index": 0, "delta": {"content": "late"}}]}
            finally:
                self.closed.set()

        return events()


class ReplacementBackend:
    def __init__(self) -> None:
        self.old_started = threading.Event()
        self.old_closed = threading.Event()

    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        async def events() -> AsyncIterator[Mapping[str, Any]]:
            if body["model"] == "old-model":
                self.old_started.set()
                try:
                    yield {"choices": [{"index": 0, "delta": {"content": "old:first"}}]}
                    await asyncio.Event().wait()
                finally:
                    self.old_closed.set()
                return
            yield {"choices": [{"index": 0, "delta": {"content": "new:complete"}}]}

        return events()


@dataclass
class FakeOperation:
    operation_id: int
    frame_id: int
    body: InitVar[bytes]
    metadata: InitVar[FrameSubmitMetadata]
    terminal_results: list[tuple[ResultPushMetadata, bytes]]
    submit: NativeRuntimeEvent = field(init=False)
    native_thread_ids: list[int] = field(default_factory=list)
    partial_results: list[tuple[PartialResultMetadata, bytes]] = field(default_factory=list)
    progress_results: list[tuple[ProgressMetadata, bytes]] = field(default_factory=list)
    result_drops: list[tuple[ResultDropReasonMetadata, bytes]] = field(default_factory=list)
    on_terminal: Callable[[], None] | None = None
    fail_next_terminal_result: bool = False

    def __post_init__(self, body: bytes, metadata: FrameSubmitMetadata) -> None:
        self.submit = NativeRuntimeEvent(
            RuntimeFrameHeader(
                message_type=MessageType.FRAME_SUBMIT,
                session_id=1,
                frame_id=self.frame_id,
            ),
            RuntimeEventMetadata(RuntimeEventMetadataKind.FRAME_SUBMIT, metadata),
            RuntimeEventTail.with_body(body),
        )

    async def send_result(self, metadata: ResultPushMetadata, body: bytes = b"") -> None:
        if self.fail_next_terminal_result:
            self.fail_next_terminal_result = False
            raise OSError("terminal profile result unavailable")
        self.terminal_results.append((metadata, body))
        if self.on_terminal is not None:
            self.on_terminal()

    async def send_partial_result(self, metadata: PartialResultMetadata, body: bytes = b"") -> None:
        self.partial_results.append((metadata, body))

    async def send_progress(self, metadata: ProgressMetadata, body: bytes = b"") -> None:
        self.progress_results.append((metadata, body))

    async def send_result_drop(self, metadata: ResultDropReasonMetadata, diagnostic: bytes = b"") -> None:
        self.result_drops.append((metadata, diagnostic))
        if self.on_terminal is not None:
            self.on_terminal()


@dataclass(frozen=True)
class FakeServerEvent:
    submit: FakeOperation | None = None
    runtime: NativeRuntimeEvent | None = None
    wait_for_backend: bool = False

    def as_submit(self) -> FakeOperation | None:
        return self.submit

    def as_runtime(self) -> NativeRuntimeEvent | None:
        return self.runtime

    def as_lifecycle(self) -> None:
        return None


class FakeSession:
    def __init__(self, operation: FakeOperation, stop_event: asyncio.Event) -> None:
        self.active_transport_name = "ipc"
        self._operation = operation
        self._stop_event = stop_event
        self._delivered = False
        self.partial_results = operation.partial_results
        self.closed = False

    def poll_events(self, *, timeout_ms: int = 0, max_events: int = 1) -> tuple[FakeServerEvent, ...]:
        self._operation.native_thread_ids.append(threading.get_ident())
        assert timeout_ms == 10
        assert max_events == 1
        if not self._delivered:
            self._delivered = True
            return (FakeServerEvent(submit=self._operation),)
        self._stop_event.set()
        return ()

    def close(self) -> None:
        self._operation.native_thread_ids.append(threading.get_ident())
        self.closed = True


class MultiOperationFakeSession:
    def __init__(
        self,
        operations: list[FakeOperation],
        *,
        operation_accepted: Callable[[], None],
    ) -> None:
        self.active_transport_name = "ipc"
        self._pending = list(operations)
        self._operation_accepted = operation_accepted
        self.partial_results: list[tuple[PartialResultMetadata, bytes]] = []
        for operation in operations:
            operation.partial_results = self.partial_results
        self.closed = False

    def poll_events(self, *, timeout_ms: int = 0, max_events: int = 1) -> tuple[FakeServerEvent, ...]:
        assert timeout_ms == 10
        assert max_events == 1
        if self._pending:
            operation = self._pending.pop(0)
            operation.native_thread_ids.append(threading.get_ident())
            self._operation_accepted()
            return (FakeServerEvent(submit=operation),)
        return ()

    def close(self) -> None:
        self.closed = True


class ScriptedEventSession:
    def __init__(
        self,
        events: list[FakeServerEvent],
        *,
        operation: FakeOperation,
        backend_started: threading.Event,
        stop_event: asyncio.Event,
        stop_after_last_event: bool = False,
    ) -> None:
        self.active_transport_name = "ipc"
        self._events = list(events)
        self._operation = operation
        self._backend_started = backend_started
        self._stop_event = stop_event
        self._loop = asyncio.get_running_loop()
        self._stop_after_last_event = stop_after_last_event
        self.closed = False

    def poll_events(self, *, timeout_ms: int = 0, max_events: int = 1) -> tuple[FakeServerEvent, ...]:
        assert timeout_ms == 10
        assert max_events == 1
        if self._events:
            event = self._events.pop(0)
            if event.wait_for_backend:
                assert self._backend_started.wait(timeout=1)
            if not self._events and self._stop_after_last_event:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            return (event,)
        return ()

    def close(self) -> None:
        self.closed = True


class FakeServer:
    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self._accepted = False
        self.closed = False

    async def accept(self, options: NativeServerAcceptOptions | None = None) -> FakeSession:
        assert options is not None
        assert options.timeout_ms == 10
        if self._accepted:
            raise NativeWouldBlockError(NativeStatus(FFI_STATUS_WOULD_BLOCK))
        self._accepted = True
        return self._session


class MultiSessionFakeServer:
    def __init__(self, sessions: list[Any]) -> None:
        self._pending = list(sessions)

    async def accept(self, options: NativeServerAcceptOptions | None = None) -> Any:
        assert options is not None
        assert options.timeout_ms == 10
        if self._pending:
            return self._pending.pop(0)
        raise NativeWouldBlockError(NativeStatus(FFI_STATUS_WOULD_BLOCK))


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

    backend = StreamingBackend()
    statistics = await serve(OpenAiNnrpAdapter(backend), config=config, stop_event=stop_event)

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
    assert [metadata.progress_sequence for metadata, _body in operation.progress_results] == list(range(1, 9))
    assert [metadata.stage_code for metadata, _body in operation.progress_results] == [
        0x0001,
        0x0003,
        0x0002,
        0x0004,
        0x0005,
        0x0007,
        0x0008,
        0x0009,
    ]
    terminal_metadata, terminal_body = operation.terminal_results[0]
    assert terminal_metadata.result_class is ResultClass.COMPLETE
    assert terminal_metadata.payload_kind_bitmap is PayloadKind.STRUCTURED_EVENT
    assert json.loads(terminal_body)["type"] == "response.completed"
    assert captured_options[0].endpoint.uri == "nnrp://runtime.local/vllm"
    assert captured_options[0].provider_routes["ipc"].provider_endpoint == "npipe://nnrp-vllm"
    assert session.closed is True
    assert server_context.exited is True
    assert operation.native_thread_ids
    assert threading.get_ident() not in operation.native_thread_ids
    assert backend.requests[0]["request_id"] == "nnrp-71-19"


@pytest.mark.parametrize("endpoint", ["tcp://127.0.0.1:7766", "unix:///tmp/nnrp.sock", "ws://host/nnrp"])
def test_server_config_rejects_provider_locator_as_application_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError, match="nnrp:// or nnrps://"):
        NnrpServerConfig(endpoint=endpoint)


@pytest.mark.asyncio
async def test_production_entrypoint_rejects_preview3_packet_session() -> None:
    class Preview3PacketSession:
        pass

    with pytest.raises(TypeError, match="OpenAiNnrpAdapter Preview4"):
        await serve(  # type: ignore[arg-type]
            Preview3PacketSession(),
            config=NnrpServerConfig(endpoint="nnrp://runtime.local/vllm"),
        )


@pytest.mark.asyncio
async def test_production_entrypoint_rejects_preview3_runtime_config() -> None:
    class Preview3RuntimeConfig:
        pass

    with pytest.raises(TypeError, match="NnrpServerConfig Preview4"):
        await serve(  # type: ignore[arg-type]
            OpenAiNnrpAdapter(StreamingBackend()),
            config=Preview3RuntimeConfig(),
        )


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
    assert [metadata.stage_code for metadata, _body in operation.progress_results] == [0x0001, 0x0003, 0x0008, 0x000B]


@pytest.mark.asyncio
async def test_native_server_runs_sessions_and_operations_concurrently_with_per_operation_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    operations = [
        FakeOperation(
            operation_id=operation_id,
            frame_id=operation_id + 100,
            body=json.dumps(_chat_request(model=f"model-{operation_id}")).encode("utf-8"),
            metadata=_submit_metadata(operation_id=operation_id),
            terminal_results=[],
        )
        for operation_id in range(1, 5)
    ]
    completed_operations = 0

    def operation_completed() -> None:
        nonlocal completed_operations
        completed_operations += 1
        if completed_operations == len(operations):
            stop_event.set()

    for operation in operations:
        operation.on_terminal = operation_completed

    sessions = [
        MultiOperationFakeSession(operations[:2], operation_accepted=lambda: None),
        MultiOperationFakeSession(operations[2:], operation_accepted=lambda: None),
    ]
    server_context = FakeServerContext(MultiSessionFakeServer(sessions))
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        lambda _options: server_context,
    )
    backend = InterleavingBackend(expected_operations=len(operations))

    statistics = await serve(
        OpenAiNnrpAdapter(backend),
        config=NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            accept_timeout_ms=10,
            receive_timeout_ms=10,
            max_active_sessions=2,
            max_operations_per_session=2,
            native_worker_count=8,
        ),
        stop_event=stop_event,
    )

    assert statistics.accepted_sessions == 2
    assert statistics.accepted_operations == 4
    assert statistics.partial_results == 8
    assert statistics.terminal_results == 4
    assert set(backend.started) == {f"model-{operation_id}" for operation_id in range(1, 5)}
    for session in sessions:
        results_by_operation: dict[int, list[tuple[PartialResultMetadata, bytes]]] = {}
        for metadata, body in session.partial_results:
            results_by_operation.setdefault(metadata.operation_id, []).append((metadata, body))
        for operation_id, results in results_by_operation.items():
            assert [metadata.result_sequence for metadata, _body in results] == [1, 2]
            assert [json.loads(body)["delta"] for _metadata, body in results] == [
                f"model-{operation_id}:1",
                f"model-{operation_id}:2",
            ]
        assert session.closed is True
    assert all(len(operation.terminal_results) == 1 for operation in operations)


@pytest.mark.asyncio
async def test_native_server_rejects_duplicate_operation_without_corrupting_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    operations = [
        FakeOperation(
            operation_id=91,
            frame_id=frame_id,
            body=json.dumps(_chat_request()).encode("utf-8"),
            metadata=_submit_metadata(operation_id=91),
            terminal_results=[],
        )
        for frame_id in (191, 192)
    ]
    received = 0

    def operation_received() -> None:
        nonlocal received
        received += 1
        if received == len(operations):
            loop.call_soon_threadsafe(stop_event.set)

    session = MultiOperationFakeSession(operations, operation_accepted=operation_received)
    server_context = FakeServerContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", lambda _options: server_context)

    statistics = await serve(
        OpenAiNnrpAdapter(StreamingBackend()),
        config=NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            accept_timeout_ms=10,
            receive_timeout_ms=10,
            max_active_sessions=1,
            max_operations_per_session=2,
            native_worker_count=4,
        ),
        stop_event=stop_event,
    )

    assert statistics.accepted_operations == 1
    assert statistics.terminal_results == 2
    assert json.loads(operations[0].terminal_results[0][1])["type"] == "response.completed"
    duplicate_event = json.loads(operations[1].terminal_results[0][1])
    assert duplicate_event["type"] == "response.error"
    assert duplicate_event["error"]["code"] == "duplicate_operation_id"


@pytest.mark.parametrize(("message_type", "expect_drop"), [(MessageType.CANCEL, False), (MessageType.ABORT, True)])
@pytest.mark.asyncio
async def test_native_control_stops_backend_and_emits_one_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
    message_type: MessageType,
    expect_drop: bool,
) -> None:
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=101,
        frame_id=201,
        body=json.dumps(_chat_request()).encode("utf-8"),
        metadata=_submit_metadata(operation_id=101),
        terminal_results=[],
    )
    control = _control_event(message_type, operation_id=101, sequence=1)
    operation.on_terminal = stop_event.set
    session = ScriptedEventSession(
        [FakeServerEvent(submit=operation), FakeServerEvent(runtime=control, wait_for_backend=True)],
        operation=operation,
        backend_started=backend.started,
        stop_event=stop_event,
    )
    server_context = FakeServerContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", lambda _options: server_context)

    statistics = await serve(
        OpenAiNnrpAdapter(backend),
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

    assert statistics.accepted_operations == 1
    assert statistics.terminal_results == 1
    assert backend.closed.is_set()
    assert [json.loads(body)["delta"] for _metadata, body in operation.partial_results] == ["first"]
    if expect_drop:
        assert operation.terminal_results == []
        assert len(operation.result_drops) == 1
        assert operation.result_drops[0][0].operation_id == 101
    else:
        assert operation.result_drops == []
        assert len(operation.terminal_results) == 1
        assert json.loads(operation.terminal_results[0][1]) == {
            "reason": "peer_cancelled",
            "type": "response.cancelled",
        }
    assert operation.progress_results[-1][0].stage_code == 0x000A


@pytest.mark.asyncio
async def test_native_cancel_falls_back_to_typed_drop_when_profile_terminal_cannot_be_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=107,
        frame_id=207,
        body=json.dumps(_chat_request()).encode("utf-8"),
        metadata=_submit_metadata(operation_id=107),
        terminal_results=[],
        on_terminal=stop_event.set,
        fail_next_terminal_result=True,
    )
    session = ScriptedEventSession(
        [
            FakeServerEvent(submit=operation),
            FakeServerEvent(
                runtime=_control_event(MessageType.CANCEL, operation_id=107, sequence=1),
                wait_for_backend=True,
            ),
        ],
        operation=operation,
        backend_started=backend.started,
        stop_event=stop_event,
    )
    server_context = FakeServerContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", lambda _options: server_context)

    statistics = await serve(
        OpenAiNnrpAdapter(backend),
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
    assert backend.closed.is_set()
    assert operation.terminal_results == []
    assert len(operation.result_drops) == 1
    metadata, diagnostic = operation.result_drops[0]
    assert metadata.drop_reason_code is ResultDropReasonCode.PEER_CANCELLED
    assert diagnostic == b"obsolete"


@pytest.mark.asyncio
async def test_server_shutdown_drops_active_operation_before_closing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=102,
        frame_id=202,
        body=json.dumps(_chat_request()).encode("utf-8"),
        metadata=_submit_metadata(operation_id=102),
        terminal_results=[],
    )
    session = ScriptedEventSession(
        [FakeServerEvent(submit=operation)],
        operation=operation,
        backend_started=backend.started,
        stop_event=stop_event,
    )
    server_context = FakeServerContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", lambda _options: server_context)

    async def request_shutdown() -> None:
        await asyncio.to_thread(backend.started.wait, 1)
        stop_event.set()

    shutdown_task = asyncio.create_task(request_shutdown())

    try:
        statistics = await serve(
            OpenAiNnrpAdapter(backend),
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
    finally:
        await shutdown_task

    assert statistics.terminal_results == 1
    assert backend.closed.is_set()
    assert len(operation.result_drops) == 1
    assert operation.result_drops[0][0].drop_reason_code is ResultDropReasonCode.TRANSPORT_CLOSED
    assert operation.progress_results[-1][0].stage_code == 0x000A
    assert session.closed is True
    assert server_context.exited is True


@pytest.mark.asyncio
async def test_peer_disconnect_stops_backend_without_sending_late_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=103,
        frame_id=203,
        body=json.dumps(_chat_request()).encode("utf-8"),
        metadata=_submit_metadata(operation_id=103),
        terminal_results=[],
    )
    session = ScriptedEventSession(
        [
            FakeServerEvent(submit=operation),
            FakeServerEvent(runtime=_session_close_event(last_operation_id=103), wait_for_backend=True),
        ],
        operation=operation,
        backend_started=backend.started,
        stop_event=stop_event,
        stop_after_last_event=True,
    )
    server_context = FakeServerContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", lambda _options: server_context)

    statistics = await serve(
        OpenAiNnrpAdapter(backend),
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

    assert statistics.terminal_results == 0
    assert backend.closed.is_set()
    assert operation.terminal_results == []
    assert operation.result_drops == []
    assert session.closed is True


@pytest.mark.asyncio
async def test_native_deadline_update_stops_backend_and_drops_late_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=104,
        frame_id=204,
        body=json.dumps(_chat_request()).encode("utf-8"),
        metadata=_submit_metadata(operation_id=104),
        terminal_results=[],
        on_terminal=stop_event.set,
    )
    deadline = _deadline_event(
        MessageType.DEADLINE,
        operation_id=104,
        sequence=1,
        deadline_unix_ms=int(time.time() * 1000) - 1,
    )
    session = ScriptedEventSession(
        [
            FakeServerEvent(submit=operation),
            FakeServerEvent(runtime=deadline, wait_for_backend=True),
        ],
        operation=operation,
        backend_started=backend.started,
        stop_event=stop_event,
    )
    server_context = FakeServerContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", lambda _options: server_context)

    statistics = await serve(
        OpenAiNnrpAdapter(backend),
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

    assert statistics.accepted_operations == 1
    assert statistics.terminal_results == 1
    assert backend.closed.is_set()
    assert [json.loads(body)["delta"] for _metadata, body in operation.partial_results] == ["first"]
    assert operation.terminal_results == []
    assert len(operation.result_drops) == 1
    drop_metadata, diagnostic = operation.result_drops[0]
    assert drop_metadata.drop_reason_code is ResultDropReasonCode.DEADLINE_EXPIRED
    assert diagnostic == b"deadline_expired"
    assert operation.progress_results[-1][0].stage_code == 0x000A


@pytest.mark.asyncio
async def test_native_supersede_admits_replacement_before_dropping_old_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    backend = ReplacementBackend()
    old_operation = FakeOperation(
        operation_id=105,
        frame_id=205,
        body=json.dumps(_chat_request(model="old-model")).encode("utf-8"),
        metadata=_submit_metadata(operation_id=105),
        terminal_results=[],
    )
    new_operation = FakeOperation(
        operation_id=106,
        frame_id=206,
        body=json.dumps(_chat_request(model="new-model")).encode("utf-8"),
        metadata=_submit_metadata(operation_id=106),
        terminal_results=[],
    )
    completed = 0

    def operation_completed() -> None:
        nonlocal completed
        completed += 1
        if completed == 2:
            stop_event.set()

    old_operation.on_terminal = operation_completed
    new_operation.on_terminal = operation_completed
    session = ScriptedEventSession(
        [
            FakeServerEvent(submit=old_operation),
            FakeServerEvent(
                runtime=_supersede_event(old_operation_id=105, new_operation_id=106, sequence=1),
                wait_for_backend=True,
            ),
            FakeServerEvent(submit=new_operation),
        ],
        operation=old_operation,
        backend_started=backend.old_started,
        stop_event=stop_event,
    )
    server_context = FakeServerContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", lambda _options: server_context)

    statistics = await serve(
        OpenAiNnrpAdapter(backend),
        config=NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            accept_timeout_ms=10,
            receive_timeout_ms=10,
            max_active_sessions=1,
            max_operations_per_session=2,
            native_worker_count=2,
        ),
        stop_event=stop_event,
    )

    assert statistics.accepted_operations == 2
    assert statistics.terminal_results == 2
    assert backend.old_closed.is_set()
    assert old_operation.terminal_results == []
    assert len(old_operation.result_drops) == 1
    assert old_operation.result_drops[0][0].drop_reason_code is ResultDropReasonCode.SUPERSEDED
    assert old_operation.result_drops[0][1] == b"newer_request"
    assert len(new_operation.terminal_results) == 1
    assert json.loads(new_operation.terminal_results[0][1])["type"] == "response.completed"


def _control_event(message_type: MessageType, *, operation_id: int, sequence: int) -> NativeRuntimeEvent:
    metadata = ControlRequestMetadata(
        operation_id=operation_id,
        control_sequence=sequence,
        reason_code=3,
        source_role=RuntimeRole.CLIENT,
        flags=0,
        diagnostic_bytes=len(b"obsolete"),
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=message_type, session_id=1, frame_id=201),
        RuntimeEventMetadata(RuntimeEventMetadataKind.CONTROL_REQUEST, metadata),
        RuntimeEventTail.with_body(b"obsolete"),
    )


def _session_close_event(*, last_operation_id: int) -> NativeRuntimeEvent:
    metadata = SessionCloseMetadata(
        close_reason=SessionCloseReason.CLIENT_SHUTDOWN,
        in_flight_policy=InFlightPolicy.ABORT,
        drain_timeout_ms=0,
        last_operation_id=last_operation_id,
        session_error_code=0,
        session_close_tag=1,
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.SESSION_CLOSE, session_id=1),
        RuntimeEventMetadata(RuntimeEventMetadataKind.SESSION_CLOSE, metadata),
        RuntimeEventTail.none(),
    )


def _deadline_event(
    message_type: MessageType,
    *,
    operation_id: int,
    sequence: int,
    deadline_unix_ms: int,
) -> NativeRuntimeEvent:
    metadata = SchedulingMetadata(
        operation_id=operation_id,
        control_sequence=sequence,
        priority_class=0,
        priority_delta=0,
        deadline_unix_ms=deadline_unix_ms,
        flags=0,
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=message_type, session_id=1),
        RuntimeEventMetadata(RuntimeEventMetadataKind.SCHEDULING, metadata),
        RuntimeEventTail.none(),
    )


def _supersede_event(
    *,
    old_operation_id: int,
    new_operation_id: int,
    sequence: int,
) -> NativeRuntimeEvent:
    metadata = SupersedeMetadata(
        old_operation_id=old_operation_id,
        new_operation_id=new_operation_id,
        control_sequence=sequence,
        drop_reason_code=int(ResultDropReasonCode.SUPERSEDED),
        flags=1,
        diagnostic_bytes=len(b"newer_request"),
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.SUPERSEDE, session_id=1),
        RuntimeEventMetadata(RuntimeEventMetadataKind.SUPERSEDE, metadata),
        RuntimeEventTail.with_body(b"newer_request"),
    )


def _chat_request(*, model: str = "mock-model") -> dict[str, Any]:
    return {
        "schema_version": "openai-compatible/1",
        "operation": "chat.completions.create",
        "body": {
            "model": model,
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
