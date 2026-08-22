from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import InitVar, dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from nnrp import (
    NativeTransportBinding,
    NativeTransportEndpoint,
    NativeTransportProvider,
    NativeTransportProviderCost,
    NativeTransportProviderKind,
    NativeTransportProviderLimitation,
    NativeTransportProviderLimits,
    NativeTransportProviderMetadata,
    NativeTransportSelectionError,
    NativeWouldBlockError,
    StreamSemantics,
    TransportId,
)
from nnrp.core import (
    FrameSubmitMetadata,
    InputProfile,
    MessageType,
    PayloadKind,
    ResultClass,
    ResultPushMetadata,
    build_typed_payload_frame,
    pack_body,
    pack_typed_payload_frames,
    unpack_typed_payload_frames,
    validate_result_push_body,
)
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
    TraceContextMetadata,
)
from nnrp.server import NativeServerAcceptOptions, NativeServerBootstrapOptions, NativeServerProviderRoute

from vllm_nnrp_adapter import NnrpServerConfig, OpenAiNnrpAdapter, serve
from vllm_nnrp_adapter.nnrp_runtime import (
    _apply_trace_context,
    _native_handle_identity,
    _OperationObservationTracker,
    _serve_operation,
    _ServeCounters,
)
from vllm_nnrp_adapter.operation_state import OperationRegistry, OperationState
from vllm_nnrp_adapter.runtime_control import OperationControlSlot, RuntimeControlKind, RuntimeControlRequest


class StreamingBackend:
    def __init__(self) -> None:
        self.requests: list[Mapping[str, Any]] = []

    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        self.requests.append(body)

        async def events() -> AsyncIterator[Mapping[str, Any]]:
            yield {"choices": [{"index": 0, "delta": {"content": body["model"]}}]}
            yield {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        return events()


class FailingBackend:
    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        raise RuntimeError(f"backend failed for {body['model']}")


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
        self.close_calls = 0
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

        return AcceptedCloseStream(events(), self._record_close)

    def _record_close(self) -> None:
        self.close_calls += 1


class AcceptedCloseStream:
    def __init__(self, iterator: Any, on_close: Callable[[], None]) -> None:
        self._iterator = iterator
        self._on_close = on_close
        self._closed = False

    def __aiter__(self) -> AcceptedCloseStream:
        return self

    async def __anext__(self) -> Mapping[str, Any]:
        if self._closed:
            raise StopAsyncIteration
        return await self._iterator.__anext__()

    async def aclose(self) -> bool:
        if self._closed:
            return True
        self._closed = True
        self._on_close()
        await self._iterator.aclose()
        return True


class ReplacementBackend:
    def __init__(self) -> None:
        self.old_started = threading.Event()
        self.old_closed = threading.Event()
        self.old_close_calls = 0

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

        stream = events()
        if body["model"] == "old-model":
            return AcceptedCloseStream(stream, self._record_old_close)
        return stream

    def _record_old_close(self) -> None:
        self.old_close_calls += 1


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
    cancel_next_terminal_result: bool = False

    def __post_init__(self, body: bytes, metadata: FrameSubmitMetadata) -> None:
        self.submit = NativeRuntimeEvent(
            RuntimeFrameHeader(
                message_type=MessageType.FRAME_SUBMIT,
                session_id=1,
                frame_id=self.frame_id,
                view_id=self.operation_id + 2_000,
                route_id=self.operation_id + 1_000,
                trace_id=self.operation_id + 3_000,
            ),
            RuntimeEventMetadata(RuntimeEventMetadataKind.FRAME_SUBMIT, metadata),
            RuntimeEventTail.with_body(body),
        )

    async def send_result(self, metadata: ResultPushMetadata, body: bytes = b"") -> None:
        if self.cancel_next_terminal_result:
            self.cancel_next_terminal_result = False
            raise asyncio.CancelledError
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
    def __init__(
        self,
        session: FakeSession,
        *,
        bound_provider_endpoints: Mapping[str, NativeTransportEndpoint] | None = None,
    ) -> None:
        self._session = session
        self._accepted = False
        self.closed = False
        self.bound_provider_endpoints = dict(bound_provider_endpoints or {})

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
        self.bound_provider_endpoints: dict[str, NativeTransportEndpoint] = {}

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


class FailingServerContext:
    def __enter__(self) -> None:
        raise OSError("listener failed before admission")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        raise AssertionError("an unopened listener context must not be exited")


def _unavailable_binding(transport_id: TransportId) -> NativeTransportBinding:
    transport_name = transport_id.name.lower()
    provider = NativeTransportProvider(
        name=f"test-{transport_name}",
        version="1",
        transport_id=transport_id,
        kind=NativeTransportProviderKind.NATIVE_DYNAMIC,
        available=False,
        library_path=None,
        metadata=NativeTransportProviderMetadata(
            id=f"test.{transport_name}",
            cost=NativeTransportProviderCost(model_id=0, units=0),
            preference_rank=1,
            limits=NativeTransportProviderLimits(max_frame_bytes=1024),
            limitations=(NativeTransportProviderLimitation.LOCAL_HOST_ONLY,),
        ),
        diagnostic="test provider is unavailable",
    )
    return NativeTransportBinding(
        entrypoints=None,
        provider=provider,
        role_entrypoints=None,
        unavailable_diagnostic=provider.diagnostic,
    )


def _listen_with_context(server_context: FakeServerContext) -> Callable[..., FakeServerContext]:
    def listen(
        _options: NativeServerBootstrapOptions,
        *,
        transports: tuple[NativeTransportBinding, ...] | None = None,
    ) -> FakeServerContext:
        return server_context

    return listen


@pytest.mark.asyncio
async def test_native_server_emits_ordered_partial_results_and_one_terminal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    operation = FakeOperation(
        operation_id=71,
        frame_id=19,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=71),
        terminal_results=[],
    )
    session = FakeSession(operation, stop_event)
    server_context = FakeServerContext(
        FakeServer(
            session,
            bound_provider_endpoints={
                "ipc": NativeTransportEndpoint(
                    uri="npipe://nnrp-vllm",
                    scheme="npipe",
                    transport_name="ipc",
                    transport_id=TransportId.IPC,
                    address="nnrp-vllm",
                    secure=False,
                )
            },
        )
    )
    captured_options: list[NativeServerBootstrapOptions] = []
    captured_transports: list[tuple[NativeTransportBinding, ...] | None] = []

    def fake_listen(
        options: NativeServerBootstrapOptions,
        *,
        transports: tuple[NativeTransportBinding, ...] | None = None,
    ) -> FakeServerContext:
        captured_options.append(options)
        captured_transports.append(transports)
        return server_context

    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", fake_listen)
    bindings = (
        _unavailable_binding(TransportId.IPC),
        _unavailable_binding(TransportId.WEBSOCKET),
    )
    config = NnrpServerConfig(
        endpoint="nnrp://runtime.local/vllm",
        provider_routes={"ipc": NativeServerProviderRoute(provider_endpoint="npipe://nnrp-vllm")},
        transports=list(bindings),
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
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == "completed"
    assert observation["output_event_count"] == 3
    assert observation["total_tokens"] == 2
    terminal_metadata, terminal_body = operation.terminal_results[0]
    assert terminal_metadata.result_class is ResultClass.COMPLETE
    assert terminal_metadata.payload_kind_bitmap == PayloadKind.STRUCTURED_EVENT
    assert terminal_metadata.payload_frame_count == 1
    assert _decode_terminal_profile_body(terminal_metadata, terminal_body)["type"] == "response.completed"
    assert captured_options[0].endpoint.uri == "nnrp://runtime.local/vllm"
    assert captured_options[0].provider_routes["ipc"].provider_endpoint == "npipe://nnrp-vllm"
    startup = _startup_observation_records(caplog)[0]
    assert startup == {
        "application_endpoint": "nnrp://runtime.local/vllm",
        "bound_provider_endpoints": {"ipc": "npipe://nnrp-vllm"},
        "eligible_providers": ["ipc"],
        "transport_policy": "auto",
    }
    assert config.transports == bindings
    assert captured_transports == [bindings]
    assert session.closed is True
    assert server_context.exited is True
    assert operation.native_thread_ids
    assert threading.get_ident() not in operation.native_thread_ids
    assert backend.requests[0]["request_id"] == "nnrp-71-19"


@pytest.mark.asyncio
async def test_terminal_send_cancellation_preserves_completed_operation_state() -> None:
    operation = FakeOperation(
        operation_id=72,
        frame_id=20,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=72),
        terminal_results=[],
        cancel_next_terminal_result=True,
    )
    registry = OperationRegistry()
    record = registry.register(operation.operation_id, "request-72")
    record.transition(OperationState.QUEUED)
    control = OperationControlSlot(operation.operation_id)
    control.terminal_request = RuntimeControlRequest(
        kind=RuntimeControlKind.PEER_DISCONNECT,
        operation_id=operation.operation_id,
        control_sequence=1,
        reason_code=0,
        source_role=RuntimeRole.CLIENT,
        flags=0,
        diagnostic=b"peer_disconnect",
    )
    observation = _OperationObservationTracker.from_operation(
        operation,
        selected_transport="tcp",
        backend_family="StreamingBackend",
        backend_binding=None,
        vllm_version=None,
    )
    counters = _ServeCounters()

    await _serve_operation(
        OpenAiNnrpAdapter(StreamingBackend()),
        operation,
        record=record,
        control=control,
        observation=observation,
        observation_sinks=(),
        counters=counters,
    )

    assert record.state is OperationState.COMPLETED
    assert record.resources_released is True
    assert counters.terminal_results == 0


def _trace_context_event(
    metadata: TraceContextMetadata,
    body: bytes,
    *,
    frame_id: int,
    header_trace_id: int | None = None,
) -> NativeRuntimeEvent:
    return NativeRuntimeEvent(
        RuntimeFrameHeader(
            message_type=MessageType.TRACE_CONTEXT,
            session_id=1,
            frame_id=frame_id,
            trace_id=metadata.trace_id if header_trace_id is None else header_trace_id,
        ),
        RuntimeEventMetadata(RuntimeEventMetadataKind.TRACE_CONTEXT, metadata),
        RuntimeEventTail.with_body(body),
    )


def test_trace_context_correlates_session_and_active_operation_frames() -> None:
    operation = FakeOperation(
        operation_id=73,
        frame_id=23,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=73),
        terminal_results=[],
    )
    record = OperationRegistry().register(operation.operation_id, "request-73")
    record.transition(OperationState.QUEUED)
    observation = _OperationObservationTracker.from_operation(
        operation,
        selected_transport="ipc",
        clock_ns=lambda: 0,
    )
    session_metadata = TraceContextMetadata(81, 82, 80, 3, 1, 7)
    session_context = _apply_trace_context(
        _trace_context_event(session_metadata, b"session", frame_id=0),
        observations_by_frame={},
        session_trace_context=None,
    )

    assert session_context == (session_metadata, b"session")

    operation_metadata = TraceContextMetadata(91, 92, 90, 5, 3, 9)
    retained_session_context = _apply_trace_context(
        _trace_context_event(operation_metadata, b"operation", frame_id=23),
        observations_by_frame={23: (record, observation)},
        session_trace_context=session_context,
    )
    result = observation.finish(OperationState.FAILED)

    assert retained_session_context == session_context
    assert result.identity.trace_id == 91
    assert result.trace_span_id == 92
    assert result.trace_parent_span_id == 90
    assert result.trace_stage_code == 5
    assert result.trace_flags == 3
    assert result.trace_attribute_bytes == 9


@pytest.mark.parametrize(
    ("event", "message"),
    (
        (_trace_context_event(TraceContextMetadata(1, 2, 0, 0, 0, 0), b"", frame_id=99), "unknown active frame"),
        (
            _trace_context_event(
                TraceContextMetadata(1, 2, 0, 0, 0, 0),
                b"",
                frame_id=0,
                header_trace_id=2,
            ),
            "does not match",
        ),
        (_trace_context_event(TraceContextMetadata(1, 2, 0, 0, 0, 1), b"", frame_id=0), "body_bytes"),
    ),
)
def test_trace_context_rejects_invalid_correlation(
    event: NativeRuntimeEvent,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _apply_trace_context(event, observations_by_frame={}, session_trace_context=None)


@pytest.mark.asyncio
async def test_native_server_applies_session_trace_context_to_operation_observation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=74,
        frame_id=24,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=74),
        terminal_results=[],
        on_terminal=stop_event.set,
    )
    trace_metadata = TraceContextMetadata(101, 102, 100, 3, 1, 12)
    session = ScriptedEventSession(
        [
            FakeServerEvent(runtime=_trace_context_event(trace_metadata, b"private-attr", frame_id=0)),
            FakeServerEvent(submit=operation),
            FakeServerEvent(
                runtime=_control_event(MessageType.CANCEL, operation_id=74, sequence=1),
                wait_for_backend=True,
            ),
        ],
        operation=operation,
        backend_started=backend.started,
        stop_event=stop_event,
    )
    server_context = FakeServerContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

    await serve(
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

    observation = _observation_records(caplog)[0]
    assert observation["trace_id"] == 101
    assert observation["trace_span_id"] == 102
    assert observation["trace_parent_span_id"] == 100
    assert observation["trace_stage_code"] == 3
    assert observation["trace_flags"] == 1
    assert observation["trace_attribute_bytes"] == 12
    assert "private-attr" not in json.dumps(observation)


@pytest.mark.parametrize("endpoint", ["tcp://127.0.0.1:7766", "unix:///tmp/nnrp.sock", "ws://host/nnrp"])
def test_server_config_rejects_provider_locator_as_application_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError, match="nnrp:// or nnrps://"):
        NnrpServerConfig(endpoint=endpoint)


def test_server_config_uses_installed_provider_discovery_when_transports_are_omitted() -> None:
    assert NnrpServerConfig(endpoint="nnrp://runtime.local/vllm").transports is None


def test_server_config_rejects_non_binding_transport_entries() -> None:
    with pytest.raises(TypeError, match="transports values must be NativeTransportBinding"):
        NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            transports=[object()],  # type: ignore[list-item]
        )


def test_server_config_rejects_invalid_observation_sink() -> None:
    with pytest.raises(TypeError, match="observation_sinks values must implement ObservationSink"):
        NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            observation_sinks=[object()],  # type: ignore[list-item]
        )


def test_native_handle_identity_preserves_available_local_identity() -> None:
    wrapped = SimpleNamespace(handle=SimpleNamespace(id=41, generation=3))

    assert _native_handle_identity(wrapped) == (41, 3)
    assert _native_handle_identity(object()) == (None, None)


@pytest.mark.asyncio
async def test_unavailable_explicit_binding_never_starts_a_provider_listener() -> None:
    with pytest.raises(NativeTransportSelectionError):
        await serve(
            OpenAiNnrpAdapter(StreamingBackend()),
            config=NnrpServerConfig(
                endpoint="nnrp://runtime.local/vllm",
                provider_routes={"ipc": NativeServerProviderRoute(provider_endpoint="npipe://nnrp-vllm")},
                transports=[_unavailable_binding(TransportId.IPC)],
            ),
        )


@pytest.mark.asyncio
async def test_listener_failure_remains_a_server_failure_without_profile_terminal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        lambda *_args, **_kwargs: FailingServerContext(),
    )

    with pytest.raises(OSError, match="listener failed before admission"):
        await serve(
            OpenAiNnrpAdapter(StreamingBackend()),
            config=NnrpServerConfig(endpoint="nnrp://runtime.local/vllm"),
        )

    assert _startup_observation_records(caplog) == []
    assert _observation_records(caplog) == []


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
@pytest.mark.parametrize("invalid_body_kind", ["raw-json", "non-snapshot", "trailing-byte"])
async def test_invalid_submit_body_produces_one_terminal_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    invalid_body_kind: str,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    if invalid_body_kind == "raw-json":
        invalid_body = json.dumps(_chat_request()).encode("utf-8")
    elif invalid_body_kind == "non-snapshot":
        invalid_body = _typed_profile_body(_chat_request(), stream_semantics=StreamSemantics.APPEND)
    else:
        invalid_body = _typed_profile_body(_chat_request()) + b"unexpected"
    operation = FakeOperation(
        operation_id=72,
        frame_id=20,
        body=invalid_body,
        metadata=_submit_metadata(operation_id=72),
        terminal_results=[],
    )
    session = FakeSession(operation, stop_event)
    server_context = FakeServerContext(FakeServer(session))
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
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
    assert _decode_terminal_profile_body(metadata, body)["error"]["code"] == "invalid_submit_body"
    assert [metadata.stage_code for metadata, _body in operation.progress_results] == [0x0001, 0x0003, 0x0008, 0x000B]
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == "failed"
    assert observation["error_family"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_backend_failure_produces_one_typed_application_terminal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    operation = FakeOperation(
        operation_id=73,
        frame_id=21,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=73),
        terminal_results=[],
        on_terminal=stop_event.set,
    )
    session = FakeSession(operation, stop_event)
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(FakeServerContext(FakeServer(session))),
    )

    statistics = await serve(
        OpenAiNnrpAdapter(FailingBackend()),
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
    event = _decode_terminal_profile_body(metadata, body)
    assert event["type"] == "response.error"
    assert event["error"]["code"] == "backend_error"
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == "failed"
    assert observation["error_family"] == "RuntimeError"


@pytest.mark.asyncio
async def test_native_server_runs_sessions_and_operations_concurrently_with_per_operation_order(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    operations = [
        FakeOperation(
            operation_id=operation_id,
            frame_id=operation_id + 100,
            body=_typed_profile_body(_chat_request(model=f"model-{operation_id}")),
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
        _listen_with_context(server_context),
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
    observation_records = [
        json.loads(record.getMessage().removeprefix("nnrp_operation_observation "))
        for record in caplog.records
        if record.getMessage().startswith("nnrp_operation_observation ")
    ]
    assert len(observation_records) == len(operations)
    assert {record["operation_id"] for record in observation_records} == {1, 2, 3, 4}
    for observation in observation_records:
        operation_id = observation["operation_id"]
        assert observation["frame_id"] == operation_id + 100
        assert observation["route_id"] == operation_id + 1_000
        assert observation["view_id"] == operation_id + 2_000
        assert observation["trace_id"] == operation_id + 3_000
        assert observation["model_id"] == f"model-{operation_id}"
        assert observation["profile_operation"] == "chat.completions.create"
        assert observation["backend_family"] == "InterleavingBackend"
        assert observation["backend_binding"] is None
        assert observation["vllm_version"] is None
        assert observation["selected_transport"] == "ipc"
        assert observation["output_event_count"] == 3
        assert observation["terminal_outcome"] == "completed"


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
            body=_typed_profile_body(_chat_request()),
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
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

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
    assert _decode_terminal_profile_body(*operations[0].terminal_results[0])["type"] == "response.completed"
    duplicate_event = _decode_terminal_profile_body(*operations[1].terminal_results[0])
    assert duplicate_event["type"] == "response.error"
    assert duplicate_event["error"]["code"] == "duplicate_operation_id"


@pytest.mark.parametrize(("message_type", "expect_drop"), [(MessageType.CANCEL, False), (MessageType.ABORT, True)])
@pytest.mark.asyncio
async def test_native_control_stops_backend_and_emits_one_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
    message_type: MessageType,
    expect_drop: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=101,
        frame_id=201,
        body=_typed_profile_body(_chat_request()),
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
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

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
    assert backend.close_calls == 1
    assert [json.loads(body)["delta"] for _metadata, body in operation.partial_results] == ["first"]
    if expect_drop:
        assert operation.terminal_results == []
        assert len(operation.result_drops) == 1
        assert operation.result_drops[0][0].operation_id == 101
    else:
        assert operation.result_drops == []
        assert len(operation.terminal_results) == 1
        assert _decode_terminal_profile_body(*operation.terminal_results[0]) == {
            "reason": "peer_cancelled",
            "type": "response.cancelled",
        }
    assert operation.progress_results[-1][0].stage_code == 0x000A
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == ("dropped" if expect_drop else "cancelled")
    assert observation["cancellation_kind"] == message_type.name.lower()
    assert observation["cancellation_source"] == "client"
    assert observation["cancellation_reason_code"] == 3
    assert observation["backend_abort_accepted"] is True
    assert observation["drop_reason"] == ("peer_cancelled" if expect_drop else None)


@pytest.mark.asyncio
async def test_native_cancel_falls_back_to_typed_drop_when_profile_terminal_cannot_be_delivered(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=107,
        frame_id=207,
        body=_typed_profile_body(_chat_request()),
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
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

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
    assert backend.close_calls == 1
    assert operation.terminal_results == []
    assert len(operation.result_drops) == 1
    metadata, diagnostic = operation.result_drops[0]
    assert metadata.drop_reason_code is ResultDropReasonCode.PEER_CANCELLED
    assert diagnostic == b"obsolete"
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == "cancelled"
    assert observation["drop_reason"] == "peer_cancelled"


@pytest.mark.asyncio
async def test_server_shutdown_drops_active_operation_before_closing_session(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=102,
        frame_id=202,
        body=_typed_profile_body(_chat_request()),
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
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

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
    assert backend.close_calls == 1
    assert len(operation.result_drops) == 1
    assert operation.result_drops[0][0].drop_reason_code is ResultDropReasonCode.TRANSPORT_CLOSED
    assert operation.progress_results[-1][0].stage_code == 0x000A
    assert session.closed is True
    assert server_context.exited is True
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == "dropped"
    assert observation["cancellation_kind"] == "server_shutdown"
    assert observation["drop_reason"] == "transport_closed"


@pytest.mark.asyncio
async def test_cancel_shutdown_race_emits_exactly_one_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=108,
        frame_id=208,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=108),
        terminal_results=[],
    )
    session = ScriptedEventSession(
        [
            FakeServerEvent(submit=operation),
            FakeServerEvent(
                runtime=_control_event(MessageType.CANCEL, operation_id=108, sequence=1),
                wait_for_backend=True,
            ),
        ],
        operation=operation,
        backend_started=backend.started,
        stop_event=stop_event,
    )
    server_context = FakeServerContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

    async def request_shutdown() -> None:
        assert await asyncio.to_thread(backend.started.wait, 1)
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
    assert backend.close_calls == 1
    assert len(operation.terminal_results) + len(operation.result_drops) == 1
    assert session.closed is True
    assert server_context.exited is True
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] in {"cancelled", "dropped"}
    assert observation["cancellation_kind"] in {"cancel", "server_shutdown"}


@pytest.mark.asyncio
async def test_peer_disconnect_stops_backend_without_sending_late_terminal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=103,
        frame_id=203,
        body=_typed_profile_body(_chat_request()),
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
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

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
    assert backend.close_calls == 1
    assert operation.terminal_results == []
    assert operation.result_drops == []
    assert session.closed is True
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == "dropped"
    assert observation["cancellation_kind"] == "peer_disconnect"
    assert observation["cancellation_source"] == "client"


@pytest.mark.asyncio
async def test_native_server_restarts_after_clean_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    contexts: list[FakeServerContext] = []

    def listen(
        _options: NativeServerBootstrapOptions,
        *,
        transports: tuple[NativeTransportBinding, ...] | None = None,
    ) -> FakeServerContext:
        assert transports is None
        return contexts.pop(0)

    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", listen)

    for operation_id in (109, 110):
        stop_event = asyncio.Event()
        operation = FakeOperation(
            operation_id=operation_id,
            frame_id=operation_id + 100,
            body=_typed_profile_body(_chat_request(model=f"restart-{operation_id}")),
            metadata=_submit_metadata(operation_id=operation_id),
            terminal_results=[],
            on_terminal=stop_event.set,
        )
        session = ScriptedEventSession(
            [FakeServerEvent(submit=operation)],
            operation=operation,
            backend_started=threading.Event(),
            stop_event=stop_event,
        )
        server_context = FakeServerContext(MultiSessionFakeServer([session]))
        contexts.append(server_context)

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

        assert statistics.accepted_sessions == 1
        assert statistics.accepted_operations == 1
        assert statistics.terminal_results == 1
        assert len(operation.terminal_results) == 1
        assert _decode_terminal_profile_body(*operation.terminal_results[0])["type"] == "response.completed"
        assert session.closed is True
        assert server_context.exited is True
        assert contexts == []


@pytest.mark.asyncio
async def test_native_deadline_update_stops_backend_and_drops_late_output(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=104,
        frame_id=204,
        body=_typed_profile_body(_chat_request()),
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
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

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
    assert backend.close_calls == 1
    assert [json.loads(body)["delta"] for _metadata, body in operation.partial_results] == ["first"]
    assert operation.terminal_results == []
    assert len(operation.result_drops) == 1
    drop_metadata, diagnostic = operation.result_drops[0]
    assert drop_metadata.drop_reason_code is ResultDropReasonCode.DEADLINE_EXPIRED
    assert diagnostic == b"deadline_expired"
    assert operation.progress_results[-1][0].stage_code == 0x000A
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == "dropped"
    assert observation["cancellation_kind"] == "deadline_expired"
    assert observation["drop_reason"] == "deadline_expired"


@pytest.mark.asyncio
async def test_native_supersede_admits_replacement_before_dropping_old_operation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    backend = ReplacementBackend()
    old_operation = FakeOperation(
        operation_id=105,
        frame_id=205,
        body=_typed_profile_body(_chat_request(model="old-model")),
        metadata=_submit_metadata(operation_id=105),
        terminal_results=[],
    )
    new_operation = FakeOperation(
        operation_id=106,
        frame_id=206,
        body=_typed_profile_body(_chat_request(model="new-model")),
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
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

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
    assert backend.old_close_calls == 1
    assert old_operation.terminal_results == []
    assert len(old_operation.result_drops) == 1
    assert old_operation.result_drops[0][0].drop_reason_code is ResultDropReasonCode.SUPERSEDED
    assert old_operation.result_drops[0][1] == b"newer_request"
    assert len(new_operation.terminal_results) == 1
    assert _decode_terminal_profile_body(*new_operation.terminal_results[0])["type"] == "response.completed"
    observations = {record["operation_id"]: record for record in _observation_records(caplog)}
    assert observations[105]["terminal_outcome"] == "dropped"
    assert observations[105]["cancellation_kind"] == "supersede"
    assert observations[105]["drop_reason"] == "superseded"
    assert observations[106]["terminal_outcome"] == "completed"
    assert observations[106]["cancellation_kind"] is None


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


def _typed_profile_body(
    document: Mapping[str, Any],
    *,
    stream_semantics: StreamSemantics = StreamSemantics.SNAPSHOT,
) -> bytes:
    payload = json.dumps(dict(document), separators=(",", ":"), sort_keys=True).encode("utf-8")
    frame = build_typed_payload_frame(
        PayloadKind.STRUCTURED_EVENT,
        payload,
        profile_id=0,
        descriptor_flags=0,
        schema_id=0,
        schema_version=0,
        stream_semantics=stream_semantics,
    )
    descriptors, frames = pack_typed_payload_frames((frame,))
    return pack_body(
        typed_payload_descriptor_region=descriptors,
        typed_payload_frame_region=frames,
    )


def _decode_terminal_profile_body(metadata: ResultPushMetadata, body: bytes) -> dict[str, Any]:
    body_view = validate_result_push_body(metadata, body)
    frames = unpack_typed_payload_frames(
        body_view.typed_payload_descriptor_region,
        body_view.typed_payload_frame_region,
        payload_kind_bitmap=metadata.payload_kind_bitmap,
    )
    assert len(frames) == 1
    frame = frames[0]
    assert frame.payload_kind is PayloadKind.STRUCTURED_EVENT
    assert frame.profile_id == 0
    assert int(frame.descriptor_flags) == 0
    assert frame.schema_id == 0
    assert frame.schema_version == 0
    assert int(frame.stream_semantics) == int(StreamSemantics.SNAPSHOT)
    value = json.loads(frame.payload)
    assert isinstance(value, dict)
    return value


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


def _capture_observations(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="vllm_nnrp_adapter.operation")
    caplog.set_level(logging.INFO, logger="vllm_nnrp_adapter.server")


def _observation_records(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage().removeprefix("nnrp_operation_observation "))
        for record in caplog.records
        if record.getMessage().startswith("nnrp_operation_observation ")
    ]


def _startup_observation_records(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage().removeprefix("nnrp_server_startup "))
        for record in caplog.records
        if record.getMessage().startswith("nnrp_server_startup ")
    ]
