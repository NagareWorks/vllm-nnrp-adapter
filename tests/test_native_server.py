from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import InitVar, dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from nnrp import (
    ERROR_FAMILY_LIFECYCLE,
    NativeInvalidStateError,
    NativeProtocolError,
    NativeTransportBinding,
    NativeTransportEndpoint,
    NativeTransportProvider,
    NativeTransportProviderCost,
    NativeTransportProviderKind,
    NativeTransportProviderLimitation,
    NativeTransportProviderLimits,
    NativeTransportProviderMetadata,
    NativeTransportSelectionError,
    NativeTransportServerSecurity,
    NativeWouldBlockError,
    StreamSemantics,
    TransportId,
)
from nnrp.core import (
    ErrorCode,
    ErrorScope,
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
from nnrp.native import (
    FFI_STATUS_INVALID_STATE,
    FFI_STATUS_PROTOCOL_ERROR,
    FFI_STATUS_WOULD_BLOCK,
    NativeStatus,
)
from nnrp.runtime import (
    BudgetMetadata,
    CacheMissMetadata,
    CacheReferenceMetadata,
    CacheReuseScope,
    CapabilityMetadata,
    ControlRequestMetadata,
    InFlightPolicy,
    MemoryLocationHint,
    NativeRuntimeEvent,
    ObjectDescriptorMetadata,
    ObjectReferenceMetadata,
    ObjectReleaseMetadata,
    ObjectReleaseReason,
    OwnershipHint,
    PartialResultMetadata,
    PressureMetadata,
    ProgressMetadata,
    RecoverableErrorMetadata,
    ResultDropReasonCode,
    ResultDropReasonMetadata,
    RetryAfterMetadata,
    RouteHintMetadata,
    RuntimeEventMetadata,
    RuntimeEventMetadataKind,
    RuntimeEventTail,
    RuntimeFrameHeader,
    RuntimeObjectKind,
    RuntimeRole,
    SchedulingMetadata,
    SessionCloseMetadata,
    SessionCloseReason,
    SupersedeMetadata,
    TraceContextMetadata,
)
from nnrp.server import NativeServerAcceptOptions, NativeServerBootstrapOptions, NativeServerProviderRoute

from vllm_nnrp_adapter import (
    NnrpServerConfig,
    OpenAiNnrpAdapter,
    StructuredLogObservationSink,
    serve,
)
from vllm_nnrp_adapter.nnrp_runtime import (
    _AdmissionWindowReporter,
    _apply_trace_context,
    _BackendTraceContextSlot,
    _decode_request,
    _decode_structured_event,
    _finalize_cancelled_operation,
    _handle_runtime_control,
    _native_handle_identity,
    _NativeCallExecutor,
    _OperationObservationTracker,
    _RecoveryReporter,
    _retire_completed_tasks,
    _serve,
    _serve_operation,
    _ServeCounters,
    _w3c_trace_headers,
)
from vllm_nnrp_adapter.operation_state import OperationRegistry, OperationState
from vllm_nnrp_adapter.pressure import OutboundCreditController
from vllm_nnrp_adapter.runtime_control import (
    OperationControlSlot,
    RuntimeControlDisposition,
    RuntimeControlKind,
    RuntimeControlRegistry,
    RuntimeControlRequest,
    RuntimePriorityUpdate,
)
from vllm_nnrp_adapter.vllm_backend import VllmBackend
from vllm_nnrp_adapter.vllm_compat import VLLM_COMPATIBILITY_BINDINGS

LOCAL_IPC_ENDPOINT = "npipe://nnrp-vllm" if os.name == "nt" else "unix:///tmp/nnrp-vllm.sock"
LOCAL_IPC_SCHEME = "npipe" if os.name == "nt" else "unix"


def test_structured_event_decoder_enforces_byte_and_depth_limits_before_json_decode() -> None:
    assert _decode_structured_event(b'{"body":{"messages":[]}}', max_bytes=64, max_depth=3) == {
        "body": {"messages": []}
    }
    assert _decode_structured_event(b'{"text":"[{\\\"nested\\\":true}]"}', max_depth=1)["text"] == (
        '[{"nested":true}]'
    )

    with pytest.raises(ValueError, match="8-byte adapter limit"):
        _decode_structured_event(b'{"body":{}}', max_bytes=8)
    with pytest.raises(ValueError, match="2-level nesting limit"):
        _decode_structured_event(b'{"body":{"messages":[]}}', max_depth=2)
    with pytest.raises(ValueError, match="limits must be positive"):
        _decode_structured_event(b"{}", max_depth=0)


def test_structured_event_decoder_normalizes_invalid_utf8_json_and_extreme_nesting() -> None:
    with pytest.raises(ValueError, match="UTF-8 OpenAI profile request"):
        _decode_structured_event(b"\xff")
    with pytest.raises(ValueError, match="UTF-8 OpenAI profile request"):
        _decode_structured_event(b'{"body":]')
    with pytest.raises(ValueError, match="64-level nesting limit"):
        _decode_structured_event(b"[" * 65 + b"]" * 65)


def test_native_submit_decode_applies_structured_event_depth_limit() -> None:
    nested: object = "leaf"
    for _ in range(65):
        nested = [nested]
    request = _chat_request()
    request["body"]["metadata"] = nested

    with pytest.raises(ValueError, match="64-level nesting limit"):
        _decode_request(_submit_metadata(operation_id=1), _typed_profile_body(request))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "diagnostic"),
    (
        (
            NativeRuntimeEvent(
                RuntimeFrameHeader(message_type=MessageType.BUDGET_UPDATE, session_id=7, frame_id=9),
                RuntimeEventMetadata(
                    RuntimeEventMetadataKind.BUDGET,
                    BudgetMetadata(41, 0, 0, 0, 64, 1),
                ),
                RuntimeEventTail.none(),
            ),
            b"budget_update_unsupported",
        ),
        (
            NativeRuntimeEvent(
                RuntimeFrameHeader(message_type=MessageType.ROUTE_HINT, session_id=7, frame_id=9),
                RuntimeEventMetadata(
                    RuntimeEventMetadataKind.ROUTE_HINT,
                    RouteHintMetadata(41, 3, 7, 3, 0, 0, 1),
                ),
                RuntimeEventTail.none(),
            ),
            b"route_hint_unsupported",
        ),
        (
            NativeRuntimeEvent(
                RuntimeFrameHeader(message_type=MessageType.EXECUTION_HINT, session_id=7, frame_id=9),
                RuntimeEventMetadata(
                    RuntimeEventMetadataKind.ROUTE_HINT,
                    RouteHintMetadata(41, 0, 7, 0, 0, 0, 1),
                ),
                RuntimeEventTail.none(),
            ),
            b"execution_hint_unsupported",
        ),
    ),
)
async def test_unsupported_runtime_control_returns_typed_error(
    event: NativeRuntimeEvent,
    diagnostic: bytes,
) -> None:
    operation = FakeOperation(
        operation_id=41,
        frame_id=9,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=41),
        terminal_results=[],
    )
    session = FakeSession(operation, asyncio.Event())
    native = _NativeCallExecutor(1)
    counters = _ServeCounters()
    try:
        await _handle_runtime_control(
            event,
            adapter=OpenAiNnrpAdapter(StreamingBackend()),
            session=session,  # type: ignore[arg-type]
            native=native,
            registry=OperationRegistry(),
            controls=RuntimeControlRegistry(),
            counters=counters,
        )
    finally:
        native.close()

    assert counters.applied_control_events == 0
    assert counters.rejected_control_events == 1
    assert len(session.recoverable_errors) == 1
    metadata, body = session.recoverable_errors[0]
    assert metadata.error_code is ErrorCode.UNSUPPORTED_CAPABILITY
    assert metadata.error_scope is ErrorScope.FRAME
    assert metadata.related_session_id == 7
    assert metadata.related_frame_id == 41
    assert metadata.related_view_id == 0
    assert metadata.diagnostic_bytes == len(diagnostic)
    assert body == diagnostic


class StreamingBackend:
    def __init__(self) -> None:
        self.requests: list[Mapping[str, Any]] = []

    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        self.requests.append(body)

        async def events() -> AsyncIterator[Mapping[str, Any]]:
            yield {"choices": [{"index": 0, "delta": {"content": body["model"]}}]}
            yield {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        return events()


class TraceAwareStreamingBackend:
    def __init__(self) -> None:
        self.requests: list[Mapping[str, Any]] = []
        self.trace_headers: list[Mapping[str, str]] = []
        self.started = threading.Event()

    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        raise AssertionError("trace-aware backend must receive the explicit context call")

    def create_chat_completion_with_context(
        self,
        body: Mapping[str, Any],
        *,
        trace_headers: Mapping[str, str],
    ) -> object:
        self.requests.append(body)
        self.trace_headers.append(trace_headers)
        self.started.set()

        async def events() -> AsyncIterator[Mapping[str, Any]]:
            yield {"choices": [{"index": 0, "delta": {"content": "traced"}}]}

        return events()


class PriorityAwareStreamingBackend:
    supports_runtime_priority = True

    def __init__(self) -> None:
        self.requests: list[Mapping[str, Any]] = []
        self.priorities: list[int] = []

    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        raise AssertionError("priority-aware backend must receive the explicit context call")

    def create_chat_completion_with_context(
        self,
        body: Mapping[str, Any],
        *,
        priority: int,
    ) -> object:
        self.requests.append(body)
        self.priorities.append(priority)

        async def events() -> AsyncIterator[Mapping[str, Any]]:
            yield {"choices": [{"index": 0, "delta": {"content": "prioritized"}}]}

        return events()


class FailingBackend:
    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        raise RuntimeError(f"backend failed for {body['model']}")


class OverloadedBackend:
    def create_chat_completion(self, body: Mapping[str, Any]) -> object:
        raise RuntimeError(f"scheduler full: reject request for {body['model']}")


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


class LivePriorityCancellableBackend(CancellableBackend):
    supports_runtime_priority = True
    supports_live_runtime_priority = True

    def __init__(self) -> None:
        super().__init__()
        self.priority_updates: list[tuple[str, int]] = []

    def create_chat_completion_with_context(
        self,
        body: Mapping[str, Any],
        *,
        trace_headers: Mapping[str, str] | None = None,
        priority: int | None = None,
    ) -> object:
        del trace_headers, priority
        return self.create_chat_completion(body)

    async def update_runtime_priority(self, request_id: str, priority: int) -> bool:
        self.priority_updates.append((request_id, priority))
        return True


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
    fail_next_terminal_lifecycle_error: bool = False
    cancel_next_terminal_result: bool = False
    fail_next_progress_invalid_state: bool = False
    partial_send_gate: asyncio.Event | None = None
    partial_send_observer: Callable[[bool], None] | None = None
    terminal_send_observer: Callable[[], None] | None = None
    drop_send_observer: Callable[[], None] | None = None

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
        if self.terminal_send_observer is not None:
            self.terminal_send_observer()
        if self.cancel_next_terminal_result:
            self.cancel_next_terminal_result = False
            raise asyncio.CancelledError
        if self.fail_next_terminal_lifecycle_error:
            self.fail_next_terminal_lifecycle_error = False
            raise NativeProtocolError(
                NativeStatus(
                    FFI_STATUS_PROTOCOL_ERROR,
                    error_family=ERROR_FAMILY_LIFECYCLE,
                ),
                "native runtime operation became terminal before RESULT_PUSH",
            )
        if self.fail_next_terminal_result:
            self.fail_next_terminal_result = False
            raise OSError("terminal profile result unavailable")
        self.terminal_results.append((metadata, body))
        if self.on_terminal is not None:
            self.on_terminal()

    async def send_partial_result(self, metadata: PartialResultMetadata, body: bytes = b"") -> None:
        if self.partial_send_observer is not None:
            self.partial_send_observer(True)
        try:
            if self.partial_send_gate is not None:
                await self.partial_send_gate.wait()
            self.partial_results.append((metadata, body))
        finally:
            if self.partial_send_observer is not None:
                self.partial_send_observer(False)

    async def send_progress(self, metadata: ProgressMetadata, body: bytes = b"") -> None:
        if self.fail_next_progress_invalid_state:
            self.fail_next_progress_invalid_state = False
            raise NativeInvalidStateError(
                NativeStatus(FFI_STATUS_INVALID_STATE, detail_code=105),
                "native runtime operation is already terminal",
            )
        self.progress_results.append((metadata, body))

    async def send_result_drop(self, metadata: ResultDropReasonMetadata, diagnostic: bytes = b"") -> None:
        if self.drop_send_observer is not None:
            self.drop_send_observer()
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
    def __init__(
        self,
        operation: FakeOperation,
        stop_event: asyncio.Event,
        *,
        negotiate_progress: bool = False,
    ) -> None:
        self.active_transport_name = "ipc"
        self._operation = operation
        self._stop_event = stop_event
        self._delivered = False
        self._negotiate_progress = negotiate_progress
        self._progress_negotiation_delivered = False
        self.partial_results = operation.partial_results
        self.backpressure_updates: list[PressureMetadata] = []
        self.credit_updates: list[PressureMetadata] = []
        self.recoverable_errors: list[tuple[RecoverableErrorMetadata, bytes]] = []
        self.retry_after_updates: list[tuple[RetryAfterMetadata, bytes]] = []
        self.capability_negotiations: list[tuple[CapabilityMetadata, bytes]] = []
        self.trace_contexts: list[tuple[TraceContextMetadata, bytes, int | None]] = []
        self.closed = False

    def poll_events(self, *, timeout_ms: int = 0, max_events: int = 1) -> tuple[FakeServerEvent, ...]:
        self._operation.native_thread_ids.append(threading.get_ident())
        assert timeout_ms == 10
        assert max_events == 1
        if self._negotiate_progress and not self._progress_negotiation_delivered:
            self._progress_negotiation_delivered = True
            return (FakeServerEvent(runtime=_capability_event("control.progress_partial")),)
        if not self._delivered:
            self._delivered = True
            return (FakeServerEvent(submit=self._operation),)
        self._stop_event.set()
        return ()

    def close(self) -> None:
        self._operation.native_thread_ids.append(threading.get_ident())
        self.closed = True

    def send_backpressure(self, metadata: PressureMetadata) -> None:
        self.backpressure_updates.append(metadata)

    def send_credit_update(self, metadata: PressureMetadata) -> None:
        self.credit_updates.append(metadata)

    def send_recoverable_error(
        self,
        metadata: RecoverableErrorMetadata,
        diagnostic: bytes = b"",
    ) -> None:
        self.recoverable_errors.append((metadata, diagnostic))

    def send_retry_after(
        self,
        metadata: RetryAfterMetadata,
        diagnostic: bytes = b"",
    ) -> None:
        self.retry_after_updates.append((metadata, diagnostic))

    def negotiate_capabilities(self, metadata: CapabilityMetadata, body: bytes = b"") -> None:
        self.capability_negotiations.append((metadata, body))

    def send_trace_context(
        self,
        metadata: TraceContextMetadata,
        body: bytes = b"",
        *,
        operation_id: int | None = None,
    ) -> None:
        self.trace_contexts.append((metadata, body, operation_id))


class ClosingFakeSession(FakeSession):
    def __init__(
        self,
        operation: FakeOperation,
        stop_event: asyncio.Event,
        *,
        detail_code: int,
    ) -> None:
        super().__init__(operation, stop_event)
        self._detail_code = detail_code

    def poll_events(self, *, timeout_ms: int = 0, max_events: int = 1) -> tuple[FakeServerEvent, ...]:
        assert timeout_ms == 10
        assert max_events == 1
        raise NativeInvalidStateError(
            NativeStatus(FFI_STATUS_INVALID_STATE, detail_code=self._detail_code),
            "native runtime session closed",
        )

    def close(self) -> None:
        super().close()
        self._stop_event.set()


class MultiOperationFakeSession:
    def __init__(
        self,
        operations: list[FakeOperation],
        *,
        operation_accepted: Callable[[], None],
        connection_identity: tuple[int, int] = (10_001, 1),
        session_identity: tuple[int, int] = (20_001, 1),
    ) -> None:
        self.active_transport_name = "ipc"
        self.server = SimpleNamespace(
            handle=SimpleNamespace(id=connection_identity[0], generation=connection_identity[1])
        )
        self.handle = SimpleNamespace(
            handle=SimpleNamespace(id=session_identity[0], generation=session_identity[1])
        )
        self._pending = list(operations)
        self._operation_accepted = operation_accepted
        self.partial_results: list[tuple[PartialResultMetadata, bytes]] = []
        for operation in operations:
            operation.partial_results = self.partial_results
        self.backpressure_updates: list[PressureMetadata] = []
        self.credit_updates: list[PressureMetadata] = []
        self.trace_contexts: list[tuple[TraceContextMetadata, bytes, int | None]] = []
        self.recoverable_errors: list[tuple[RecoverableErrorMetadata, bytes]] = []
        self.retry_after_updates: list[tuple[RetryAfterMetadata, bytes]] = []
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

    def send_backpressure(self, metadata: PressureMetadata) -> None:
        self.backpressure_updates.append(metadata)

    def send_credit_update(self, metadata: PressureMetadata) -> None:
        self.credit_updates.append(metadata)

    def send_recoverable_error(
        self,
        metadata: RecoverableErrorMetadata,
        diagnostic: bytes = b"",
    ) -> None:
        self.recoverable_errors.append((metadata, diagnostic))

    def send_retry_after(
        self,
        metadata: RetryAfterMetadata,
        diagnostic: bytes = b"",
    ) -> None:
        self.retry_after_updates.append((metadata, diagnostic))


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
        self.capability_negotiations: list[tuple[CapabilityMetadata, bytes]] = []
        self.profile_degradations: list[tuple[CapabilityMetadata, bytes]] = []
        self.recoverable_errors: list[tuple[RecoverableErrorMetadata, bytes]] = []
        self.retry_after_updates: list[tuple[RetryAfterMetadata, bytes]] = []
        self.backpressure_updates: list[PressureMetadata] = []
        self.credit_updates: list[PressureMetadata] = []
        self.trace_contexts: list[tuple[TraceContextMetadata, bytes, int | None]] = []
        self.cache_misses: list[tuple[CacheMissMetadata, bytes]] = []
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

    def send_backpressure(self, metadata: PressureMetadata) -> None:
        self.backpressure_updates.append(metadata)

    def send_credit_update(self, metadata: PressureMetadata) -> None:
        self.credit_updates.append(metadata)

    def send_recoverable_error(
        self,
        metadata: RecoverableErrorMetadata,
        diagnostic: bytes = b"",
    ) -> None:
        self.recoverable_errors.append((metadata, diagnostic))

    def send_retry_after(
        self,
        metadata: RetryAfterMetadata,
        diagnostic: bytes,
    ) -> None:
        self.retry_after_updates.append((metadata, diagnostic))

    def negotiate_capabilities(self, metadata: CapabilityMetadata, body: bytes = b"") -> None:
        self.capability_negotiations.append((metadata, body))

    def degrade_profile(self, metadata: CapabilityMetadata, body: bytes = b"") -> None:
        self.profile_degradations.append((metadata, body))

    def send_trace_context(
        self,
        metadata: TraceContextMetadata,
        body: bytes = b"",
        *,
        operation_id: int | None = None,
    ) -> None:
        self.trace_contexts.append((metadata, body, operation_id))

    def report_cache_miss(self, metadata: CacheMissMetadata, diagnostic: bytes = b"") -> None:
        self.cache_misses.append((metadata, diagnostic))


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
        self.accepted_count = 0
        self.bound_provider_endpoints: dict[str, NativeTransportEndpoint] = {}

    async def accept(self, options: NativeServerAcceptOptions | None = None) -> Any:
        assert options is not None
        assert options.timeout_ms == 10
        if self._pending:
            self.accepted_count += 1
            return self._pending.pop(0)
        raise NativeWouldBlockError(NativeStatus(FFI_STATUS_WOULD_BLOCK))


class ClosingFakeServer:
    def __init__(self, *, detail_code: int) -> None:
        self.bound_provider_endpoints: dict[str, NativeTransportEndpoint] = {}
        self._detail_code = detail_code

    async def accept(self, options: NativeServerAcceptOptions | None = None) -> Any:
        assert options is not None
        assert options.timeout_ms == 10
        raise NativeInvalidStateError(
            NativeStatus(FFI_STATUS_INVALID_STATE, detail_code=self._detail_code),
            "native runtime server closed",
        )


class ShutdownRaceFakeServer:
    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self.accept_started = asyncio.Event()
        self.release_accept = asyncio.Event()
        self.bound_provider_endpoints: dict[str, NativeTransportEndpoint] = {}

    async def accept(self, options: NativeServerAcceptOptions | None = None) -> FakeSession:
        assert options is not None
        assert options.timeout_ms == 10
        self.accept_started.set()
        await self.release_accept.wait()
        return self._session


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
async def test_native_server_accept_treats_transport_closed_as_clean_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_context = FakeServerContext(ClosingFakeServer(detail_code=105))  # type: ignore[arg-type]
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
            native_worker_count=1,
        ),
    )

    assert statistics.accepted_sessions == 0
    assert statistics.accepted_operations == 0
    assert server_context.exited is True


@pytest.mark.asyncio
async def test_native_server_accept_propagates_other_invalid_state_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_context = FakeServerContext(ClosingFakeServer(detail_code=106))  # type: ignore[arg-type]
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

    with pytest.raises(NativeInvalidStateError) as error:
        await serve(
            OpenAiNnrpAdapter(StreamingBackend()),
            config=NnrpServerConfig(
                endpoint="nnrp://runtime.local/vllm",
                accept_timeout_ms=10,
                receive_timeout_ms=10,
                max_active_sessions=1,
                max_operations_per_session=1,
                native_worker_count=1,
            ),
        )

    assert error.value.status.detail_code == 106
    assert server_context.exited is True


@pytest.mark.asyncio
async def test_native_session_poll_treats_transport_closed_as_clean_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    operation = FakeOperation(
        operation_id=69,
        frame_id=17,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=69),
        terminal_results=[],
    )
    session = ClosingFakeSession(operation, stop_event, detail_code=105)
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
            native_worker_count=1,
        ),
        stop_event=stop_event,
    )

    assert statistics.accepted_sessions == 1
    assert statistics.accepted_operations == 0
    assert session.closed is True


@pytest.mark.asyncio
async def test_native_session_poll_propagates_other_invalid_state_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    operation = FakeOperation(
        operation_id=70,
        frame_id=18,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=70),
        terminal_results=[],
    )
    session = ClosingFakeSession(operation, stop_event, detail_code=106)
    server_context = FakeServerContext(FakeServer(session))
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

    with pytest.raises(NativeInvalidStateError) as error:
        await serve(
            OpenAiNnrpAdapter(StreamingBackend()),
            config=NnrpServerConfig(
                endpoint="nnrp://runtime.local/vllm",
                accept_timeout_ms=10,
                receive_timeout_ms=10,
                max_active_sessions=1,
                max_operations_per_session=1,
                native_worker_count=1,
            ),
            stop_event=stop_event,
        )

    assert error.value.status.detail_code == 106
    assert session.closed is True


@pytest.mark.asyncio
async def test_admission_window_reports_only_after_capability_is_enabled() -> None:
    operation = FakeOperation(
        operation_id=70,
        frame_id=18,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=70),
        terminal_results=[],
    )
    session = FakeSession(operation, asyncio.Event())
    native = _NativeCallExecutor(1)
    reporter = _AdmissionWindowReporter(
        session=session,  # type: ignore[arg-type]
        native=native,
        capacity=1,
        retry_after_ms=25,
    )
    try:
        await reporter.update(0)
        assert session.credit_updates == []
        assert session.backpressure_updates == []

        await reporter.enable(0, scope_id=1)
        await reporter.update(1)
        await reporter.update(0)
    finally:
        native.close()

    assert [metadata.credit_window for metadata in session.credit_updates] == [1, 1]
    assert [metadata.credit_window for metadata in session.backpressure_updates] == [0]
    assert all(metadata.scope_id == 1 for metadata in session.credit_updates)
    assert session.backpressure_updates[0].scope_id == 1
    assert session.backpressure_updates[0].pressure_reason == 1
    assert session.backpressure_updates[0].retry_after_ms == 25


@pytest.mark.asyncio
async def test_recovery_reporter_emits_correlated_error_and_retry_after_only_when_enabled() -> None:
    operation = FakeOperation(
        operation_id=71,
        frame_id=19,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=71),
        terminal_results=[],
    )
    session = FakeSession(operation, asyncio.Event())
    native = _NativeCallExecutor(1)
    reporter = _RecoveryReporter(session=session, native=native, retry_after_ms=40)  # type: ignore[arg-type]
    try:
        await reporter.report(
            operation,  # type: ignore[arg-type]
            diagnostic=b"adapter_operation_limit",
            source_role=RuntimeRole.SCHEDULER,
        )
        assert session.recoverable_errors == []
        assert session.retry_after_updates == []

        reporter.enable()
        await asyncio.gather(
            reporter.report(
                operation,  # type: ignore[arg-type]
                diagnostic=b"adapter_operation_limit",
                source_role=RuntimeRole.SCHEDULER,
            ),
            reporter.report(
                operation,  # type: ignore[arg-type]
                diagnostic=b"adapter_operation_limit",
                source_role=RuntimeRole.SCHEDULER,
            ),
        )
    finally:
        native.close()

    error, error_diagnostic = session.recoverable_errors[0]
    retry, retry_diagnostic = session.retry_after_updates[0]
    assert error.error_code is ErrorCode.SERVER_BUSY
    assert error.error_scope is ErrorScope.FRAME
    assert error.source_role is RuntimeRole.SCHEDULER
    assert error.flags == 0x02
    assert error.retry_after_ms == 40
    assert error.related_frame_id == operation.frame_id
    assert error_diagnostic == b"adapter_operation_limit"
    assert retry.scope_id == operation.operation_id
    assert retry.control_sequence == 1
    assert retry.retry_after_ms == 40
    assert retry.flags == 0x02
    assert retry_diagnostic == error_diagnostic
    assert [metadata.control_sequence for metadata, _body in session.retry_after_updates] == [1, 2]


@pytest.mark.asyncio
async def test_negotiated_transient_backend_rejection_emits_recovery_controls(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    operation = FakeOperation(
        operation_id=72,
        frame_id=20,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=72),
        terminal_results=[],
        on_terminal=stop_event.set,
    )
    session = ScriptedEventSession(
        [
            FakeServerEvent(runtime=_capability_event("control.recoverable_error")),
            FakeServerEvent(submit=operation),
        ],
        operation=operation,
        backend_started=threading.Event(),
        stop_event=stop_event,
    )
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(FakeServerContext(FakeServer(session))),  # type: ignore[arg-type]
    )

    statistics = await serve(
        OpenAiNnrpAdapter(OverloadedBackend()),
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
    assert len(session.capability_negotiations) == 1
    assert len(session.recoverable_errors) == 1
    assert session.recoverable_errors[0][1] == b"scheduler_rejected"
    assert len(session.retry_after_updates) == 1
    assert session.retry_after_updates[0][0].scope_id == operation.operation_id
    assert session.retry_after_updates[0][0].retry_after_ms == 10
    metadata, body = operation.terminal_results[0]
    assert metadata.status_code == 503
    assert _decode_terminal_profile_body(metadata, body)["error"]["code"] == "scheduler_rejected"
    observation = _observation_records(caplog)[0]
    assert observation["retry_after_ms"] == 10
    assert observation["retry_reason_code"] == 0
    assert observation["retry_source"] == "runtime"


@pytest.mark.asyncio
async def test_unnegotiated_backend_failure_does_not_emit_recovery_controls() -> None:
    operation = FakeOperation(
        operation_id=74,
        frame_id=22,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=74),
        terminal_results=[],
    )
    registry = OperationRegistry()
    record = registry.register(operation.operation_id, "request-74")
    record.transition(OperationState.QUEUED)
    control = OperationControlSlot(operation.operation_id)
    observation = _OperationObservationTracker.from_operation(
        operation,
        selected_transport="ipc",
        active_profile_id=0,
        backend_family="OverloadedBackend",
    )
    session = FakeSession(operation, asyncio.Event())
    native = _NativeCallExecutor(1)
    reporter = _RecoveryReporter(session=session, native=native, retry_after_ms=30)  # type: ignore[arg-type]
    try:
        await _serve_operation(
            OpenAiNnrpAdapter(OverloadedBackend()),
            operation,  # type: ignore[arg-type]
            record=record,
            control=control,
            observation=observation,
            observation_sinks=(),
            counters=_ServeCounters(),
            recovery=reporter,
        )
    finally:
        native.close()

    assert session.recoverable_errors == []
    assert session.retry_after_updates == []


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
    session = FakeSession(operation, stop_event, negotiate_progress=True)
    server_context = FakeServerContext(
        FakeServer(
            session,
            bound_provider_endpoints={
                "ipc": NativeTransportEndpoint(
                    uri=LOCAL_IPC_ENDPOINT,
                    scheme=LOCAL_IPC_SCHEME,
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
        provider_routes={"ipc": NativeServerProviderRoute(provider_endpoint=LOCAL_IPC_ENDPOINT)},
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
    assert captured_options[0].provider_routes["ipc"].provider_endpoint == LOCAL_IPC_ENDPOINT
    startup = _startup_observation_records(caplog)[0]
    assert startup == {
        "application_endpoint": "nnrp://runtime.local/vllm",
        "bound_provider_endpoints": {"ipc": LOCAL_IPC_ENDPOINT},
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
    assert session.credit_updates == []
    assert session.backpressure_updates == []


@pytest.mark.asyncio
async def test_output_credit_blocks_backend_pull_until_peer_grants_window() -> None:
    operation = FakeOperation(
        operation_id=73,
        frame_id=21,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=73),
        terminal_results=[],
    )
    registry = OperationRegistry()
    record = registry.register(operation.operation_id, "request-73")
    record.transition(OperationState.QUEUED)
    control = OperationControlSlot(operation.operation_id)
    observation = _OperationObservationTracker.from_operation(
        operation,
        selected_transport="ipc",
        active_profile_id=0,
        backend_family="StreamingBackend",
        backend_binding=None,
        vllm_version=None,
    )
    credits = OutboundCreditController()
    await credits.apply(
        MessageType.CREDIT_UPDATE,
        _pressure_metadata(scope_id=operation.operation_id, credit_window=0, flags=0x2),
    )
    backend = StreamingBackend()

    task = asyncio.create_task(
        _serve_operation(
            OpenAiNnrpAdapter(backend),
            operation,
            record=record,
            control=control,
            observation=observation,
            observation_sinks=(),
            counters=_ServeCounters(),
            output_credits=credits,
        )
    )
    await asyncio.sleep(0)
    assert backend.requests == []

    await credits.apply(
        MessageType.CREDIT_UPDATE,
        _pressure_metadata(scope_id=operation.operation_id, credit_window=1, flags=0x2),
    )
    for _ in range(100):
        if len(operation.partial_results) == 1:
            break
        await asyncio.sleep(0.001)
    assert len(operation.partial_results) == 1
    assert task.done() is False

    await credits.apply(
        MessageType.CREDIT_UPDATE,
        _pressure_metadata(scope_id=operation.operation_id, credit_window=2, flags=0x2),
    )
    await asyncio.wait_for(task, timeout=1)

    assert [json.loads(body)["type"] for _metadata, body in operation.partial_results] == [
        "response.output_text.delta",
        "response.usage",
    ]
    assert _decode_terminal_profile_body(*operation.terminal_results[0])["type"] == "response.completed"


@pytest.mark.asyncio
async def test_output_credit_uses_effective_connection_session_and_operation_window() -> None:
    credits = OutboundCreditController()
    connection = await credits.apply(
        MessageType.CREDIT_UPDATE,
        _pressure_metadata(credit_window=2, flags=0x1),
    )
    session = await credits.apply(MessageType.CREDIT_UPDATE, _pressure_metadata(credit_window=1))
    operation = await credits.apply(
        MessageType.CREDIT_UPDATE,
        _pressure_metadata(scope_id=91, credit_window=1, flags=0x2),
    )
    assert (connection.scope_kind, connection.scope_id) == ("connection", 0)
    assert (session.scope_kind, session.scope_id) == ("session", 0)
    assert (operation.scope_kind, operation.scope_id) == ("operation", 91)

    await credits.reserve(91)
    blocked = asyncio.create_task(credits.reserve(91))
    await asyncio.sleep(0)
    assert blocked.done() is False

    await credits.apply(MessageType.CREDIT_UPDATE, _pressure_metadata(credit_window=1))
    await asyncio.sleep(0)
    assert blocked.done() is False
    await credits.apply(
        MessageType.CREDIT_UPDATE,
        _pressure_metadata(scope_id=91, credit_window=1, flags=0x2),
    )
    await asyncio.wait_for(blocked, timeout=1)


@pytest.mark.asyncio
async def test_paused_pressure_and_invalid_scope_do_not_leak_credit() -> None:
    credits = OutboundCreditController()
    with pytest.raises(ValueError, match="cannot target connection and operation"):
        await credits.apply(
            MessageType.CREDIT_UPDATE,
            _pressure_metadata(scope_id=1, credit_window=1, flags=0x3),
        )
    with pytest.raises(ValueError, match="non-zero scope_id"):
        await credits.apply(
            MessageType.CREDIT_UPDATE,
            _pressure_metadata(credit_window=1, flags=0x2),
        )

    await credits.apply(
        MessageType.BACKPRESSURE,
        _pressure_metadata(credit_window=4, pressure_level=3),
    )
    blocked = asyncio.create_task(credits.reserve(92))
    await asyncio.sleep(0)
    assert blocked.done() is False
    await credits.apply(MessageType.CREDIT_UPDATE, _pressure_metadata(credit_window=1))
    await asyncio.wait_for(blocked, timeout=1)


@pytest.mark.asyncio
async def test_output_credit_honors_retry_delay_and_refunds_unused_reservation() -> None:
    credits = OutboundCreditController()
    await credits.apply(
        MessageType.BACKPRESSURE,
        _pressure_metadata(credit_window=1, pressure_level=1, retry_after_ms=20),
    )
    started = asyncio.get_running_loop().time()
    reservation = await asyncio.wait_for(credits.reserve(93), timeout=1)
    assert asyncio.get_running_loop().time() - started >= 0.015

    blocked = asyncio.create_task(credits.reserve(93))
    await asyncio.sleep(0)
    assert blocked.done() is False
    await credits.refund(reservation)
    await asyncio.wait_for(blocked, timeout=1)


@pytest.mark.asyncio
async def test_output_credit_rejects_invalid_frames_and_closes_waiters() -> None:
    credits = OutboundCreditController()
    with pytest.raises(ValueError, match="requires BACKPRESSURE or CREDIT_UPDATE"):
        await credits.apply(MessageType.PROGRESS, _pressure_metadata(credit_window=1))
    with pytest.raises(ValueError, match="non-zero pressure level"):
        await credits.apply(MessageType.BACKPRESSURE, _pressure_metadata(credit_window=1))

    await credits.apply(MessageType.CREDIT_UPDATE, _pressure_metadata(credit_window=0))
    blocked = asyncio.create_task(credits.reserve(94))
    await asyncio.sleep(0)
    await credits.close()
    with pytest.raises(asyncio.CancelledError):
        await blocked


@pytest.mark.asyncio
async def test_operation_observation_records_effective_output_pressure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=95,
        frame_id=195,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=95),
        terminal_results=[],
        on_terminal=stop_event.set,
    )
    session = ScriptedEventSession(
        [
            FakeServerEvent(runtime=_capability_event("control.credit_backpressure")),
            FakeServerEvent(submit=operation),
            FakeServerEvent(
                runtime=_pressure_event(
                    MessageType.BACKPRESSURE,
                    scope_id=95,
                    credit_window=0,
                    pressure_level=3,
                    pressure_reason=7,
                    retry_after_ms=25,
                    flags=0x02,
                ),
                wait_for_backend=True,
            ),
            FakeServerEvent(
                runtime=_control_event(MessageType.CANCEL, operation_id=95, sequence=1),
                wait_for_backend=True,
            ),
        ],
        operation=operation,
        backend_started=backend.started,
        stop_event=stop_event,
    )
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(FakeServerContext(MultiSessionFakeServer([session]))),
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
    assert observation["pressure_scope"] == "operation"
    assert observation["pressure_scope_id"] == 95
    assert observation["pressure_credit_window"] == 0
    assert observation["pressure_level"] == 3
    assert observation["pressure_reason"] == 7
    assert observation["pressure_retry_after_ms"] == 25


@pytest.mark.asyncio
async def test_terminal_send_cancellation_before_write_preserves_disconnect_drop() -> None:
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
        active_profile_id=0,
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

    assert record.state is OperationState.DROPPED
    assert record.resources_released is True
    assert counters.terminal_results == 0
    assert operation.terminal_results == []
    assert operation.result_drops == []


@pytest.mark.asyncio
async def test_partial_send_cancellation_after_write_preserves_output_observation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    operation = FakeOperation(
        operation_id=73,
        frame_id=21,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=73),
        terminal_results=[],
    )
    registry = OperationRegistry()
    record = registry.register(operation.operation_id, "request-73")
    record.transition(OperationState.QUEUED)
    control = OperationControlSlot(operation.operation_id)
    control.terminal_request = RuntimeControlRequest(
        kind=RuntimeControlKind.CANCEL,
        operation_id=operation.operation_id,
        control_sequence=1,
        reason_code=1,
        source_role=RuntimeRole.CLIENT,
        flags=0,
        diagnostic=b"cancelled",
    )
    observation = _OperationObservationTracker.from_operation(
        operation,
        selected_transport="tcp",
        active_profile_id=0,
        backend_family="StreamingBackend",
        backend_binding=None,
        vllm_version=None,
    )
    counters = _ServeCounters()
    serving_task: asyncio.Task[None] | None = None

    def cancel_after_write(active: bool) -> None:
        if not active:
            assert serving_task is not None
            serving_task.cancel()

    operation.partial_send_observer = cancel_after_write
    with caplog.at_level(logging.INFO, logger="vllm_nnrp_adapter.operations"):
        serving_task = asyncio.create_task(
            _serve_operation(
                OpenAiNnrpAdapter(StreamingBackend()),
                operation,
                record=record,
                control=control,
                observation=observation,
                observation_sinks=(StructuredLogObservationSink(),),
                counters=counters,
            )
        )
        await serving_task

    assert record.state is OperationState.CANCELLED
    assert len(operation.partial_results) == 1
    assert counters.partial_results == 1
    emitted = _observation_records(caplog)[0]
    assert emitted["terminal_outcome"] == "cancelled"
    assert emitted["output_event_count"] == 1
    assert len(operation.result_drops) == 1


@pytest.mark.asyncio
async def test_priority_update_reaches_backend_admission_without_request_body_pollution() -> None:
    operation = FakeOperation(
        operation_id=76,
        frame_id=26,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=76),
        terminal_results=[],
    )
    registry = OperationRegistry()
    record = registry.register(operation.operation_id, "request-76")
    record.transition(OperationState.QUEUED)
    control = OperationControlSlot(operation.operation_id)
    assert (
        control.apply_priority(
            RuntimePriorityUpdate(operation.operation_id, 1, 2, -5, 0),
            terminal=False,
        )
        is RuntimeControlDisposition.APPLIED
    )
    observation = _OperationObservationTracker.from_operation(
        operation,
        selected_transport="ipc",
        active_profile_id=0,
        backend_family="PriorityAwareStreamingBackend",
        backend_binding=None,
        vllm_version=None,
    )
    backend = PriorityAwareStreamingBackend()

    await _serve_operation(
        OpenAiNnrpAdapter(backend),
        operation,
        record=record,
        control=control,
        observation=observation,
        observation_sinks=(),
        counters=_ServeCounters(),
    )

    assert backend.priorities == [-3]
    assert len(backend.requests) == 1
    assert "priority" not in backend.requests[0]
    assert _decode_terminal_profile_body(*operation.terminal_results[0])["type"] == "response.completed"


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


@pytest.mark.asyncio
async def test_capability_negotiation_returns_only_supported_runtime_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = await _serve_capability_request(
        monkeypatch,
        _capability_event(
            "control.capability_costs",
            "control.route_execution_hint",
            cost_model_id=6,
            preference_rank=3,
            limit_bytes=8_192,
            limit_units=512,
        ),
        backend=StreamingBackend(),
    )

    assert session.profile_degradations == []
    assert len(session.capability_negotiations) == 1
    metadata, body = session.capability_negotiations[0]
    assert _decode_capability_body(body) == ("control.capability_costs",)
    assert metadata == CapabilityMetadata(
        profile_id=0,
        capability_count=1,
        cost_model_id=0,
        preference_rank=0,
        limit_bytes=0,
        limit_units=0,
        body_bytes=len(body),
        flags=0,
    )
    assert session.recoverable_errors == []


@pytest.mark.asyncio
async def test_capability_negotiation_advertises_credit_backpressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = await _serve_capability_request(
        monkeypatch,
        _capability_event("control.credit_backpressure"),
        backend=StreamingBackend(),
    )

    assert len(session.capability_negotiations) == 1
    _metadata, body = session.capability_negotiations[0]
    assert _decode_capability_body(body) == ("control.credit_backpressure",)
    assert [metadata.credit_window for metadata in session.credit_updates] == [1]
    assert session.backpressure_updates == []


@pytest.mark.asyncio
async def test_capability_negotiation_emits_degrade_only_when_request_allows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = await _serve_capability_request(
        monkeypatch,
        _capability_event(
            "control.capability_costs",
            "control.route_execution_hint",
            flags=0x0000_0002,
        ),
        backend=StreamingBackend(),
    )

    assert session.capability_negotiations == []
    assert len(session.profile_degradations) == 1
    metadata, body = session.profile_degradations[0]
    assert metadata.flags == 0x0000_0002
    assert _decode_capability_body(body) == ("control.capability_costs",)
    assert session.recoverable_errors == []


@pytest.mark.asyncio
async def test_hard_capability_mismatch_returns_empty_subset_and_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = await _serve_capability_request(
        monkeypatch,
        _capability_event("control.route_execution_hint", flags=0x0000_0001),
        backend=StreamingBackend(),
    )

    assert session.profile_degradations == []
    assert len(session.capability_negotiations) == 1
    response, body = session.capability_negotiations[0]
    assert response.capability_count == 0
    assert response.body_bytes == 0
    assert body == b""
    assert len(session.recoverable_errors) == 1
    error, diagnostic = session.recoverable_errors[0]
    assert error.error_code is ErrorCode.UNSUPPORTED_CAPABILITY
    assert error.error_scope is ErrorScope.SESSION
    assert diagnostic == b"capability_mismatch"


@pytest.mark.asyncio
async def test_capability_negotiation_advertises_priority_only_for_supporting_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = await _serve_capability_request(
        monkeypatch,
        _capability_event("control.priority_update"),
        backend=PriorityAwareStreamingBackend(),
    )

    assert len(session.capability_negotiations) == 1
    _metadata, body = session.capability_negotiations[0]
    assert _decode_capability_body(body) == ("control.priority_update",)


@pytest.mark.asyncio
async def test_unresolved_runtime_object_is_rejected_before_vllm_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    backend = StreamingBackend()
    operation = FakeOperation(
        operation_id=901,
        frame_id=902,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=901),
        terminal_results=[],
        on_terminal=stop_event.set,
    )
    session = ScriptedEventSession(
        [
            FakeServerEvent(runtime=_object_descriptor_event()),
            FakeServerEvent(runtime=_object_reference_event(operation.operation_id)),
            FakeServerEvent(submit=operation),
        ],
        operation=operation,
        backend_started=threading.Event(),
        stop_event=stop_event,
    )
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(FakeServerContext(MultiSessionFakeServer([session]))),
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

    assert backend.requests == []
    assert statistics.terminal_results == 1
    event = _decode_terminal_profile_body(*operation.terminal_results[0])
    assert event["error"]["code"] == "runtime_object_unavailable"


@pytest.mark.asyncio
async def test_cache_reference_without_provider_returns_typed_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    operation = FakeOperation(
        operation_id=901,
        frame_id=902,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=901),
        terminal_results=[],
    )
    session = ScriptedEventSession(
        [FakeServerEvent(runtime=_cache_reference_event())],
        operation=operation,
        backend_started=threading.Event(),
        stop_event=stop_event,
        stop_after_last_event=True,
    )
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(FakeServerContext(MultiSessionFakeServer([session]))),
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

    assert statistics.terminal_results == 0
    assert len(session.cache_misses) == 1
    miss, diagnostic = session.cache_misses[0]
    assert miss.cache_namespace == 7
    assert miss.cache_key_hi == 11
    assert miss.cache_key_lo == 12
    assert diagnostic == b"adapter_cache_provider_unavailable"


@pytest.mark.asyncio
async def test_runtime_object_invalidation_stops_live_backend_and_reports_drop(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=901,
        frame_id=902,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=901),
        terminal_results=[],
        on_terminal=stop_event.set,
    )
    session = ScriptedEventSession(
        [
            FakeServerEvent(submit=operation),
            FakeServerEvent(runtime=_object_descriptor_event(), wait_for_backend=True),
            FakeServerEvent(runtime=_object_reference_event(operation.operation_id)),
            FakeServerEvent(runtime=_object_release_event(operation.operation_id)),
        ],
        operation=operation,
        backend_started=backend.started,
        stop_event=stop_event,
    )
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(FakeServerContext(MultiSessionFakeServer([session]))),
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
    drop, diagnostic = operation.result_drops[0]
    assert drop.drop_reason_code is ResultDropReasonCode.OBJECT_INVALIDATED
    assert diagnostic == b"runtime_object_33_invalidated"
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == "dropped"
    assert observation["cancellation_kind"] == "object_invalidated"
    assert observation["cancellation_reason_code"] == 5
    assert observation["drop_reason"] == "object_invalidated"


def _object_descriptor_event() -> NativeRuntimeEvent:
    metadata = ObjectDescriptorMetadata(
        object_id=33,
        object_kind=RuntimeObjectKind.IMAGE_TILE,
        producer_role=RuntimeRole.CLIENT,
        consumer_role=RuntimeRole.RUNTIME,
        session_id=1,
        byte_size=4_096,
        compute_cost_units=8,
        memory_location_hint=MemoryLocationHint.HOST_MEMORY,
        ownership_hint=OwnershipHint.SESSION_OWNED,
        lifetime_hint_ms=1_000,
        metadata_bytes=0,
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.OBJECT_DECLARE, session_id=1, frame_id=801),
        RuntimeEventMetadata(RuntimeEventMetadataKind.OBJECT_DESCRIPTOR, metadata),
        RuntimeEventTail.with_body(b""),
    )


def _object_reference_event(operation_id: int) -> NativeRuntimeEvent:
    metadata = ObjectReferenceMetadata(33, operation_id, 0, 0, 0, 0, 0)
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.OBJECT_REF, session_id=1, frame_id=802),
        RuntimeEventMetadata(RuntimeEventMetadataKind.OBJECT_REFERENCE, metadata),
        RuntimeEventTail.with_body(b""),
    )


def _object_release_event(operation_id: int) -> NativeRuntimeEvent:
    diagnostic = b"object_invalidated"
    metadata = ObjectReleaseMetadata(
        33,
        operation_id,
        ObjectReleaseReason.INVALIDATED,
        RuntimeRole.CLIENT,
        0x02,
        len(diagnostic),
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.OBJECT_RELEASE, session_id=1, frame_id=804),
        RuntimeEventMetadata(RuntimeEventMetadataKind.OBJECT_RELEASE, metadata),
        RuntimeEventTail.with_diagnostic(diagnostic),
    )


def _cache_reference_event() -> NativeRuntimeEvent:
    metadata = CacheReferenceMetadata(7, 11, 12, 3, CacheReuseScope.SESSION, 0, 0, 0, 0, 0)
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.CACHE_REFERENCE, session_id=1, frame_id=803),
        RuntimeEventMetadata(RuntimeEventMetadataKind.CACHE_REFERENCE, metadata),
        RuntimeEventTail.with_body(b""),
    )


async def _serve_capability_request(
    monkeypatch: pytest.MonkeyPatch,
    event: NativeRuntimeEvent,
    *,
    backend: object,
) -> ScriptedEventSession:
    stop_event = asyncio.Event()
    operation = FakeOperation(
        operation_id=901,
        frame_id=902,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=901),
        terminal_results=[],
    )
    session = ScriptedEventSession(
        [FakeServerEvent(runtime=event)],
        operation=operation,
        backend_started=threading.Event(),
        stop_event=stop_event,
        stop_after_last_event=True,
    )
    server_context = FakeServerContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

    await serve(
        OpenAiNnrpAdapter(backend),  # type: ignore[arg-type]
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
    return session


def _capability_event(
    *tokens: str,
    cost_model_id: int = 0,
    preference_rank: int = 0,
    limit_bytes: int = 0,
    limit_units: int = 0,
    flags: int = 0,
) -> NativeRuntimeEvent:
    body = _capability_body(*tokens)
    metadata = CapabilityMetadata(
        profile_id=0,
        capability_count=len(tokens),
        cost_model_id=cost_model_id,
        preference_rank=preference_rank,
        limit_bytes=limit_bytes,
        limit_units=limit_units,
        body_bytes=len(body),
        flags=flags,
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.CAPABILITY_NEGOTIATION, session_id=1),
        RuntimeEventMetadata(RuntimeEventMetadataKind.CAPABILITY, metadata),
        RuntimeEventTail.with_body(body),
    )


def _capability_body(*tokens: str) -> bytes:
    body = bytearray()
    for token in sorted(tokens, key=lambda value: value.encode("ascii")):
        encoded = token.encode("ascii")
        body.extend(len(encoded).to_bytes(2, "little"))
        body.extend(encoded)
    return bytes(body)


def _pressure_metadata(
    *,
    scope_id: int = 0,
    credit_window: int,
    pressure_level: int = 0,
    pressure_reason: int = 0,
    retry_after_ms: int = 0,
    flags: int = 0,
) -> PressureMetadata:
    return PressureMetadata(
        scope_id=scope_id,
        credit_window=credit_window,
        pressure_level=pressure_level,
        pressure_reason=pressure_reason,
        retry_after_ms=retry_after_ms,
        flags=flags,
    )


def _pressure_event(
    message_type: MessageType,
    *,
    scope_id: int = 0,
    credit_window: int,
    pressure_level: int = 0,
    pressure_reason: int = 0,
    retry_after_ms: int = 0,
    flags: int = 0,
) -> NativeRuntimeEvent:
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=message_type, session_id=1),
        RuntimeEventMetadata(
            RuntimeEventMetadataKind.PRESSURE,
            _pressure_metadata(
                scope_id=scope_id,
                credit_window=credit_window,
                pressure_level=pressure_level,
                pressure_reason=pressure_reason,
                retry_after_ms=retry_after_ms,
                flags=flags,
            ),
        ),
        RuntimeEventTail.none(),
    )


def _decode_capability_body(body: bytes) -> tuple[str, ...]:
    tokens: list[str] = []
    offset = 0
    while offset < len(body):
        token_length = int.from_bytes(body[offset : offset + 2], "little")
        token_start = offset + 2
        token_end = token_start + token_length
        tokens.append(body[token_start:token_end].decode("ascii"))
        offset = token_end
    return tuple(tokens)


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
        active_profile_id=0,
        clock_ns=lambda: 0,
    )
    backend_trace = _BackendTraceContextSlot()
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
        observations_by_frame={23: (record, observation, backend_trace)},
        session_trace_context=session_context,
    )

    assert retained_session_context == session_context
    assert backend_trace.begin_dispatch() == {
        "traceparent": "00-0000000000000000000000000000005b-000000000000005c-01"
    }

    late_metadata = TraceContextMetadata(111, 112, 110, 7, 0, 0)
    _apply_trace_context(
        _trace_context_event(late_metadata, b"", frame_id=23),
        observations_by_frame={23: (record, observation, backend_trace)},
        session_trace_context=session_context,
    )
    result = observation.finish(OperationState.FAILED)

    assert backend_trace.metadata is operation_metadata
    assert result.identity.trace_id == 111
    assert result.trace_span_id == 112
    assert result.trace_parent_span_id == 110
    assert result.trace_stage_code == 7
    assert result.trace_flags == 0
    assert result.trace_attribute_bytes == 0


@pytest.mark.parametrize(
    ("metadata", "expected"),
    (
        (None, None),
        (TraceContextMetadata(0, 2, 0, 0, 0, 0), None),
        (TraceContextMetadata(1, 0, 0, 0, 0, 0), None),
        (
            TraceContextMetadata(0x1234, 0x5678, 0x9ABC, 4, 3, 0),
            {"traceparent": "00-00000000000000000000000000001234-0000000000005678-01"},
        ),
    ),
)
def test_trace_context_maps_to_frozen_w3c_header(
    metadata: TraceContextMetadata | None,
    expected: Mapping[str, str] | None,
) -> None:
    assert _w3c_trace_headers(metadata) == expected


@pytest.mark.asyncio
async def test_retire_completed_tasks_consumes_expected_operation_cancellation() -> None:
    task = asyncio.create_task(asyncio.sleep(60))
    tasks = {task}
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    _retire_completed_tasks(tasks)

    assert tasks == set()


@pytest.mark.asyncio
async def test_retire_completed_tasks_still_propagates_operation_failures() -> None:
    async def fail() -> None:
        raise RuntimeError("operation failed")

    task = asyncio.create_task(fail())
    tasks = {task}
    await asyncio.wait({task})

    with pytest.raises(RuntimeError, match="operation failed"):
        _retire_completed_tasks(tasks)

    assert tasks == set()


@pytest.mark.asyncio
async def test_native_terminal_race_waits_for_control_before_emitting_drop_reason() -> None:
    operation = FakeOperation(
        operation_id=67,
        frame_id=167,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=67),
        terminal_results=[],
        fail_next_progress_invalid_state=True,
    )
    registry = OperationRegistry()
    record = registry.register(operation.operation_id, "request-67")
    record.transition(OperationState.QUEUED)
    control = OperationControlSlot(operation.operation_id)
    observation = _OperationObservationTracker.from_operation(
        operation,
        selected_transport="ipc",
        active_profile_id=0,
        backend_family="StreamingBackend",
    )
    counters = _ServeCounters()
    task = asyncio.create_task(
        _serve_operation(
            OpenAiNnrpAdapter(StreamingBackend()),
            operation,  # type: ignore[arg-type]
            record=record,
            control=control,
            observation=observation,
            observation_sinks=(),
            counters=counters,
        )
    )
    control.bind(task)
    await asyncio.sleep(0)

    disposition = await control.apply(
        RuntimeControlRequest(
            kind=RuntimeControlKind.CANCEL,
            operation_id=operation.operation_id,
            control_sequence=1,
            reason_code=3,
            source_role=RuntimeRole.CLIENT,
            flags=0,
            diagnostic=b"peer_cancelled",
        ),
        terminal=False,
    )
    await task

    assert disposition is RuntimeControlDisposition.APPLIED
    assert operation.terminal_results == []
    assert len(operation.result_drops) == 1
    assert operation.result_drops[0][0].drop_reason_code is ResultDropReasonCode.PEER_CANCELLED
    assert counters.terminal_results == 1
    assert counters.result_drops == 1


@pytest.mark.asyncio
async def test_native_terminal_result_lifecycle_race_becomes_cancelled_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    terminal_attempted = asyncio.Event()
    operation = FakeOperation(
        operation_id=68,
        frame_id=168,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=68),
        terminal_results=[],
        fail_next_terminal_lifecycle_error=True,
        terminal_send_observer=terminal_attempted.set,
    )
    registry = OperationRegistry()
    record = registry.register(operation.operation_id, "request-68")
    record.transition(OperationState.QUEUED)
    control = OperationControlSlot(operation.operation_id)
    observation = _OperationObservationTracker.from_operation(
        operation,
        selected_transport="ipc",
        active_profile_id=0,
        backend_family="StreamingBackend",
    )
    counters = _ServeCounters()
    task = asyncio.create_task(
        _serve_operation(
            OpenAiNnrpAdapter(StreamingBackend()),
            operation,  # type: ignore[arg-type]
            record=record,
            control=control,
            observation=observation,
            observation_sinks=(StructuredLogObservationSink(),),
            counters=counters,
        )
    )
    control.bind(task)
    await asyncio.wait_for(terminal_attempted.wait(), timeout=1)

    disposition = await control.apply(
        RuntimeControlRequest(
            kind=RuntimeControlKind.CANCEL,
            operation_id=operation.operation_id,
            control_sequence=1,
            reason_code=3,
            source_role=RuntimeRole.CLIENT,
            flags=0,
            diagnostic=b"peer_cancelled",
        ),
        terminal=False,
    )
    await task

    assert disposition is RuntimeControlDisposition.APPLIED
    assert operation.terminal_results == []
    assert len(operation.partial_results) == 2
    assert len(operation.result_drops) == 1
    assert operation.result_drops[0][0].drop_reason_code is ResultDropReasonCode.PEER_CANCELLED
    assert counters.partial_results == 2
    assert counters.terminal_results == 1
    assert counters.result_drops == 1
    observation_record = _observation_records(caplog)[0]
    assert observation_record["terminal_outcome"] == "cancelled"
    assert observation_record["output_event_count"] == 2
    assert observation_record["cancellation_kind"] == "cancel"
    assert observation_record["drop_reason"] == "peer_cancelled"


@pytest.mark.asyncio
async def test_late_disconnect_does_not_reclassify_completed_observation() -> None:
    operation = FakeOperation(
        operation_id=69,
        frame_id=169,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=69),
        terminal_results=[],
    )
    registry = OperationRegistry()
    record = registry.register(operation.operation_id, "request-69")
    record.transition(OperationState.QUEUED)
    record.transition(OperationState.ADMITTED)
    record.terminate(OperationState.COMPLETED)
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
        active_profile_id=0,
    )

    finalized = await _finalize_cancelled_operation(
        operation,  # type: ignore[arg-type]
        record=record,
        control=control,
        observation=observation,
        result_sequence=0,
        counters=_ServeCounters(),
        session=None,
        native=None,
    )
    result = observation.finish(record.state)

    assert finalized is True
    assert result.terminal_outcome == "completed"
    assert result.cancellation_kind is None
    assert result.cancellation_source is None
    assert result.drop_reason is None


@pytest.mark.asyncio
async def test_operation_trace_can_replace_session_default_before_backend_dispatch() -> None:
    backend = TraceAwareStreamingBackend()
    adapter = OpenAiNnrpAdapter(backend)
    session_metadata = TraceContextMetadata(0x11, 0x12, 0, 0, 0, 0)
    operation_metadata = TraceContextMetadata(0x21, 0x22, 0, 0, 1, 0)
    backend_trace = _BackendTraceContextSlot(session_metadata)
    request = _chat_request()
    request["nnrp"] = {"diagnostics": True}
    events = adapter._handle_native_request(
        request,
        backend_trace_headers_factory=backend_trace.begin_dispatch,
    )

    diagnostics = await anext(events)
    assert diagnostics["type"] == "response.diagnostics"
    assert backend_trace.dispatched is False
    assert backend_trace.update(operation_metadata) is True

    first_result = await anext(events)
    assert first_result["type"] == "response.output_text.delta"
    assert backend.trace_headers == [
        {"traceparent": "00-00000000000000000000000000000021-0000000000000022-01"}
    ]
    assert backend_trace.dispatched is True
    await events.aclose()


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


@pytest.mark.asyncio
async def test_native_server_forwards_session_trace_to_trace_aware_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    backend = TraceAwareStreamingBackend()
    operation = FakeOperation(
        operation_id=75,
        frame_id=25,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=75),
        terminal_results=[],
        on_terminal=stop_event.set,
    )
    trace_metadata = TraceContextMetadata(0x1234, 0x5678, 0, 3, 3, 0)
    session = ScriptedEventSession(
        [
            FakeServerEvent(runtime=_trace_context_event(trace_metadata, b"", frame_id=0)),
            FakeServerEvent(submit=operation),
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
            observation_sinks=(),
        ),
        stop_event=stop_event,
    )

    assert backend.trace_headers == [
        {"traceparent": "00-00000000000000000000000000001234-0000000000005678-01"}
    ]
    assert len(backend.requests) == 1
    assert "traceparent" not in backend.requests[0]
    assert "trace_headers" not in backend.requests[0]


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


@pytest.mark.parametrize(
    ("transport_name", "provider_endpoint", "message"),
    [
        ("udp", "tcp://127.0.0.1:7766", "unsupported transport names"),
        ("tcp", "ws://127.0.0.1:7766/nnrp", "cannot use websocket carrier endpoint"),
        ("ipc", None, "requires an explicit provider endpoint"),
        ("websocket", None, "requires an explicit provider endpoint"),
    ],
)
def test_server_config_rejects_invalid_provider_route_identity(
    transport_name: str,
    provider_endpoint: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            provider_routes={transport_name: NativeServerProviderRoute(provider_endpoint=provider_endpoint)},
        )


@pytest.mark.parametrize(
    ("route", "message"),
    [
        (NativeServerProviderRoute(provider_endpoint=object()), "provider route endpoint"),  # type: ignore[arg-type]
        (NativeServerProviderRoute(security=object()), "provider route security"),  # type: ignore[arg-type]
    ],
)
def test_server_config_rejects_invalid_provider_route_values(
    route: NativeServerProviderRoute,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            provider_routes={"tcp": route},
        )


@pytest.mark.parametrize(
    ("transport_name", "provider_endpoint", "security", "message"),
    [
        ("quic", "quic://127.0.0.1:7767", None, "quic provider route requires"),
        ("websocket", "wss://127.0.0.1:7768/nnrp", None, "wss provider route requires"),
        ("websocket", "ws://127.0.0.1:7768/nnrp", "server", "ws provider route must not"),
        ("ipc", LOCAL_IPC_ENDPOINT, "server", "ipc provider route must not"),
    ],
)
def test_server_config_enforces_route_local_security(
    transport_name: str,
    provider_endpoint: str,
    security: str | None,
    message: str,
) -> None:
    server_security = (
        NativeTransportServerSecurity(certificate_der=b"certificate", private_key_pkcs8_der=b"private-key")
        if security is not None
        else None
    )
    with pytest.raises(ValueError, match=message):
        NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            provider_routes={
                transport_name: NativeServerProviderRoute(
                    provider_endpoint=provider_endpoint,
                    security=server_security,
                )
            },
        )


@pytest.mark.parametrize(
    ("platform_name", "provider_endpoint", "required_scheme"),
    [
        ("nt", "unix:///tmp/nnrp-vllm.sock", "npipe"),
        ("posix", "npipe://nnrp-vllm", "unix"),
    ],
)
def test_server_config_rejects_ipc_endpoint_for_another_platform(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    provider_endpoint: str,
    required_scheme: str,
) -> None:
    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.os", SimpleNamespace(name=platform_name))

    with pytest.raises(ValueError, match=rf"requires a {required_scheme}:// endpoint"):
        NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            provider_routes={"ipc": NativeServerProviderRoute(provider_endpoint=provider_endpoint)},
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
                provider_routes={"ipc": NativeServerProviderRoute(provider_endpoint=LOCAL_IPC_ENDPOINT)},
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
async def test_multi_provider_listener_failure_keeps_logical_server_atomic(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    captured_options: list[NativeServerBootstrapOptions] = []
    ready_endpoints: list[Mapping[str, NativeTransportEndpoint]] = []
    executors: list[_NativeCallExecutor] = []

    class TrackingNativeCallExecutor(_NativeCallExecutor):
        def __init__(self, max_workers: int) -> None:
            super().__init__(max_workers)
            self.closed = False
            executors.append(self)

        def close(self) -> None:
            self.closed = True
            super().close()

    def fail_logical_listener(
        options: NativeServerBootstrapOptions,
        *,
        transports: object = None,
    ) -> FailingServerContext:
        assert transports is None
        captured_options.append(options)
        return FailingServerContext()

    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime._NativeCallExecutor", TrackingNativeCallExecutor)
    monkeypatch.setattr("vllm_nnrp_adapter.nnrp_runtime.listen_native_server", fail_logical_listener)

    routes = {
        "tcp": NativeServerProviderRoute(provider_endpoint="tcp://127.0.0.1:0"),
        "ipc": NativeServerProviderRoute(provider_endpoint=LOCAL_IPC_ENDPOINT),
    }
    with pytest.raises(OSError, match="listener failed before admission"):
        await _serve(
            OpenAiNnrpAdapter(StreamingBackend()),
            config=NnrpServerConfig(
                endpoint="nnrp://runtime.local/vllm",
                provider_routes=routes,
            ),
            stop_event=None,
            on_ready=ready_endpoints.append,
        )

    assert len(captured_options) == 1
    assert captured_options[0].provider_routes == routes
    assert ready_endpoints == []
    assert len(executors) == 1
    assert executors[0].closed is True
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
    assert operation.progress_results == []
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
        MultiOperationFakeSession(
            operations[:2],
            operation_accepted=lambda: None,
            connection_identity=(10_101, 11),
            session_identity=(20_101, 21),
        ),
        MultiOperationFakeSession(
            operations[2:],
            operation_accepted=lambda: None,
            connection_identity=(10_102, 12),
            session_identity=(20_102, 22),
        ),
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
    assert all(operation.terminal_results[0][0].active_profile_id == 0 for operation in operations)
    observation_records = [
        json.loads(record.getMessage().removeprefix("nnrp_operation_observation "))
        for record in caplog.records
        if record.getMessage().startswith("nnrp_operation_observation ")
    ]
    assert len(observation_records) == len(operations)
    assert {record["operation_id"] for record in observation_records} == {1, 2, 3, 4}
    for observation in observation_records:
        operation_id = observation["operation_id"]
        session_index = 1 if operation_id <= 2 else 2
        assert observation["connection_id"] == 10_100 + session_index
        assert observation["connection_generation"] == 10 + session_index
        assert observation["session_handle_id"] == 20_100 + session_index
        assert observation["session_generation"] == 20 + session_index
        assert observation["session_id"] == 1
        assert observation["frame_id"] == operation_id + 100
        assert observation["route_id"] == operation_id + 1_000
        assert observation["view_id"] == operation_id + 2_000
        assert observation["trace_id"] == operation_id + 3_000
        assert observation["profile_id"] == 0
        assert observation["model_id"] == f"model-{operation_id}"
        assert observation["profile_operation"] == "chat.completions.create"
        assert observation["backend_family"] == "InterleavingBackend"
        assert observation["backend_binding"] is None
        assert observation["vllm_version"] is None
        assert observation["selected_transport"] == "ipc"
        assert observation["output_event_count"] == 3
        assert observation["terminal_outcome"] == "completed"


@pytest.mark.asyncio
async def test_native_server_never_accepts_more_sessions_than_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    first_terminal = threading.Event()
    second_terminal = threading.Event()
    first = FakeOperation(
        operation_id=81,
        frame_id=181,
        body=_typed_profile_body(_chat_request(model="first")),
        metadata=_submit_metadata(operation_id=81),
        terminal_results=[],
        on_terminal=first_terminal.set,
    )

    def finish_second() -> None:
        second_terminal.set()
        stop_event.set()

    second = FakeOperation(
        operation_id=82,
        frame_id=182,
        body=_typed_profile_body(_chat_request(model="second")),
        metadata=_submit_metadata(operation_id=82),
        terminal_results=[],
        on_terminal=finish_second,
    )
    sessions = [
        ScriptedEventSession(
            [
                FakeServerEvent(submit=first),
                FakeServerEvent(runtime=_session_close_event(last_operation_id=81), wait_for_backend=True),
            ],
            operation=first,
            backend_started=first_terminal,
            stop_event=stop_event,
        ),
        ScriptedEventSession(
            [
                FakeServerEvent(submit=second),
                FakeServerEvent(runtime=_session_close_event(last_operation_id=82), wait_for_backend=True),
            ],
            operation=second,
            backend_started=second_terminal,
            stop_event=stop_event,
        ),
    ]
    server = MultiSessionFakeServer(sessions)
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(FakeServerContext(server)),
    )

    serve_task = asyncio.create_task(
        serve(
            OpenAiNnrpAdapter(backend),
            config=NnrpServerConfig(
                endpoint="nnrp://runtime.local/vllm",
                accept_timeout_ms=10,
                receive_timeout_ms=10,
                max_active_sessions=1,
                max_operations_per_session=1,
                native_worker_count=3,
                observation_sinks=(),
            ),
            stop_event=stop_event,
        )
    )
    assert await asyncio.to_thread(backend.started.wait, 1)
    await asyncio.sleep(0.02)
    assert server.accepted_count == 1

    backend._release.set()
    statistics = await asyncio.wait_for(serve_task, timeout=2)

    assert server.accepted_count == 2
    assert statistics.accepted_sessions == 2
    assert statistics.accepted_operations == 2
    assert all(session.closed for session in sessions)


@pytest.mark.asyncio
async def test_native_server_bounds_operations_and_pending_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    output_gate = asyncio.Event()
    pending_ready = asyncio.Event()
    third_rejected = asyncio.Event()
    pending_outputs = 0
    max_pending_outputs = 0
    completed = 0

    def observe_pending(started: bool) -> None:
        nonlocal pending_outputs, max_pending_outputs
        pending_outputs += 1 if started else -1
        max_pending_outputs = max(max_pending_outputs, pending_outputs)
        if max_pending_outputs == 2:
            pending_ready.set()

    def finish_admitted() -> None:
        nonlocal completed
        completed += 1
        if completed == 2:
            stop_event.set()

    operations = [
        FakeOperation(
            operation_id=operation_id,
            frame_id=operation_id + 100,
            body=_typed_profile_body(_chat_request(model=f"model-{operation_id}")),
            metadata=_submit_metadata(operation_id=operation_id),
            terminal_results=[],
            on_terminal=finish_admitted if operation_id < 3 else third_rejected.set,
            partial_send_gate=output_gate if operation_id < 3 else None,
            partial_send_observer=observe_pending if operation_id < 3 else None,
        )
        for operation_id in range(1, 4)
    ]
    session = MultiOperationFakeSession(operations, operation_accepted=lambda: None)
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(FakeServerContext(MultiSessionFakeServer([session]))),
    )
    backend = StreamingBackend()
    serve_task = asyncio.create_task(
        serve(
            OpenAiNnrpAdapter(backend),
            config=NnrpServerConfig(
                endpoint="nnrp://runtime.local/vllm",
                accept_timeout_ms=10,
                receive_timeout_ms=10,
                max_active_sessions=1,
                max_operations_per_session=2,
                native_worker_count=4,
                observation_sinks=(),
            ),
            stop_event=stop_event,
        )
    )

    await asyncio.wait_for(pending_ready.wait(), timeout=1)
    await asyncio.wait_for(third_rejected.wait(), timeout=1)
    assert pending_outputs == 2
    assert max_pending_outputs == 2
    assert len(backend.requests) == 2

    output_gate.set()
    statistics = await asyncio.wait_for(serve_task, timeout=2)

    assert pending_outputs == 0
    assert statistics.accepted_sessions == 1
    assert statistics.accepted_operations == 2
    assert statistics.terminal_results == 3
    rejected = _decode_terminal_profile_body(*operations[2].terminal_results[0])
    assert rejected["error"]["code"] == "adapter_operation_limit"


@pytest.mark.asyncio
async def test_native_server_rejects_duplicate_operation_without_corrupting_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    terminal_count = 0

    def operation_terminal() -> None:
        nonlocal terminal_count
        terminal_count += 1
        if terminal_count == 2:
            loop.call_soon_threadsafe(stop_event.set)

    operations = [
        FakeOperation(
            operation_id=91,
            frame_id=frame_id,
            body=_typed_profile_body(_chat_request()),
            metadata=_submit_metadata(operation_id=91),
            terminal_results=[],
            on_terminal=operation_terminal,
        )
        for frame_id in (191, 192)
    ]
    session = MultiOperationFakeSession(operations, operation_accepted=lambda: None)
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


@pytest.mark.parametrize("message_type", [MessageType.CANCEL, MessageType.ABORT])
@pytest.mark.asyncio
async def test_native_control_stops_backend_and_emits_one_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
    message_type: MessageType,
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
    assert operation.terminal_results == []
    assert len(operation.result_drops) == 1
    assert operation.result_drops[0][0].operation_id == 101
    assert operation.result_drops[0][0].drop_reason_code is ResultDropReasonCode.PEER_CANCELLED
    assert len(session.trace_contexts) == 1
    trace_metadata, trace_body, trace_operation_id = session.trace_contexts[0]
    assert trace_metadata.trace_id == operation.submit.header.trace_id
    assert trace_metadata.span_id == 201
    assert trace_body == b""
    assert trace_operation_id is None
    assert all(metadata.stage_code != 0x000A for metadata, _body in operation.progress_results)
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == (
        "dropped" if message_type is MessageType.ABORT else "cancelled"
    )
    assert observation["cancellation_kind"] == message_type.name.lower()
    assert observation["cancellation_source"] == "client"
    assert observation["cancellation_reason_code"] == 3
    assert observation["backend_abort_accepted"] is True
    assert observation["drop_reason"] == "peer_cancelled"


@pytest.mark.asyncio
async def test_native_cancel_aborts_vllm_engine_with_derived_request_id(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture_observations(caplog)
    binding = VLLM_COMPATIBILITY_BINDINGS[0].engine_direct
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.utils",
        SimpleNamespace(
            get_max_tokens=lambda _max_model_len, requested, _prompt_len, _defaults, _override: requested
        ),
    )

    class DirectRequest:
        stream = True
        use_beam_search = False
        tools = None
        priority = 0
        max_tokens = 16
        max_completion_tokens = None
        truncate_prompt_tokens = None
        include_reasoning = True

        def __init__(self, body: Mapping[str, Any]) -> None:
            self.request_id = body["request_id"]

        def to_sampling_params(
            self,
            max_tokens: int,
            default_sampling_params: Mapping[str, Any],
        ) -> dict[str, Any]:
            return {"max_tokens": max_tokens, **default_sampling_params}

    class DirectRequestFactory:
        engine_direct_binding = binding

        def __call__(self, body: Mapping[str, Any]) -> DirectRequest:
            return DirectRequest(body)

    class RecordingEngineClient:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.aborted: list[str] = []
            self._release = asyncio.Event()

        def generate(
            self,
            _engine_prompt: object,
            _sampling_params: object,
            _request_id: str,
            **_kwargs: object,
        ) -> AsyncIterator[object]:
            async def outputs() -> AsyncIterator[object]:
                self.started.set()
                yield SimpleNamespace(
                    prompt_token_ids=[1, 2, 3],
                    encoder_prompt_token_ids=None,
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            text="first",
                            token_ids=[10],
                            finish_reason=None,
                            stop_reason=None,
                        )
                    ],
                    finished=False,
                )
                await self._release.wait()

            return outputs()

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    class DirectServingChat:
        model_config = SimpleNamespace(max_model_len=1_024)
        default_sampling_params = {"temperature": 0.2}
        override_max_tokens = None
        reasoning_parser_cls = None
        tool_parser = None
        use_harmony = False
        renderer = SimpleNamespace(tokenizer=object())
        models = SimpleNamespace(model_name=lambda _adapter: "mock-model")

        def __init__(self) -> None:
            self.engine_client = RecordingEngineClient()

        async def render_chat_request(
            self,
            _request: DirectRequest,
        ) -> tuple[list[dict[str, str]], list[dict[str, list[int]]]]:
            return ([{"role": "user", "content": "hello"}], [{"prompt_token_ids": [1, 2, 3]}])

        def _extract_prompt_components(self, _engine_prompt: object) -> object:
            return object()

        def _extract_prompt_len(self, engine_prompt: Mapping[str, list[int]]) -> int:
            return len(engine_prompt["prompt_token_ids"])

        def create_chat_completion(self, _request: object) -> Mapping[str, Any]:
            raise AssertionError("streaming request must use the engine-direct path")

    stop_event = asyncio.Event()
    serving_chat = DirectServingChat()
    backend = VllmBackend(serving_chat, request_factory=DirectRequestFactory())
    operation = FakeOperation(
        operation_id=101,
        frame_id=201,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=101),
        terminal_results=[],
    )
    operation.on_terminal = stop_event.set
    session = ScriptedEventSession(
        [
            FakeServerEvent(submit=operation),
            FakeServerEvent(
                runtime=_control_event(MessageType.CANCEL, operation_id=101, sequence=1),
                wait_for_backend=True,
            ),
        ],
        operation=operation,
        backend_started=serving_chat.engine_client.started,
        stop_event=stop_event,
    )
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(FakeServerContext(MultiSessionFakeServer([session]))),
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
    assert serving_chat.engine_client.aborted == ["chatcmpl-nnrp-101-201"]
    assert operation.terminal_results == []
    assert operation.result_drops[0][0].drop_reason_code is ResultDropReasonCode.PEER_CANCELLED
    observation = _observation_records(caplog)[0]
    assert observation["backend_abort_accepted"] is True
    assert observation["terminal_outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_live_priority_update_returns_typed_recoverable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=109,
        frame_id=209,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=109),
        terminal_results=[],
    )
    session = ScriptedEventSession(
        [
            FakeServerEvent(submit=operation),
            FakeServerEvent(
                runtime=_priority_event(
                    operation_id=109,
                    sequence=1,
                    priority_class=0,
                    priority_delta=-4,
                ),
                wait_for_backend=True,
            ),
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

    await serve(
        OpenAiNnrpAdapter(backend),
        config=NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            accept_timeout_ms=10,
            receive_timeout_ms=10,
            max_active_sessions=1,
            max_operations_per_session=1,
            native_worker_count=2,
            observation_sinks=(),
        ),
        stop_event=stop_event,
    )

    assert len(session.recoverable_errors) == 1
    metadata, diagnostic = session.recoverable_errors[0]
    assert metadata.error_code is ErrorCode.UNSUPPORTED_CAPABILITY
    assert metadata.error_scope is ErrorScope.FRAME
    assert metadata.related_session_id == 1
    assert metadata.related_frame_id == 109
    assert diagnostic == b"live_priority_update_unsupported"


@pytest.mark.asyncio
async def test_live_priority_update_uses_explicit_backend_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    backend = LivePriorityCancellableBackend()
    operation = FakeOperation(
        operation_id=110,
        frame_id=210,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=110),
        terminal_results=[],
    )
    session = ScriptedEventSession(
        [
            FakeServerEvent(submit=operation),
            FakeServerEvent(
                runtime=_priority_event(
                    operation_id=110,
                    sequence=1,
                    priority_class=1,
                    priority_delta=-5,
                ),
                wait_for_backend=True,
            ),
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

    await serve(
        OpenAiNnrpAdapter(backend),
        config=NnrpServerConfig(
            endpoint="nnrp://runtime.local/vllm",
            accept_timeout_ms=10,
            receive_timeout_ms=10,
            max_active_sessions=1,
            max_operations_per_session=1,
            native_worker_count=2,
            observation_sinks=(),
        ),
        stop_event=stop_event,
    )

    assert backend.priority_updates == [("nnrp-110-210", -4)]
    assert session.recoverable_errors == []


@pytest.mark.asyncio
async def test_native_cancel_sends_typed_drop_reason(
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
    lifecycle: list[str] = []
    stop_event = asyncio.Event()
    backend = CancellableBackend()
    operation = FakeOperation(
        operation_id=102,
        frame_id=202,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=102),
        terminal_results=[],
        drop_send_observer=lambda: lifecycle.append("terminal-diagnostic"),
    )

    class OrderedShutdownSession(ScriptedEventSession):
        def close(self) -> None:
            lifecycle.append("session-close")
            super().close()

    class OrderedShutdownContext(FakeServerContext):
        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            lifecycle.append("listener-close")
            super().__exit__(exc_type, exc, traceback)

    session = OrderedShutdownSession(
        [FakeServerEvent(submit=operation)],
        operation=operation,
        backend_started=backend.started,
        stop_event=stop_event,
    )
    server_context = OrderedShutdownContext(MultiSessionFakeServer([session]))
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

    async def request_shutdown() -> None:
        await asyncio.to_thread(backend.started.wait, 1)
        lifecycle.append("stop-admission")
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
    assert all(metadata.stage_code != 0x000A for metadata, _body in operation.progress_results)
    assert session.closed is True
    assert server_context.exited is True
    assert lifecycle == [
        "stop-admission",
        "terminal-diagnostic",
        "session-close",
        "listener-close",
    ]
    observation = _observation_records(caplog)[0]
    assert observation["terminal_outcome"] == "dropped"
    assert observation["cancellation_kind"] == "server_shutdown"
    assert observation["drop_reason"] == "transport_closed"


@pytest.mark.asyncio
async def test_shutdown_during_accept_closes_unadmitted_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    operation = FakeOperation(
        operation_id=103,
        frame_id=203,
        body=_typed_profile_body(_chat_request()),
        metadata=_submit_metadata(operation_id=103),
        terminal_results=[],
    )
    session = FakeSession(operation, stop_event)
    server = ShutdownRaceFakeServer(session)
    server_context = FakeServerContext(server)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_runtime.listen_native_server",
        _listen_with_context(server_context),
    )

    async def request_shutdown() -> None:
        await server.accept_started.wait()
        stop_event.set()
        server.release_accept.set()

    shutdown_task = asyncio.create_task(request_shutdown())
    try:
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
    finally:
        await shutdown_task

    assert statistics.accepted_sessions == 0
    assert statistics.accepted_operations == 0
    assert session.closed is True
    assert server_context.exited is True


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
    assert all(metadata.stage_code != 0x000A for metadata, _body in operation.progress_results)
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


def _priority_event(
    *,
    operation_id: int,
    sequence: int,
    priority_class: int,
    priority_delta: int,
) -> NativeRuntimeEvent:
    metadata = SchedulingMetadata(
        operation_id=operation_id,
        control_sequence=sequence,
        priority_class=priority_class,
        priority_delta=priority_delta,
        deadline_unix_ms=0,
        flags=0,
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.PRIORITY_UPDATE, session_id=1),
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
