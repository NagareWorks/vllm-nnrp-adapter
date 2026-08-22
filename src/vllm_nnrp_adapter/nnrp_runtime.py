from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar

from nnrp import (  # type: ignore[import-untyped]
    NativeRuntimeServerOperation,
    NativeRuntimeServerSession,
    NativeTransportBinding,
    NativeTransportEndpoint,
    NativeWouldBlockError,
    PayloadKind,
    StreamSemantics,
    TransportPolicy,
)
from nnrp.core import (  # type: ignore[import-untyped]
    FrameSubmitMetadata,
    MessageType,
    ResultClass,
    ResultFlags,
    ResultPushMetadata,
    build_typed_payload_frame,
    pack_body,
    pack_typed_payload_frames,
    unpack_typed_payload_frames,
    validate_frame_submit_body,
)
from nnrp.runtime import (  # type: ignore[import-untyped]
    NativeRuntimeEvent,
    PartialResultMetadata,
    ResultDropReasonCode,
    ResultDropReasonMetadata,
    RuntimeRole,
)
from nnrp.server import (  # type: ignore[import-untyped]
    NativeServerAcceptOptions,
    NativeServerBootstrapOptions,
    NativeServerProviderRoute,
    NativeServerSessionOptions,
    listen_native_server,
)

from .adapter import OpenAiNnrpAdapter
from .nnrp_contract import validate_nnrp_runtime_contract
from .observability import (
    ObservationSink,
    ServerStartupObservation,
    StructuredLogObservationSink,
    _emit_operation_observation,
    _emit_server_startup_observation,
    _OperationObservationTracker,
)
from .operation_progress import OperationProgressReporter, OperationProgressStage
from .operation_state import OperationRecord, OperationRegistry, OperationState, OperationStateError
from .profile import build_cancelled_event
from .runtime_control import (
    OperationControlSlot,
    RuntimeControlDisposition,
    RuntimeControlKind,
    RuntimeControlRegistry,
    RuntimeControlRequest,
    decode_deadline_update,
    decode_operation_control,
)

_TERMINAL_EVENT_TYPES = frozenset({"response.completed", "response.error", "response.cancelled"})
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class NnrpServerConfig:
    endpoint: str
    provider_routes: Mapping[str, NativeServerProviderRoute] = field(default_factory=dict)
    transports: Sequence[NativeTransportBinding] | None = None
    transport_policy: TransportPolicy = TransportPolicy.AUTO
    session_options: NativeServerSessionOptions = field(default_factory=NativeServerSessionOptions)
    accept_timeout_ms: int = 100
    receive_timeout_ms: int = 100
    max_active_sessions: int = 8
    max_operations_per_session: int = 4
    native_worker_count: int = 9
    observation_sinks: Sequence[ObservationSink] = field(
        default_factory=lambda: (StructuredLogObservationSink(),)
    )

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("nnrp://", "nnrps://")):
            raise ValueError("endpoint must use nnrp:// or nnrps://")
        for name, value in (
            ("accept_timeout_ms", self.accept_timeout_ms),
            ("receive_timeout_ms", self.receive_timeout_ms),
            ("max_active_sessions", self.max_active_sessions),
            ("max_operations_per_session", self.max_operations_per_session),
            ("native_worker_count", self.native_worker_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        routes = MappingProxyType(dict(self.provider_routes))
        if any(not isinstance(route, NativeServerProviderRoute) for route in routes.values()):
            raise TypeError("provider_routes values must be NativeServerProviderRoute")
        transports = None if self.transports is None else tuple(self.transports)
        if transports is not None and any(not isinstance(binding, NativeTransportBinding) for binding in transports):
            raise TypeError("transports values must be NativeTransportBinding")
        if not isinstance(self.transport_policy, TransportPolicy):
            raise TypeError("transport_policy must be TransportPolicy")
        observation_sinks = tuple(self.observation_sinks)
        if any(not isinstance(sink, ObservationSink) for sink in observation_sinks):
            raise TypeError("observation_sinks values must implement ObservationSink")
        object.__setattr__(self, "provider_routes", routes)
        object.__setattr__(self, "transports", transports)
        object.__setattr__(self, "observation_sinks", observation_sinks)


@dataclass(frozen=True, slots=True)
class NnrpServeStatistics:
    accepted_sessions: int
    accepted_operations: int
    partial_results: int
    terminal_results: int


@dataclass(slots=True)
class _ServeCounters:
    accepted_sessions: int = 0
    accepted_operations: int = 0
    partial_results: int = 0
    terminal_results: int = 0
    result_drops: int = 0
    applied_control_events: int = 0
    rejected_control_events: int = 0

    def snapshot(self) -> NnrpServeStatistics:
        return NnrpServeStatistics(
            accepted_sessions=self.accepted_sessions,
            accepted_operations=self.accepted_operations,
            partial_results=self.partial_results,
            terminal_results=self.terminal_results,
        )


class _NativeCallExecutor:
    def __init__(self, max_workers: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vllm-nnrp-native")

    async def call(self, operation: Callable[..., _T], /, *args: object, **kwargs: object) -> _T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: operation(*args, **kwargs))

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


async def serve(
    adapter: OpenAiNnrpAdapter,
    *,
    config: NnrpServerConfig,
    stop_event: asyncio.Event | None = None,
) -> NnrpServeStatistics:
    return await _serve(adapter, config=config, stop_event=stop_event, on_ready=None)


async def _serve_with_ready(
    adapter: OpenAiNnrpAdapter,
    *,
    config: NnrpServerConfig,
    on_ready: Callable[[Mapping[str, NativeTransportEndpoint]], None],
    stop_event: asyncio.Event | None = None,
) -> NnrpServeStatistics:
    return await _serve(adapter, config=config, stop_event=stop_event, on_ready=on_ready)


async def _serve(
    adapter: OpenAiNnrpAdapter,
    *,
    config: NnrpServerConfig,
    stop_event: asyncio.Event | None,
    on_ready: Callable[[Mapping[str, NativeTransportEndpoint]], None] | None,
) -> NnrpServeStatistics:
    if not isinstance(adapter, OpenAiNnrpAdapter):
        raise TypeError("adapter must be an OpenAiNnrpAdapter Preview4 profile mapper")
    if not isinstance(config, NnrpServerConfig):
        raise TypeError("config must be an NnrpServerConfig Preview4 native-role configuration")
    validate_nnrp_runtime_contract()
    shutdown = stop_event or asyncio.Event()
    counters = _ServeCounters()
    native = _NativeCallExecutor(config.native_worker_count)
    server_context = listen_native_server(
        NativeServerBootstrapOptions(
            endpoint=config.endpoint,
            provider_routes=config.provider_routes,
            transport_policy=config.transport_policy,
            session_defaults=config.session_options,
        ),
        transports=config.transports,
    )
    sessions: set[asyncio.Task[None]] = set()
    server = None
    try:
        server = await native.call(server_context.__enter__)
        _emit_server_startup_observation(
            ServerStartupObservation.from_bound_endpoints(
                application_endpoint=config.endpoint,
                transport_policy=config.transport_policy.name.lower(),
                bound_provider_endpoints=server.bound_provider_endpoints,
            ),
            config.observation_sinks,
        )
        if on_ready is not None:
            on_ready(server.bound_provider_endpoints)
        while not shutdown.is_set():
            _retire_completed_tasks(sessions)
            if len(sessions) >= config.max_active_sessions:
                await _wait_for_task_or_shutdown(sessions, shutdown)
                continue
            try:
                session = await server.accept(NativeServerAcceptOptions(timeout_ms=config.accept_timeout_ms))
            except NativeWouldBlockError:
                continue
            counters.accepted_sessions += 1
            task = asyncio.create_task(
                _serve_session(adapter, session, config=config, shutdown=shutdown, native=native, counters=counters)
            )
            sessions.add(task)
    finally:
        try:
            await _finish_tasks(sessions, cancel=not shutdown.is_set())
        finally:
            try:
                if server is not None:
                    await native.call(server_context.__exit__, None, None, None)
            finally:
                native.close()
    return counters.snapshot()


async def _serve_session(
    adapter: OpenAiNnrpAdapter,
    session: NativeRuntimeServerSession,
    *,
    config: NnrpServerConfig,
    shutdown: asyncio.Event,
    native: _NativeCallExecutor,
    counters: _ServeCounters,
) -> None:
    operations: set[asyncio.Task[None]] = set()
    registry = OperationRegistry()
    controls = RuntimeControlRegistry()
    backend_family, backend_binding, vllm_version = adapter._backend_observation_identity()
    try:
        while not shutdown.is_set():
            _retire_completed_tasks(operations)
            try:
                events = await native.call(
                    session.poll_events,
                    max_events=1,
                    timeout_ms=config.receive_timeout_ms,
                )
            except NativeWouldBlockError:
                continue
            if not events:
                continue
            event = events[0]
            operation = event.as_submit()
            if operation is None:
                runtime_event = event.as_runtime()
                if runtime_event is not None:
                    if runtime_event.header.message_type is MessageType.SESSION_CLOSE:
                        await controls.terminate_all(
                            RuntimeControlKind.PEER_DISCONNECT,
                            source_role=RuntimeRole.CLIENT,
                            diagnostic=b"peer_disconnect",
                        )
                        break
                    await _handle_runtime_control(
                        runtime_event,
                        registry=registry,
                        controls=controls,
                        counters=counters,
                    )
                continue
            if len(operations) >= config.max_operations_per_session:
                capacity_event = _operation_capacity_event(config.max_operations_per_session)
                metadata, body = _terminal_reply(operation, capacity_event)
                await operation.send_result(
                    metadata,
                    body,
                )
                counters.terminal_results += 1
                continue
            try:
                record = registry.register(
                    operation.operation_id,
                    _backend_request_id(operation.operation_id, operation.frame_id),
                )
                record.transition(OperationState.QUEUED)
            except OperationStateError as error:
                event = _duplicate_operation_event(error)
                metadata, body = _terminal_reply(operation, event)
                await operation.send_result(metadata, body)
                counters.terminal_results += 1
                continue
            counters.accepted_operations += 1
            control = controls.register(operation.operation_id)
            connection_id, connection_generation = _native_handle_identity(getattr(session, "server", None))
            session_handle_id, session_generation = _native_handle_identity(getattr(session, "handle", None))
            observation = _OperationObservationTracker.from_operation(
                operation,
                selected_transport=session.active_transport_name,
                backend_family=backend_family,
                backend_binding=backend_binding,
                vllm_version=vllm_version,
                connection_id=connection_id,
                connection_generation=connection_generation,
                session_handle_id=session_handle_id,
                session_generation=session_generation,
            )
            pending_supersede = controls.pending_supersede(operation.operation_id)
            if pending_supersede is not None:
                try:
                    old_terminal = registry.get(pending_supersede.operation_id).is_terminal
                except OperationStateError:
                    old_terminal = False
                await controls.activate_replacement(operation.operation_id, old_terminal=old_terminal)
            task = asyncio.create_task(
                _serve_operation(
                    adapter,
                    operation,
                    record=record,
                    control=control,
                    observation=observation,
                    observation_sinks=config.observation_sinks,
                    counters=counters,
                )
            )
            control.bind(task)
            operations.add(task)
    except asyncio.CancelledError:
        await controls.terminate_all(
            RuntimeControlKind.SERVER_SHUTDOWN,
            source_role=RuntimeRole.SERVER,
            diagnostic=b"server_shutdown",
        )
        await _finish_tasks(operations, cancel=False)
        raise
    finally:
        try:
            if operations:
                termination_kind = (
                    RuntimeControlKind.SERVER_SHUTDOWN if shutdown.is_set() else RuntimeControlKind.PEER_DISCONNECT
                )
                await controls.terminate_all(
                    termination_kind,
                    source_role=(RuntimeRole.SERVER if shutdown.is_set() else RuntimeRole.CLIENT),
                    diagnostic=termination_kind.value.encode("ascii"),
                )
            await _finish_tasks(operations, cancel=False)
        finally:
            try:
                await native.call(session.close)
            finally:
                await controls.clear()
                registry.clear()


async def _serve_operation(
    adapter: OpenAiNnrpAdapter,
    operation: NativeRuntimeServerOperation,
    *,
    record: OperationRecord,
    control: OperationControlSlot,
    observation: _OperationObservationTracker,
    observation_sinks: Sequence[ObservationSink],
    counters: _ServeCounters,
) -> None:
    result_sequence = 0
    terminal_sent = False
    progress = OperationProgressReporter(operation, observer=observation.record_progress_stage)
    try:
        try:
            await progress.emit(OperationProgressStage.QUEUED)
            await progress.emit(OperationProgressStage.INPUT_RECEIVED)
            request = _decode_request(operation.submit.metadata.value, operation.submit.tail.body)
            observation.record_request(request)
            request.setdefault("request_id", record.backend_request_id)
            record.transition(OperationState.ADMITTED)
            observation.mark_admitted()
            await progress.emit(OperationProgressStage.ADMITTED)
            await progress.emit(OperationProgressStage.PREPROCESSING)
            await progress.emit(OperationProgressStage.EXECUTING)
            async for event in adapter._handle_native_request(
                request,
                backend_abort_observer=observation.record_backend_abort,
            ):
                body = _encode_event(event)
                observation.record_event(event, body_bytes=len(body))
                if _is_terminal_event(event):
                    await progress.emit(OperationProgressStage.FINALIZING)
                    record.terminate(_terminal_operation_state(event))
                    await progress.emit(_terminal_progress_stage(event))
                    terminal_sent = True
                    metadata, terminal_body = _terminal_reply(operation, event, encoded_event=body)
                    await operation.send_result(metadata, terminal_body)
                    counters.terminal_results += 1
                    break
                record.mark_partial()
                await progress.emit(OperationProgressStage.PRODUCING_PARTIAL)
                result_sequence += 1
                await operation.send_partial_result(
                    PartialResultMetadata(
                        operation_id=operation.operation_id,
                        result_sequence=result_sequence,
                        object_id=0,
                        delta_sequence=result_sequence,
                        body_bytes=len(body),
                        flags=0,
                    ),
                    body,
                )
                counters.partial_results += 1
        except asyncio.CancelledError:
            control_request = control.terminal_request
            if record.is_terminal:
                if control_request is not None:
                    observation.record_control(
                        control_request,
                        drop_reason=ResultDropReasonCode.TRANSPORT_CLOSED,
                    )
                return
            if control_request is None:
                if not record.is_terminal:
                    record.terminate(OperationState.CANCELLED)
                raise
            if control_request.kind is RuntimeControlKind.CANCEL:
                event = build_cancelled_event("peer_cancelled")
                record.terminate(OperationState.CANCELLED)
                await progress.emit(OperationProgressStage.DROPPED)
                observation.record_event(event, body_bytes=len(_encode_event(event)))
                fallback_drop = await _send_cancelled_outcome(
                    operation,
                    event,
                    control_request=control_request,
                    result_sequence=result_sequence,
                    counters=counters,
                )
                observation.record_control(control_request, drop_reason=fallback_drop)
            elif control_request.kind in {
                RuntimeControlKind.ABORT,
                RuntimeControlKind.SERVER_SHUTDOWN,
                RuntimeControlKind.DEADLINE_EXPIRED,
                RuntimeControlKind.SUPERSEDE,
            }:
                record.terminate(OperationState.DROPPED)
                diagnostic = control_request.diagnostic or {
                    RuntimeControlKind.ABORT: b"peer_abort",
                    RuntimeControlKind.SERVER_SHUTDOWN: b"server_shutdown",
                    RuntimeControlKind.DEADLINE_EXPIRED: b"deadline_expired",
                    RuntimeControlKind.SUPERSEDE: b"superseded",
                }[control_request.kind]
                drop_reason = {
                    RuntimeControlKind.ABORT: ResultDropReasonCode.PEER_CANCELLED,
                    RuntimeControlKind.SERVER_SHUTDOWN: ResultDropReasonCode.TRANSPORT_CLOSED,
                    RuntimeControlKind.DEADLINE_EXPIRED: ResultDropReasonCode.DEADLINE_EXPIRED,
                    RuntimeControlKind.SUPERSEDE: ResultDropReasonCode.SUPERSEDED,
                }[control_request.kind]
                observation.record_control(control_request, drop_reason=drop_reason)
                await progress.emit(OperationProgressStage.DROPPED)
                await operation.send_result_drop(
                    ResultDropReasonMetadata(
                        operation_id=operation.operation_id,
                        result_sequence=max(1, result_sequence + 1),
                        drop_reason_code=drop_reason,
                        source_role=RuntimeRole.RUNTIME,
                        flags=0,
                        diagnostic_bytes=len(diagnostic),
                    ),
                    diagnostic,
                )
                counters.result_drops += 1
                counters.terminal_results += 1
            else:
                record.terminate(OperationState.DROPPED)
                observation.record_control(control_request)
            terminal_sent = True
            return
        except Exception as error:
            observation.record_exception(error)
            if terminal_sent:
                raise
            event = _runtime_error_event(error)
        else:
            if not terminal_sent:
                await progress.emit(OperationProgressStage.FINALIZING)
                record.terminate(OperationState.COMPLETED)
                await progress.emit(OperationProgressStage.COMPLETED)
                terminal_sent = True
                await operation.send_result(_terminal_metadata(operation, None), b"")
                counters.terminal_results += 1

        if not terminal_sent:
            await progress.emit(OperationProgressStage.FINALIZING)
            record.terminate(_terminal_operation_state(event))
            await progress.emit(_terminal_progress_stage(event))
            body = _encode_event(event)
            observation.record_event(event, body_bytes=len(body))
            terminal_sent = True
            metadata, terminal_body = _terminal_reply(operation, event, encoded_event=body)
            await operation.send_result(metadata, terminal_body)
            counters.terminal_results += 1
    finally:
        try:
            await control.complete()
        finally:
            try:
                if record.is_terminal and not record.resources_released:
                    record.release_resources()
            finally:
                _emit_operation_observation(observation.finish(record.state), observation_sinks)


async def _handle_runtime_control(
    event: NativeRuntimeEvent,
    *,
    registry: OperationRegistry,
    controls: RuntimeControlRegistry,
    counters: _ServeCounters,
) -> None:
    request = decode_operation_control(event)
    deadline = None if request is not None else decode_deadline_update(event)
    if request is None and deadline is None:
        return
    if request is not None:
        operation_id = request.operation_id
    else:
        assert deadline is not None
        operation_id = deadline.operation_id
    try:
        terminal = registry.get(operation_id).is_terminal
    except OperationStateError:
        terminal = False
    if request is not None:
        if request.kind is RuntimeControlKind.SUPERSEDE:
            try:
                replacement_active = not registry.get(request.replacement_operation_id).is_terminal
            except OperationStateError:
                replacement_active = False
            disposition = await controls.apply_supersede(
                request,
                old_terminal=terminal,
                replacement_active=replacement_active,
            )
        else:
            disposition = await controls.apply(request, terminal=terminal)
    else:
        assert deadline is not None
        disposition = await controls.apply_deadline(deadline, terminal=terminal)
    if disposition is RuntimeControlDisposition.APPLIED:
        counters.applied_control_events += 1
    else:
        counters.rejected_control_events += 1


async def _send_cancelled_outcome(
    operation: NativeRuntimeServerOperation,
    event: Mapping[str, Any],
    *,
    control_request: RuntimeControlRequest,
    result_sequence: int,
    counters: _ServeCounters,
) -> ResultDropReasonCode | None:
    fallback_drop = None
    try:
        metadata, body = _terminal_reply(operation, event)
        await operation.send_result(metadata, body)
    except Exception:
        diagnostic = control_request.diagnostic or b"peer_cancelled"
        await operation.send_result_drop(
            ResultDropReasonMetadata(
                operation_id=operation.operation_id,
                result_sequence=max(1, result_sequence + 1),
                drop_reason_code=ResultDropReasonCode.PEER_CANCELLED,
                source_role=RuntimeRole.RUNTIME,
                flags=0,
                diagnostic_bytes=len(diagnostic),
            ),
            diagnostic,
        )
        counters.result_drops += 1
        fallback_drop = ResultDropReasonCode.PEER_CANCELLED
    counters.terminal_results += 1
    return fallback_drop


def _decode_request(metadata: object, body: bytes) -> dict[str, Any]:
    if not isinstance(metadata, FrameSubmitMetadata):
        raise ValueError("native FRAME_SUBMIT metadata must use the current data-plane contract")
    if metadata.payload_kind_bitmap != PayloadKind.STRUCTURED_EVENT or metadata.payload_frame_count != 1:
        raise ValueError("OpenAI profile submit must declare exactly one STRUCTURED_EVENT payload frame")
    try:
        body_view = validate_frame_submit_body(metadata, body)
        frames = unpack_typed_payload_frames(
            body_view.typed_payload_descriptor_region,
            body_view.typed_payload_frame_region,
            payload_kind_bitmap=metadata.payload_kind_bitmap,
        )
    except ValueError as error:
        raise ValueError("OpenAI profile submit has an invalid typed payload body") from error
    if len(frames) != 1:
        raise ValueError("OpenAI profile submit must contain exactly one typed payload frame")
    frame = frames[0]
    if (
        frame.payload_kind is not PayloadKind.STRUCTURED_EVENT
        or frame.profile_id != 0
        or int(frame.descriptor_flags) != 0
        or frame.schema_id != 0
        or frame.schema_version != 0
        or int(frame.stream_semantics) != int(StreamSemantics.SNAPSHOT)
    ):
        raise ValueError("OpenAI profile submit descriptor does not match the frozen wire mapping")
    try:
        value = json.loads(frame.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("native FRAME_SUBMIT body must contain a UTF-8 OpenAI profile request") from error
    if not isinstance(value, dict):
        raise ValueError("native FRAME_SUBMIT body must contain an OpenAI profile request object")
    return value


def _encode_event(event: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(event), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _encode_typed_profile_body(payload: bytes) -> bytes:
    frame = build_typed_payload_frame(
        PayloadKind.STRUCTURED_EVENT,
        payload,
        profile_id=0,
        descriptor_flags=0,
        schema_id=0,
        schema_version=0,
        stream_semantics=StreamSemantics.SNAPSHOT,
    )
    descriptors, frames = pack_typed_payload_frames((frame,))
    return bytes(
        pack_body(
            typed_payload_descriptor_region=descriptors,
            typed_payload_frame_region=frames,
        )
    )


def _terminal_reply(
    operation: NativeRuntimeServerOperation,
    event: Mapping[str, Any] | None,
    *,
    encoded_event: bytes | None = None,
) -> tuple[ResultPushMetadata, bytes]:
    metadata = _terminal_metadata(operation, event)
    if event is None:
        return metadata, b""
    payload = _encode_event(event) if encoded_event is None else encoded_event
    return metadata, _encode_typed_profile_body(payload)


def _is_terminal_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") in _TERMINAL_EVENT_TYPES


def _terminal_operation_state(event: Mapping[str, Any]) -> OperationState:
    event_type = event.get("type")
    if event_type == "response.completed":
        return OperationState.COMPLETED
    if event_type == "response.cancelled":
        return OperationState.CANCELLED
    return OperationState.FAILED


def _terminal_progress_stage(event: Mapping[str, Any]) -> OperationProgressStage:
    event_type = event.get("type")
    if event_type == "response.completed":
        return OperationProgressStage.COMPLETED
    if event_type == "response.cancelled":
        return OperationProgressStage.DROPPED
    return OperationProgressStage.FAILED


def _backend_request_id(operation_id: int, frame_id: int) -> str:
    return f"nnrp-{operation_id}-{frame_id}"


def _native_handle_identity(value: object) -> tuple[int | None, int | None]:
    handle = getattr(value, "handle", None)
    handle_id = getattr(handle, "id", None)
    generation = getattr(handle, "generation", None)
    return (
        handle_id if type(handle_id) is int else None,
        generation if type(generation) is int else None,
    )


def _duplicate_operation_event(error: OperationStateError) -> dict[str, Any]:
    return {
        "type": "response.error",
        "error": {
            "type": "invalid_request_error",
            "code": "duplicate_operation_id",
            "message": str(error),
        },
    }


def _operation_capacity_event(limit: int) -> dict[str, Any]:
    return {
        "type": "response.error",
        "error": {
            "type": "server_error",
            "code": "adapter_operation_limit",
            "message": f"Adapter session operation limit {limit} is active.",
        },
    }


def _terminal_metadata(
    operation: NativeRuntimeServerOperation,
    event: Mapping[str, Any] | None,
) -> ResultPushMetadata:
    has_profile_body = event is not None
    return ResultPushMetadata(
        status_code=_status_code(event),
        result_flags=ResultFlags(0),
        section_count=0,
        tile_count=0,
        active_profile_id=int(operation.submit.metadata.value.input_profile),
        reserved0=0,
        inference_ms=0,
        queue_ms=0,
        server_total_ms=0,
        reserved1=0,
        tile_base_id=0,
        tile_index_bytes=0,
        result_class=ResultClass.COMPLETE,
        payload_kind_bitmap=PayloadKind.STRUCTURED_EVENT if has_profile_body else PayloadKind(0),
        payload_frame_count=1 if has_profile_body else 0,
    )


def _status_code(event: Mapping[str, Any] | None) -> int:
    if event is None:
        return 200
    if event.get("type") == "response.cancelled":
        return 499
    if event.get("type") != "response.error":
        return 200
    error = event.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    if code == "request_timeout":
        return 504
    if code in {"backend_overload", "scheduler_rejected"}:
        return 503
    return 500


def _runtime_error_event(error: Exception) -> dict[str, Any]:
    if isinstance(error, ValueError):
        return {
            "type": "response.error",
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_submit_body",
                "message": str(error),
            },
        }
    return {
        "type": "response.error",
        "error": {
            "type": "server_error",
            "code": "adapter_runtime_error",
            "message": str(error),
        },
    }


async def _wait_for_task_or_shutdown(tasks: set[asyncio.Task[None]], shutdown: asyncio.Event) -> None:
    if not tasks:
        await shutdown.wait()
        return
    stop_task = asyncio.create_task(shutdown.wait())
    try:
        await asyncio.wait((*tasks, stop_task), return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


def _retire_completed_tasks(tasks: set[asyncio.Task[None]]) -> None:
    for task in tuple(tasks):
        if task.done():
            tasks.remove(task)
            task.result()


async def _finish_tasks(tasks: set[asyncio.Task[None]], *, cancel: bool) -> None:
    if cancel:
        for task in tasks:
            task.cancel()
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if not cancel:
            for result in results:
                if isinstance(result, BaseException):
                    raise result
    tasks.clear()
