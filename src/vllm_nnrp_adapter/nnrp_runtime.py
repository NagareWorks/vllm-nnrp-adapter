from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Any, TypeVar

from nnrp import (  # type: ignore[import-untyped]
    ERROR_FAMILY_LIFECYCLE,
    PREVIEW4_CAPABILITY_TOKENS,
    NativeInvalidStateError,
    NativeProtocolError,
    NativeRuntimeServerOperation,
    NativeRuntimeServerSession,
    NativeTransportBinding,
    NativeTransportEndpoint,
    NativeTransportServerSecurity,
    NativeWouldBlockError,
    PayloadKind,
    StreamSemantics,
    TransportPolicy,
    parse_native_transport_endpoint,
)
from nnrp.core import (  # type: ignore[import-untyped]
    ErrorCode,
    ErrorScope,
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
    CapabilityMetadata,
    NativeRuntimeEvent,
    PartialResultMetadata,
    PressureMetadata,
    RecoverableErrorMetadata,
    ResultDropReasonCode,
    ResultDropReasonMetadata,
    RetryAfterMetadata,
    RuntimeEventMetadataKind,
    RuntimeEventTailKind,
    RuntimeRole,
    TraceContextMetadata,
)
from nnrp.server import (  # type: ignore[import-untyped]
    NativeServerAcceptOptions,
    NativeServerBootstrapOptions,
    NativeServerProviderRoute,
    NativeServerSessionOptions,
    listen_native_server,
)

from .adapter import OpenAiNnrpAdapter
from .capability_ledger import supported_runtime_capabilities
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
from .pressure import AppliedPressureState, OutboundCreditController
from .runtime_control import (
    OperationControlSlot,
    RuntimeControlDisposition,
    RuntimeControlKind,
    RuntimeControlRegistry,
    RuntimeControlRequest,
    decode_deadline_update,
    decode_operation_control,
    decode_priority_update,
)

_TERMINAL_EVENT_TYPES = frozenset({"response.completed", "response.error", "response.cancelled"})
_CAPABILITY_FLAG_HARD_REQUIREMENT = 0x0000_0001
_CAPABILITY_FLAG_DOWNGRADE_ALLOWED = 0x0000_0002
_OPENAI_COMPATIBLE_PROFILE_ID = 0
_PROVIDER_NAMES = frozenset({"tcp", "quic", "ipc", "websocket"})
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
        _validate_server_provider_routes(routes)
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


def _validate_server_provider_routes(routes: Mapping[str, NativeServerProviderRoute]) -> None:
    unknown_names = set(routes) - _PROVIDER_NAMES
    if unknown_names:
        names = ", ".join(sorted(unknown_names))
        raise ValueError(f"provider_routes contains unsupported transport names: {names}")

    for transport_name, route in routes.items():
        endpoint = route.provider_endpoint
        if endpoint is not None and not isinstance(endpoint, (str, NativeTransportEndpoint)):
            raise TypeError("provider route endpoint must be str, NativeTransportEndpoint, or None")
        if route.security is not None and not isinstance(route.security, NativeTransportServerSecurity):
            raise TypeError("provider route security must be NativeTransportServerSecurity or None")
        resolved = None
        if endpoint is not None:
            resolved = (
                endpoint
                if isinstance(endpoint, NativeTransportEndpoint)
                else parse_native_transport_endpoint(endpoint)
            )
            if resolved.transport_name != transport_name:
                raise ValueError(
                    f"{transport_name} provider route cannot use {resolved.transport_name} carrier endpoint"
                )

        if transport_name in {"ipc", "websocket"} and resolved is None:
            raise ValueError(f"{transport_name} provider route requires an explicit provider endpoint")
        if transport_name == "ipc":
            assert resolved is not None
            expected_scheme = "npipe" if os.name == "nt" else "unix"
            if resolved.scheme != expected_scheme:
                raise ValueError(
                    f"ipc provider route on this platform requires a {expected_scheme}:// endpoint"
                )
        if transport_name == "quic" and route.security is None:
            raise ValueError("quic provider route requires server security material")
        if transport_name == "ipc" and route.security is not None:
            raise ValueError("ipc provider route must not carry server security material")
        if transport_name == "websocket":
            assert resolved is not None
            if resolved.secure and route.security is None:
                raise ValueError("wss provider route requires server security material")
            if not resolved.secure and route.security is not None:
                raise ValueError("ws provider route must not carry server security material")


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


@dataclass(slots=True)
class _BackendTraceContextSlot:
    metadata: TraceContextMetadata | None = None
    dispatched: bool = False

    def update(self, metadata: TraceContextMetadata) -> bool:
        if self.dispatched:
            return False
        self.metadata = metadata
        return True

    def begin_dispatch(self) -> Mapping[str, str] | None:
        self.dispatched = True
        return _w3c_trace_headers(self.metadata)


class _NativeCallExecutor:
    def __init__(self, max_workers: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vllm-nnrp-native")

    async def call(self, operation: Callable[..., _T], /, *args: object, **kwargs: object) -> _T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: operation(*args, **kwargs))

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


@dataclass(slots=True)
class _AdmissionWindowReporter:
    session: NativeRuntimeServerSession
    native: _NativeCallExecutor
    capacity: int
    retry_after_ms: int
    last_available: int | None = None
    enabled: bool = False
    scope_id: int = 0

    async def enable(self, active_operations: int, *, scope_id: int) -> None:
        if self.enabled:
            return
        self.enabled = True
        self.scope_id = scope_id
        await self.update(active_operations, force=True)

    async def update(self, active_operations: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        available = max(0, self.capacity - active_operations)
        if not force and available == self.last_available:
            return
        metadata = PressureMetadata(
            scope_id=self.scope_id,
            credit_window=available,
            pressure_level=0 if available else 2,
            pressure_reason=0 if available else 1,
            retry_after_ms=0 if available else self.retry_after_ms,
            flags=0,
        )
        if available:
            await self.native.call(self.session.send_credit_update, metadata)
        else:
            await self.native.call(self.session.send_backpressure, metadata)
        self.last_available = available


@dataclass(slots=True)
class _RecoveryReporter:
    session: NativeRuntimeServerSession
    native: _NativeCallExecutor
    retry_after_ms: int
    enabled: bool = False
    control_sequence: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def enable(self) -> None:
        self.enabled = True

    async def report(
        self,
        operation: NativeRuntimeServerOperation,
        *,
        diagnostic: bytes,
        source_role: RuntimeRole,
    ) -> RetryAfterMetadata | None:
        if not self.enabled:
            return None
        async with self._lock:
            self.control_sequence += 1
            control_sequence = self.control_sequence
            header = operation.submit.header
            await self.native.call(
                self.session.send_recoverable_error,
                RecoverableErrorMetadata(
                    error_code=ErrorCode.SERVER_BUSY,
                    error_scope=ErrorScope.FRAME,
                    recovery_action=0,
                    source_role=source_role,
                    flags=0x02,
                    retry_after_ms=self.retry_after_ms,
                    related_session_id=header.session_id,
                    related_frame_id=header.frame_id,
                    related_view_id=header.view_id,
                    diagnostic_bytes=len(diagnostic),
                ),
                diagnostic,
            )
            retry = RetryAfterMetadata(
                scope_id=operation.operation_id,
                control_sequence=control_sequence,
                retry_after_ms=self.retry_after_ms,
                jitter_ms=0,
                reason_code=0,
                source_role=source_role,
                flags=0x02,
                diagnostic_bytes=len(diagnostic),
            )
            await self.native.call(self.session.send_retry_after, retry, diagnostic)
            return retry


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
            except NativeInvalidStateError as error:
                if error.status.detail_code != 105:
                    raise
                break
            if shutdown.is_set():
                await native.call(session.close)
                break
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
    observations_by_frame: dict[
        int,
        tuple[OperationRecord, _OperationObservationTracker, _BackendTraceContextSlot],
    ] = {}
    observations_by_operation: dict[int, _OperationObservationTracker] = {}
    session_trace_context: tuple[TraceContextMetadata, bytes] | None = None
    credit_backpressure_negotiated = False
    progress_partial_negotiated = False
    output_credits = OutboundCreditController()
    admission_window = _AdmissionWindowReporter(
        session=session,
        native=native,
        capacity=config.max_operations_per_session,
        retry_after_ms=config.receive_timeout_ms,
    )
    recovery = _RecoveryReporter(
        session=session,
        native=native,
        retry_after_ms=config.receive_timeout_ms,
    )
    backend_family, backend_binding, vllm_version = adapter._backend_observation_identity()
    try:
        await admission_window.update(0)
        while not shutdown.is_set():
            previous_operation_count = len(operations)
            _retire_completed_tasks(operations)
            if len(operations) != previous_operation_count:
                await admission_window.update(len(operations))
            try:
                events = await native.call(
                    session.poll_events,
                    max_events=1,
                    timeout_ms=config.receive_timeout_ms,
                )
            except NativeWouldBlockError:
                continue
            except NativeInvalidStateError as error:
                if error.status.detail_code != 105:
                    raise
                break
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
                    if runtime_event.header.message_type is MessageType.TRACE_CONTEXT:
                        session_trace_context = _apply_trace_context(
                            runtime_event,
                            observations_by_frame=observations_by_frame,
                            session_trace_context=session_trace_context,
                        )
                        counters.applied_control_events += 1
                        continue
                    if runtime_event.header.message_type is MessageType.CAPABILITY_NEGOTIATION:
                        accepted_capabilities = await _handle_capability_negotiation(
                            runtime_event,
                            adapter=adapter,
                            session=session,
                            native=native,
                            counters=counters,
                        )
                        if (
                            not credit_backpressure_negotiated
                            and "control.credit_backpressure" in accepted_capabilities
                        ):
                            credit_backpressure_negotiated = True
                            await admission_window.enable(
                                len(operations),
                                scope_id=runtime_event.header.session_id,
                            )
                        if "control.recoverable_error" in accepted_capabilities:
                            recovery.enable()
                        if "control.progress_partial" in accepted_capabilities:
                            progress_partial_negotiated = True
                        continue
                    if runtime_event.header.message_type in {
                        MessageType.BACKPRESSURE,
                        MessageType.CREDIT_UPDATE,
                    }:
                        if not credit_backpressure_negotiated:
                            raise ValueError(
                                "BACKPRESSURE and CREDIT_UPDATE require negotiated "
                                "control.credit_backpressure"
                            )
                        pressure = await _apply_output_pressure(
                            runtime_event,
                            output_credits=output_credits,
                        )
                        if pressure.scope_kind == "operation":
                            observation = observations_by_operation.get(pressure.scope_id)
                            if observation is not None:
                                observation.record_pressure(
                                    pressure.metadata,
                                    scope_kind=pressure.scope_kind,
                                    scope_id=pressure.scope_id,
                                )
                        else:
                            for observation in observations_by_operation.values():
                                observation.record_pressure(
                                    pressure.metadata,
                                    scope_kind=pressure.scope_kind,
                                    scope_id=pressure.scope_id,
                                )
                        counters.applied_control_events += 1
                        continue
                    await _handle_runtime_control(
                        runtime_event,
                        adapter=adapter,
                        session=session,
                        native=native,
                        registry=registry,
                        controls=controls,
                        counters=counters,
                    )
                continue
            if len(operations) >= config.max_operations_per_session:
                await admission_window.update(len(operations), force=True)
                await recovery.report(
                    operation,
                    diagnostic=b"adapter_operation_limit",
                    source_role=RuntimeRole.SCHEDULER,
                )
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
                active_profile_id=_OPENAI_COMPATIBLE_PROFILE_ID,
                backend_family=backend_family,
                backend_binding=backend_binding,
                vllm_version=vllm_version,
                connection_id=connection_id,
                connection_generation=connection_generation,
                session_handle_id=session_handle_id,
                session_generation=session_generation,
            )
            if session_trace_context is not None:
                observation.record_trace_context(*session_trace_context)
            backend_trace = _BackendTraceContextSlot(
                None if session_trace_context is None else session_trace_context[0]
            )
            observations_by_frame[operation.frame_id] = (record, observation, backend_trace)
            observations_by_operation[operation.operation_id] = observation
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
                    backend_trace=backend_trace,
                    output_credits=output_credits,
                    recovery=recovery,
                    emit_progress=progress_partial_negotiated,
                    session=session,
                    native=native,
                )
            )
            control.bind(task)
            operations.add(task)
            task.add_done_callback(
                partial(
                    _forget_operation_observation,
                    observations_by_operation,
                    operation.operation_id,
                )
            )
            await admission_window.update(len(operations))
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
            await output_credits.close()
            await _finish_tasks(operations, cancel=False)
        finally:
            try:
                await native.call(session.close)
            finally:
                await controls.clear()
                registry.clear()


async def _handle_capability_negotiation(
    event: NativeRuntimeEvent,
    *,
    adapter: OpenAiNnrpAdapter,
    session: NativeRuntimeServerSession,
    native: _NativeCallExecutor,
    counters: _ServeCounters,
) -> tuple[str, ...]:
    metadata, requested = _decode_capability_offer(event)
    supported = supported_runtime_capabilities(
        supports_runtime_priority=adapter._supports_runtime_priority()
    )
    accepted = tuple(token for token in requested if token in supported)
    accepted_body = _encode_capability_tokens(accepted)
    response = CapabilityMetadata(
        profile_id=metadata.profile_id,
        capability_count=len(accepted),
        cost_model_id=0,
        preference_rank=0,
        limit_bytes=0,
        limit_units=0,
        body_bytes=len(accepted_body),
        flags=metadata.flags,
    )
    downgraded = len(accepted) != len(requested)
    if downgraded and accepted and metadata.flags & _CAPABILITY_FLAG_DOWNGRADE_ALLOWED:
        await native.call(session.degrade_profile, response, accepted_body)
    else:
        await native.call(session.negotiate_capabilities, response, accepted_body)
    if not accepted and metadata.flags & _CAPABILITY_FLAG_HARD_REQUIREMENT:
        await _send_capability_mismatch(event, session=session, native=native)
        counters.rejected_control_events += 1
    else:
        counters.applied_control_events += 1
    return accepted


async def _apply_output_pressure(
    event: NativeRuntimeEvent,
    *,
    output_credits: OutboundCreditController,
) -> AppliedPressureState:
    if event.metadata.kind is not RuntimeEventMetadataKind.PRESSURE or not isinstance(
        event.metadata.value, PressureMetadata
    ):
        raise ValueError("BACKPRESSURE and CREDIT_UPDATE require PressureMetadata")
    if event.tail.kind is not RuntimeEventTailKind.NONE:
        raise ValueError("BACKPRESSURE and CREDIT_UPDATE cannot carry a body")
    return await output_credits.apply(event.header.message_type, event.metadata.value)


def _decode_capability_offer(event: NativeRuntimeEvent) -> tuple[CapabilityMetadata, tuple[str, ...]]:
    if event.metadata.kind is not RuntimeEventMetadataKind.CAPABILITY or not isinstance(
        event.metadata.value, CapabilityMetadata
    ):
        raise ValueError("CAPABILITY_NEGOTIATION requires CapabilityMetadata")
    metadata = event.metadata.value
    if event.tail.kind is RuntimeEventTailKind.NONE:
        body = b""
    elif event.tail.kind is RuntimeEventTailKind.BODY:
        body = event.tail.body
    else:
        raise ValueError("CAPABILITY_NEGOTIATION requires a capability token body")
    if metadata.body_bytes != len(body):
        raise ValueError("CAPABILITY_NEGOTIATION body_bytes does not match the capability token body")
    tokens: list[str] = []
    offset = 0
    while offset < len(body):
        if offset + 2 > len(body):
            raise ValueError("capability entry is missing its token length")
        token_length = int.from_bytes(body[offset : offset + 2], "little")
        if token_length == 0:
            raise ValueError("capability token length must be non-zero")
        token_start = offset + 2
        token_end = token_start + token_length
        if token_end > len(body):
            raise ValueError("capability token exceeds the declared body")
        try:
            token = body[token_start:token_end].decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("capability token must be ASCII") from error
        if token not in PREVIEW4_CAPABILITY_TOKENS:
            raise ValueError(f"unknown Preview4 capability token {token!r}")
        if tokens and tokens[-1].encode("ascii") >= token.encode("ascii"):
            raise ValueError("capability tokens must be unique and use canonical byte order")
        tokens.append(token)
        offset = token_end
    if len(tokens) != metadata.capability_count:
        raise ValueError("CAPABILITY_NEGOTIATION capability_count does not match the capability token body")
    if metadata.profile_id != _OPENAI_COMPATIBLE_PROFILE_ID:
        return metadata, ()
    return metadata, tuple(tokens)


def _encode_capability_tokens(tokens: Sequence[str]) -> bytes:
    body = bytearray()
    for token in sorted(tokens, key=lambda value: value.encode("ascii")):
        encoded = token.encode("ascii")
        body.extend(len(encoded).to_bytes(2, "little"))
        body.extend(encoded)
    return bytes(body)


async def _send_capability_mismatch(
    event: NativeRuntimeEvent,
    *,
    session: NativeRuntimeServerSession,
    native: _NativeCallExecutor,
) -> None:
    diagnostic = b"capability_mismatch"
    metadata = RecoverableErrorMetadata(
        error_code=ErrorCode.UNSUPPORTED_CAPABILITY,
        error_scope=ErrorScope.SESSION,
        recovery_action=0,
        source_role=RuntimeRole.RUNTIME,
        flags=0,
        retry_after_ms=0,
        related_session_id=event.header.session_id,
        related_frame_id=event.header.frame_id,
        related_view_id=event.header.view_id,
        diagnostic_bytes=len(diagnostic),
    )
    await native.call(session.send_recoverable_error, metadata, diagnostic)


def _apply_trace_context(
    event: NativeRuntimeEvent,
    *,
    observations_by_frame: Mapping[
        int,
        tuple[OperationRecord, _OperationObservationTracker, _BackendTraceContextSlot],
    ],
    session_trace_context: tuple[TraceContextMetadata, bytes] | None,
) -> tuple[TraceContextMetadata, bytes] | None:
    if event.metadata.kind is not RuntimeEventMetadataKind.TRACE_CONTEXT or not isinstance(
        event.metadata.value, TraceContextMetadata
    ):
        raise ValueError("TRACE_CONTEXT requires TraceContextMetadata")
    if event.tail.kind is not RuntimeEventTailKind.BODY:
        raise ValueError("TRACE_CONTEXT requires a trace attribute body")
    metadata = event.metadata.value
    attributes = event.tail.body
    if metadata.body_bytes != len(attributes):
        raise ValueError("TRACE_CONTEXT body_bytes does not match the trace attribute body")
    if event.header.trace_id not in {0, metadata.trace_id}:
        raise ValueError("TRACE_CONTEXT header trace_id does not match metadata trace_id")
    if event.header.frame_id == 0:
        return metadata, attributes
    try:
        record, observation, backend_trace = observations_by_frame[event.header.frame_id]
    except KeyError as error:
        raise ValueError(f"TRACE_CONTEXT references unknown active frame {event.header.frame_id}") from error
    if record.is_terminal:
        raise ValueError(f"TRACE_CONTEXT references terminal frame {event.header.frame_id}")
    observation.record_trace_context(metadata, attributes)
    backend_trace.update(metadata)
    return session_trace_context


def _w3c_trace_headers(metadata: TraceContextMetadata | None) -> Mapping[str, str] | None:
    if metadata is None or metadata.trace_id == 0 or metadata.span_id == 0:
        return None
    traceparent = (
        f"00-{metadata.trace_id:032x}-{metadata.span_id:016x}-{metadata.flags & 0x01:02x}"
    )
    return MappingProxyType({"traceparent": traceparent})


async def _serve_operation(
    adapter: OpenAiNnrpAdapter,
    operation: NativeRuntimeServerOperation,
    *,
    record: OperationRecord,
    control: OperationControlSlot,
    observation: _OperationObservationTracker,
    observation_sinks: Sequence[ObservationSink],
    counters: _ServeCounters,
    backend_trace: _BackendTraceContextSlot | None = None,
    output_credits: OutboundCreditController | None = None,
    recovery: _RecoveryReporter | None = None,
    emit_progress: bool = True,
    session: NativeRuntimeServerSession | None = None,
    native: _NativeCallExecutor | None = None,
) -> None:
    result_sequence = 0
    terminal_sent = False
    progress = OperationProgressReporter(
        operation,
        observer=observation.record_progress_stage,
        enabled=emit_progress,
    )
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
            backend_events = adapter._handle_native_request(
                request,
                backend_abort_observer=observation.record_backend_abort,
                backend_trace_headers_factory=(
                    None if backend_trace is None else backend_trace.begin_dispatch
                ),
                backend_priority_factory=control.begin_backend_dispatch,
            ).__aiter__()
            while True:
                reservation = (
                    None
                    if output_credits is None
                    else await output_credits.reserve(operation.operation_id)
                )
                try:
                    event = await anext(backend_events)
                except StopAsyncIteration:
                    if output_credits is not None and reservation is not None:
                        await output_credits.refund(reservation)
                    break
                except BaseException:
                    if output_credits is not None and reservation is not None:
                        await output_credits.refund(reservation)
                    raise
                body = _encode_event(event)
                recovery_diagnostic = _recoverable_event_diagnostic(event)
                if recovery is not None and recovery_diagnostic is not None:
                    retry = await recovery.report(
                        operation,
                        diagnostic=recovery_diagnostic,
                        source_role=RuntimeRole.RUNTIME,
                    )
                    if retry is not None:
                        observation.record_retry_hint(retry)
                if _is_terminal_event(event):
                    await progress.emit(OperationProgressStage.FINALIZING)
                    await progress.emit(_terminal_progress_stage(event))
                    metadata, terminal_body = _terminal_reply(operation, event, encoded_event=body)
                    await _send_terminal_result(operation, metadata, terminal_body)
                    observation.record_event(event, body_bytes=len(body))
                    record.terminate(_terminal_operation_state(event))
                    terminal_sent = True
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
                observation.record_event(event, body_bytes=len(body))
                counters.partial_results += 1
        except asyncio.CancelledError:
            if not await _finalize_cancelled_operation(
                operation,
                record=record,
                control=control,
                observation=observation,
                result_sequence=result_sequence,
                counters=counters,
                session=session,
                native=native,
            ):
                raise
            return
        except NativeInvalidStateError as error:
            if error.status.detail_code != 105:
                raise
            try:
                await control.wait_for_terminal_request()
            except asyncio.CancelledError:
                pass
            if await _finalize_cancelled_operation(
                operation,
                record=record,
                control=control,
                observation=observation,
                result_sequence=result_sequence,
                counters=counters,
                session=session,
                native=native,
            ):
                return
            raise
        except _NativeTerminalRaceError:
            raise
        except Exception as error:
            observation.record_exception(error)
            if terminal_sent:
                raise
            event = _runtime_error_event(error)
        else:
            if not terminal_sent:
                await progress.emit(OperationProgressStage.FINALIZING)
                await progress.emit(OperationProgressStage.COMPLETED)
                await _send_terminal_result(operation, _terminal_metadata(operation, None), b"")
                record.terminate(OperationState.COMPLETED)
                terminal_sent = True
                counters.terminal_results += 1

        if not terminal_sent:
            await progress.emit(OperationProgressStage.FINALIZING)
            await progress.emit(_terminal_progress_stage(event))
            body = _encode_event(event)
            metadata, terminal_body = _terminal_reply(operation, event, encoded_event=body)
            await _send_terminal_result(operation, metadata, terminal_body)
            observation.record_event(event, body_bytes=len(body))
            record.terminate(_terminal_operation_state(event))
            terminal_sent = True
            counters.terminal_results += 1
    except _NativeTerminalRaceError:
        try:
            await control.wait_for_terminal_request()
        except asyncio.CancelledError:
            pass
        if await _finalize_cancelled_operation(
            operation,
            record=record,
            control=control,
            observation=observation,
            result_sequence=result_sequence,
            counters=counters,
            session=session,
            native=native,
        ):
            return
        raise
    finally:
        try:
            await control.complete()
        finally:
            try:
                if output_credits is not None:
                    await output_credits.retire(operation.operation_id)
                if record.is_terminal and not record.resources_released:
                    record.release_resources()
            finally:
                _emit_operation_observation(observation.finish(record.state), observation_sinks)


async def _handle_runtime_control(
    event: NativeRuntimeEvent,
    *,
    adapter: OpenAiNnrpAdapter,
    session: NativeRuntimeServerSession,
    native: _NativeCallExecutor,
    registry: OperationRegistry,
    controls: RuntimeControlRegistry,
    counters: _ServeCounters,
) -> None:
    request = decode_operation_control(event)
    deadline = None if request is not None else decode_deadline_update(event)
    priority = (
        None
        if request is not None or deadline is not None
        else decode_priority_update(event)
    )
    if request is None and deadline is None and priority is None:
        return
    if request is not None:
        operation_id = request.operation_id
    elif deadline is not None:
        assert deadline is not None
        operation_id = deadline.operation_id
    else:
        assert priority is not None
        operation_id = priority.operation_id
    try:
        operation_record = registry.get(operation_id)
        terminal = operation_record.is_terminal
    except OperationStateError:
        operation_record = None
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
    elif deadline is not None:
        assert deadline is not None
        disposition = await controls.apply_deadline(deadline, terminal=terminal)
    else:
        assert priority is not None
        disposition = controls.apply_priority(
            priority,
            terminal=terminal,
            backend_supported=adapter._supports_runtime_priority(),
        )
        if disposition is RuntimeControlDisposition.LIVE_UPDATE_REQUIRED:
            live_update_accepted = (
                operation_record is not None
                and await adapter._update_runtime_priority(
                    operation_record.backend_request_id,
                    priority.backend_priority,
                )
            )
            disposition = (
                RuntimeControlDisposition.APPLIED
                if live_update_accepted
                else RuntimeControlDisposition.UNSUPPORTED_LIVE_UPDATE
            )
        if disposition is RuntimeControlDisposition.UNSUPPORTED_LIVE_UPDATE:
            await _send_unsupported_priority_update(
                event,
                operation_id=operation_id,
                session=session,
                native=native,
            )
    if disposition is RuntimeControlDisposition.APPLIED:
        counters.applied_control_events += 1
    else:
        counters.rejected_control_events += 1


async def _send_unsupported_priority_update(
    event: NativeRuntimeEvent,
    *,
    operation_id: int,
    session: NativeRuntimeServerSession,
    native: _NativeCallExecutor,
) -> None:
    diagnostic = b"live_priority_update_unsupported"
    metadata = RecoverableErrorMetadata(
        error_code=ErrorCode.UNSUPPORTED_CAPABILITY,
        error_scope=ErrorScope.FRAME,
        recovery_action=0,
        source_role=RuntimeRole.RUNTIME,
        flags=0,
        retry_after_ms=0,
        related_session_id=event.header.session_id,
        related_frame_id=operation_id & 0xFFFF_FFFF,
        related_view_id=event.header.view_id,
        diagnostic_bytes=len(diagnostic),
    )
    await native.call(session.send_recoverable_error, metadata, diagnostic)


async def _send_cancelled_outcome(
    operation: NativeRuntimeServerOperation,
    *,
    control_request: RuntimeControlRequest,
    result_sequence: int,
    counters: _ServeCounters,
) -> ResultDropReasonCode:
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
    counters.terminal_results += 1
    return ResultDropReasonCode.PEER_CANCELLED


async def _finalize_cancelled_operation(
    operation: NativeRuntimeServerOperation,
    *,
    record: OperationRecord,
    control: OperationControlSlot,
    observation: _OperationObservationTracker,
    result_sequence: int,
    counters: _ServeCounters,
    session: NativeRuntimeServerSession | None,
    native: _NativeCallExecutor | None,
) -> bool:
    control_request = control.terminal_request
    if record.is_terminal:
        return True
    if control_request is None:
        record.terminate(OperationState.CANCELLED)
        return False
    if control_request.kind in {
        RuntimeControlKind.CANCEL,
        RuntimeControlKind.ABORT,
        RuntimeControlKind.DEADLINE_EXPIRED,
        RuntimeControlKind.SUPERSEDE,
    }:
        await _send_terminal_trace_context(operation, session=session, native=native)
    if control_request.kind is RuntimeControlKind.CANCEL:
        drop_reason = await _send_cancelled_outcome(
            operation,
            control_request=control_request,
            result_sequence=result_sequence,
            counters=counters,
        )
        record.terminate(OperationState.CANCELLED)
        observation.record_control(control_request, drop_reason=drop_reason)
        return True
    if control_request.kind in {
        RuntimeControlKind.ABORT,
        RuntimeControlKind.SERVER_SHUTDOWN,
        RuntimeControlKind.DEADLINE_EXPIRED,
        RuntimeControlKind.SUPERSEDE,
    }:
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
        record.terminate(OperationState.DROPPED)
        counters.result_drops += 1
        counters.terminal_results += 1
        return True
    record.terminate(OperationState.DROPPED)
    observation.record_control(control_request)
    return True


class _NativeTerminalRaceError(RuntimeError):
    pass


async def _send_terminal_result(
    operation: NativeRuntimeServerOperation,
    metadata: ResultPushMetadata,
    body: bytes,
) -> None:
    try:
        await operation.send_result(metadata, body)
    except NativeProtocolError as error:
        status = error.status
        if (
            status.error_family != ERROR_FAMILY_LIFECYCLE
            or status.protocol_error_code != 0
            or status.detail_code != 0
        ):
            raise
        raise _NativeTerminalRaceError from error


async def _send_terminal_trace_context(
    operation: NativeRuntimeServerOperation,
    *,
    session: NativeRuntimeServerSession | None,
    native: _NativeCallExecutor | None,
) -> None:
    if session is None or native is None:
        return
    metadata = TraceContextMetadata(
        trace_id=operation.submit.header.trace_id or operation.operation_id,
        span_id=operation.frame_id or 1,
        parent_span_id=0,
        stage_code=0,
        flags=0,
        body_bytes=0,
    )
    await native.call(session.send_trace_context, metadata, b"", operation_id=None)


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


def _recoverable_event_diagnostic(event: Mapping[str, Any]) -> bytes | None:
    if event.get("type") != "response.error":
        return None
    error = event.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    if code not in {"backend_overload", "scheduler_rejected", "request_timeout"}:
        return None
    return str(code).encode("ascii")


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
        active_profile_id=_OPENAI_COMPATIBLE_PROFILE_ID,
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
            try:
                task.result()
            except asyncio.CancelledError:
                pass


def _forget_operation_observation(
    observations: dict[int, _OperationObservationTracker],
    operation_id: int,
    _task: asyncio.Task[None],
) -> None:
    observations.pop(operation_id, None)


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
