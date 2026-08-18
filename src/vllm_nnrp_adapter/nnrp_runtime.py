from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar

from nnrp import (  # type: ignore[import-untyped]
    NativeRuntimeServerOperation,
    NativeRuntimeServerSession,
    NativeWouldBlockError,
    PayloadKind,
    TransportPolicy,
)
from nnrp.core import MessageType, ResultClass, ResultFlags, ResultPushMetadata  # type: ignore[import-untyped]
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
from .operation_state import OperationRecord, OperationRegistry, OperationState, OperationStateError
from .profile import build_cancelled_event
from .runtime_control import (
    OperationControlSlot,
    RuntimeControlDisposition,
    RuntimeControlKind,
    RuntimeControlRegistry,
    decode_operation_control,
)

_TERMINAL_EVENT_TYPES = frozenset({"response.completed", "response.error", "response.cancelled"})
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class NnrpServerConfig:
    endpoint: str
    provider_routes: Mapping[str, NativeServerProviderRoute] = field(default_factory=dict)
    transport_policy: TransportPolicy = TransportPolicy.AUTO
    session_options: NativeServerSessionOptions = field(default_factory=NativeServerSessionOptions)
    accept_timeout_ms: int = 100
    receive_timeout_ms: int = 100
    max_active_sessions: int = 8
    max_operations_per_session: int = 4
    native_worker_count: int = 9

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
        if not isinstance(self.transport_policy, TransportPolicy):
            raise TypeError("transport_policy must be TransportPolicy")
        object.__setattr__(self, "provider_routes", routes)


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
        )
    )
    sessions: set[asyncio.Task[None]] = set()
    server = None
    try:
        server = await native.call(server_context.__enter__)
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
                await operation.send_result(
                    _terminal_metadata(operation, capacity_event),
                    _encode_event(capacity_event),
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
                await operation.send_result(_terminal_metadata(operation, event), _encode_event(event))
                counters.terminal_results += 1
                continue
            counters.accepted_operations += 1
            control = controls.register(operation.operation_id)
            task = asyncio.create_task(
                _serve_operation(
                    adapter,
                    operation,
                    record=record,
                    control=control,
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
                controls.clear()
                registry.clear()


async def _serve_operation(
    adapter: OpenAiNnrpAdapter,
    operation: NativeRuntimeServerOperation,
    *,
    record: OperationRecord,
    control: OperationControlSlot,
    counters: _ServeCounters,
) -> None:
    result_sequence = 0
    terminal_sent = False
    try:
        try:
            request = _decode_request(operation.body)
            request.setdefault("request_id", record.backend_request_id)
            record.transition(OperationState.ADMITTED)
            async for event in adapter.handle_request(request):
                body = _encode_event(event)
                if _is_terminal_event(event):
                    record.terminate(_terminal_operation_state(event))
                    terminal_sent = True
                    await operation.send_result(_terminal_metadata(operation, event), body)
                    counters.terminal_results += 1
                    break
                record.mark_partial()
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
            if control_request is None:
                if not record.is_terminal:
                    record.terminate(OperationState.CANCELLED)
                raise
            if control_request.kind is RuntimeControlKind.CANCEL:
                event = build_cancelled_event("peer_cancelled")
                record.terminate(OperationState.CANCELLED)
                await operation.send_result(_terminal_metadata(operation, event), _encode_event(event))
                counters.terminal_results += 1
            elif control_request.kind in {RuntimeControlKind.ABORT, RuntimeControlKind.SERVER_SHUTDOWN}:
                record.terminate(OperationState.DROPPED)
                diagnostic = control_request.diagnostic or b"peer_abort"
                drop_reason = (
                    ResultDropReasonCode.PEER_CANCELLED
                    if control_request.kind is RuntimeControlKind.ABORT
                    else ResultDropReasonCode.TRANSPORT_CLOSED
                )
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
            terminal_sent = True
            return
        except Exception as error:
            if terminal_sent:
                raise
            event = _runtime_error_event(error)
        else:
            if not terminal_sent:
                event = _missing_terminal_event()

        if not terminal_sent:
            record.terminate(_terminal_operation_state(event))
            terminal_sent = True
            await operation.send_result(_terminal_metadata(operation, event), _encode_event(event))
            counters.terminal_results += 1
    finally:
        if record.is_terminal and not record.resources_released:
            record.release_resources()


async def _handle_runtime_control(
    event: NativeRuntimeEvent,
    *,
    registry: OperationRegistry,
    controls: RuntimeControlRegistry,
    counters: _ServeCounters,
) -> None:
    request = decode_operation_control(event)
    if request is None:
        return
    try:
        terminal = registry.get(request.operation_id).is_terminal
    except OperationStateError:
        terminal = False
    disposition = await controls.apply(request, terminal=terminal)
    if disposition is RuntimeControlDisposition.APPLIED:
        counters.applied_control_events += 1
    else:
        counters.rejected_control_events += 1


def _decode_request(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("native FRAME_SUBMIT body must contain a UTF-8 OpenAI profile request") from error
    if not isinstance(value, dict):
        raise ValueError("native FRAME_SUBMIT body must contain an OpenAI profile request object")
    return value


def _encode_event(event: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(event), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _is_terminal_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") in _TERMINAL_EVENT_TYPES


def _terminal_operation_state(event: Mapping[str, Any]) -> OperationState:
    event_type = event.get("type")
    if event_type == "response.completed":
        return OperationState.COMPLETED
    if event_type == "response.cancelled":
        return OperationState.CANCELLED
    return OperationState.FAILED


def _backend_request_id(operation_id: int, frame_id: int) -> str:
    return f"nnrp-{operation_id}-{frame_id}"


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
    event: Mapping[str, Any],
) -> ResultPushMetadata:
    return ResultPushMetadata(
        status_code=_status_code(event),
        result_flags=ResultFlags(0),
        section_count=0,
        tile_count=0,
        active_profile_id=int(operation.metadata.input_profile),
        reserved0=0,
        inference_ms=0,
        queue_ms=0,
        server_total_ms=0,
        reserved1=0,
        tile_base_id=0,
        tile_index_bytes=0,
        result_class=ResultClass.COMPLETE,
        payload_kind_bitmap=PayloadKind.STRUCTURED_EVENT,
        payload_frame_count=1,
    )


def _status_code(event: Mapping[str, Any]) -> int:
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


def _missing_terminal_event() -> dict[str, Any]:
    return {
        "type": "response.error",
        "error": {
            "type": "server_error",
            "code": "missing_terminal_event",
            "message": "Adapter result stream ended without a terminal event.",
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
