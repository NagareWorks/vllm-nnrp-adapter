from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, cast

from nnrp.client import (  # type: ignore[import-untyped]
    NativeClientConnection,
    NativeClientOptions,
    NativeClientProviderRoute,
    NativeClientSessionOptions,
    SubmitHeaderContext,
    SubmitIdentity,
    SubmitPolicy,
    SubmitRequest,
    TypedPayloadInputFrame,
    TypedPayloadSubmitInput,
    connect_native_client_connection,
)
from nnrp.core import MessageType, PayloadKind, TransportPolicy  # type: ignore[import-untyped]
from nnrp.native import (  # type: ignore[import-untyped]
    NativeRuntimeOperation,
    NativeRuntimeSession,
    NativeWouldBlockError,
)
from nnrp.runtime import (  # type: ignore[import-untyped]
    NativeClientEvent,
    NativeRuntimeEvent,
    OperationLifecycleEvent,
    OperationState,
    ResultDropReasonCode,
    ResultDropReasonMetadata,
)
from nnrp.schema import StreamSemantics  # type: ignore[import-untyped]

from .benchmark import synthetic_prompt
from .profile import CHAT_COMPLETIONS_CREATE, OPENAI_COMPATIBLE_SCHEMA_VERSION
from .stale_work_workload import StaleWorkCase, StaleWorkResult

_CONTROL_KINDS = frozenset(("cancel", "abort", "deadline", "supersede"))
_CONTROL_OUTCOME = {
    "cancel": "cancelled",
    "abort": "aborted",
    "deadline": "expired",
    "supersede": "superseded",
}


class _HttpResponse(Protocol):
    def __iter__(self) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class OrchestratedHttpController(Protocol):
    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[StaleWorkCase],
    ) -> None: ...

    async def dispatch(self, control: OrchestratedHttpControl) -> bool: ...

    async def end_run(self) -> None: ...


@dataclass(frozen=True)
class OpenAiHttpSseDriverConfig:
    endpoint: str
    api_key: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    sample_id_header: str = "X-NNRP-Benchmark-Sample-Id"

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute http:// or https:// URL")
        if self.api_key is not None and not self.api_key:
            raise ValueError("api_key must be non-empty when provided")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        _validate_header(self.sample_id_header, "sample_id_header")
        copied_headers: dict[str, str] = {}
        for name, value in self.headers.items():
            _validate_header(name, "headers name")
            if not isinstance(value, str) or not value or "\r" in value or "\n" in value:
                raise ValueError("headers values must be non-empty single-line strings")
            copied_headers[name] = value
        object.__setattr__(self, "headers", MappingProxyType(copied_headers))


@dataclass(frozen=True)
class OrchestratedHttpControl:
    sample_id: str
    control_kind: str
    replacement_sample_id: str | None = None
    deadline_unix_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if self.control_kind not in _CONTROL_KINDS:
            raise ValueError(f"unsupported stale-work control kind: {self.control_kind}")
        if self.control_kind == "supersede":
            if not isinstance(self.replacement_sample_id, str) or not self.replacement_sample_id:
                raise ValueError("supersede control requires replacement_sample_id")
        elif self.replacement_sample_id is not None:
            raise ValueError("replacement_sample_id is only valid for supersede")
        if self.control_kind == "deadline":
            if (
                isinstance(self.deadline_unix_ms, bool)
                or not isinstance(self.deadline_unix_ms, int)
                or self.deadline_unix_ms <= 0
            ):
                raise ValueError("deadline control requires a positive deadline_unix_ms")
        elif self.deadline_unix_ms is not None:
            raise ValueError("deadline_unix_ms is only valid for deadline")


@dataclass(frozen=True)
class OrchestratedHttpSseDriverConfig:
    request: OpenAiHttpSseDriverConfig
    controller: OrchestratedHttpController

    def __post_init__(self) -> None:
        if not isinstance(self.request, OpenAiHttpSseDriverConfig):
            raise TypeError("request must be OpenAiHttpSseDriverConfig")
        for method_name in ("begin_run", "dispatch", "end_run"):
            if not callable(getattr(self.controller, method_name, None)):
                raise TypeError(f"controller must define callable {method_name}()")


class RawOpenAiHttpSseDriver:
    """Raw OpenAI HTTP/SSE baseline whose only control is client disconnect."""

    baseline = "raw_openai_http_sse"

    def __init__(self, config: OpenAiHttpSseDriverConfig) -> None:
        self._config = config
        self._active: set[_RawOpenAiHttpSseOperation] = set()
        self._begun = False

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[StaleWorkCase],
    ) -> None:
        if self._begun or self._active:
            raise RuntimeError("raw HTTP/SSE driver run is already active")
        if not schedule or len(schedule) != workload.get("sample_count"):
            raise ValueError("raw HTTP/SSE driver requires the complete workload schedule")
        self._begun = True

    async def warmup(self, case: StaleWorkCase) -> None:
        operation = await self.start(case)
        try:
            result = await operation.wait()
            if result.terminal_outcome != "completed":
                raise RuntimeError("raw HTTP/SSE warmup request did not complete")
        finally:
            await operation.close()

    async def start(self, case: StaleWorkCase) -> _RawOpenAiHttpSseOperation:
        if not self._begun:
            raise RuntimeError("raw HTTP/SSE driver has not begun a run")
        operation = _RawOpenAiHttpSseOperation(self._config, case, self._active.discard)
        self._active.add(operation)
        return operation

    async def end_run(self) -> None:
        operations = tuple(self._active)
        if operations:
            await asyncio.gather(*(operation.close() for operation in operations))
        self._begun = False


class OrchestratedHttpSseDriver:
    """HTTP/SSE baseline whose controls are supplied by deployment orchestration."""

    baseline = "orchestrated_http_sse"

    def __init__(self, config: OrchestratedHttpSseDriverConfig) -> None:
        self._config = config
        self._active: set[_OrchestratedHttpSseOperation] = set()
        self._replacement_tasks: set[asyncio.Task[None]] = set()
        self._begun = False

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[StaleWorkCase],
    ) -> None:
        if self._begun or self._active or self._replacement_tasks:
            raise RuntimeError("orchestrated HTTP/SSE driver run is already active")
        if not schedule or len(schedule) != workload.get("sample_count"):
            raise ValueError("orchestrated HTTP/SSE driver requires the complete workload schedule")
        await self._config.controller.begin_run(workload, schedule)
        self._begun = True

    async def warmup(self, case: StaleWorkCase) -> None:
        operation = await self.start(case)
        try:
            result = await operation.wait()
            if result.terminal_outcome != "completed":
                raise RuntimeError("orchestrated HTTP/SSE warmup request did not complete")
        finally:
            await operation.close()

    async def start(self, case: StaleWorkCase) -> _OrchestratedHttpSseOperation:
        if not self._begun:
            raise RuntimeError("orchestrated HTTP/SSE driver has not begun a run")
        operation = _OrchestratedHttpSseOperation(self, case)
        self._active.add(operation)
        return operation

    async def end_run(self) -> None:
        cleanup_errors: list[BaseException] = []
        if self._replacement_tasks:
            replacement_results = await asyncio.gather(*tuple(self._replacement_tasks), return_exceptions=True)
            cleanup_errors.extend(result for result in replacement_results if isinstance(result, BaseException))
        operations = tuple(self._active)
        if operations:
            close_results = await asyncio.gather(
                *(operation.close() for operation in operations),
                return_exceptions=True,
            )
            cleanup_errors.extend(result for result in close_results if isinstance(result, BaseException))
        try:
            await self._config.controller.end_run()
        except BaseException as error:
            cleanup_errors.append(error)
        self._active.clear()
        self._replacement_tasks.clear()
        self._begun = False
        if cleanup_errors:
            raise RuntimeError("orchestrated HTTP/SSE driver cleanup failed") from cleanup_errors[0]

    async def _start_replacement(self, case: StaleWorkCase, replacement_sample_id: str) -> None:
        replacement = await self.start(
            replace(
                case,
                sample_id=replacement_sample_id,
                control_kind=None,
            )
        )
        task = asyncio.create_task(self._drain_replacement(replacement))
        self._replacement_tasks.add(task)
        task.add_done_callback(self._replacement_tasks.discard)

    async def _drain_replacement(self, replacement: _OrchestratedHttpSseOperation) -> None:
        try:
            result = await replacement.wait()
            if result.terminal_outcome != "completed":
                raise RuntimeError("orchestrated HTTP/SSE supersede replacement did not complete")
        finally:
            await replacement.close()

    def _release(self, operation: _OrchestratedHttpSseOperation) -> None:
        self._active.discard(operation)


class _OrchestratedHttpSseOperation:
    def __init__(self, driver: OrchestratedHttpSseDriver, case: StaleWorkCase) -> None:
        self._driver = driver
        self._case = case
        self._control_lock = asyncio.Lock()
        self._inner = _RawOpenAiHttpSseOperation(
            driver._config.request,
            case,
            lambda _operation: driver._release(self),
            close_response_on_control=False,
        )

    async def apply_control(self, control_kind: str) -> bool:
        if control_kind not in _CONTROL_KINDS:
            raise ValueError(f"unsupported stale-work control kind: {control_kind}")
        async with self._control_lock:
            with self._inner._state_lock:
                if (
                    self._inner._finished
                    or self._inner._closed
                    or self._inner._control_kind is not None
                ):
                    return False
            replacement_sample_id = (
                f"{self._case.sample_id}:replacement" if control_kind == "supersede" else None
            )
            control = OrchestratedHttpControl(
                sample_id=self._case.sample_id,
                control_kind=control_kind,
                replacement_sample_id=replacement_sample_id,
                deadline_unix_ms=(
                    max(1, time.time_ns() // 1_000_000) if control_kind == "deadline" else None
                ),
            )
            dispatched = await self._driver._config.controller.dispatch(control)
            if not isinstance(dispatched, bool):
                raise TypeError("orchestrated HTTP/SSE controller dispatch() must return bool")
            if not dispatched:
                return False
            with self._inner._state_lock:
                self._inner._control_kind = control_kind
            if replacement_sample_id is not None:
                await self._driver._start_replacement(self._case, replacement_sample_id)
            return True

    async def wait(self) -> StaleWorkResult:
        result = await self._inner.wait()
        control_kind = self._inner._control_kind
        if result.terminal_outcome == "cancelled" and control_kind is not None:
            return replace(result, terminal_outcome=_CONTROL_OUTCOME[control_kind])
        return result

    async def close(self) -> None:
        await self._inner.close()


@dataclass(frozen=True)
class DirectNnrpDriverConfig:
    endpoint: str
    provider_routes: Mapping[str, NativeClientProviderRoute] = field(default_factory=dict)
    transport_policy: TransportPolicy = TransportPolicy.AUTO
    timeout_seconds: float = 300.0
    event_poll_seconds: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint:
            raise ValueError("endpoint must be a non-empty nnrp:// application endpoint")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if not math.isfinite(self.event_poll_seconds) or self.event_poll_seconds <= 0:
            raise ValueError("event_poll_seconds must be positive and finite")
        routes = MappingProxyType(dict(self.provider_routes))
        policy = TransportPolicy(self.transport_policy)
        NativeClientOptions(self.endpoint, provider_routes=routes, transport_policy=policy)
        object.__setattr__(self, "provider_routes", routes)
        object.__setattr__(self, "transport_policy", policy)


class DirectNnrpDriver:
    """Persistent native NNRP client for the stale-work comparison path."""

    baseline = "direct_nnrp"

    def __init__(self, config: DirectNnrpDriverConfig) -> None:
        self._config = config
        self._connection_context: AbstractContextManager[NativeClientConnection] | None = None
        self._connection: NativeClientConnection | None = None
        self._session: NativeRuntimeSession | None = None
        self._event_pump: asyncio.Task[None] | None = None
        self._operations_by_id: dict[int, _DirectNnrpOperation] = {}
        self._operations_by_frame: dict[int, _DirectNnrpOperation] = {}
        self._replacement_tasks: set[asyncio.Task[None]] = set()
        self._next_operation_id = 1
        self._next_frame_id = 1
        self._next_control_sequence = 1
        self._pump_error: BaseException | None = None
        self._begun = False

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[StaleWorkCase],
    ) -> None:
        if self._begun or self._connection_context is not None or self._operations_by_id:
            raise RuntimeError("direct NNRP driver run is already active")
        if not schedule or len(schedule) != workload.get("sample_count"):
            raise ValueError("direct NNRP driver requires the complete workload schedule")
        max_in_flight = workload.get("max_in_flight")
        if isinstance(max_in_flight, bool) or not isinstance(max_in_flight, int) or not 1 <= max_in_flight <= 0xFFFF:
            raise ValueError("direct NNRP driver requires max_in_flight in 1..65535")

        timeout_ms = min(0xFFFFFFFF, max(1, math.ceil(self._config.timeout_seconds * 1_000)))
        session_capacity = min(0xFFFF, max_in_flight * 2)
        options = NativeClientOptions(
            self._config.endpoint,
            provider_routes=self._config.provider_routes,
            transport_policy=self._config.transport_policy,
            session_defaults=NativeClientSessionOptions(
                profile_id=0,
                schema_id=0,
                schema_version=0,
                default_deadline_ms=timeout_ms,
                max_in_flight_operations=session_capacity,
                lease_ttl_hint_ms=max(30_000, timeout_ms),
            ),
        )
        context = connect_native_client_connection(options)
        connection: NativeClientConnection | None = None
        try:
            connection = cast(NativeClientConnection, await asyncio.to_thread(context.__enter__))
            session = await connection.open_session()
        except BaseException:
            if connection is not None:
                await asyncio.to_thread(context.__exit__, None, None, None)
            raise

        self._connection_context = context
        self._connection = connection
        self._session = session
        self._pump_error = None
        self._begun = True
        self._event_pump = asyncio.create_task(self._pump_events(), name="nnrp-stale-work-events")

    async def warmup(self, case: StaleWorkCase) -> None:
        operation = await self.start(case)
        try:
            result = await operation.wait()
            if result.terminal_outcome != "completed":
                raise RuntimeError("direct NNRP warmup request did not complete")
        finally:
            await operation.close()

    async def start(self, case: StaleWorkCase) -> _DirectNnrpOperation:
        return await self._start_operation(case)

    async def end_run(self) -> None:
        cleanup_errors: list[BaseException] = []
        if self._replacement_tasks:
            replacement_results = await asyncio.gather(*tuple(self._replacement_tasks), return_exceptions=True)
            cleanup_errors.extend(result for result in replacement_results if isinstance(result, BaseException))

        operations = tuple(self._operations_by_id.values())
        if operations:
            close_results = await asyncio.gather(
                *(operation.close() for operation in operations),
                return_exceptions=True,
            )
            cleanup_errors.extend(result for result in close_results if isinstance(result, BaseException))

        pump = self._event_pump
        self._event_pump = None
        if pump is not None:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)

        context = self._connection_context
        self._connection_context = None
        self._connection = None
        self._session = None
        self._operations_by_id.clear()
        self._operations_by_frame.clear()
        self._replacement_tasks.clear()
        self._begun = False
        if context is not None:
            try:
                await asyncio.to_thread(context.__exit__, None, None, None)
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise RuntimeError("direct NNRP driver cleanup failed") from cleanup_errors[0]

    async def _start_operation(
        self,
        case: StaleWorkCase,
        *,
        identity: tuple[int, int] | None = None,
    ) -> _DirectNnrpOperation:
        if not self._begun or self._session is None:
            raise RuntimeError("direct NNRP driver has not begun a run")
        if self._pump_error is not None:
            raise RuntimeError("direct NNRP event pump failed") from self._pump_error
        operation_id, frame_id = self._allocate_operation_identity() if identity is None else identity
        operation = _DirectNnrpOperation(self, case, operation_id, frame_id)
        self._operations_by_id[operation_id] = operation
        self._operations_by_frame[frame_id] = operation
        try:
            native_operation = await self._session.async_submit_operation(
                _nnrp_submit_request(case, operation_id=operation_id, frame_id=frame_id)
            )
        except BaseException:
            self._release(operation)
            raise
        try:
            operation._bind(native_operation)
        except BaseException:
            self._release(operation)
            await asyncio.to_thread(native_operation.cancel)
            raise
        return operation

    async def _start_replacement(
        self,
        case: StaleWorkCase,
        *,
        identity: tuple[int, int],
    ) -> _DirectNnrpOperation:
        replacement = await self._start_operation(
            replace(
                case,
                sample_id=f"{case.sample_id}:replacement",
                control_kind=None,
            ),
            identity=identity,
        )
        task = asyncio.create_task(self._drain_replacement(replacement))
        self._replacement_tasks.add(task)
        task.add_done_callback(self._replacement_tasks.discard)
        return replacement

    async def _drain_replacement(self, replacement: _DirectNnrpOperation) -> None:
        try:
            result = await replacement.wait()
            if result.terminal_outcome != "completed":
                raise RuntimeError("supersede replacement did not complete")
        finally:
            await replacement.close()

    async def _pump_events(self) -> None:
        assert self._session is not None
        while True:
            try:
                event = await self._session.next_event(timeout=self._config.event_poll_seconds)
            except NativeWouldBlockError:
                continue
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                self._pump_error = error
                for operation in tuple(self._operations_by_id.values()):
                    operation._fail(error)
                return
            observed_operation = self._operation_for_event(event)
            if observed_operation is not None:
                observed_operation._observe(event)

    def _operation_for_event(self, event: NativeClientEvent) -> _DirectNnrpOperation | None:
        if isinstance(event, OperationLifecycleEvent):
            return self._operations_by_id.get(event.operation_id)
        operation_id = getattr(event.metadata.value, "operation_id", 0)
        if type(operation_id) is int and operation_id > 0:
            operation = self._operations_by_id.get(operation_id)
            if operation is not None:
                return operation
        return self._operations_by_frame.get(event.header.frame_id)

    def _allocate_operation_identity(self) -> tuple[int, int]:
        operation_id = self._next_operation_id
        frame_id = self._next_frame_id
        if operation_id > 0xFFFFFFFFFFFFFFFF or frame_id > 0xFFFFFFFF:
            raise OverflowError("direct NNRP benchmark exhausted operation identity space")
        self._next_operation_id += 1
        self._next_frame_id += 1
        return operation_id, frame_id

    def _allocate_control_sequence(self) -> int:
        sequence = self._next_control_sequence
        if sequence > 0xFFFFFFFF:
            raise OverflowError("direct NNRP benchmark exhausted control sequence space")
        self._next_control_sequence += 1
        return sequence

    def _release(self, operation: _DirectNnrpOperation) -> None:
        if self._operations_by_id.get(operation.operation_id) is operation:
            self._operations_by_id.pop(operation.operation_id, None)
        if self._operations_by_frame.get(operation.frame_id) is operation:
            self._operations_by_frame.pop(operation.frame_id, None)


class _DirectNnrpOperation:
    def __init__(
        self,
        driver: DirectNnrpDriver,
        case: StaleWorkCase,
        operation_id: int,
        frame_id: int,
    ) -> None:
        self._driver = driver
        self._case = case
        self.operation_id = operation_id
        self.frame_id = frame_id
        self._native_operation: NativeRuntimeOperation | None = None
        self._result: asyncio.Future[StaleWorkResult] = asyncio.get_running_loop().create_future()
        self._control_lock = asyncio.Lock()
        self._control_kind: str | None = None
        self._late_result_count = 0
        self._closed = False

    def _bind(self, operation: NativeRuntimeOperation) -> None:
        if operation.operation_id != self.operation_id or operation.frame_id != self.frame_id:
            raise RuntimeError("native operation identity differs from the submitted identity")
        self._native_operation = operation

    async def apply_control(self, control_kind: str) -> bool:
        if control_kind not in _CONTROL_KINDS:
            raise ValueError(f"unsupported stale-work control kind: {control_kind}")
        async with self._control_lock:
            if self._closed or self._result.done() or self._control_kind is not None:
                return False
            connection = self._driver._connection
            session = self._driver._session
            if connection is None or session is None:
                return False
            sequence = self._driver._allocate_control_sequence()
            if control_kind == "cancel":
                await asyncio.to_thread(
                    connection.cancel_runtime_operation,
                    session,
                    operation_id=self.operation_id,
                    control_sequence=sequence,
                    diagnostic=b"stale_work_cancel",
                )
                self._control_kind = control_kind
            elif control_kind == "abort":
                await asyncio.to_thread(
                    connection.abort_runtime_operation,
                    session,
                    operation_id=self.operation_id,
                    control_sequence=sequence,
                    diagnostic=b"stale_work_abort",
                )
                self._control_kind = control_kind
            elif control_kind == "deadline":
                await asyncio.to_thread(
                    connection.update_runtime_deadline,
                    session,
                    operation_id=self.operation_id,
                    control_sequence=sequence,
                    deadline_unix_ms=max(1, time.time_ns() // 1_000_000),
                )
                self._control_kind = control_kind
            else:
                replacement_identity = self._driver._allocate_operation_identity()
                await asyncio.to_thread(
                    connection.supersede_runtime_operation,
                    session,
                    old_operation_id=self.operation_id,
                    new_operation_id=replacement_identity[0],
                    control_sequence=sequence,
                    diagnostic=b"stale_work_supersede",
                )
                self._control_kind = control_kind
                await self._driver._start_replacement(self._case, identity=replacement_identity)
            return True

    async def wait(self) -> StaleWorkResult:
        return await asyncio.wait_for(asyncio.shield(self._result), timeout=self._driver._config.timeout_seconds)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._driver._release(self)
        if not self._result.done():
            self._result.cancel()
            native_operation = self._native_operation
            if native_operation is not None:
                await asyncio.to_thread(native_operation.cancel)

    def _observe(self, event: NativeClientEvent) -> None:
        if self._closed or self._result.done():
            return
        if isinstance(event, NativeRuntimeEvent):
            message_type = event.header.message_type
            if self._control_kind is not None and message_type in {
                MessageType.PROGRESS,
                MessageType.PARTIAL_RESULT,
                MessageType.RESULT_PUSH,
            }:
                self._late_result_count += 1
            if message_type is MessageType.RESULT_PUSH:
                self._complete("completed")
            elif message_type is MessageType.RESULT_DROP_REASON:
                metadata = event.metadata.value
                outcome = (
                    _drop_terminal_outcome(metadata.drop_reason_code, self._control_kind)
                    if isinstance(metadata, ResultDropReasonMetadata)
                    else "failed"
                )
                self._complete(outcome)
            elif message_type is MessageType.RESULT_DROP:
                self._complete("failed")
            return

        lifecycle_outcome = {
            OperationState.COMPLETED: "completed",
            OperationState.CANCELLED: "cancelled",
            OperationState.SUPERSEDED: "superseded",
            OperationState.FAILED: "failed",
        }.get(event.state)
        if lifecycle_outcome is not None:
            self._complete(lifecycle_outcome)

    def _complete(self, outcome: str) -> None:
        useful_result_weight = 0.0 if self._case.is_stale else float(outcome == "completed")
        self._result.set_result(
            StaleWorkResult(
                terminal_outcome=outcome,
                useful_result_weight=useful_result_weight,
                late_result_count=self._late_result_count,
            )
        )

    def _fail(self, error: BaseException) -> None:
        if not self._result.done():
            self._result.set_exception(error)


def _nnrp_submit_request(case: StaleWorkCase, *, operation_id: int, frame_id: int) -> SubmitRequest:
    payload = json.dumps(
        {
            "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
            "operation": CHAT_COMPLETIONS_CREATE,
            "request_id": case.sample_id,
            "body": {
                "model": case.model,
                "messages": [{"role": "user", "content": synthetic_prompt(case.prompt_tokens)}],
                "stream": True,
                "max_tokens": case.max_completion_tokens,
                "stream_options": {"include_usage": True},
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return SubmitRequest.typed_payload(
        TypedPayloadSubmitInput(
            identity=SubmitIdentity(
                operation_id=operation_id,
                frame_id=frame_id,
                header=SubmitHeaderContext(trace_id=operation_id),
            ),
            policy=SubmitPolicy(),
            frames=(
                TypedPayloadInputFrame(
                    profile_id=0,
                    payload_kind=PayloadKind.STRUCTURED_EVENT,
                    payload=payload,
                    schema_id=0,
                    schema_version=0,
                    stream_semantics=StreamSemantics.SNAPSHOT,
                ),
            ),
        )
    )


def _drop_terminal_outcome(
    reason: ResultDropReasonCode | int,
    control_kind: str | None,
) -> str:
    try:
        normalized = ResultDropReasonCode(reason)
    except ValueError:
        return "failed"
    if normalized is ResultDropReasonCode.DEADLINE_EXPIRED:
        return "expired"
    if normalized is ResultDropReasonCode.SUPERSEDED:
        return "superseded"
    if normalized is ResultDropReasonCode.PEER_CANCELLED:
        if control_kind == "cancel":
            return "cancelled"
        if control_kind == "abort":
            return "aborted"
    return "failed"


class _RawOpenAiHttpSseOperation:
    def __init__(
        self,
        config: OpenAiHttpSseDriverConfig,
        case: StaleWorkCase,
        release: Callable[[_RawOpenAiHttpSseOperation], None],
        *,
        close_response_on_control: bool = True,
    ) -> None:
        self._config = config
        self._case = case
        self._release = release
        self._close_response_on_control = close_response_on_control
        self._state_lock = threading.Lock()
        self._wait_lock = asyncio.Lock()
        self._response: _HttpResponse | None = None
        self._control_kind: str | None = None
        self._finished = False
        self._closed = False
        self._result: StaleWorkResult | None = None

    async def apply_control(self, control_kind: str) -> bool:
        if control_kind not in _CONTROL_KINDS:
            raise ValueError(f"unsupported stale-work control kind: {control_kind}")
        with self._state_lock:
            if self._finished or self._closed or self._control_kind is not None:
                return False
            self._control_kind = control_kind
            response = self._response
        if response is not None and self._close_response_on_control:
            await asyncio.to_thread(_close_response, response)
        return True

    async def wait(self) -> StaleWorkResult:
        async with self._wait_lock:
            if self._result is None:
                self._result = await asyncio.to_thread(self._run_sync)
            return self._result

    async def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            response = self._response
        if response is not None:
            await asyncio.to_thread(_close_response, response)
        self._release(self)

    def _run_sync(self) -> StaleWorkResult:
        request = urllib.request.Request(
            self._config.endpoint,
            data=_request_body(self._case),
            headers=_request_headers(self._config, self._case),
            method="POST",
        )
        response: _HttpResponse | None = None
        stream_completed = False
        late_result_count = 0
        try:
            response = _urlopen(request, self._config.timeout_seconds)
            with self._state_lock:
                self._response = response
                close_immediately = (
                    self._closed
                    or self._control_kind is not None
                    and self._close_response_on_control
                )
            if close_immediately:
                _close_response(response)
            for raw_line in response:
                data = _sse_data(raw_line)
                if data is None:
                    continue
                if data == "[DONE]":
                    stream_completed = True
                    break
                parsed = json.loads(data)
                if not isinstance(parsed, Mapping):
                    raise ValueError("OpenAI SSE data must decode to an object")
                with self._state_lock:
                    if self._control_kind is not None:
                        late_result_count += 1
        except (UnicodeDecodeError, ValueError):
            terminal_outcome = "failed"
        except (OSError, urllib.error.URLError):
            terminal_outcome = "cancelled" if self._control_requested() else "failed"
        else:
            terminal_outcome = "completed" if stream_completed or not self._control_requested() else "cancelled"
        finally:
            if response is not None:
                _close_response(response)
            with self._state_lock:
                self._response = None
                self._finished = True

        return StaleWorkResult(
            terminal_outcome=terminal_outcome,
            useful_result_weight=0.0 if self._case.is_stale else float(terminal_outcome == "completed"),
            late_result_count=late_result_count,
        )

    def _control_requested(self) -> bool:
        with self._state_lock:
            return self._control_kind is not None


def _request_body(case: StaleWorkCase) -> bytes:
    return json.dumps(
        {
            "model": case.model,
            "messages": [{"role": "user", "content": synthetic_prompt(case.prompt_tokens)}],
            "stream": True,
            "max_tokens": case.max_completion_tokens,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _request_headers(config: OpenAiHttpSseDriverConfig, case: StaleWorkCase) -> dict[str, str]:
    headers = dict(config.headers)
    headers["Content-Type"] = "application/json"
    headers[config.sample_id_header] = case.sample_id
    if config.api_key is not None:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def _sse_data(raw_line: bytes) -> str | None:
    line = raw_line.decode("utf-8").strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[5:].strip()
    return data or None


def _urlopen(request: urllib.request.Request, timeout_seconds: float) -> _HttpResponse:
    return cast(_HttpResponse, urllib.request.urlopen(request, timeout=timeout_seconds))


def _close_response(response: _HttpResponse) -> None:
    try:
        response.close()
    except OSError:
        pass


def _validate_header(value: object, location: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or ":" in value
        or "\r" in value
        or "\n" in value
        or value != value.strip()
    ):
        raise ValueError(f"{location} must be a non-empty HTTP header name")
