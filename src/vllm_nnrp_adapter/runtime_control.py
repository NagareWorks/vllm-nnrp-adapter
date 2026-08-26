from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from nnrp.core import MessageType  # type: ignore[import-untyped]
from nnrp.runtime import (  # type: ignore[import-untyped]
    ControlRequestMetadata,
    NativeRuntimeEvent,
    RuntimeEventTailKind,
    RuntimeRole,
    SchedulingMetadata,
    SupersedeMetadata,
)


class RuntimeControlKind(StrEnum):
    CANCEL = "cancel"
    ABORT = "abort"
    PEER_DISCONNECT = "peer_disconnect"
    SERVER_SHUTDOWN = "server_shutdown"
    DEADLINE_EXPIRED = "deadline_expired"
    SUPERSEDE = "supersede"
    OBJECT_INVALIDATED = "object_invalidated"


class RuntimeControlDisposition(StrEnum):
    APPLIED = "applied"
    STALE = "stale"
    UNKNOWN_OPERATION = "unknown_operation"
    TERMINAL_OPERATION = "terminal_operation"
    LIVE_UPDATE_REQUIRED = "live_update_required"
    UNSUPPORTED_LIVE_UPDATE = "unsupported_live_update"


@dataclass(frozen=True, slots=True)
class RuntimeControlRequest:
    kind: RuntimeControlKind
    operation_id: int
    control_sequence: int
    reason_code: int
    source_role: RuntimeRole | int
    flags: int
    diagnostic: bytes
    replacement_operation_id: int = 0


@dataclass(frozen=True, slots=True)
class RuntimeDeadlineUpdate:
    operation_id: int
    control_sequence: int
    deadline_unix_ms: int
    flags: int


@dataclass(frozen=True, slots=True)
class RuntimePriorityUpdate:
    operation_id: int
    control_sequence: int
    priority_class: int
    priority_delta: int
    flags: int

    @property
    def backend_priority(self) -> int:
        return self.priority_class + self.priority_delta


@dataclass(slots=True)
class OperationControlSlot:
    operation_id: int
    task: asyncio.Task[None] | None = None
    last_control_sequence: int = 0
    terminal_request: RuntimeControlRequest | None = None
    deadline_update: RuntimeDeadlineUpdate | None = None
    priority_update: RuntimePriorityUpdate | None = None
    backend_dispatched: bool = False
    _deadline_task: asyncio.Task[None] | None = None
    _terminal_request_ready: asyncio.Event = field(
        default_factory=asyncio.Event,
        init=False,
        repr=False,
    )

    def bind(self, task: asyncio.Task[None]) -> None:
        if self.task is not None:
            raise RuntimeError(f"operation {self.operation_id} already has a bound task")
        self.task = task
        self._schedule_deadline()

    async def apply(self, request: RuntimeControlRequest, *, terminal: bool) -> RuntimeControlDisposition:
        if request.control_sequence <= self.last_control_sequence:
            return RuntimeControlDisposition.STALE
        self.last_control_sequence = request.control_sequence
        if terminal:
            return RuntimeControlDisposition.TERMINAL_OPERATION
        if self.terminal_request is not None:
            return RuntimeControlDisposition.STALE
        self.terminal_request = request
        await self._cancel_deadline()
        task = self.task
        if task is None:
            raise RuntimeError(f"operation {self.operation_id} does not have a bound task")
        await asyncio.sleep(0)
        self._terminal_request_ready.set()
        task.cancel()
        return RuntimeControlDisposition.APPLIED

    async def activate_reserved(
        self,
        request: RuntimeControlRequest,
        *,
        terminal: bool,
    ) -> RuntimeControlDisposition:
        if request.control_sequence != self.last_control_sequence:
            return RuntimeControlDisposition.STALE
        if terminal:
            return RuntimeControlDisposition.TERMINAL_OPERATION
        if self.terminal_request is not None:
            return RuntimeControlDisposition.STALE
        self.terminal_request = request
        self._terminal_request_ready.set()
        await self._cancel_deadline()
        task = self.task
        if task is None:
            raise RuntimeError(f"operation {self.operation_id} does not have a bound task")
        task.cancel()
        return RuntimeControlDisposition.APPLIED

    async def apply_deadline(
        self,
        update: RuntimeDeadlineUpdate,
        *,
        terminal: bool,
    ) -> RuntimeControlDisposition:
        if update.control_sequence <= self.last_control_sequence:
            return RuntimeControlDisposition.STALE
        self.last_control_sequence = update.control_sequence
        if terminal or self.terminal_request is not None:
            return RuntimeControlDisposition.TERMINAL_OPERATION
        self.deadline_update = update
        await self._cancel_deadline()
        self._schedule_deadline()
        return RuntimeControlDisposition.APPLIED

    def apply_priority(
        self,
        update: RuntimePriorityUpdate,
        *,
        terminal: bool,
        backend_supported: bool = True,
    ) -> RuntimeControlDisposition:
        if update.control_sequence <= self.last_control_sequence:
            return RuntimeControlDisposition.STALE
        self.last_control_sequence = update.control_sequence
        if terminal or self.terminal_request is not None:
            return RuntimeControlDisposition.TERMINAL_OPERATION
        if not backend_supported:
            return RuntimeControlDisposition.UNSUPPORTED_LIVE_UPDATE
        self.priority_update = update
        if self.backend_dispatched:
            return RuntimeControlDisposition.LIVE_UPDATE_REQUIRED
        return RuntimeControlDisposition.APPLIED

    def begin_backend_dispatch(self) -> int | None:
        self.backend_dispatched = True
        update = self.priority_update
        return None if update is None else update.backend_priority

    async def complete(self) -> None:
        await self._cancel_deadline()

    async def wait_for_terminal_request(self) -> RuntimeControlRequest:
        await self._terminal_request_ready.wait()
        request = self.terminal_request
        if request is None:
            raise RuntimeError(f"operation {self.operation_id} has no terminal request")
        return request

    def _schedule_deadline(self) -> None:
        if self.task is None or self.deadline_update is None:
            return
        self._deadline_task = asyncio.create_task(self._wait_for_deadline(self.deadline_update))

    async def _wait_for_deadline(self, update: RuntimeDeadlineUpdate) -> None:
        delay_seconds = max(0.0, (update.deadline_unix_ms - time.time() * 1000) / 1000)
        await asyncio.sleep(delay_seconds)
        if self.terminal_request is not None or self.task is None or self.task.done():
            return
        self.terminal_request = RuntimeControlRequest(
            kind=RuntimeControlKind.DEADLINE_EXPIRED,
            operation_id=self.operation_id,
            control_sequence=update.control_sequence,
            reason_code=0,
            source_role=RuntimeRole.RUNTIME,
            flags=update.flags,
            diagnostic=b"deadline_expired",
        )
        self._terminal_request_ready.set()
        self.task.cancel()

    async def _cancel_deadline(self) -> None:
        task = self._deadline_task
        self._deadline_task = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class RuntimeControlRegistry:
    def __init__(self) -> None:
        self._slots: dict[int, OperationControlSlot] = {}
        self._pending_deadlines: dict[int, RuntimeDeadlineUpdate] = {}
        self._pending_supersedes: dict[int, RuntimeControlRequest] = {}
        self._pending_supersedes_by_old: dict[int, RuntimeControlRequest] = {}

    def register(self, operation_id: int) -> OperationControlSlot:
        if operation_id in self._slots:
            raise RuntimeError(f"duplicate runtime-control operation {operation_id}")
        slot = OperationControlSlot(operation_id)
        pending_deadline = self._pending_deadlines.pop(operation_id, None)
        if pending_deadline is not None:
            slot.deadline_update = pending_deadline
            slot.last_control_sequence = pending_deadline.control_sequence
        self._slots[operation_id] = slot
        return slot

    async def apply(self, request: RuntimeControlRequest, *, terminal: bool) -> RuntimeControlDisposition:
        slot = self._slots.get(request.operation_id)
        if slot is None:
            return RuntimeControlDisposition.UNKNOWN_OPERATION
        return await slot.apply(request, terminal=terminal)

    async def apply_deadline(
        self,
        update: RuntimeDeadlineUpdate,
        *,
        terminal: bool,
    ) -> RuntimeControlDisposition:
        slot = self._slots.get(update.operation_id)
        if slot is not None:
            return await slot.apply_deadline(update, terminal=terminal)
        pending = self._pending_deadlines.get(update.operation_id)
        if pending is not None and update.control_sequence <= pending.control_sequence:
            return RuntimeControlDisposition.STALE
        self._pending_deadlines[update.operation_id] = update
        return RuntimeControlDisposition.APPLIED

    def apply_priority(
        self,
        update: RuntimePriorityUpdate,
        *,
        terminal: bool,
        backend_supported: bool = True,
    ) -> RuntimeControlDisposition:
        slot = self._slots.get(update.operation_id)
        if slot is None:
            return RuntimeControlDisposition.UNKNOWN_OPERATION
        return slot.apply_priority(
            update,
            terminal=terminal,
            backend_supported=backend_supported,
        )

    async def apply_supersede(
        self,
        request: RuntimeControlRequest,
        *,
        old_terminal: bool,
        replacement_active: bool,
    ) -> RuntimeControlDisposition:
        slot = self._slots.get(request.operation_id)
        if slot is None:
            return RuntimeControlDisposition.UNKNOWN_OPERATION
        if request.control_sequence <= slot.last_control_sequence:
            return RuntimeControlDisposition.STALE
        if old_terminal or slot.terminal_request is not None:
            return RuntimeControlDisposition.TERMINAL_OPERATION
        prior = self._pending_supersedes_by_old.get(request.operation_id)
        if prior is not None and request.control_sequence <= prior.control_sequence:
            return RuntimeControlDisposition.STALE
        if replacement_active:
            if prior is not None:
                self._pending_supersedes.pop(prior.replacement_operation_id, None)
                self._pending_supersedes_by_old.pop(request.operation_id, None)
            return await slot.apply(request, terminal=False)
        if prior is not None:
            self._pending_supersedes.pop(prior.replacement_operation_id, None)
        slot.last_control_sequence = request.control_sequence
        self._pending_supersedes[request.replacement_operation_id] = request
        self._pending_supersedes_by_old[request.operation_id] = request
        return RuntimeControlDisposition.APPLIED

    def pending_supersede(self, replacement_operation_id: int) -> RuntimeControlRequest | None:
        return self._pending_supersedes.get(replacement_operation_id)

    async def activate_replacement(
        self,
        replacement_operation_id: int,
        *,
        old_terminal: bool,
    ) -> RuntimeControlDisposition | None:
        request = self._pending_supersedes.pop(replacement_operation_id, None)
        if request is None:
            return None
        if self._pending_supersedes_by_old.get(request.operation_id) is request:
            self._pending_supersedes_by_old.pop(request.operation_id, None)
        slot = self._slots.get(request.operation_id)
        if slot is None:
            return RuntimeControlDisposition.UNKNOWN_OPERATION
        return await slot.activate_reserved(request, terminal=old_terminal)

    async def terminate_all(
        self,
        kind: RuntimeControlKind,
        *,
        source_role: RuntimeRole,
        diagnostic: bytes,
    ) -> None:
        for slot in self._slots.values():
            task = slot.task
            if task is None or task.done() or slot.terminal_request is not None:
                continue
            await slot.apply(
                RuntimeControlRequest(
                    kind=kind,
                    operation_id=slot.operation_id,
                    control_sequence=slot.last_control_sequence + 1,
                    reason_code=0,
                    source_role=source_role,
                    flags=0,
                    diagnostic=diagnostic,
                ),
                terminal=False,
            )

    async def invalidate_dependencies(
        self,
        operation_ids: tuple[int, ...],
        *,
        object_id: int,
    ) -> tuple[int, ...]:
        invalidated: list[int] = []
        diagnostic = f"runtime_object_{object_id}_invalidated".encode("ascii")
        for operation_id in operation_ids:
            slot = self._slots.get(operation_id)
            task = None if slot is None else slot.task
            if (
                slot is None
                or task is None
                or task.done()
                or slot.terminal_request is not None
            ):
                continue
            disposition = await slot.apply(
                RuntimeControlRequest(
                    kind=RuntimeControlKind.OBJECT_INVALIDATED,
                    operation_id=operation_id,
                    control_sequence=slot.last_control_sequence + 1,
                    reason_code=5,
                    source_role=RuntimeRole.RUNTIME,
                    flags=0,
                    diagnostic=diagnostic,
                ),
                terminal=False,
            )
            if disposition is RuntimeControlDisposition.APPLIED:
                invalidated.append(operation_id)
        return tuple(invalidated)

    async def clear(self) -> None:
        await asyncio.gather(*(slot.complete() for slot in self._slots.values()))
        self._slots.clear()
        self._pending_deadlines.clear()
        self._pending_supersedes.clear()
        self._pending_supersedes_by_old.clear()


def decode_operation_control(event: NativeRuntimeEvent) -> RuntimeControlRequest | None:
    message_type = event.header.message_type
    if message_type is MessageType.SUPERSEDE:
        metadata = event.metadata.value
        if not isinstance(metadata, SupersedeMetadata):
            raise TypeError("SUPERSEDE requires SupersedeMetadata")
        if metadata.old_operation_id <= 0 or metadata.new_operation_id <= 0:
            raise ValueError("SUPERSEDE requires non-zero old_operation_id and new_operation_id")
        if metadata.old_operation_id == metadata.new_operation_id:
            raise ValueError("SUPERSEDE requires distinct old_operation_id and new_operation_id")
        return RuntimeControlRequest(
            kind=RuntimeControlKind.SUPERSEDE,
            operation_id=metadata.old_operation_id,
            control_sequence=metadata.control_sequence,
            reason_code=metadata.drop_reason_code,
            source_role=RuntimeRole.CLIENT,
            flags=metadata.flags,
            diagnostic=_event_diagnostic(event),
            replacement_operation_id=metadata.new_operation_id,
        )
    if message_type not in {MessageType.CANCEL, MessageType.ABORT}:
        return None
    metadata = event.metadata.value
    if not isinstance(metadata, ControlRequestMetadata):
        raise TypeError(f"{message_type.name} requires ControlRequestMetadata")
    if metadata.operation_id <= 0:
        raise ValueError(f"{message_type.name} requires a non-zero operation_id")
    kind = RuntimeControlKind.CANCEL if message_type is MessageType.CANCEL else RuntimeControlKind.ABORT
    return RuntimeControlRequest(
        kind=kind,
        operation_id=metadata.operation_id,
        control_sequence=metadata.control_sequence,
        reason_code=metadata.reason_code,
        source_role=metadata.source_role,
        flags=metadata.flags,
        diagnostic=_event_diagnostic(event),
    )


def decode_deadline_update(event: NativeRuntimeEvent) -> RuntimeDeadlineUpdate | None:
    message_type = event.header.message_type
    if message_type not in {MessageType.DEADLINE, MessageType.EXPIRE_AT}:
        return None
    metadata = event.metadata.value
    if not isinstance(metadata, SchedulingMetadata):
        raise TypeError(f"{message_type.name} requires SchedulingMetadata")
    if metadata.operation_id <= 0:
        raise ValueError(f"{message_type.name} requires a non-zero operation_id")
    if metadata.deadline_unix_ms <= 0:
        raise ValueError(f"{message_type.name} requires a non-zero deadline_unix_ms")
    return RuntimeDeadlineUpdate(
        operation_id=metadata.operation_id,
        control_sequence=metadata.control_sequence,
        deadline_unix_ms=metadata.deadline_unix_ms,
        flags=metadata.flags,
    )


def decode_priority_update(event: NativeRuntimeEvent) -> RuntimePriorityUpdate | None:
    if event.header.message_type is not MessageType.PRIORITY_UPDATE:
        return None
    metadata = event.metadata.value
    if not isinstance(metadata, SchedulingMetadata):
        raise TypeError("PRIORITY_UPDATE requires SchedulingMetadata")
    if metadata.operation_id <= 0:
        raise ValueError("PRIORITY_UPDATE requires a non-zero operation_id")
    return RuntimePriorityUpdate(
        operation_id=metadata.operation_id,
        control_sequence=metadata.control_sequence,
        priority_class=metadata.priority_class,
        priority_delta=metadata.priority_delta,
        flags=metadata.flags,
    )


def _event_diagnostic(event: NativeRuntimeEvent) -> bytes:
    if event.tail.kind is RuntimeEventTailKind.BODY:
        return cast(bytes, event.tail.body)
    if event.tail.kind is RuntimeEventTailKind.DIAGNOSTIC:
        return cast(bytes, event.tail.diagnostic)
    return b""
