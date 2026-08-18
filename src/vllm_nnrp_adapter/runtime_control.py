from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from nnrp.core import MessageType  # type: ignore[import-untyped]
from nnrp.runtime import (  # type: ignore[import-untyped]
    ControlRequestMetadata,
    NativeRuntimeEvent,
    RuntimeEventTailKind,
    RuntimeRole,
    SchedulingMetadata,
)


class RuntimeControlKind(StrEnum):
    CANCEL = "cancel"
    ABORT = "abort"
    PEER_DISCONNECT = "peer_disconnect"
    SERVER_SHUTDOWN = "server_shutdown"
    DEADLINE_EXPIRED = "deadline_expired"


class RuntimeControlDisposition(StrEnum):
    APPLIED = "applied"
    STALE = "stale"
    UNKNOWN_OPERATION = "unknown_operation"
    TERMINAL_OPERATION = "terminal_operation"


@dataclass(frozen=True, slots=True)
class RuntimeControlRequest:
    kind: RuntimeControlKind
    operation_id: int
    control_sequence: int
    reason_code: int
    source_role: RuntimeRole | int
    flags: int
    diagnostic: bytes


@dataclass(frozen=True, slots=True)
class RuntimeDeadlineUpdate:
    operation_id: int
    control_sequence: int
    deadline_unix_ms: int
    flags: int


@dataclass(slots=True)
class OperationControlSlot:
    operation_id: int
    task: asyncio.Task[None] | None = None
    last_control_sequence: int = 0
    terminal_request: RuntimeControlRequest | None = None
    deadline_update: RuntimeDeadlineUpdate | None = None
    _deadline_task: asyncio.Task[None] | None = None

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

    async def complete(self) -> None:
        await self._cancel_deadline()

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

    async def clear(self) -> None:
        await asyncio.gather(*(slot.complete() for slot in self._slots.values()))
        self._slots.clear()
        self._pending_deadlines.clear()


def decode_operation_control(event: NativeRuntimeEvent) -> RuntimeControlRequest | None:
    message_type = event.header.message_type
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


def _event_diagnostic(event: NativeRuntimeEvent) -> bytes:
    if event.tail.kind is RuntimeEventTailKind.BODY:
        return cast(bytes, event.tail.body)
    if event.tail.kind is RuntimeEventTailKind.DIAGNOSTIC:
        return cast(bytes, event.tail.diagnostic)
    return b""
