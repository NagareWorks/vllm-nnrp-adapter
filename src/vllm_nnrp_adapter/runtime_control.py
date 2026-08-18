from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from nnrp.core import MessageType  # type: ignore[import-untyped]
from nnrp.runtime import (  # type: ignore[import-untyped]
    ControlRequestMetadata,
    NativeRuntimeEvent,
    RuntimeEventTailKind,
    RuntimeRole,
)


class RuntimeControlKind(StrEnum):
    CANCEL = "cancel"
    ABORT = "abort"


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


@dataclass(slots=True)
class OperationControlSlot:
    operation_id: int
    task: asyncio.Task[None] | None = None
    last_control_sequence: int = 0
    terminal_request: RuntimeControlRequest | None = None

    def bind(self, task: asyncio.Task[None]) -> None:
        if self.task is not None:
            raise RuntimeError(f"operation {self.operation_id} already has a bound task")
        self.task = task

    async def apply(self, request: RuntimeControlRequest, *, terminal: bool) -> RuntimeControlDisposition:
        if request.control_sequence <= self.last_control_sequence:
            return RuntimeControlDisposition.STALE
        self.last_control_sequence = request.control_sequence
        if terminal:
            return RuntimeControlDisposition.TERMINAL_OPERATION
        if self.terminal_request is not None:
            return RuntimeControlDisposition.STALE
        self.terminal_request = request
        task = self.task
        if task is None:
            raise RuntimeError(f"operation {self.operation_id} does not have a bound task")
        await asyncio.sleep(0)
        task.cancel()
        return RuntimeControlDisposition.APPLIED


class RuntimeControlRegistry:
    def __init__(self) -> None:
        self._slots: dict[int, OperationControlSlot] = {}

    def register(self, operation_id: int) -> OperationControlSlot:
        if operation_id in self._slots:
            raise RuntimeError(f"duplicate runtime-control operation {operation_id}")
        slot = OperationControlSlot(operation_id)
        self._slots[operation_id] = slot
        return slot

    async def apply(self, request: RuntimeControlRequest, *, terminal: bool) -> RuntimeControlDisposition:
        slot = self._slots.get(request.operation_id)
        if slot is None:
            return RuntimeControlDisposition.UNKNOWN_OPERATION
        return await slot.apply(request, terminal=terminal)

    def clear(self) -> None:
        self._slots.clear()


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


def _event_diagnostic(event: NativeRuntimeEvent) -> bytes:
    if event.tail.kind is RuntimeEventTailKind.BODY:
        return cast(bytes, event.tail.body)
    if event.tail.kind is RuntimeEventTailKind.DIAGNOSTIC:
        return cast(bytes, event.tail.diagnostic)
    return b""
