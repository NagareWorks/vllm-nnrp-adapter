from __future__ import annotations

import asyncio
import time

import pytest
from nnrp.core import MessageType
from nnrp.runtime import (
    ControlRequestMetadata,
    NativeRuntimeEvent,
    RuntimeEventMetadata,
    RuntimeEventMetadataKind,
    RuntimeEventTail,
    RuntimeFrameHeader,
    RuntimeRole,
    SchedulingMetadata,
)

from vllm_nnrp_adapter.runtime_control import (
    OperationControlSlot,
    RuntimeControlDisposition,
    RuntimeControlKind,
    RuntimeControlRegistry,
    RuntimeControlRequest,
    RuntimeDeadlineUpdate,
    decode_deadline_update,
    decode_operation_control,
)


def test_decode_operation_control_preserves_frozen_metadata_and_tail() -> None:
    request = decode_operation_control(_control_event(MessageType.ABORT, operation_id=7, sequence=9))

    assert request == RuntimeControlRequest(
        kind=RuntimeControlKind.ABORT,
        operation_id=7,
        control_sequence=9,
        reason_code=3,
        source_role=RuntimeRole.CLIENT,
        flags=0,
        diagnostic=b"obsolete",
    )
    assert decode_operation_control(_runtime_event(MessageType.PROGRESS)) is None


def test_decode_operation_control_rejects_wrong_metadata_and_session_scope() -> None:
    wrong_metadata = NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.CANCEL),
        RuntimeEventMetadata(RuntimeEventMetadataKind.NONE),
        RuntimeEventTail.none(),
    )
    with pytest.raises(TypeError, match="CANCEL requires ControlRequestMetadata"):
        decode_operation_control(wrong_metadata)

    with pytest.raises(ValueError, match="non-zero operation_id"):
        decode_operation_control(_control_event(MessageType.CANCEL, operation_id=0, sequence=1))


def test_decode_deadline_update_preserves_frozen_metadata() -> None:
    update = decode_deadline_update(
        _deadline_event(MessageType.EXPIRE_AT, operation_id=8, sequence=10, deadline_unix_ms=123_456)
    )

    assert update == RuntimeDeadlineUpdate(
        operation_id=8,
        control_sequence=10,
        deadline_unix_ms=123_456,
        flags=1,
    )
    assert decode_deadline_update(_runtime_event(MessageType.PROGRESS)) is None


def test_decode_deadline_update_rejects_wrong_metadata_and_zero_values() -> None:
    with pytest.raises(TypeError, match="DEADLINE requires SchedulingMetadata"):
        decode_deadline_update(_runtime_event(MessageType.DEADLINE))
    with pytest.raises(ValueError, match="non-zero operation_id"):
        decode_deadline_update(_deadline_event(MessageType.DEADLINE, operation_id=0, sequence=1))
    with pytest.raises(ValueError, match="non-zero deadline_unix_ms"):
        decode_deadline_update(_deadline_event(MessageType.DEADLINE, operation_id=1, sequence=1, deadline_unix_ms=0))


@pytest.mark.asyncio
async def test_control_slot_cancels_once_and_rejects_stale_or_terminal_updates() -> None:
    started = asyncio.Event()

    async def worker() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(worker())
    await started.wait()
    slot = OperationControlSlot(11)
    slot.bind(task)
    request = _request(operation_id=11, sequence=2)

    assert await slot.apply(request, terminal=False) is RuntimeControlDisposition.APPLIED
    assert await slot.apply(_request(operation_id=11, sequence=1), terminal=False) is RuntimeControlDisposition.STALE
    terminal_disposition = await slot.apply(_request(operation_id=11, sequence=3), terminal=True)
    assert terminal_disposition is RuntimeControlDisposition.TERMINAL_OPERATION
    with pytest.raises(asyncio.CancelledError):
        await task
    duplicate_task = asyncio.create_task(asyncio.sleep(0))
    try:
        with pytest.raises(RuntimeError, match="already has a bound task"):
            slot.bind(duplicate_task)
    finally:
        duplicate_task.cancel()
        await asyncio.gather(duplicate_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_control_registry_rejects_unknown_duplicate_and_unbound_operations() -> None:
    registry = RuntimeControlRegistry()
    slot = registry.register(21)
    with pytest.raises(RuntimeError, match="duplicate runtime-control operation 21"):
        registry.register(21)
    assert (
        await registry.apply(_request(operation_id=99, sequence=1), terminal=False)
        is RuntimeControlDisposition.UNKNOWN_OPERATION
    )
    with pytest.raises(RuntimeError, match="does not have a bound task"):
        await slot.apply(_request(operation_id=21, sequence=1), terminal=False)
    await registry.clear()


@pytest.mark.asyncio
async def test_control_registry_terminates_every_bound_operation() -> None:
    registry = RuntimeControlRegistry()

    async def worker() -> None:
        await asyncio.Event().wait()

    tasks = [asyncio.create_task(worker()) for _ in range(2)]
    for operation_id, task in enumerate(tasks, start=1):
        registry.register(operation_id).bind(task)

    await registry.terminate_all(
        RuntimeControlKind.SERVER_SHUTDOWN,
        source_role=RuntimeRole.SERVER,
        diagnostic=b"server_shutdown",
    )

    await asyncio.gather(*tasks, return_exceptions=True)
    assert all(task.cancelled() for task in tasks)


@pytest.mark.asyncio
async def test_deadline_arriving_before_submit_is_bound_and_expires_operation() -> None:
    registry = RuntimeControlRegistry()
    update = RuntimeDeadlineUpdate(
        operation_id=31,
        control_sequence=4,
        deadline_unix_ms=int(time.time() * 1000) - 1,
        flags=1,
    )

    assert await registry.apply_deadline(update, terminal=False) is RuntimeControlDisposition.APPLIED
    slot = registry.register(31)

    async def worker() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(worker())
    slot.bind(task)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert slot.terminal_request is not None
    assert slot.terminal_request.kind is RuntimeControlKind.DEADLINE_EXPIRED
    await registry.clear()


@pytest.mark.asyncio
async def test_deadline_update_reschedules_and_rejects_stale_sequence() -> None:
    slot = OperationControlSlot(32)

    async def worker() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(worker())
    slot.bind(task)
    later = RuntimeDeadlineUpdate(32, 5, int(time.time() * 1000) + 60_000, 0)
    sooner = RuntimeDeadlineUpdate(32, 6, int(time.time() * 1000) - 1, 0)

    assert await slot.apply_deadline(later, terminal=False) is RuntimeControlDisposition.APPLIED
    assert await slot.apply_deadline(later, terminal=False) is RuntimeControlDisposition.STALE
    assert await slot.apply_deadline(sooner, terminal=False) is RuntimeControlDisposition.APPLIED
    with pytest.raises(asyncio.CancelledError):
        await task
    assert slot.terminal_request is not None
    assert slot.terminal_request.control_sequence == 6
    await slot.complete()


@pytest.mark.asyncio
async def test_completed_operation_cancels_pending_deadline_timer() -> None:
    slot = OperationControlSlot(33)
    task = asyncio.create_task(asyncio.sleep(0))
    slot.bind(task)
    await slot.apply_deadline(
        RuntimeDeadlineUpdate(33, 1, int(time.time() * 1000) + 60_000, 0),
        terminal=False,
    )
    await task
    await slot.complete()
    assert slot.terminal_request is None


def _request(*, operation_id: int, sequence: int) -> RuntimeControlRequest:
    return RuntimeControlRequest(
        kind=RuntimeControlKind.CANCEL,
        operation_id=operation_id,
        control_sequence=sequence,
        reason_code=3,
        source_role=RuntimeRole.CLIENT,
        flags=0,
        diagnostic=b"obsolete",
    )


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
        RuntimeFrameHeader(message_type=message_type),
        RuntimeEventMetadata(RuntimeEventMetadataKind.CONTROL_REQUEST, metadata),
        RuntimeEventTail.with_body(b"obsolete"),
    )


def _deadline_event(
    message_type: MessageType,
    *,
    operation_id: int,
    sequence: int,
    deadline_unix_ms: int = 123_456,
) -> NativeRuntimeEvent:
    metadata = SchedulingMetadata(
        operation_id=operation_id,
        control_sequence=sequence,
        priority_class=0,
        priority_delta=0,
        deadline_unix_ms=deadline_unix_ms,
        flags=1,
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=message_type),
        RuntimeEventMetadata(RuntimeEventMetadataKind.SCHEDULING, metadata),
        RuntimeEventTail.none(),
    )


def _runtime_event(message_type: MessageType) -> NativeRuntimeEvent:
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=message_type),
        RuntimeEventMetadata(RuntimeEventMetadataKind.NONE),
        RuntimeEventTail.with_diagnostic(b"ignored"),
    )
