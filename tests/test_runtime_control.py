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
    SupersedeMetadata,
)

from vllm_nnrp_adapter.runtime_control import (
    OperationControlSlot,
    RuntimeControlDisposition,
    RuntimeControlKind,
    RuntimeControlRegistry,
    RuntimeControlRequest,
    RuntimeDeadlineUpdate,
    RuntimePriorityUpdate,
    decode_deadline_update,
    decode_operation_control,
    decode_priority_update,
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


def test_decode_supersede_preserves_replacement_and_drop_metadata() -> None:
    request = decode_operation_control(_supersede_event(old_operation_id=7, new_operation_id=8, sequence=11))

    assert request == RuntimeControlRequest(
        kind=RuntimeControlKind.SUPERSEDE,
        operation_id=7,
        control_sequence=11,
        reason_code=2,
        source_role=RuntimeRole.CLIENT,
        flags=1,
        diagnostic=b"newer_request",
        replacement_operation_id=8,
    )


def test_decode_supersede_rejects_wrong_zero_and_identical_operations() -> None:
    wrong_metadata = NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.SUPERSEDE),
        RuntimeEventMetadata(RuntimeEventMetadataKind.NONE),
        RuntimeEventTail.none(),
    )
    with pytest.raises(TypeError, match="SUPERSEDE requires SupersedeMetadata"):
        decode_operation_control(wrong_metadata)
    with pytest.raises(ValueError, match="non-zero"):
        decode_operation_control(_supersede_event(old_operation_id=0, new_operation_id=8, sequence=1))
    with pytest.raises(ValueError, match="distinct"):
        decode_operation_control(_supersede_event(old_operation_id=8, new_operation_id=8, sequence=1))


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


def test_decode_priority_update_preserves_frozen_metadata() -> None:
    update = decode_priority_update(
        _priority_event(operation_id=9, sequence=12, priority_class=2, priority_delta=-5)
    )

    assert update == RuntimePriorityUpdate(
        operation_id=9,
        control_sequence=12,
        priority_class=2,
        priority_delta=-5,
        flags=3,
    )
    assert update.backend_priority == -3
    assert decode_priority_update(_runtime_event(MessageType.PROGRESS)) is None


def test_decode_priority_update_rejects_wrong_metadata_and_session_scope() -> None:
    with pytest.raises(TypeError, match="PRIORITY_UPDATE requires SchedulingMetadata"):
        decode_priority_update(_runtime_event(MessageType.PRIORITY_UPDATE))
    with pytest.raises(ValueError, match="non-zero operation_id"):
        decode_priority_update(
            _priority_event(operation_id=0, sequence=1, priority_class=1, priority_delta=0)
        )


def test_control_slot_applies_priority_before_dispatch_and_rejects_live_update() -> None:
    slot = OperationControlSlot(10)
    initial = RuntimePriorityUpdate(10, 1, 2, -4, 0)

    assert slot.apply_priority(initial, terminal=False) is RuntimeControlDisposition.APPLIED
    assert slot.begin_backend_dispatch() == -2

    live = RuntimePriorityUpdate(10, 2, 0, -8, 0)
    assert (
        slot.apply_priority(live, terminal=False)
        is RuntimeControlDisposition.LIVE_UPDATE_REQUIRED
    )
    assert slot.apply_priority(live, terminal=False) is RuntimeControlDisposition.STALE


def test_control_slot_consumes_unsupported_backend_priority_sequence() -> None:
    slot = OperationControlSlot(11)
    update = RuntimePriorityUpdate(11, 3, 1, 0, 0)

    assert (
        slot.apply_priority(update, terminal=False, backend_supported=False)
        is RuntimeControlDisposition.UNSUPPORTED_LIVE_UPDATE
    )
    assert (
        slot.apply_priority(update, terminal=False, backend_supported=False)
        is RuntimeControlDisposition.STALE
    )
    assert slot.priority_update is None


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
async def test_control_registry_invalidates_only_live_object_dependents() -> None:
    registry = RuntimeControlRegistry()

    async def worker() -> None:
        await asyncio.Event().wait()

    live_task = asyncio.create_task(worker())
    completed_task = asyncio.create_task(asyncio.sleep(0))
    live_slot = registry.register(11)
    live_slot.bind(live_task)
    registry.register(12).bind(completed_task)
    await completed_task

    invalidated = await registry.invalidate_dependencies((11, 12, 99), object_id=33)

    await asyncio.gather(live_task, return_exceptions=True)
    assert invalidated == (11,)
    assert live_task.cancelled()
    assert live_slot.terminal_request == RuntimeControlRequest(
        kind=RuntimeControlKind.OBJECT_INVALIDATED,
        operation_id=11,
        control_sequence=1,
        reason_code=5,
        source_role=RuntimeRole.RUNTIME,
        flags=0,
        diagnostic=b"runtime_object_33_invalidated",
    )
    await registry.clear()


@pytest.mark.asyncio
async def test_supersede_waits_for_replacement_admission_before_cancelling_old_operation() -> None:
    registry = RuntimeControlRegistry()

    async def worker() -> None:
        await asyncio.Event().wait()

    old_task = asyncio.create_task(worker())
    registry.register(41).bind(old_task)
    request = RuntimeControlRequest(
        kind=RuntimeControlKind.SUPERSEDE,
        operation_id=41,
        control_sequence=3,
        reason_code=2,
        source_role=RuntimeRole.CLIENT,
        flags=1,
        diagnostic=b"newer_request",
        replacement_operation_id=42,
    )

    assert (
        await registry.apply_supersede(request, old_terminal=False, replacement_active=False)
        is RuntimeControlDisposition.APPLIED
    )
    assert not old_task.done()
    assert registry.pending_supersede(42) is request
    assert (
        await registry.apply(_request(operation_id=41, sequence=2), terminal=False) is RuntimeControlDisposition.STALE
    )
    assert await registry.activate_replacement(42, old_terminal=False) is RuntimeControlDisposition.APPLIED
    with pytest.raises(asyncio.CancelledError):
        await old_task
    await registry.clear()


@pytest.mark.asyncio
async def test_newer_pending_supersede_replaces_old_target_and_invalidates_prior_activation() -> None:
    registry = RuntimeControlRegistry()

    async def worker() -> None:
        await asyncio.Event().wait()

    old_task = asyncio.create_task(worker())
    registry.register(51).bind(old_task)
    first = RuntimeControlRequest(
        RuntimeControlKind.SUPERSEDE,
        51,
        4,
        2,
        RuntimeRole.CLIENT,
        1,
        b"first",
        52,
    )
    second = RuntimeControlRequest(
        RuntimeControlKind.SUPERSEDE,
        51,
        5,
        2,
        RuntimeRole.CLIENT,
        1,
        b"second",
        53,
    )

    assert (
        await registry.apply_supersede(first, old_terminal=False, replacement_active=False)
        is RuntimeControlDisposition.APPLIED
    )
    assert (
        await registry.apply_supersede(second, old_terminal=False, replacement_active=False)
        is RuntimeControlDisposition.APPLIED
    )
    assert await registry.activate_replacement(52, old_terminal=False) is None
    assert not old_task.done()
    assert await registry.activate_replacement(53, old_terminal=False) is RuntimeControlDisposition.APPLIED
    await asyncio.gather(old_task, return_exceptions=True)
    assert old_task.cancelled()
    await registry.clear()


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
        flags=3,
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
        drop_reason_code=2,
        flags=1,
        diagnostic_bytes=len(b"newer_request"),
    )
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=MessageType.SUPERSEDE),
        RuntimeEventMetadata(RuntimeEventMetadataKind.SUPERSEDE, metadata),
        RuntimeEventTail.with_body(b"newer_request"),
    )


def _runtime_event(message_type: MessageType) -> NativeRuntimeEvent:
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=message_type),
        RuntimeEventMetadata(RuntimeEventMetadataKind.NONE),
        RuntimeEventTail.with_diagnostic(b"ignored"),
    )
