from __future__ import annotations

import asyncio

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
)

from vllm_nnrp_adapter.runtime_control import (
    OperationControlSlot,
    RuntimeControlDisposition,
    RuntimeControlKind,
    RuntimeControlRegistry,
    RuntimeControlRequest,
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
    registry.clear()


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


def _runtime_event(message_type: MessageType) -> NativeRuntimeEvent:
    return NativeRuntimeEvent(
        RuntimeFrameHeader(message_type=message_type),
        RuntimeEventMetadata(RuntimeEventMetadataKind.NONE),
        RuntimeEventTail.with_diagnostic(b"ignored"),
    )
