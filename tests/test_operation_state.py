from __future__ import annotations

import pytest

from vllm_nnrp_adapter.operation_state import (
    OperationRegistry,
    OperationState,
    OperationStateError,
)


def test_operation_registry_enforces_lifecycle_and_releases_terminal_resources() -> None:
    registry = OperationRegistry()
    record = registry.register(7, "request-7")

    record.transition(OperationState.QUEUED)
    record.transition(OperationState.ADMITTED)
    record.mark_partial()
    record.mark_partial()
    record.terminate(OperationState.COMPLETED)
    record.release_resources()

    assert record.state is OperationState.COMPLETED
    assert record.resources_released is True
    assert registry.get(7) is record


def test_operation_registry_rejects_duplicate_without_replacing_original_record() -> None:
    registry = OperationRegistry()
    original = registry.register(11, "request-original")

    with pytest.raises(OperationStateError, match="duplicate operation_id 11"):
        registry.register(11, "request-duplicate")

    assert registry.get(11) is original
    assert original.backend_request_id == "request-original"


def test_operation_record_rejects_partial_and_second_terminal_after_completion() -> None:
    record = OperationRegistry().register(13, "request-13")
    record.transition(OperationState.QUEUED)
    record.transition(OperationState.ADMITTED)
    record.terminate(OperationState.CANCELLED)

    with pytest.raises(OperationStateError, match="cannot emit a partial result"):
        record.mark_partial()
    with pytest.raises(OperationStateError, match="cancelled -> failed"):
        record.terminate(OperationState.FAILED)


def test_operation_registry_rejects_invalid_and_unknown_ids() -> None:
    registry = OperationRegistry()

    with pytest.raises(OperationStateError, match="must be positive"):
        registry.register(0, "request-zero")
    with pytest.raises(OperationStateError, match="unknown operation_id 99"):
        registry.get(99)


def test_operation_registry_clear_releases_session_tombstones() -> None:
    registry = OperationRegistry()
    registry.register(17, "request-17")

    registry.clear()

    with pytest.raises(OperationStateError, match="unknown operation_id 17"):
        registry.get(17)
