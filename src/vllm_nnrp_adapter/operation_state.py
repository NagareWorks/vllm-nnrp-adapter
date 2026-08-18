from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OperationState(StrEnum):
    ACCEPTED = "accepted"
    QUEUED = "queued"
    ADMITTED = "admitted"
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DROPPED = "dropped"
    FAILED = "failed"


TERMINAL_OPERATION_STATES = frozenset(
    {
        OperationState.COMPLETED,
        OperationState.CANCELLED,
        OperationState.DROPPED,
        OperationState.FAILED,
    }
)

_TRANSITIONS = {
    OperationState.ACCEPTED: frozenset(
        {OperationState.QUEUED, OperationState.CANCELLED, OperationState.DROPPED, OperationState.FAILED}
    ),
    OperationState.QUEUED: frozenset(
        {OperationState.ADMITTED, OperationState.CANCELLED, OperationState.DROPPED, OperationState.FAILED}
    ),
    OperationState.ADMITTED: frozenset(TERMINAL_OPERATION_STATES | {OperationState.STREAMING}),
    OperationState.STREAMING: TERMINAL_OPERATION_STATES,
}


class OperationStateError(RuntimeError):
    pass


@dataclass(slots=True)
class OperationRecord:
    operation_id: int
    backend_request_id: str
    state: OperationState = OperationState.ACCEPTED
    resources_released: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_OPERATION_STATES

    def transition(self, next_state: OperationState) -> None:
        allowed = _TRANSITIONS.get(self.state, frozenset())
        if next_state not in allowed:
            raise OperationStateError(
                f"illegal operation state transition for {self.operation_id}: {self.state.value} -> {next_state.value}"
            )
        self.state = next_state

    def mark_partial(self) -> None:
        if self.state is OperationState.ADMITTED:
            self.transition(OperationState.STREAMING)
            return
        if self.state is not OperationState.STREAMING:
            raise OperationStateError(
                f"operation {self.operation_id} cannot emit a partial result in state {self.state.value}"
            )

    def terminate(self, terminal_state: OperationState) -> None:
        if terminal_state not in TERMINAL_OPERATION_STATES:
            raise OperationStateError(f"operation terminal state required, got {terminal_state.value}")
        self.transition(terminal_state)

    def release_resources(self) -> None:
        if not self.is_terminal:
            raise OperationStateError(
                f"operation {self.operation_id} resources cannot be released before terminal state"
            )
        self.resources_released = True


class OperationRegistry:
    def __init__(self) -> None:
        self._records: dict[int, OperationRecord] = {}

    def register(self, operation_id: int, backend_request_id: str) -> OperationRecord:
        if operation_id <= 0:
            raise OperationStateError("operation_id must be positive")
        if operation_id in self._records:
            raise OperationStateError(f"duplicate operation_id {operation_id}")
        record = OperationRecord(operation_id=operation_id, backend_request_id=backend_request_id)
        self._records[operation_id] = record
        return record

    def get(self, operation_id: int) -> OperationRecord:
        try:
            return self._records[operation_id]
        except KeyError as error:
            raise OperationStateError(f"unknown operation_id {operation_id}") from error

    def clear(self) -> None:
        self._records.clear()
