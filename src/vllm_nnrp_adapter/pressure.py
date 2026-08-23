from __future__ import annotations

import asyncio
from dataclasses import dataclass

from nnrp.core import MessageType  # type: ignore[import-untyped]
from nnrp.runtime import PressureMetadata  # type: ignore[import-untyped]

_PRESSURE_SCOPE_CONNECTION = 0x0000_0001
_PRESSURE_SCOPE_OPERATION = 0x0000_0002
_BACKPRESSURE_LEVEL_PAUSED = 0x0003


@dataclass(slots=True)
class _CreditScope:
    remaining: int
    generation: int
    not_before: float = 0.0


@dataclass(frozen=True, slots=True)
class _CreditReservation:
    scopes: tuple[tuple[str, int, int], ...]


class OutboundCreditController:
    """Applies peer-advertised frame credits before backend output is pulled."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._connection: _CreditScope | None = None
        self._session: _CreditScope | None = None
        self._operations: dict[int, _CreditScope] = {}
        self._next_generation = 1
        self._closed = False

    async def apply(self, message_type: MessageType, metadata: PressureMetadata) -> None:
        scope_kind, scope_id = _pressure_scope(metadata)
        if message_type is MessageType.CREDIT_UPDATE:
            remaining = metadata.credit_window
            not_before = 0.0
        elif message_type is MessageType.BACKPRESSURE:
            if metadata.pressure_level == 0:
                raise ValueError("BACKPRESSURE requires a non-zero pressure level")
            remaining = (
                0
                if metadata.pressure_level == _BACKPRESSURE_LEVEL_PAUSED
                else metadata.credit_window
            )
            not_before = (
                asyncio.get_running_loop().time() + metadata.retry_after_ms / 1000
                if metadata.retry_after_ms
                else 0.0
            )
        else:
            raise ValueError("pressure control requires BACKPRESSURE or CREDIT_UPDATE")

        async with self._condition:
            scope = _CreditScope(
                remaining=remaining,
                generation=self._next_generation,
                not_before=not_before,
            )
            self._next_generation += 1
            if scope_kind == "connection":
                self._connection = scope
            elif scope_kind == "session":
                self._session = scope
            else:
                self._operations[scope_id] = scope
            self._condition.notify_all()

    async def reserve(self, operation_id: int) -> _CreditReservation:
        async with self._condition:
            while True:
                if self._closed:
                    raise asyncio.CancelledError
                scopes = self._applicable_scopes(operation_id)
                now = asyncio.get_running_loop().time()
                blocked_for_credit = any(scope.remaining == 0 for _kind, _scope_id, scope in scopes)
                retry_at = max((scope.not_before for _kind, _scope_id, scope in scopes), default=0.0)
                if not blocked_for_credit and retry_at <= now:
                    reservations: list[tuple[str, int, int]] = []
                    for kind, scope_id, scope in scopes:
                        scope.remaining -= 1
                        reservations.append((kind, scope_id, scope.generation))
                    return _CreditReservation(tuple(reservations))
                if blocked_for_credit or retry_at == 0.0:
                    await self._condition.wait()
                    continue
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=retry_at - now)
                except TimeoutError:
                    pass

    async def refund(self, reservation: _CreditReservation) -> None:
        async with self._condition:
            for kind, scope_id, generation in reservation.scopes:
                scope = self._lookup_scope(kind, scope_id)
                if scope is not None and scope.generation == generation:
                    scope.remaining += 1
            self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    async def retire(self, operation_id: int) -> None:
        async with self._condition:
            self._operations.pop(operation_id, None)
            self._condition.notify_all()

    def _applicable_scopes(self, operation_id: int) -> tuple[tuple[str, int, _CreditScope], ...]:
        scopes: list[tuple[str, int, _CreditScope]] = []
        if self._connection is not None:
            scopes.append(("connection", 0, self._connection))
        if self._session is not None:
            scopes.append(("session", 0, self._session))
        operation = self._operations.get(operation_id)
        if operation is not None:
            scopes.append(("operation", operation_id, operation))
        return tuple(scopes)

    def _lookup_scope(self, kind: str, scope_id: int) -> _CreditScope | None:
        if kind == "connection":
            return self._connection
        if kind == "session":
            return self._session
        return self._operations.get(scope_id)


def _pressure_scope(metadata: PressureMetadata) -> tuple[str, int]:
    scope_flags = metadata.flags & (_PRESSURE_SCOPE_CONNECTION | _PRESSURE_SCOPE_OPERATION)
    if scope_flags == (_PRESSURE_SCOPE_CONNECTION | _PRESSURE_SCOPE_OPERATION):
        raise ValueError("pressure control cannot target connection and operation simultaneously")
    if scope_flags == _PRESSURE_SCOPE_CONNECTION:
        return "connection", 0
    if scope_flags == _PRESSURE_SCOPE_OPERATION:
        if metadata.scope_id == 0:
            raise ValueError("operation pressure control requires a non-zero scope_id")
        return "operation", metadata.scope_id
    return "session", metadata.scope_id
