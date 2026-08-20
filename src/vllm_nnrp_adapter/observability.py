from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from nnrp import NativeRuntimeServerOperation  # type: ignore[import-untyped]
from nnrp.core import FrameSubmitMetadata  # type: ignore[import-untyped]

from .operation_state import OperationState
from .runtime_control import RuntimeControlRequest

_LOGGER = logging.getLogger("vllm_nnrp_adapter.operation")


@dataclass(frozen=True, slots=True)
class _OperationIdentity:
    selected_transport: str
    session_id: int
    operation_id: int
    frame_id: int
    route_id: int
    view_id: int
    trace_id: int
    profile_id: int


@dataclass(frozen=True, slots=True)
class _OperationObservation:
    identity: _OperationIdentity
    model_id: str | None
    profile_operation: str | None
    backend_family: str
    backend_binding: str | None
    vllm_version: str | None
    queue_delay_ms: float | None
    first_event_latency_ms: float | None
    inter_event_latency_ms: tuple[float, ...]
    terminal_latency_ms: float
    output_event_count: int
    output_bytes: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    error_family: str | None
    cancellation_kind: str | None
    cancellation_source: str | None
    cancellation_reason_code: int | None
    backend_abort_accepted: bool | None
    drop_reason: str | None
    terminal_outcome: str

    def to_log_fields(self) -> dict[str, object]:
        return {
            "selected_transport": self.identity.selected_transport,
            "session_id": self.identity.session_id,
            "operation_id": self.identity.operation_id,
            "frame_id": self.identity.frame_id,
            "route_id": self.identity.route_id,
            "view_id": self.identity.view_id,
            "trace_id": self.identity.trace_id,
            "profile_id": self.identity.profile_id,
            "model_id": self.model_id,
            "profile_operation": self.profile_operation,
            "backend_family": self.backend_family,
            "backend_binding": self.backend_binding,
            "vllm_version": self.vllm_version,
            "queue_delay_ms": self.queue_delay_ms,
            "first_event_latency_ms": self.first_event_latency_ms,
            "inter_event_latency_ms": self.inter_event_latency_ms,
            "terminal_latency_ms": self.terminal_latency_ms,
            "output_event_count": self.output_event_count,
            "output_bytes": self.output_bytes,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error_family": self.error_family,
            "cancellation_kind": self.cancellation_kind,
            "cancellation_source": self.cancellation_source,
            "cancellation_reason_code": self.cancellation_reason_code,
            "backend_abort_accepted": self.backend_abort_accepted,
            "drop_reason": self.drop_reason,
            "terminal_outcome": self.terminal_outcome,
        }


@dataclass(slots=True)
class _OperationObservationTracker:
    identity: _OperationIdentity
    backend_family: str
    backend_binding: str | None
    vllm_version: str | None
    _clock_ns: Callable[[], int] = field(repr=False)
    _accepted_ns: int = field(repr=False)
    _admitted_ns: int | None = field(default=None, repr=False)
    _first_event_ns: int | None = field(default=None, repr=False)
    _last_event_ns: int | None = field(default=None, repr=False)
    _inter_event_ns: list[int] = field(default_factory=list, repr=False)
    _model_id: str | None = field(default=None, repr=False)
    _profile_operation: str | None = field(default=None, repr=False)
    _output_event_count: int = field(default=0, repr=False)
    _output_bytes: int = field(default=0, repr=False)
    _prompt_tokens: int | None = field(default=None, repr=False)
    _completion_tokens: int | None = field(default=None, repr=False)
    _total_tokens: int | None = field(default=None, repr=False)
    _error_family: str | None = field(default=None, repr=False)
    _cancellation_kind: str | None = field(default=None, repr=False)
    _cancellation_source: str | None = field(default=None, repr=False)
    _cancellation_reason_code: int | None = field(default=None, repr=False)
    _backend_abort_accepted: bool | None = field(default=None, repr=False)
    _drop_reason: str | None = field(default=None, repr=False)
    _finished: bool = field(default=False, repr=False)

    @classmethod
    def from_operation(
        cls,
        operation: NativeRuntimeServerOperation,
        *,
        selected_transport: str,
        backend_family: str = "unknown",
        backend_binding: str | None = None,
        vllm_version: str | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> _OperationObservationTracker:
        submit = operation.submit
        metadata = submit.metadata.value
        if not isinstance(metadata, FrameSubmitMetadata):
            raise TypeError("FRAME_SUBMIT observation requires FrameSubmitMetadata")
        header = submit.header
        return cls(
            identity=_OperationIdentity(
                selected_transport=selected_transport,
                session_id=header.session_id,
                operation_id=operation.operation_id,
                frame_id=operation.frame_id,
                route_id=header.route_id,
                view_id=header.view_id,
                trace_id=header.trace_id,
                profile_id=int(metadata.input_profile),
            ),
            backend_family=backend_family,
            backend_binding=backend_binding,
            vllm_version=vllm_version,
            _clock_ns=clock_ns,
            _accepted_ns=clock_ns(),
        )

    def record_request(self, request: Mapping[str, Any]) -> None:
        operation = request.get("operation")
        self._profile_operation = operation if isinstance(operation, str) else None
        body = request.get("body")
        model = body.get("model") if isinstance(body, Mapping) else None
        self._model_id = model if isinstance(model, str) else None

    def mark_admitted(self) -> None:
        if self._admitted_ns is None:
            self._admitted_ns = self._clock_ns()

    def record_event(self, event: Mapping[str, Any], *, body_bytes: int) -> None:
        now = self._clock_ns()
        if self._first_event_ns is None:
            self._first_event_ns = now
        if self._last_event_ns is not None:
            self._inter_event_ns.append(now - self._last_event_ns)
        self._last_event_ns = now
        self._output_event_count += 1
        self._output_bytes += body_bytes
        self._record_usage(event)
        if event.get("type") == "response.error":
            diagnostics = event.get("diagnostics")
            backend_family = diagnostics.get("backend_error_family") if isinstance(diagnostics, Mapping) else None
            error = event.get("error")
            profile_family = error.get("type") if isinstance(error, Mapping) else None
            self._error_family = str(backend_family or profile_family or "unknown")

    def record_exception(self, error: Exception) -> None:
        self._error_family = type(error).__name__

    def record_control(self, request: RuntimeControlRequest, *, drop_reason: object | None = None) -> None:
        self._cancellation_kind = request.kind.value
        self._cancellation_source = _enum_value(request.source_role)
        self._cancellation_reason_code = request.reason_code
        if drop_reason is not None:
            self._drop_reason = _enum_value(drop_reason)

    def record_drop(self, drop_reason: object) -> None:
        self._drop_reason = _enum_value(drop_reason)

    def record_backend_abort(self, accepted: bool | None) -> None:
        self._backend_abort_accepted = accepted

    def finish(self, terminal_state: OperationState) -> _OperationObservation:
        if self._finished:
            raise RuntimeError(f"operation {self.identity.operation_id} observation already finished")
        self._finished = True
        terminal_ns = self._clock_ns()
        return _OperationObservation(
            identity=self.identity,
            model_id=self._model_id,
            profile_operation=self._profile_operation,
            backend_family=self.backend_family,
            backend_binding=self.backend_binding,
            vllm_version=self.vllm_version,
            queue_delay_ms=_duration_ms(self._accepted_ns, self._admitted_ns),
            first_event_latency_ms=_duration_ms(self._accepted_ns, self._first_event_ns),
            inter_event_latency_ms=tuple(_nanoseconds_to_ms(value) for value in self._inter_event_ns),
            terminal_latency_ms=_nanoseconds_to_ms(terminal_ns - self._accepted_ns),
            output_event_count=self._output_event_count,
            output_bytes=self._output_bytes,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._total_tokens,
            error_family=self._error_family,
            cancellation_kind=self._cancellation_kind,
            cancellation_source=self._cancellation_source,
            cancellation_reason_code=self._cancellation_reason_code,
            backend_abort_accepted=self._backend_abort_accepted,
            drop_reason=self._drop_reason,
            terminal_outcome=terminal_state.value,
        )

    def _record_usage(self, event: Mapping[str, Any]) -> None:
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            return
        self._prompt_tokens = _optional_int(usage.get("prompt_tokens"))
        self._completion_tokens = _optional_int(usage.get("completion_tokens"))
        self._total_tokens = _optional_int(usage.get("total_tokens"))


def _emit_operation_observation(observation: _OperationObservation) -> None:
    fields = observation.to_log_fields()
    _LOGGER.info("nnrp_operation_observation %s", json.dumps(fields, sort_keys=True, separators=(",", ":")))


def _optional_int(value: object) -> int | None:
    return value if type(value) is int else None


def _enum_value(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    raw_value = getattr(value, "value", value)
    return str(raw_value)


def _duration_ms(start_ns: int, end_ns: int | None) -> float | None:
    if end_ns is None:
        return None
    return _nanoseconds_to_ms(end_ns - start_ns)


def _nanoseconds_to_ms(value: int) -> float:
    return value / 1_000_000
