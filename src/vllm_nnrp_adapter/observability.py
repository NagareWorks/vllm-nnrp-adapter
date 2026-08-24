from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from nnrp import NativeRuntimeServerOperation, NativeTransportEndpoint  # type: ignore[import-untyped]
from nnrp.core import FrameSubmitMetadata  # type: ignore[import-untyped]
from nnrp.runtime import PressureMetadata, RetryAfterMetadata, TraceContextMetadata  # type: ignore[import-untyped]

from .operation_progress import OperationProgressStage
from .operation_state import OperationState
from .runtime_control import RuntimeControlRequest

if TYPE_CHECKING:
    from prometheus_client.registry import CollectorRegistry

_LOGGER = logging.getLogger("vllm_nnrp_adapter.operation")
_SERVER_LOGGER = logging.getLogger("vllm_nnrp_adapter.server")
_SINK_LOGGER = logging.getLogger("vllm_nnrp_adapter.observability")

_KNOWN_TRANSPORTS = frozenset({"ipc", "quic", "tcp", "websocket"})
_KNOWN_TRANSPORT_POLICIES = frozenset(
    {
        "auto",
        "force_ipc",
        "force_quic",
        "force_tcp",
        "force_websocket",
        "prefer_ipc",
        "prefer_quic",
        "prefer_tcp",
        "prefer_websocket",
    }
)
_KNOWN_TERMINAL_OUTCOMES = frozenset({"cancelled", "completed", "dropped", "failed"})
_KNOWN_CANCELLATION_KINDS = frozenset(
    {"abort", "cancel", "deadline_expired", "peer_disconnect", "server_shutdown", "supersede"}
)
_KNOWN_DROP_REASONS = frozenset(
    {
        "backpressure",
        "budget_exceeded",
        "capability_mismatch",
        "conformance_injection",
        "deadline_expired",
        "object_invalidated",
        "peer_cancelled",
        "superseded",
        "transport_closed",
    }
)
_KNOWN_PROFILE_OPERATIONS = {"chat.completions.create": "chat_completions_create"}


@dataclass(frozen=True, slots=True)
class ServerStartupObservation:
    application_endpoint: str
    transport_policy: str
    bound_provider_endpoints: tuple[tuple[str, str], ...]

    @classmethod
    def from_bound_endpoints(
        cls,
        *,
        application_endpoint: str,
        transport_policy: str,
        bound_provider_endpoints: Mapping[str, NativeTransportEndpoint],
    ) -> ServerStartupObservation:
        return cls(
            application_endpoint=application_endpoint,
            transport_policy=transport_policy,
            bound_provider_endpoints=tuple(
                sorted((provider, endpoint.uri) for provider, endpoint in bound_provider_endpoints.items())
            ),
        )

    def to_log_fields(self) -> dict[str, object]:
        return {
            "application_endpoint": self.application_endpoint,
            "transport_policy": self.transport_policy,
            "eligible_providers": [provider for provider, _endpoint in self.bound_provider_endpoints],
            "bound_provider_endpoints": dict(self.bound_provider_endpoints),
        }


@dataclass(frozen=True, slots=True)
class OperationIdentity:
    selected_transport: str
    connection_id: int | None
    connection_generation: int | None
    session_handle_id: int | None
    session_generation: int | None
    session_id: int
    operation_id: int
    frame_id: int
    route_id: int
    view_id: int
    trace_id: int
    profile_id: int


@dataclass(frozen=True, slots=True)
class OperationStageTransition:
    stage_code: int
    stage_name: str
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class OperationObservation:
    identity: OperationIdentity
    model_id: str | None
    profile_operation: str | None
    backend_family: str
    backend_binding: str | None
    vllm_version: str | None
    queue_delay_ms: float | None
    admission_latency_ms: float | None
    preprocessing_latency_ms: float | None
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
    stage_transitions: tuple[OperationStageTransition, ...]
    retry_after_ms: int | None = None
    retry_reason_code: int | None = None
    retry_source: str | None = None
    pressure_scope: str | None = None
    pressure_scope_id: int | None = None
    pressure_credit_window: int | None = None
    pressure_level: int | None = None
    pressure_reason: int | None = None
    pressure_retry_after_ms: int | None = None
    trace_span_id: int | None = None
    trace_parent_span_id: int | None = None
    trace_stage_code: int | None = None
    trace_flags: int | None = None
    trace_attribute_bytes: int = 0

    def to_log_fields(self) -> dict[str, object]:
        return {
            "selected_transport": self.identity.selected_transport,
            "connection_id": self.identity.connection_id,
            "connection_generation": self.identity.connection_generation,
            "session_handle_id": self.identity.session_handle_id,
            "session_generation": self.identity.session_generation,
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
            "admission_latency_ms": self.admission_latency_ms,
            "preprocessing_latency_ms": self.preprocessing_latency_ms,
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
            "retry_after_ms": self.retry_after_ms,
            "retry_reason_code": self.retry_reason_code,
            "retry_source": self.retry_source,
            "pressure_scope": self.pressure_scope,
            "pressure_scope_id": self.pressure_scope_id,
            "pressure_credit_window": self.pressure_credit_window,
            "pressure_level": self.pressure_level,
            "pressure_reason": self.pressure_reason,
            "pressure_retry_after_ms": self.pressure_retry_after_ms,
            "trace_span_id": self.trace_span_id,
            "trace_parent_span_id": self.trace_parent_span_id,
            "trace_stage_code": self.trace_stage_code,
            "trace_flags": self.trace_flags,
            "trace_attribute_bytes": self.trace_attribute_bytes,
            "stage_transitions": [
                {
                    "stage_code": transition.stage_code,
                    "stage_name": transition.stage_name,
                    "elapsed_ms": transition.elapsed_ms,
                }
                for transition in self.stage_transitions
            ],
        }


@dataclass(slots=True)
class _OperationObservationTracker:
    identity: OperationIdentity
    backend_family: str
    backend_binding: str | None
    vllm_version: str | None
    _clock_ns: Callable[[], int] = field(repr=False)
    _accepted_ns: int = field(repr=False)
    _input_received_ns: int | None = field(default=None, repr=False)
    _admitted_ns: int | None = field(default=None, repr=False)
    _preprocessing_ns: int | None = field(default=None, repr=False)
    _executing_ns: int | None = field(default=None, repr=False)
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
    _retry_after_ms: int | None = field(default=None, repr=False)
    _retry_reason_code: int | None = field(default=None, repr=False)
    _retry_source: str | None = field(default=None, repr=False)
    _pressure_scope: str | None = field(default=None, repr=False)
    _pressure_scope_id: int | None = field(default=None, repr=False)
    _pressure_credit_window: int | None = field(default=None, repr=False)
    _pressure_level: int | None = field(default=None, repr=False)
    _pressure_reason: int | None = field(default=None, repr=False)
    _pressure_retry_after_ms: int | None = field(default=None, repr=False)
    _trace_span_id: int | None = field(default=None, repr=False)
    _trace_parent_span_id: int | None = field(default=None, repr=False)
    _trace_stage_code: int | None = field(default=None, repr=False)
    _trace_flags: int | None = field(default=None, repr=False)
    _trace_attribute_bytes: int = field(default=0, repr=False)
    _stage_transitions: list[tuple[int, str, int]] = field(default_factory=list, repr=False)
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
        connection_id: int | None = None,
        connection_generation: int | None = None,
        session_handle_id: int | None = None,
        session_generation: int | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> _OperationObservationTracker:
        submit = operation.submit
        metadata = submit.metadata.value
        if not isinstance(metadata, FrameSubmitMetadata):
            raise TypeError("FRAME_SUBMIT observation requires FrameSubmitMetadata")
        header = submit.header
        return cls(
            identity=OperationIdentity(
                selected_transport=selected_transport,
                connection_id=connection_id,
                connection_generation=connection_generation,
                session_handle_id=session_handle_id,
                session_generation=session_generation,
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

    def record_progress_stage(self, stage: OperationProgressStage) -> None:
        now = self._clock_ns()
        stage_code = stage.value
        stage_name = _enum_value(stage)
        self._stage_transitions.append((stage_code, stage_name, now))
        if stage_name == "input_received" and self._input_received_ns is None:
            self._input_received_ns = now
        elif stage_name == "preprocessing" and self._preprocessing_ns is None:
            self._preprocessing_ns = now
        elif stage_name == "executing" and self._executing_ns is None:
            self._executing_ns = now

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

    def record_retry_hint(self, metadata: RetryAfterMetadata) -> None:
        self._retry_after_ms = metadata.retry_after_ms
        self._retry_reason_code = metadata.reason_code
        self._retry_source = _enum_value(metadata.source_role)

    def record_pressure(
        self,
        metadata: PressureMetadata,
        *,
        scope_kind: str,
        scope_id: int,
    ) -> None:
        self._pressure_scope = scope_kind
        self._pressure_scope_id = scope_id
        self._pressure_credit_window = metadata.credit_window
        self._pressure_level = metadata.pressure_level
        self._pressure_reason = metadata.pressure_reason
        self._pressure_retry_after_ms = metadata.retry_after_ms

    def record_trace_context(self, metadata: TraceContextMetadata, attributes: bytes) -> None:
        if metadata.body_bytes != len(attributes):
            raise ValueError("TRACE_CONTEXT body_bytes does not match the trace attribute body")
        self.identity = replace(self.identity, trace_id=metadata.trace_id)
        self._trace_span_id = metadata.span_id
        self._trace_parent_span_id = metadata.parent_span_id
        self._trace_stage_code = metadata.stage_code
        self._trace_flags = metadata.flags
        self._trace_attribute_bytes = len(attributes)

    def finish(self, terminal_state: OperationState) -> OperationObservation:
        if self._finished:
            raise RuntimeError(f"operation {self.identity.operation_id} observation already finished")
        self._finished = True
        terminal_ns = self._clock_ns()
        return OperationObservation(
            identity=self.identity,
            model_id=self._model_id,
            profile_operation=self._profile_operation,
            backend_family=self.backend_family,
            backend_binding=self.backend_binding,
            vllm_version=self.vllm_version,
            queue_delay_ms=_duration_ms(self._accepted_ns, self._admitted_ns),
            admission_latency_ms=_optional_duration_ms(self._input_received_ns, self._admitted_ns),
            preprocessing_latency_ms=_optional_duration_ms(self._preprocessing_ns, self._executing_ns),
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
            retry_after_ms=self._retry_after_ms,
            retry_reason_code=self._retry_reason_code,
            retry_source=self._retry_source,
            pressure_scope=self._pressure_scope,
            pressure_scope_id=self._pressure_scope_id,
            pressure_credit_window=self._pressure_credit_window,
            pressure_level=self._pressure_level,
            pressure_reason=self._pressure_reason,
            pressure_retry_after_ms=self._pressure_retry_after_ms,
            trace_span_id=self._trace_span_id,
            trace_parent_span_id=self._trace_parent_span_id,
            trace_stage_code=self._trace_stage_code,
            trace_flags=self._trace_flags,
            trace_attribute_bytes=self._trace_attribute_bytes,
            stage_transitions=tuple(
                OperationStageTransition(
                    stage_code=stage_code,
                    stage_name=stage_name,
                    elapsed_ms=_nanoseconds_to_ms(timestamp_ns - self._accepted_ns),
                )
                for stage_code, stage_name, timestamp_ns in self._stage_transitions
            ),
        )

    def _record_usage(self, event: Mapping[str, Any]) -> None:
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            return
        self._prompt_tokens = _optional_int(usage.get("prompt_tokens"))
        self._completion_tokens = _optional_int(usage.get("completion_tokens"))
        self._total_tokens = _optional_int(usage.get("total_tokens"))


@runtime_checkable
class ObservationSink(Protocol):
    def observe_server_startup(self, observation: ServerStartupObservation) -> None: ...

    def observe_operation(self, observation: OperationObservation) -> None: ...


class StructuredLogObservationSink:
    def __init__(
        self,
        *,
        operation_logger: logging.Logger = _LOGGER,
        server_logger: logging.Logger = _SERVER_LOGGER,
    ) -> None:
        self._operation_logger = operation_logger
        self._server_logger = server_logger

    def observe_server_startup(self, observation: ServerStartupObservation) -> None:
        self._server_logger.info(
            "nnrp_server_startup %s",
            json.dumps(observation.to_log_fields(), sort_keys=True, separators=(",", ":")),
        )

    def observe_operation(self, observation: OperationObservation) -> None:
        self._operation_logger.info(
            "nnrp_operation_observation %s",
            json.dumps(observation.to_log_fields(), sort_keys=True, separators=(",", ":")),
        )


class PrometheusObservationSink:
    """Export bounded adapter metrics into a caller-owned Prometheus registry."""

    def __init__(self, registry: CollectorRegistry) -> None:
        try:
            from prometheus_client import Counter, Histogram
        except ImportError as error:
            raise RuntimeError(
                "PrometheusObservationSink requires the 'prometheus' optional dependency"
            ) from error

        self._server_starts = Counter(
            "vllm_nnrp_adapter_server_starts_total",
            "NNRP adapter server provider bindings started.",
            ("transport_policy", "provider"),
            registry=registry,
        )
        self._operations = Counter(
            "vllm_nnrp_adapter_operations_total",
            "NNRP adapter operations reaching a terminal outcome.",
            ("transport", "operation", "outcome"),
            registry=registry,
        )
        self._latency = Histogram(
            "vllm_nnrp_adapter_operation_latency_seconds",
            "NNRP adapter operation stage latency.",
            ("transport", "operation", "outcome", "stage"),
            registry=registry,
        )
        self._output_events = Counter(
            "vllm_nnrp_adapter_output_events_total",
            "NNRP adapter output events.",
            ("transport", "operation"),
            registry=registry,
        )
        self._output_bytes = Counter(
            "vllm_nnrp_adapter_output_bytes_total",
            "NNRP adapter encoded output bytes.",
            ("transport", "operation"),
            registry=registry,
        )
        self._tokens = Counter(
            "vllm_nnrp_adapter_tokens_total",
            "NNRP adapter token usage reported by the backend.",
            ("transport", "operation", "kind"),
            registry=registry,
        )
        self._cancellations = Counter(
            "vllm_nnrp_adapter_cancellations_total",
            "NNRP adapter operation cancellations.",
            ("transport", "kind"),
            registry=registry,
        )
        self._drops = Counter(
            "vllm_nnrp_adapter_result_drops_total",
            "NNRP adapter result drops.",
            ("transport", "reason"),
            registry=registry,
        )

    def observe_server_startup(self, observation: ServerStartupObservation) -> None:
        policy = _bounded_label(observation.transport_policy, _KNOWN_TRANSPORT_POLICIES)
        for provider, _endpoint in observation.bound_provider_endpoints:
            self._server_starts.labels(
                transport_policy=policy,
                provider=_bounded_transport(provider),
            ).inc()

    def observe_operation(self, observation: OperationObservation) -> None:
        transport = _bounded_transport(observation.identity.selected_transport)
        operation = _KNOWN_PROFILE_OPERATIONS.get(observation.profile_operation or "", "other")
        outcome = _bounded_label(observation.terminal_outcome, _KNOWN_TERMINAL_OUTCOMES)
        labels = {"transport": transport, "operation": operation, "outcome": outcome}

        self._operations.labels(**labels).inc()
        self._observe_latency(labels, "queue", observation.queue_delay_ms)
        self._observe_latency(labels, "admission", observation.admission_latency_ms)
        self._observe_latency(labels, "preprocessing", observation.preprocessing_latency_ms)
        self._observe_latency(labels, "first_event", observation.first_event_latency_ms)
        self._observe_latency(labels, "terminal", observation.terminal_latency_ms)
        event_labels = {"transport": transport, "operation": operation}
        self._output_events.labels(**event_labels).inc(observation.output_event_count)
        self._output_bytes.labels(**event_labels).inc(observation.output_bytes)
        for kind, value in (
            ("prompt", observation.prompt_tokens),
            ("completion", observation.completion_tokens),
            ("total", observation.total_tokens),
        ):
            if value is not None:
                self._tokens.labels(**event_labels, kind=kind).inc(value)
        if observation.cancellation_kind is not None:
            self._cancellations.labels(
                transport=transport,
                kind=_bounded_label(observation.cancellation_kind, _KNOWN_CANCELLATION_KINDS),
            ).inc()
        if observation.drop_reason is not None:
            self._drops.labels(
                transport=transport,
                reason=_bounded_label(observation.drop_reason, _KNOWN_DROP_REASONS),
            ).inc()

    def _observe_latency(self, labels: Mapping[str, str], stage: str, value_ms: float | None) -> None:
        if value_ms is not None:
            self._latency.labels(**labels, stage=stage).observe(value_ms / 1_000)


def _emit_operation_observation(
    observation: OperationObservation,
    sinks: Sequence[ObservationSink],
) -> None:
    for sink in sinks:
        try:
            sink.observe_operation(observation)
        except Exception:
            _SINK_LOGGER.exception("operation observation sink failed")


def _emit_server_startup_observation(
    observation: ServerStartupObservation,
    sinks: Sequence[ObservationSink],
) -> None:
    for sink in sinks:
        try:
            sink.observe_server_startup(observation)
        except Exception:
            _SINK_LOGGER.exception("server startup observation sink failed")


def _bounded_transport(value: str) -> str:
    return _bounded_label(value, _KNOWN_TRANSPORTS)


def _bounded_label(value: str, known_values: frozenset[str]) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in known_values else "other"


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


def _optional_duration_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None:
        return None
    return _duration_ms(start_ns, end_ns)


def _nanoseconds_to_ms(value: int) -> float:
    return value / 1_000_000
