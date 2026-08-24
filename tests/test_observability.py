from __future__ import annotations

import json
import logging
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest
from nnrp.core import FrameSubmitMetadata, InputProfile, MessageType, PayloadKind
from nnrp.runtime import (
    NativeRuntimeEvent,
    PressureMetadata,
    ResultDropReasonCode,
    RetryAfterMetadata,
    RuntimeEventMetadata,
    RuntimeEventMetadataKind,
    RuntimeEventTail,
    RuntimeFrameHeader,
    RuntimeRole,
    TraceContextMetadata,
)
from prometheus_client import CollectorRegistry, generate_latest

from vllm_nnrp_adapter.observability import (
    OperationIdentity,
    OperationObservation,
    OperationStageTransition,
    PrometheusObservationSink,
    ServerStartupObservation,
    StructuredLogObservationSink,
    _emit_operation_observation,
    _emit_server_startup_observation,
    _OperationObservationTracker,
)
from vllm_nnrp_adapter.operation_progress import OperationProgressStage
from vllm_nnrp_adapter.operation_state import OperationState
from vllm_nnrp_adapter.runtime_control import RuntimeControlKind, RuntimeControlRequest


def test_operation_observation_is_immutable_complete_and_structured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ticks = iter(
        (
            0,
            1_000_000,
            2_000_000,
            3_000_000,
            4_000_000,
            5_000_000,
            8_000_000,
            11_000_000,
            14_000_000,
            17_000_000,
        )
    )
    operation = _operation()
    tracker = _OperationObservationTracker.from_operation(
        operation,
        selected_transport="ipc",
        backend_family="VllmBackend",
        backend_binding="current-0.26",
        vllm_version="0.26.0",
        connection_id=101,
        connection_generation=2,
        session_handle_id=202,
        session_generation=3,
        clock_ns=lambda: next(ticks),
    )
    tracker.record_request(
        {
            "operation": "chat.completions.create",
            "body": {"model": "test-model"},
        }
    )
    tracker.record_progress_stage(OperationProgressStage.QUEUED)
    tracker.record_progress_stage(OperationProgressStage.INPUT_RECEIVED)
    tracker.mark_admitted()
    tracker.record_progress_stage(OperationProgressStage.ADMITTED)
    tracker.record_progress_stage(OperationProgressStage.PREPROCESSING)
    tracker.record_progress_stage(OperationProgressStage.EXECUTING)
    tracker.record_event(
        {
            "type": "response.usage",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
        body_bytes=17,
    )
    tracker.record_event(
        {
            "type": "response.error",
            "error": {"type": "server_error"},
            "diagnostics": {"backend_error_family": "BackendBusy"},
        },
        body_bytes=23,
    )
    tracker.record_control(
        RuntimeControlRequest(
            kind=RuntimeControlKind.ABORT,
            operation_id=7,
            control_sequence=3,
            reason_code=19,
            source_role=RuntimeRole.CLIENT,
            flags=0,
            diagnostic=b"obsolete",
        ),
        drop_reason=ResultDropReasonCode.PEER_CANCELLED,
    )
    tracker.record_backend_abort(True)
    tracker.record_retry_hint(
        RetryAfterMetadata(
            scope_id=7,
            control_sequence=9,
            retry_after_ms=40,
            jitter_ms=0,
            reason_code=3,
            source_role=RuntimeRole.RUNTIME,
            flags=0x02,
            diagnostic_bytes=0,
        )
    )
    tracker.record_pressure(
        PressureMetadata(
            scope_id=7,
            credit_window=0,
            pressure_level=3,
            pressure_reason=5,
            retry_after_ms=25,
            flags=0x02,
        ),
        scope_kind="operation",
        scope_id=7,
    )
    observation = tracker.finish(OperationState.DROPPED)

    assert observation.identity.selected_transport == "ipc"
    assert observation.identity.connection_id == 101
    assert observation.identity.connection_generation == 2
    assert observation.identity.session_handle_id == 202
    assert observation.identity.session_generation == 3
    assert observation.identity.session_id == 11
    assert observation.identity.operation_id == 7
    assert observation.identity.frame_id == 17
    assert observation.identity.route_id == 27
    assert observation.identity.view_id == 37
    assert observation.identity.trace_id == 47
    assert observation.identity.profile_id == int(InputProfile.UNSPECIFIED)
    assert observation.model_id == "test-model"
    assert observation.profile_operation == "chat.completions.create"
    assert observation.backend_family == "VllmBackend"
    assert observation.backend_binding == "current-0.26"
    assert observation.vllm_version == "0.26.0"
    assert observation.queue_delay_ms == 3.0
    assert observation.admission_latency_ms == 1.0
    assert observation.preprocessing_latency_ms == 3.0
    assert observation.first_event_latency_ms == 11.0
    assert observation.inter_event_latency_ms == (3.0,)
    assert observation.terminal_latency_ms == 17.0
    assert observation.output_event_count == 2
    assert observation.output_bytes == 40
    assert observation.prompt_tokens == 4
    assert observation.completion_tokens == 2
    assert observation.total_tokens == 6
    assert observation.error_family == "BackendBusy"
    assert observation.cancellation_kind == "abort"
    assert observation.cancellation_source == "client"
    assert observation.cancellation_reason_code == 19
    assert observation.backend_abort_accepted is True
    assert observation.drop_reason == "peer_cancelled"
    assert observation.terminal_outcome == "dropped"
    assert observation.retry_after_ms == 40
    assert observation.retry_reason_code == 3
    assert observation.retry_source == "runtime"
    assert observation.pressure_scope == "operation"
    assert observation.pressure_scope_id == 7
    assert observation.pressure_credit_window == 0
    assert observation.pressure_level == 3
    assert observation.pressure_reason == 5
    assert observation.pressure_retry_after_ms == 25
    assert observation.stage_transitions == (
        OperationStageTransition(0x0001, "queued", 1.0),
        OperationStageTransition(0x0003, "input_received", 2.0),
        OperationStageTransition(0x0002, "admitted", 4.0),
        OperationStageTransition(0x0004, "preprocessing", 5.0),
        OperationStageTransition(0x0005, "executing", 8.0),
    )
    with pytest.raises(FrozenInstanceError):
        observation.model_id = "mutated"  # type: ignore[misc]
    with pytest.raises(RuntimeError, match="already finished"):
        tracker.finish(OperationState.DROPPED)

    caplog.set_level(logging.INFO, logger="vllm_nnrp_adapter.operation")
    _emit_operation_observation(observation, (StructuredLogObservationSink(),))
    payload = json.loads(caplog.records[-1].getMessage().removeprefix("nnrp_operation_observation "))
    assert payload["operation_id"] == 7
    assert payload["connection_id"] == 101
    assert payload["session_handle_id"] == 202
    assert payload["model_id"] == "test-model"
    assert payload["backend_binding"] == "current-0.26"
    assert payload["vllm_version"] == "0.26.0"
    assert payload["admission_latency_ms"] == 1.0
    assert payload["preprocessing_latency_ms"] == 3.0
    assert payload["stage_transitions"][-1] == {
        "elapsed_ms": 8.0,
        "stage_code": 5,
        "stage_name": "executing",
    }
    assert payload["backend_abort_accepted"] is True
    assert payload["terminal_outcome"] == "dropped"
    assert payload["retry_after_ms"] == 40
    assert payload["retry_source"] == "runtime"
    assert payload["pressure_scope"] == "operation"
    assert payload["pressure_level"] == 3


def test_operation_observation_records_trace_context_without_attribute_contents() -> None:
    tracker = _OperationObservationTracker.from_operation(
        _operation(),
        selected_transport="ipc",
        clock_ns=lambda: 0,
    )

    tracker.record_trace_context(
        TraceContextMetadata(
            trace_id=71,
            span_id=72,
            parent_span_id=70,
            stage_code=5,
            flags=3,
            body_bytes=6,
        ),
        b"secret",
    )
    observation = tracker.finish(OperationState.FAILED)

    assert observation.identity.trace_id == 71
    assert observation.trace_span_id == 72
    assert observation.trace_parent_span_id == 70
    assert observation.trace_stage_code == 5
    assert observation.trace_flags == 3
    assert observation.trace_attribute_bytes == 6
    assert "secret" not in json.dumps(observation.to_log_fields())

    with pytest.raises(ValueError, match="body_bytes"):
        tracker = _OperationObservationTracker.from_operation(
            _operation(),
            selected_transport="ipc",
            clock_ns=lambda: 0,
        )
        tracker.record_trace_context(TraceContextMetadata(1, 2, 0, 0, 0, 1), b"")


def test_operation_observation_keeps_unavailable_values_absent() -> None:
    ticks = iter((100, 200, 300))
    tracker = _OperationObservationTracker.from_operation(
        _operation(),
        selected_transport="websocket",
        clock_ns=lambda: next(ticks),
    )
    tracker.record_request({"body": {"model": 1}})
    tracker.record_event(
        {"type": "response.usage", "usage": {"prompt_tokens": "unknown"}},
        body_bytes=0,
    )
    tracker.record_exception(ValueError("bad request"))
    observation = tracker.finish(OperationState.FAILED)

    assert observation.model_id is None
    assert observation.profile_operation is None
    assert observation.queue_delay_ms is None
    assert observation.admission_latency_ms is None
    assert observation.preprocessing_latency_ms is None
    assert observation.first_event_latency_ms == 0.0001
    assert observation.prompt_tokens is None
    assert observation.error_family == "ValueError"
    assert observation.backend_abort_accepted is None


def test_server_startup_observation_is_immutable_sorted_and_structured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    observation = ServerStartupObservation.from_bound_endpoints(
        application_endpoint="nnrp://runtime.local/vllm",
        transport_policy="auto",
        bound_provider_endpoints={
            "websocket": SimpleNamespace(uri="ws://127.0.0.1:9001/nnrp"),
            "ipc": SimpleNamespace(uri="unix:///tmp/nnrp.sock"),
        },  # type: ignore[arg-type]
    )

    assert observation.bound_provider_endpoints == (
        ("ipc", "unix:///tmp/nnrp.sock"),
        ("websocket", "ws://127.0.0.1:9001/nnrp"),
    )
    with pytest.raises(FrozenInstanceError):
        observation.transport_policy = "force_tcp"  # type: ignore[misc]

    caplog.set_level(logging.INFO, logger="vllm_nnrp_adapter.server")
    _emit_server_startup_observation(observation, (StructuredLogObservationSink(),))
    payload = json.loads(caplog.records[-1].getMessage().removeprefix("nnrp_server_startup "))
    assert payload == {
        "application_endpoint": "nnrp://runtime.local/vllm",
        "bound_provider_endpoints": {
            "ipc": "unix:///tmp/nnrp.sock",
            "websocket": "ws://127.0.0.1:9001/nnrp",
        },
        "eligible_providers": ["ipc", "websocket"],
        "transport_policy": "auto",
    }


def test_prometheus_sink_registers_bounded_metrics_in_existing_registry() -> None:
    registry = CollectorRegistry()
    sink = PrometheusObservationSink(registry)
    sink.observe_server_startup(
        ServerStartupObservation(
            application_endpoint="nnrp://runtime.local/vllm",
            transport_policy="auto",
            bound_provider_endpoints=(("ipc", "unix:///tmp/nnrp.sock"),),
        )
    )
    sink.observe_operation(_public_observation())

    metrics = generate_latest(registry).decode("utf-8")
    assert 'vllm_nnrp_adapter_server_starts_total{provider="ipc",transport_policy="auto"} 1.0' in metrics
    assert (
        'vllm_nnrp_adapter_operations_total{operation="chat_completions_create",outcome="completed",'
        'transport="ipc"} 1.0'
    ) in metrics
    assert 'vllm_nnrp_adapter_output_events_total{operation="chat_completions_create",transport="ipc"} 2.0' in metrics
    assert 'vllm_nnrp_adapter_output_bytes_total{operation="chat_completions_create",transport="ipc"} 40.0' in metrics
    assert (
        'vllm_nnrp_adapter_tokens_total{kind="total",operation="chat_completions_create",transport="ipc"} 6.0'
        in metrics
    )
    assert "test-model" not in metrics
    assert "trace_id" not in metrics
    assert "operation_id" not in metrics


def test_observation_sink_failure_isolated_from_following_sinks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorded: list[OperationObservation] = []

    class FailingSink:
        def observe_server_startup(self, observation: ServerStartupObservation) -> None:
            raise RuntimeError(observation.application_endpoint)

        def observe_operation(self, observation: OperationObservation) -> None:
            raise RuntimeError(str(observation.identity.operation_id))

    class RecordingSink:
        def observe_server_startup(self, observation: ServerStartupObservation) -> None:
            pass

        def observe_operation(self, observation: OperationObservation) -> None:
            recorded.append(observation)

    observation = _public_observation()
    caplog.set_level(logging.ERROR, logger="vllm_nnrp_adapter.observability")
    _emit_operation_observation(observation, (FailingSink(), RecordingSink()))

    assert recorded == [observation]
    assert "operation observation sink failed" in caplog.records[-1].getMessage()


def _public_observation() -> OperationObservation:
    return OperationObservation(
        identity=OperationIdentity(
            selected_transport="ipc",
            connection_id=101,
            connection_generation=2,
            session_handle_id=202,
            session_generation=3,
            session_id=11,
            operation_id=7,
            frame_id=17,
            route_id=27,
            view_id=37,
            trace_id=47,
            profile_id=int(InputProfile.UNSPECIFIED),
        ),
        model_id="test-model",
        profile_operation="chat.completions.create",
        backend_family="VllmBackend",
        backend_binding="current-0.26",
        vllm_version="0.26.0",
        queue_delay_ms=2.0,
        admission_latency_ms=1.0,
        preprocessing_latency_ms=3.0,
        first_event_latency_ms=5.0,
        inter_event_latency_ms=(3.0,),
        terminal_latency_ms=11.0,
        output_event_count=2,
        output_bytes=40,
        prompt_tokens=4,
        completion_tokens=2,
        total_tokens=6,
        error_family=None,
        cancellation_kind=None,
        cancellation_source=None,
        cancellation_reason_code=None,
        backend_abort_accepted=None,
        drop_reason=None,
        terminal_outcome="completed",
        stage_transitions=(OperationStageTransition(0x0005, "executing", 4.0),),
    )


def _operation() -> Any:
    metadata = FrameSubmitMetadata(
        src_width=0,
        src_height=0,
        tile_width=0,
        tile_height=0,
        tile_count=0,
        section_count=0,
        frame_class=0,
        input_profile=InputProfile.UNSPECIFIED,
        tile_index_mode=0,
        reserved0=0,
        latency_budget_ms=0,
        target_fps_x100=0,
        retry_of_frame=0,
        tile_base_id=0,
        camera_bytes=0,
        tile_index_bytes=0,
        operation_id=7,
        payload_kind_bitmap=PayloadKind.STRUCTURED_EVENT,
        payload_frame_count=1,
    )
    submit = NativeRuntimeEvent(
        RuntimeFrameHeader(
            message_type=MessageType.FRAME_SUBMIT,
            session_id=11,
            frame_id=17,
            route_id=27,
            view_id=37,
            trace_id=47,
        ),
        RuntimeEventMetadata(RuntimeEventMetadataKind.FRAME_SUBMIT, metadata),
        RuntimeEventTail.with_body(b"{}"),
    )
    return SimpleNamespace(operation_id=7, frame_id=17, submit=submit)
