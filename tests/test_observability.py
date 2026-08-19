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
    ResultDropReasonCode,
    RuntimeEventMetadata,
    RuntimeEventMetadataKind,
    RuntimeEventTail,
    RuntimeFrameHeader,
    RuntimeRole,
)

from vllm_nnrp_adapter.observability import (
    _emit_operation_observation,
    _OperationObservationTracker,
)
from vllm_nnrp_adapter.operation_state import OperationState
from vllm_nnrp_adapter.runtime_control import RuntimeControlKind, RuntimeControlRequest


def test_operation_observation_is_immutable_complete_and_structured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ticks = iter((0, 2_000_000, 5_000_000, 8_000_000, 11_000_000))
    operation = _operation()
    tracker = _OperationObservationTracker.from_operation(
        operation,
        selected_transport="ipc",
        clock_ns=lambda: next(ticks),
    )
    tracker.record_request(
        {
            "operation": "chat.completions.create",
            "body": {"model": "test-model"},
        }
    )
    tracker.mark_admitted()
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
    observation = tracker.finish(OperationState.DROPPED)

    assert observation.identity.selected_transport == "ipc"
    assert observation.identity.session_id == 11
    assert observation.identity.operation_id == 7
    assert observation.identity.frame_id == 17
    assert observation.identity.route_id == 27
    assert observation.identity.view_id == 37
    assert observation.identity.trace_id == 47
    assert observation.identity.profile_id == int(InputProfile.UNSPECIFIED)
    assert observation.model_id == "test-model"
    assert observation.profile_operation == "chat.completions.create"
    assert observation.queue_delay_ms == 2.0
    assert observation.first_event_latency_ms == 5.0
    assert observation.inter_event_latency_ms == (3.0,)
    assert observation.terminal_latency_ms == 11.0
    assert observation.output_event_count == 2
    assert observation.output_bytes == 40
    assert observation.prompt_tokens == 4
    assert observation.completion_tokens == 2
    assert observation.total_tokens == 6
    assert observation.error_family == "BackendBusy"
    assert observation.cancellation_kind == "abort"
    assert observation.cancellation_source == "client"
    assert observation.cancellation_reason_code == 19
    assert observation.drop_reason == "peer_cancelled"
    assert observation.terminal_outcome == "dropped"
    with pytest.raises(FrozenInstanceError):
        observation.model_id = "mutated"  # type: ignore[misc]
    with pytest.raises(RuntimeError, match="already finished"):
        tracker.finish(OperationState.DROPPED)

    caplog.set_level(logging.INFO, logger="vllm_nnrp_adapter.operation")
    _emit_operation_observation(observation)
    payload = json.loads(caplog.records[-1].getMessage().removeprefix("nnrp_operation_observation "))
    assert payload["operation_id"] == 7
    assert payload["model_id"] == "test-model"
    assert payload["terminal_outcome"] == "dropped"


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
    assert observation.first_event_latency_ms == 0.0001
    assert observation.prompt_tokens is None
    assert observation.error_family == "ValueError"


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
