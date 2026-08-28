from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from nnrp import PayloadKind
from nnrp.core import MessageType, unpack_typed_payload_frames, validate_frame_submit_body
from nnrp.runtime import (
    NativeRuntimeEvent,
    ProgressMetadata,
    ResultDropReasonCode,
    ResultDropReasonMetadata,
    RuntimeEventMetadata,
    RuntimeEventMetadataKind,
    RuntimeEventTail,
    RuntimeFrameHeader,
    RuntimeRole,
)

from vllm_nnrp_adapter.stale_work_drivers import (
    DirectNnrpDriver,
    DirectNnrpDriverConfig,
    OpenAiHttpSseDriverConfig,
    OrchestratedHttpControl,
    OrchestratedHttpSseDriver,
    OrchestratedHttpSseDriverConfig,
    RawOpenAiHttpSseDriver,
)
from vllm_nnrp_adapter.stale_work_workload import StaleWorkCase


def _case(*, control_kind: str | None = None) -> StaleWorkCase:
    return StaleWorkCase(
        sample_id="sample-000007",
        ordinal=7,
        scheduled_offset_seconds=0.0,
        model="public-test-model",
        prompt_tokens=8,
        max_completion_tokens=16,
        control_kind=control_kind,
        control_delay_seconds=0.001,
    )


async def _start(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
    *,
    case: StaleWorkCase,
) -> tuple[RawOpenAiHttpSseDriver, object, list[urllib.request.Request]]:
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout_seconds: float) -> object:
        assert timeout_seconds == 12.0
        requests.append(request)
        return response

    monkeypatch.setattr("vllm_nnrp_adapter.stale_work_drivers._urlopen", fake_urlopen)
    driver = RawOpenAiHttpSseDriver(
        OpenAiHttpSseDriverConfig(
            "https://example.invalid/v1/chat/completions",
            api_key="test-key",
            headers={"X-Deployment": "test"},
            timeout_seconds=12.0,
        )
    )
    await driver.begin_run({"sample_count": 1}, (case,))
    return driver, await driver.start(case), requests


class _StaticResponse:
    def __init__(self, *lines: bytes) -> None:
        self.lines = lines
        self.close_count = 0

    def __iter__(self) -> Iterator[bytes]:
        yield from self.lines

    def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_raw_http_driver_sends_correlated_openai_stream_request(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case()
    response = _StaticResponse(
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
        b"data: [DONE]\n",
    )
    driver, operation, requests = await _start(monkeypatch, response, case=case)

    result = await operation.wait()  # type: ignore[attr-defined]
    await operation.close()  # type: ignore[attr-defined]
    await driver.end_run()

    assert result.terminal_outcome == "completed"
    assert result.useful_result_weight == 1.0
    assert result.late_result_count == 0
    assert response.close_count >= 1
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == "https://example.invalid/v1/chat/completions"
    assert dict(request.header_items()) == {
        "Authorization": "Bearer test-key",
        "Content-type": "application/json",
        "X-deployment": "test",
        "X-nnrp-benchmark-sample-id": case.sample_id,
    }
    body = json.loads(request.data)
    assert body["model"] == case.model
    assert body["stream"] is True
    assert body["max_tokens"] == case.max_completion_tokens
    assert body["messages"][0]["content"].split() == ["a"] * case.prompt_tokens


class _BlockingResponse:
    def __init__(self) -> None:
        self.iterating = threading.Event()
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        self.iterating.set()
        if not self.closed.wait(2.0):
            raise AssertionError("response was not closed")
        raise OSError("connection closed by client")
        yield b""  # pragma: no cover

    def close(self) -> None:
        self.closed.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("control_kind", ["cancel", "abort", "deadline", "supersede"])
async def test_raw_http_controls_are_honestly_client_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    control_kind: str,
) -> None:
    case = _case(control_kind=control_kind)
    response = _BlockingResponse()
    driver, operation, _requests = await _start(monkeypatch, response, case=case)
    wait_task = asyncio.create_task(operation.wait())  # type: ignore[attr-defined]
    assert await asyncio.to_thread(response.iterating.wait, 1.0)

    assert await operation.apply_control(control_kind) is True  # type: ignore[attr-defined]
    result = await wait_task
    await operation.close()  # type: ignore[attr-defined]
    await driver.end_run()

    assert result.terminal_outcome == "cancelled"
    assert result.useful_result_weight == 0.0
    assert result.late_result_count == 0


class _LateResponse:
    def __init__(self) -> None:
        self.iterating = threading.Event()
        self.release = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        self.iterating.set()
        if not self.release.wait(2.0):
            raise AssertionError("late response was not released")
        yield b'data: {"choices":[{"delta":{"content":"late"}}]}\n'
        yield b"data: [DONE]\n"

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_raw_http_driver_preserves_post_dispatch_result_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(control_kind="cancel")
    response = _LateResponse()
    driver, operation, _requests = await _start(monkeypatch, response, case=case)
    wait_task = asyncio.create_task(operation.wait())  # type: ignore[attr-defined]
    assert await asyncio.to_thread(response.iterating.wait, 1.0)

    assert await operation.apply_control("cancel") is True  # type: ignore[attr-defined]
    response.release.set()
    result = await wait_task
    await operation.close()  # type: ignore[attr-defined]
    await driver.end_run()

    assert result.terminal_outcome == "completed"
    assert result.useful_result_weight == 0.0
    assert result.late_result_count == 1


@pytest.mark.asyncio
async def test_control_before_response_open_is_applied_when_response_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(control_kind="cancel")
    response = _BlockingResponse()
    entered = threading.Event()
    release = threading.Event()

    def delayed_urlopen(request: urllib.request.Request, timeout_seconds: float) -> _BlockingResponse:
        entered.set()
        if not release.wait(2.0):
            raise AssertionError("urlopen was not released")
        return response

    monkeypatch.setattr("vllm_nnrp_adapter.stale_work_drivers._urlopen", delayed_urlopen)
    driver = RawOpenAiHttpSseDriver(OpenAiHttpSseDriverConfig("https://example.invalid/v1/chat/completions"))
    await driver.begin_run({"sample_count": 1}, (case,))
    operation = await driver.start(case)
    wait_task = asyncio.create_task(operation.wait())
    assert await asyncio.to_thread(entered.wait, 1.0)

    assert await operation.apply_control("cancel") is True
    release.set()
    result = await wait_task
    await operation.close()
    await driver.end_run()

    assert result.terminal_outcome == "cancelled"
    assert response.closed.is_set()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"endpoint": "nnrp://service"}, "absolute http"),
        ({"endpoint": "https://example.invalid", "api_key": ""}, "api_key"),
        ({"endpoint": "https://example.invalid", "timeout_seconds": 0}, "positive"),
        ({"endpoint": "https://example.invalid", "sample_id_header": "Bad:Name"}, "header name"),
        ({"endpoint": "https://example.invalid", "headers": {"X-Test": "bad\nvalue"}}, "single-line"),
    ],
)
def test_raw_http_driver_config_rejects_ambiguous_network_settings(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OpenAiHttpSseDriverConfig(**kwargs)  # type: ignore[arg-type]


class _FakeOrchestrationController:
    def __init__(self, *, dispatch_result: bool = True) -> None:
        self.dispatch_result = dispatch_result
        self.begun = 0
        self.ended = 0
        self.controls: list[OrchestratedHttpControl] = []

    async def begin_run(
        self,
        workload: dict[str, object],
        schedule: tuple[StaleWorkCase, ...],
    ) -> None:
        assert workload["sample_count"] == len(schedule)
        self.begun += 1

    async def dispatch(self, control: OrchestratedHttpControl) -> bool:
        self.controls.append(control)
        return self.dispatch_result

    async def end_run(self) -> None:
        self.ended += 1


class _ControlledHttpResponse:
    def __init__(self) -> None:
        self.iterating = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        self.iterating.set()
        if not self.release.wait(2.0):
            raise AssertionError("orchestrated response was not released")
        raise OSError("server ended controlled stream")
        yield b""  # pragma: no cover

    def close(self) -> None:
        self.closed.set()


async def _start_orchestrated(
    monkeypatch: pytest.MonkeyPatch,
    case: StaleWorkCase,
    response_factory: Any,
    *,
    dispatch_result: bool = True,
) -> tuple[OrchestratedHttpSseDriver, Any, _FakeOrchestrationController]:
    monkeypatch.setattr("vllm_nnrp_adapter.stale_work_drivers._urlopen", response_factory)
    controller = _FakeOrchestrationController(dispatch_result=dispatch_result)
    driver = OrchestratedHttpSseDriver(
        OrchestratedHttpSseDriverConfig(
            request=OpenAiHttpSseDriverConfig("https://example.invalid/v1/chat/completions"),
            controller=controller,
        )
    )
    await driver.begin_run({"sample_count": 1}, (case,))
    return driver, await driver.start(case), controller


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("control_kind", "expected_outcome"),
    [
        ("cancel", "cancelled"),
        ("abort", "aborted"),
        ("deadline", "expired"),
    ],
)
async def test_orchestrated_http_dispatches_without_client_disconnect(
    monkeypatch: pytest.MonkeyPatch,
    control_kind: str,
    expected_outcome: str,
) -> None:
    case = _case(control_kind=control_kind)
    response = _ControlledHttpResponse()
    driver, operation, controller = await _start_orchestrated(
        monkeypatch,
        case,
        lambda _request, _timeout: response,
    )
    wait_task = asyncio.create_task(operation.wait())
    assert await asyncio.to_thread(response.iterating.wait, 1.0)

    assert await operation.apply_control(control_kind) is True
    assert not response.closed.is_set()
    assert controller.controls[0].sample_id == case.sample_id
    assert controller.controls[0].control_kind == control_kind
    if control_kind == "deadline":
        assert controller.controls[0].deadline_unix_ms is not None
    response.release.set()
    result = await wait_task
    await operation.close()
    await driver.end_run()

    assert result.terminal_outcome == expected_outcome
    assert result.useful_result_weight == 0.0
    assert controller.begun == 1
    assert controller.ended == 1


@pytest.mark.asyncio
async def test_orchestrated_http_supersede_submits_real_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(control_kind="supersede")
    original = _ControlledHttpResponse()
    requests: list[str] = []

    def response_for_request(request: urllib.request.Request, _timeout: float) -> object:
        headers = {name.lower(): value for name, value in request.header_items()}
        request_id = headers["x-nnrp-benchmark-sample-id"]
        requests.append(request_id)
        if request_id == case.sample_id:
            return original
        return _StaticResponse(b"data: [DONE]\n")

    driver, operation, controller = await _start_orchestrated(monkeypatch, case, response_for_request)
    wait_task = asyncio.create_task(operation.wait())
    assert await asyncio.to_thread(original.iterating.wait, 1.0)

    assert await operation.apply_control("supersede") is True
    original.release.set()
    result = await wait_task
    await operation.close()
    await driver.end_run()

    replacement_id = f"{case.sample_id}:replacement"
    assert controller.controls == [
        OrchestratedHttpControl(case.sample_id, "supersede", replacement_sample_id=replacement_id)
    ]
    assert requests == [case.sample_id, replacement_id]
    assert result.terminal_outcome == "superseded"


@pytest.mark.asyncio
async def test_orchestrated_http_rejected_dispatch_does_not_claim_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(control_kind="abort")
    response = _ControlledHttpResponse()
    driver, operation, controller = await _start_orchestrated(
        monkeypatch,
        case,
        lambda _request, _timeout: response,
        dispatch_result=False,
    )
    wait_task = asyncio.create_task(operation.wait())
    assert await asyncio.to_thread(response.iterating.wait, 1.0)

    assert await operation.apply_control("abort") is False
    response.release.set()
    result = await wait_task
    await operation.close()
    await driver.end_run()

    assert result.terminal_outcome == "failed"
    assert controller.controls[0].control_kind == "abort"


@pytest.mark.asyncio
async def test_orchestrated_http_malformed_sse_remains_failed_after_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MalformedResponse:
        def __init__(self) -> None:
            self.iterating = threading.Event()
            self.release = threading.Event()

        def __iter__(self) -> Iterator[bytes]:
            self.iterating.set()
            if not self.release.wait(2.0):
                raise AssertionError("malformed response was not released")
            yield b"data: {not-json}\n"

        def close(self) -> None:
            pass

    case = _case(control_kind="cancel")
    response = MalformedResponse()
    driver, operation, _controller = await _start_orchestrated(
        monkeypatch,
        case,
        lambda _request, _timeout: response,
    )
    wait_task = asyncio.create_task(operation.wait())
    assert await asyncio.to_thread(response.iterating.wait, 1.0)
    assert await operation.apply_control("cancel") is True
    response.release.set()
    result = await wait_task
    await operation.close()
    await driver.end_run()

    assert result.terminal_outcome == "failed"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_id": "", "control_kind": "cancel"}, "sample_id"),
        ({"sample_id": "sample", "control_kind": "supersede"}, "replacement"),
        (
            {"sample_id": "sample", "control_kind": "cancel", "replacement_sample_id": "other"},
            "only valid",
        ),
        ({"sample_id": "sample", "control_kind": "deadline"}, "deadline_unix_ms"),
    ],
)
def test_orchestrated_http_control_rejects_ambiguous_plan(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OrchestratedHttpControl(**kwargs)  # type: ignore[arg-type]


class _FakeNativeOperation:
    def __init__(self, operation_id: int, frame_id: int) -> None:
        self.operation_id = operation_id
        self.frame_id = frame_id
        self.cancel_count = 0

    def cancel(self) -> None:
        self.cancel_count += 1


class _FakeNativeSession:
    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.operations: list[_FakeNativeOperation] = []
        self.events: asyncio.Queue[Any] = asyncio.Queue()

    async def async_submit_operation(self, request: Any) -> _FakeNativeOperation:
        self.requests.append(request)
        operation = _FakeNativeOperation(request.operation_id, request.frame_id)
        self.operations.append(operation)
        return operation

    async def next_event(self, timeout: float | None = None) -> Any:
        del timeout
        return await self.events.get()


class _FakeNativeConnection:
    def __init__(self, session: _FakeNativeSession) -> None:
        self.session = session
        self.control_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def open_session(self) -> _FakeNativeSession:
        return self.session

    def cancel_runtime_operation(self, *args: Any, **kwargs: Any) -> None:
        self.control_calls.append(("cancel", args, kwargs))

    def abort_runtime_operation(self, *args: Any, **kwargs: Any) -> None:
        self.control_calls.append(("abort", args, kwargs))

    def update_runtime_deadline(self, *args: Any, **kwargs: Any) -> None:
        self.control_calls.append(("deadline", args, kwargs))

    def supersede_runtime_operation(self, *args: Any, **kwargs: Any) -> None:
        self.control_calls.append(("supersede", args, kwargs))


class _FakeNativeConnectionContext:
    def __init__(self, connection: _FakeNativeConnection) -> None:
        self.connection = connection
        self.exit_count = 0

    def __enter__(self) -> _FakeNativeConnection:
        return self.connection

    def __exit__(self, *args: Any) -> None:
        self.exit_count += 1


async def _start_direct(
    monkeypatch: pytest.MonkeyPatch,
    case: StaleWorkCase,
) -> tuple[DirectNnrpDriver, Any, _FakeNativeSession, _FakeNativeConnection, _FakeNativeConnectionContext]:
    session = _FakeNativeSession()
    connection = _FakeNativeConnection(session)
    context = _FakeNativeConnectionContext(connection)
    monkeypatch.setattr(
        "vllm_nnrp_adapter.stale_work_drivers.connect_native_client_connection",
        lambda _options: context,
    )
    driver = DirectNnrpDriver(
        DirectNnrpDriverConfig(
            "nnrp://benchmark-service",
            timeout_seconds=2.0,
            event_poll_seconds=0.01,
        )
    )
    await driver.begin_run({"sample_count": 1, "max_in_flight": 4}, (case,))
    return driver, await driver.start(case), session, connection, context


def _progress_event(operation_id: int, frame_id: int) -> NativeRuntimeEvent:
    return NativeRuntimeEvent(
        RuntimeFrameHeader(MessageType.PROGRESS, session_id=1, frame_id=frame_id),
        RuntimeEventMetadata(
            RuntimeEventMetadataKind.PROGRESS,
            ProgressMetadata(operation_id, 1, 1, 5_000, 0, 0),
        ),
        RuntimeEventTail.with_body(b"progress"),
    )


def _result_event(operation_id: int, frame_id: int) -> NativeRuntimeEvent:
    return NativeRuntimeEvent(
        RuntimeFrameHeader(MessageType.RESULT_PUSH, session_id=1, frame_id=frame_id),
        RuntimeEventMetadata(RuntimeEventMetadataKind.RESULT_PUSH, SimpleNamespace(operation_id=operation_id)),
        RuntimeEventTail.with_body(b"result"),
    )


def _drop_event(
    operation_id: int,
    frame_id: int,
    reason: ResultDropReasonCode,
) -> NativeRuntimeEvent:
    return NativeRuntimeEvent(
        RuntimeFrameHeader(MessageType.RESULT_DROP_REASON, session_id=1, frame_id=frame_id),
        RuntimeEventMetadata(
            RuntimeEventMetadataKind.RESULT_DROP_REASON,
            ResultDropReasonMetadata(operation_id, 1, reason, RuntimeRole.RUNTIME, 0, 0),
        ),
        RuntimeEventTail.with_diagnostic(b""),
    )


def _submit_envelope(request: Any) -> dict[str, Any]:
    body = validate_frame_submit_body(request.metadata, request.body)
    frames = unpack_typed_payload_frames(
        body.typed_payload_descriptor_region,
        body.typed_payload_frame_region,
        payload_kind_bitmap=request.metadata.payload_kind_bitmap,
    )
    assert len(frames) == 1
    assert frames[0].payload_kind is PayloadKind.STRUCTURED_EVENT
    return json.loads(frames[0].payload)


@pytest.mark.asyncio
async def test_direct_nnrp_driver_reuses_session_and_sends_correlated_profile_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    driver, operation, session, _connection, context = await _start_direct(monkeypatch, case)
    await session.events.put(_result_event(operation.operation_id, operation.frame_id))

    result = await operation.wait()
    await operation.close()
    await driver.end_run()

    assert result.terminal_outcome == "completed"
    assert result.useful_result_weight == 1.0
    assert len(session.requests) == 1
    envelope = _submit_envelope(session.requests[0])
    assert envelope["request_id"] == case.sample_id
    assert envelope["operation"] == "chat.completions.create"
    assert envelope["body"]["stream"] is True
    assert envelope["body"]["messages"][0]["content"].split() == ["a"] * case.prompt_tokens
    assert context.exit_count == 1


@pytest.mark.asyncio
async def test_direct_nnrp_event_pump_routes_concurrent_operations_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_case = _case()
    second_case = StaleWorkCase(
        sample_id="sample-000008",
        ordinal=8,
        scheduled_offset_seconds=0.0,
        model=first_case.model,
        prompt_tokens=first_case.prompt_tokens,
        max_completion_tokens=first_case.max_completion_tokens,
        control_kind=None,
        control_delay_seconds=first_case.control_delay_seconds,
    )
    driver, first, session, _connection, _context = await _start_direct(monkeypatch, first_case)
    second = await driver.start(second_case)

    await session.events.put(_result_event(second.operation_id, second.frame_id))
    await session.events.put(_result_event(first.operation_id, first.frame_id))
    second_result, first_result = await asyncio.gather(second.wait(), first.wait())
    await asyncio.gather(first.close(), second.close())
    await driver.end_run()

    assert first_result.terminal_outcome == "completed"
    assert second_result.terminal_outcome == "completed"
    assert [_submit_envelope(request)["request_id"] for request in session.requests] == [
        first_case.sample_id,
        second_case.sample_id,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("control_kind", "drop_reason", "expected_outcome"),
    [
        ("cancel", ResultDropReasonCode.PEER_CANCELLED, "cancelled"),
        ("abort", ResultDropReasonCode.PEER_CANCELLED, "aborted"),
        ("deadline", ResultDropReasonCode.DEADLINE_EXPIRED, "expired"),
    ],
)
async def test_direct_nnrp_controls_use_server_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
    control_kind: str,
    drop_reason: ResultDropReasonCode,
    expected_outcome: str,
) -> None:
    case = _case(control_kind=control_kind)
    driver, operation, session, connection, _context = await _start_direct(monkeypatch, case)

    assert await operation.apply_control(control_kind) is True
    await session.events.put(_progress_event(operation.operation_id, operation.frame_id))
    await session.events.put(_drop_event(operation.operation_id, operation.frame_id, drop_reason))
    result = await operation.wait()
    await operation.close()
    await driver.end_run()

    assert [call[0] for call in connection.control_calls] == [control_kind]
    assert result.terminal_outcome == expected_outcome
    assert result.useful_result_weight == 0.0
    assert result.late_result_count == 1


@pytest.mark.asyncio
async def test_direct_nnrp_does_not_infer_control_semantics_from_unrelated_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(control_kind="deadline")
    driver, operation, session, _connection, _context = await _start_direct(monkeypatch, case)

    assert await operation.apply_control("deadline") is True
    await session.events.put(
        _drop_event(operation.operation_id, operation.frame_id, ResultDropReasonCode.BACKPRESSURE)
    )
    result = await operation.wait()
    await operation.close()
    await driver.end_run()

    assert result.terminal_outcome == "failed"
    assert result.useful_result_weight == 0.0


@pytest.mark.asyncio
async def test_direct_nnrp_supersede_submits_and_drains_real_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(control_kind="supersede")
    driver, operation, session, connection, _context = await _start_direct(monkeypatch, case)

    assert await operation.apply_control("supersede") is True
    assert len(session.requests) == 2
    replacement = session.operations[1]
    control = connection.control_calls[0]
    assert control[0] == "supersede"
    assert control[2]["old_operation_id"] == operation.operation_id
    assert control[2]["new_operation_id"] == replacement.operation_id
    assert _submit_envelope(session.requests[1])["request_id"] == f"{case.sample_id}:replacement"

    await session.events.put(
        _drop_event(operation.operation_id, operation.frame_id, ResultDropReasonCode.SUPERSEDED)
    )
    await session.events.put(_result_event(replacement.operation_id, replacement.frame_id))
    result = await operation.wait()
    await operation.close()
    await asyncio.sleep(0)
    await driver.end_run()

    assert result.terminal_outcome == "superseded"
    assert result.useful_result_weight == 0.0
    assert replacement.cancel_count == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"endpoint": ""}, "non-empty"),
        ({"endpoint": "nnrp://service", "timeout_seconds": float("inf")}, "finite"),
        ({"endpoint": "nnrp://service", "event_poll_seconds": 0}, "positive"),
    ],
)
def test_direct_nnrp_driver_config_rejects_unbounded_runtime_settings(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DirectNnrpDriverConfig(**kwargs)  # type: ignore[arg-type]
