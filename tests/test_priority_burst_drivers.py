from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from vllm_nnrp_adapter.openai_http_sse import OpenAiHttpSseDriverConfig
from vllm_nnrp_adapter.priority_burst_drivers import (
    OrchestratedPriorityHttpSseDriver,
    RawPriorityHttpSseDriver,
)
from vllm_nnrp_adapter.priority_burst_workload import PriorityBurstCase


def _case(*, urgent: bool = True) -> PriorityBurstCase:
    return PriorityBurstCase(
        sample_id="sample-000007",
        ordinal=7,
        scheduled_offset_seconds=0.01,
        model="public-test-model",
        prompt_tokens=8,
        max_completion_tokens=16,
        traffic_class="urgent" if urgent else "normal",
        backend_priority=0 if urgent else 1,
    )


def _config() -> OpenAiHttpSseDriverConfig:
    return OpenAiHttpSseDriverConfig(
        "https://example.invalid/v1/chat/completions",
        api_key="test-key",
        headers={"X-Deployment": "test"},
        timeout_seconds=12.0,
    )


class _StaticResponse:
    def __init__(self, *lines: bytes) -> None:
        self.lines = lines
        self.close_count = 0

    def __iter__(self) -> Iterator[bytes]:
        yield from self.lines

    def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("driver_type", "expected_priority"),
    [
        (RawPriorityHttpSseDriver, None),
        (OrchestratedPriorityHttpSseDriver, 0),
    ],
)
async def test_priority_http_drivers_preserve_admission_boundary(
    monkeypatch: pytest.MonkeyPatch,
    driver_type: type[RawPriorityHttpSseDriver],
    expected_priority: int | None,
) -> None:
    requests: list[urllib.request.Request] = []
    response = _StaticResponse(
        b'event: message\n',
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
        b'data: [DONE]\n',
    )

    def fake_urlopen(request: urllib.request.Request, timeout_seconds: float) -> _StaticResponse:
        assert timeout_seconds == 12.0
        requests.append(request)
        return response

    monkeypatch.setattr("vllm_nnrp_adapter.priority_burst_drivers._urlopen", fake_urlopen)
    case = _case()
    driver = driver_type(_config())
    await driver.begin_run({"sample_count": 1}, (case,))
    operation = await driver.start(case)
    assert (await operation.wait()).terminal_outcome == "completed"
    assert await operation.wait() == await operation.wait()
    await operation.close()
    await operation.close()
    await driver.end_run()

    assert response.close_count >= 1
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == "https://example.invalid/v1/chat/completions"
    assert dict(request.header_items()) == {
        "Accept": "text/event-stream",
        "Authorization": "Bearer test-key",
        "Content-type": "application/json",
        "X-deployment": "test",
        "X-nnrp-benchmark-sample-id": case.sample_id,
    }
    body = json.loads(request.data)
    assert body["model"] == case.model
    assert body["stream"] is True
    assert body["max_tokens"] == case.max_completion_tokens
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"][0]["content"].split() == ["a"] * case.prompt_tokens
    if expected_priority is None:
        assert "priority" not in body
    else:
        assert body["priority"] == expected_priority


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (_StaticResponse(b"data: not-json\n"), "failed"),
        (_StaticResponse(b"data: []\n"), "failed"),
        (_StaticResponse(b'data: {"choices":[]}\n'), "failed"),
        (_StaticResponse(b"\xff\n"), "failed"),
    ],
)
async def test_priority_http_driver_rejects_incomplete_or_invalid_sse(
    monkeypatch: pytest.MonkeyPatch,
    response: _StaticResponse,
    expected: str,
) -> None:
    monkeypatch.setattr(
        "vllm_nnrp_adapter.priority_burst_drivers._urlopen",
        lambda _request, _timeout: response,
    )
    case = _case()
    driver = RawPriorityHttpSseDriver(_config())
    await driver.begin_run({"sample_count": 1}, (case,))
    operation = await driver.start(case)

    assert (await operation.wait()).terminal_outcome == expected

    await operation.close()
    await driver.end_run()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [TimeoutError(), urllib.error.URLError(TimeoutError())],
)
async def test_priority_http_driver_classifies_timeouts(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    def fail(_request: urllib.request.Request, _timeout: float) -> _StaticResponse:
        raise error

    monkeypatch.setattr("vllm_nnrp_adapter.priority_burst_drivers._urlopen", fail)
    case = _case()
    driver = RawPriorityHttpSseDriver(_config())
    await driver.begin_run({"sample_count": 1}, (case,))
    operation = await driver.start(case)

    assert (await operation.wait()).terminal_outcome == "timed_out"

    await operation.close()
    await driver.end_run()


@pytest.mark.asyncio
async def test_priority_http_driver_classifies_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_request: urllib.request.Request, _timeout: float) -> _StaticResponse:
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr("vllm_nnrp_adapter.priority_burst_drivers._urlopen", fail)
    case = _case()
    driver = RawPriorityHttpSseDriver(_config())
    await driver.begin_run({"sample_count": 1}, (case,))
    operation = await driver.start(case)

    assert (await operation.wait()).terminal_outcome == "failed"

    await operation.close()
    await driver.end_run()


class _BlockingResponse:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        self.started.set()
        if not self.closed.wait(2.0):
            raise AssertionError("response was not closed")
        raise OSError("closed")
        yield b""  # pragma: no cover

    def close(self) -> None:
        self.closed.set()


@pytest.mark.asyncio
async def test_priority_http_driver_end_run_closes_active_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _BlockingResponse()
    monkeypatch.setattr(
        "vllm_nnrp_adapter.priority_burst_drivers._urlopen",
        lambda _request, _timeout: response,
    )
    case = _case()
    driver = RawPriorityHttpSseDriver(_config())
    await driver.begin_run({"sample_count": 1}, (case,))
    operation = await driver.start(case)
    wait_task = asyncio.create_task(operation.wait())
    assert await asyncio.to_thread(response.started.wait, 1.0)

    await driver.end_run()

    assert (await wait_task).terminal_outcome == "failed"
    assert response.closed.is_set()


@pytest.mark.asyncio
async def test_priority_http_driver_enforces_run_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case()
    driver = RawPriorityHttpSseDriver(_config())
    with pytest.raises(RuntimeError, match="has not begun"):
        await driver.start(case)
    with pytest.raises(ValueError, match="complete workload schedule"):
        await driver.begin_run({"sample_count": 2}, (case,))

    await driver.begin_run({"sample_count": 1}, (case,))
    with pytest.raises(RuntimeError, match="already active"):
        await driver.begin_run({"sample_count": 1}, (case,))
    monkeypatch.setattr(
        "vllm_nnrp_adapter.priority_burst_drivers._urlopen",
        lambda _request, _timeout: _StaticResponse(b"data: [DONE]\n"),
    )
    await driver.warmup(case)
    await driver.end_run()


@pytest.mark.asyncio
async def test_priority_http_driver_rejects_failed_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vllm_nnrp_adapter.priority_burst_drivers._urlopen",
        lambda _request, _timeout: _StaticResponse(b'data: {"choices":[]}\n'),
    )
    case = _case()
    driver = RawPriorityHttpSseDriver(_config())
    await driver.begin_run({"sample_count": 1}, (case,))

    with pytest.raises(RuntimeError, match="warmup request did not complete"):
        await driver.warmup(case)

    await driver.end_run()
