from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from collections.abc import Iterator

import pytest

from vllm_nnrp_adapter.stale_work_drivers import OpenAiHttpSseDriverConfig, RawOpenAiHttpSseDriver
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
