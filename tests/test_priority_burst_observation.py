from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from typing import Any

import pytest

from vllm_nnrp_adapter.priority_burst_observation import (
    HttpPriorityBurstObservationConfig,
    HttpPriorityBurstObservationProbe,
    _post_json,
)
from vllm_nnrp_adapter.priority_burst_workload import PriorityBurstCase, PriorityBurstResult


def _case() -> PriorityBurstCase:
    return PriorityBurstCase(
        sample_id="sample-000001",
        ordinal=1,
        scheduled_offset_seconds=0.1,
        model="public-test-model",
        prompt_tokens=8,
        max_completion_tokens=16,
        traffic_class="urgent",
        backend_priority=0,
    )


def _config() -> HttpPriorityBurstObservationConfig:
    return HttpPriorityBurstObservationConfig(
        endpoint="https://observation.example.invalid/v1/priority-burst",
        method="vllm_scheduler_trace",
        scope="dedicated_engine",
        source="deployment-scheduler-events",
        api_key="test-key",
        headers={"X-Deployment": "test"},
        timeout_seconds=12.0,
    )


def _finish_response(**overrides: object) -> dict[str, object]:
    response: dict[str, object] = {
        "baseline": "direct_nnrp",
        "sample_id": "sample-000001",
        "method": "vllm_scheduler_trace",
        "scope": "dedicated_engine",
        "source": "deployment-scheduler-events",
        "queued_after_seconds": 0.001,
        "backend_started_after_seconds": 0.01,
        "backend_completed_after_seconds": 0.05,
        "observed_backend_priority": 0,
        "queue_depth_at_submit": 4,
        "continuously_runnable": True,
    }
    response.update(overrides)
    return response


@pytest.mark.asyncio
async def test_http_priority_observation_enforces_correlated_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, object]] = []

    async def fake_post(
        config: HttpPriorityBurstObservationConfig,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        assert config == _config()
        request = dict(payload)
        requests.append(request)
        if request["action"] == "begin_run":
            return {"run_id": "run-1"}
        if request["action"] == "start_sample":
            return {"sample_token": "token-1"}
        if request["action"] == "finish_sample":
            return _finish_response()
        return {}

    monkeypatch.setattr("vllm_nnrp_adapter.priority_burst_observation._post_json_async", fake_post)
    case = _case()
    probe = HttpPriorityBurstObservationProbe(_config())
    workload = {"scenario": "priority_burst", "sample_count": 1}

    await probe.begin_run(workload, (case,))
    session = await probe.start_sample("direct_nnrp", case)
    await session.operation_submitted(123.5)
    observation = await session.finish(PriorityBurstResult("completed"))
    await session.close()
    await session.close()
    await probe.end_run()

    assert observation.queued_after_seconds == 0.001
    assert observation.backend_started_after_seconds == 0.01
    assert observation.backend_completed_after_seconds == 0.05
    assert observation.observed_backend_priority == 0
    assert observation.queue_depth_at_submit == 4
    assert observation.continuously_runnable is True
    assert [request["action"] for request in requests] == [
        "begin_run",
        "start_sample",
        "operation_submitted",
        "finish_sample",
        "close_sample",
        "end_run",
    ]
    assert requests[0]["schema_version"] == "nnrp-priority-burst-observation/v1"
    assert requests[0]["observation"] == {
        "method": "vllm_scheduler_trace",
        "scope": "dedicated_engine",
        "source": "deployment-scheduler-events",
    }
    assert requests[0]["workload"] == workload
    assert requests[0]["schedule"] == [
        {
            "sample_id": case.sample_id,
            "ordinal": 1,
            "scheduled_offset_seconds": 0.1,
            "model": "public-test-model",
            "prompt_tokens": 8,
            "max_completion_tokens": 16,
            "traffic_class": "urgent",
            "backend_priority": 0,
        }
    ]
    assert requests[2]["client_monotonic_seconds"] == 123.5
    assert requests[3]["result"] == {"terminal_outcome": "completed"}
    assert requests[4]["finished"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"baseline": "raw_openai_http_sse"}, "baseline does not match"),
        ({"sample_id": "wrong"}, "sample_id does not match"),
        ({"method": "engine_request_events"}, "method does not match"),
        ({"scope": "shared_engine"}, "scope does not match"),
        ({"source": "wrong"}, "source does not match"),
        ({"queued_after_seconds": -1}, "non-negative finite"),
        ({"backend_started_after_seconds": 0.0}, "backend start precedes"),
        ({"backend_completed_after_seconds": 0.0}, "backend completion precedes"),
        ({"observed_backend_priority": True}, "integer or null"),
        ({"queue_depth_at_submit": -1}, "non-negative integer"),
        ({"continuously_runnable": 1}, "must be a boolean"),
    ],
)
async def test_http_priority_observation_rejects_invalid_evidence(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    async def fake_post(
        _config: HttpPriorityBurstObservationConfig,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        if payload["action"] == "begin_run":
            return {"run_id": "run-1"}
        if payload["action"] == "start_sample":
            return {"sample_token": "token-1"}
        if payload["action"] == "finish_sample":
            return _finish_response(**overrides)
        return {}

    monkeypatch.setattr("vllm_nnrp_adapter.priority_burst_observation._post_json_async", fake_post)
    case = _case()
    probe = HttpPriorityBurstObservationProbe(_config())
    await probe.begin_run({"sample_count": 1}, (case,))
    session = await probe.start_sample("direct_nnrp", case)
    await session.operation_submitted(1.0)

    with pytest.raises(ValueError, match=message):
        await session.finish(PriorityBurstResult("completed"))

    await session.close()
    await probe.end_run()


@pytest.mark.asyncio
async def test_http_priority_observation_requires_ordered_state(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(
        _config: HttpPriorityBurstObservationConfig,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        if payload["action"] == "begin_run":
            return {"run_id": "run-1"}
        if payload["action"] == "start_sample":
            return {"sample_token": "token-1"}
        if payload["action"] == "finish_sample":
            return _finish_response()
        return {}

    monkeypatch.setattr("vllm_nnrp_adapter.priority_burst_observation._post_json_async", fake_post)
    case = _case()
    probe = HttpPriorityBurstObservationProbe(_config())
    with pytest.raises(RuntimeError, match="has not begun"):
        await probe.start_sample("direct_nnrp", case)
    await probe.begin_run({"sample_count": 1}, (case,))
    with pytest.raises(RuntimeError, match="already active"):
        await probe.begin_run({"sample_count": 1}, (case,))
    session = await probe.start_sample("direct_nnrp", case)
    with pytest.raises(RuntimeError, match="has not been submitted"):
        await session.finish(PriorityBurstResult("completed"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        await session.operation_submitted(-1.0)
    await session.operation_submitted(1.0)
    with pytest.raises(RuntimeError, match="already been submitted"):
        await session.operation_submitted(2.0)
    with pytest.raises(RuntimeError, match="sample sessions are active"):
        await probe.end_run()
    await session.finish(PriorityBurstResult("completed"))
    with pytest.raises(RuntimeError, match="already finished"):
        await session.finish(PriorityBurstResult("completed"))
    await session.close()
    with pytest.raises(RuntimeError, match="sample is closed"):
        await session.operation_submitted(3.0)
    await probe.end_run()
    with pytest.raises(RuntimeError, match="has not begun"):
        await probe.end_run()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"endpoint": "nnrp://observation"}, "absolute http"),
        ({"method": "wall_clock"}, "method must be"),
        ({"scope": "scheduled_batch"}, "scope must be"),
        ({"source": ""}, "source must be"),
        ({"api_key": ""}, "api_key"),
        ({"timeout_seconds": 0}, "finite and positive"),
        ({"headers": {"Bad:Name": "value"}}, "header name"),
        ({"headers": {"X-Test": "bad\nvalue"}}, "single-line"),
    ],
)
def test_http_priority_observation_config_rejects_ambiguous_settings(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "endpoint": "https://observation.example.invalid/v1/priority-burst",
        "method": "vllm_scheduler_trace",
        "scope": "dedicated_engine",
        "source": "deployment-scheduler-events",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        HttpPriorityBurstObservationConfig(**values)  # type: ignore[arg-type]


class _JsonResponse:
    def __init__(self, value: bytes) -> None:
        self.value = value

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def read(self) -> bytes:
        return self.value


def test_post_json_sends_authenticated_sidecar_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> _JsonResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _JsonResponse(b'{"run_id":"run-1"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = _post_json(_config(), {"action": "begin_run"})

    assert result == {"run_id": "run-1"}
    assert captured["timeout"] == 12.0
    request = captured["request"]
    assert request.full_url == _config().endpoint
    assert request.method == "POST"
    assert json.loads(request.data) == {"action": "begin_run"}
    assert dict(request.header_items()) == {
        "Accept": "application/json",
        "Authorization": "Bearer test-key",
        "Content-type": "application/json",
        "X-deployment": "test",
    }


@pytest.mark.parametrize(
    ("raw", "error_type", "message"),
    [
        (b"not-json", ValueError, "UTF-8 JSON"),
        (b"[]", TypeError, "JSON object"),
    ],
)
def test_post_json_rejects_invalid_sidecar_response(
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _request, *, timeout: _JsonResponse(raw),
    )
    with pytest.raises(error_type, match=message):
        _post_json(_config(), {"action": "begin_run"})
