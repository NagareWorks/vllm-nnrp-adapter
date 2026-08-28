from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from typing import Any

import pytest

from vllm_nnrp_adapter.adoption_evidence import GPU_ACCOUNTING_METHODS, GPU_ACCOUNTING_SCOPES
from vllm_nnrp_adapter.stale_work_accounting import (
    HttpAccountingProbeConfig,
    HttpStaleWorkAccountingProbe,
    _post_json,
)
from vllm_nnrp_adapter.stale_work_workload import StaleWorkCase, StaleWorkResult


def _case() -> StaleWorkCase:
    return StaleWorkCase(
        sample_id="sample-000001",
        ordinal=1,
        scheduled_offset_seconds=0.1,
        model="public-test-model",
        prompt_tokens=8,
        max_completion_tokens=16,
        control_kind="cancel",
        control_delay_seconds=0.01,
    )


def _config() -> HttpAccountingProbeConfig:
    return HttpAccountingProbeConfig(
        endpoint="https://accounting.example.invalid/v1/stale-work",
        method="cuda_event_attribution",
        scope="scheduled_batch",
        source="deployment-cuda-events",
        api_key="test-key",
        headers={"X-Deployment": "test"},
        timeout_seconds=12.0,
    )


@pytest.mark.asyncio
async def test_http_accounting_probe_enforces_correlated_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict[str, object]] = []

    async def fake_post(
        config: HttpAccountingProbeConfig,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        assert config == _config()
        request = dict(payload)
        requests.append(request)
        action = request["action"]
        if action == "begin_run":
            return {"run_id": "run-1"}
        if action == "start_sample":
            return {"sample_token": "sample-token-1"}
        if action == "finish_sample":
            return {
                "baseline": "direct_nnrp",
                "sample_id": "sample-000001",
                "method": "cuda_event_attribution",
                "scope": "scheduled_batch",
                "source": "deployment-cuda-events",
                "control_accepted": True,
                "control_accepted_after_seconds": 0.012,
                "backend_stopped_after_seconds": 0.018,
                "gpu_seconds": 0.015,
            }
        return {}

    monkeypatch.setattr("vllm_nnrp_adapter.stale_work_accounting._post_json_async", fake_post)
    case = _case()
    probe = HttpStaleWorkAccountingProbe(_config())
    workload = {"scenario": "stale_work", "sample_count": 1}

    await probe.begin_run(workload, (case,))
    session = await probe.start_sample("direct_nnrp", case)
    await session.operation_started(123.5)
    accounting = await session.finish(
        StaleWorkResult("cancelled", 0.0, 0),
        control_kind="cancel",
        control_dispatched=True,
    )
    await session.close()
    await probe.end_run()

    assert accounting.control_accepted is True
    assert accounting.control_accepted_after_seconds == 0.012
    assert accounting.backend_stopped_after_seconds == 0.018
    assert accounting.gpu_seconds == 0.015
    assert [request["action"] for request in requests] == [
        "begin_run",
        "start_sample",
        "operation_started",
        "finish_sample",
        "close_sample",
        "end_run",
    ]
    assert requests[0]["accounting"] == {
        "method": "cuda_event_attribution",
        "scope": "scheduled_batch",
        "source": "deployment-cuda-events",
    }
    assert requests[0]["workload"] == workload
    assert requests[0]["schedule"] == [
        {
            "sample_id": "sample-000001",
            "ordinal": 1,
            "scheduled_offset_seconds": 0.1,
            "model": "public-test-model",
            "prompt_tokens": 8,
            "max_completion_tokens": 16,
            "control_kind": "cancel",
            "control_delay_seconds": 0.01,
        }
    ]
    assert requests[2]["client_monotonic_seconds"] == 123.5
    assert requests[3]["result"] == {
        "terminal_outcome": "cancelled",
        "useful_result_weight": 0.0,
        "late_result_count": 0,
    }
    assert requests[4]["finished"] is True


@pytest.mark.asyncio
async def test_http_accounting_probe_rejects_mismatched_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(
        _config: HttpAccountingProbeConfig,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        if payload["action"] == "begin_run":
            return {"run_id": "run-1"}
        if payload["action"] == "start_sample":
            return {"sample_token": "sample-token-1"}
        if payload["action"] == "finish_sample":
            return {
                "baseline": "raw_openai_http_sse",
                "sample_id": "sample-000001",
                "method": "cuda_event_attribution",
                "scope": "scheduled_batch",
                "source": "deployment-cuda-events",
                "control_accepted": True,
                "control_accepted_after_seconds": 0.01,
                "backend_stopped_after_seconds": 0.02,
                "gpu_seconds": 0.01,
            }
        return {}

    monkeypatch.setattr("vllm_nnrp_adapter.stale_work_accounting._post_json_async", fake_post)
    case = _case()
    probe = HttpStaleWorkAccountingProbe(_config())
    await probe.begin_run({"sample_count": 1}, (case,))
    session = await probe.start_sample("direct_nnrp", case)
    await session.operation_started(1.0)

    with pytest.raises(ValueError, match="baseline does not match"):
        await session.finish(
            StaleWorkResult("cancelled", 0.0, 0),
            control_kind="cancel",
            control_dispatched=True,
        )
    await session.close()
    await probe.end_run()


@pytest.mark.asyncio
async def test_http_accounting_probe_requires_ordered_sample_state(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(
        _config: HttpAccountingProbeConfig,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        if payload["action"] == "begin_run":
            return {"run_id": "run-1"}
        if payload["action"] == "start_sample":
            return {"sample_token": "sample-token-1"}
        return {}

    monkeypatch.setattr("vllm_nnrp_adapter.stale_work_accounting._post_json_async", fake_post)
    case = _case()
    probe = HttpStaleWorkAccountingProbe(_config())
    await probe.begin_run({"sample_count": 1}, (case,))
    session = await probe.start_sample("direct_nnrp", case)

    with pytest.raises(RuntimeError, match="has not started"):
        await session.finish(
            StaleWorkResult("cancelled", 0.0, 0),
            control_kind="cancel",
            control_dispatched=True,
        )
    await session.operation_started(1.0)
    with pytest.raises(RuntimeError, match="already started"):
        await session.operation_started(2.0)
    with pytest.raises(RuntimeError, match="sample sessions are active"):
        await probe.end_run()
    await session.close()
    await probe.end_run()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"endpoint": "nnrp://accounting"}, "absolute http"),
        ({"method": "wall_clock"}, "method must be"),
        ({"scope": "host"}, "scope must be"),
        ({"source": ""}, "source must be"),
        ({"api_key": ""}, "api_key"),
        ({"timeout_seconds": 0}, "finite and positive"),
        ({"headers": {"Bad:Name": "value"}}, "header name"),
        ({"headers": {"X-Test": "bad\nvalue"}}, "single-line"),
    ],
)
def test_http_accounting_config_rejects_ambiguous_settings(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "endpoint": "https://accounting.example.invalid/v1/stale-work",
        "method": "cuda_event_attribution",
        "scope": "scheduled_batch",
        "source": "deployment-cuda-events",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        HttpAccountingProbeConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", GPU_ACCOUNTING_METHODS)
@pytest.mark.parametrize("scope", GPU_ACCOUNTING_SCOPES)
def test_http_accounting_config_uses_manifest_accounting_vocabulary(method: str, scope: str) -> None:
    config = HttpAccountingProbeConfig(
        endpoint="https://accounting.example.invalid/v1/stale-work",
        method=method,
        scope=scope,
        source="deployment-accounting",
    )

    assert config.method == method
    assert config.scope == scope


class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_http_accounting_transport_posts_private_configuration_without_persisting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 12.0
        requests.append(request)
        return _Response({"run_id": "run-1"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    response = _post_json(_config(), {"schema_version": "test/v1", "action": "begin_run"})

    assert response == {"run_id": "run-1"}
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == "https://accounting.example.invalid/v1/stale-work"
    assert request.method == "POST"
    assert dict(request.header_items()) == {
        "Accept": "application/json",
        "Authorization": "Bearer test-key",
        "Content-type": "application/json",
        "X-deployment": "test",
    }
    assert json.loads(request.data) == {"schema_version": "test/v1", "action": "begin_run"}


@pytest.mark.parametrize("payload", [[], "not-an-object", 1])
def test_http_accounting_transport_requires_object_response(
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
) -> None:
    def fake_urlopen(_request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 12.0
        return _Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(TypeError, match="JSON object"):
        _post_json(_config(), {"action": "begin_run"})
