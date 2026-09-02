from __future__ import annotations

import asyncio
import json
import math
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import cast

from .priority_burst_workload import (
    PRIORITY_BURST_SCHEDULER_METHODS,
    PRIORITY_BURST_SCHEDULER_SCOPES,
    PriorityBurstCase,
    PriorityBurstObservationResult,
    PriorityBurstResult,
)

_SCHEMA_VERSION = "nnrp-priority-burst-observation/v1"


@dataclass(frozen=True)
class HttpPriorityBurstObservationConfig:
    endpoint: str
    method: str
    scope: str
    source: str
    api_key: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute http:// or https:// URL")
        if self.method not in PRIORITY_BURST_SCHEDULER_METHODS:
            raise ValueError(f"method must be one of {', '.join(PRIORITY_BURST_SCHEDULER_METHODS)}")
        if self.scope not in PRIORITY_BURST_SCHEDULER_SCOPES:
            raise ValueError(f"scope must be one of {', '.join(PRIORITY_BURST_SCHEDULER_SCOPES)}")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be non-empty")
        if self.api_key is not None and not self.api_key:
            raise ValueError("api_key must be non-empty when provided")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        copied_headers: dict[str, str] = {}
        for name, value in self.headers.items():
            _validate_header(name, "headers name")
            if not isinstance(value, str) or not value or "\r" in value or "\n" in value:
                raise ValueError("headers values must be non-empty single-line strings")
            copied_headers[name] = value
        object.__setattr__(self, "headers", MappingProxyType(copied_headers))


class HttpPriorityBurstObservationProbe:
    """Independent scheduler observation backed by a deployment-owned sidecar."""

    def __init__(self, config: HttpPriorityBurstObservationConfig) -> None:
        self._config = config
        self.method = config.method
        self.scope = config.scope
        self.source = config.source
        self._run_id: str | None = None
        self._sessions: set[_HttpPriorityBurstObservationSession] = set()

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[PriorityBurstCase],
    ) -> None:
        if self._run_id is not None or self._sessions:
            raise RuntimeError("HTTP priority observation run is already active")
        response = await self._request(
            "begin_run",
            {
                "observation": self._identity(),
                "workload": deepcopy(dict(workload)),
                "schedule": [asdict(case) for case in schedule],
            },
        )
        self._run_id = _required_string(response, "run_id", "begin_run response")

    async def start_sample(
        self,
        baseline: str,
        case: PriorityBurstCase,
    ) -> _HttpPriorityBurstObservationSession:
        run_id = self._require_run()
        response = await self._request(
            "start_sample",
            {"run_id": run_id, "baseline": baseline, "sample": asdict(case)},
        )
        session = _HttpPriorityBurstObservationSession(
            probe=self,
            run_id=run_id,
            sample_token=_required_string(response, "sample_token", "start_sample response"),
            baseline=baseline,
            case=case,
        )
        self._sessions.add(session)
        return session

    async def end_run(self) -> None:
        run_id = self._require_run()
        if self._sessions:
            raise RuntimeError("HTTP priority observation cannot end while sample sessions are active")
        await self._request("end_run", {"run_id": run_id})
        self._run_id = None

    async def _request(self, action: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        return await _post_json_async(
            self._config,
            {"schema_version": _SCHEMA_VERSION, "action": action, **payload},
        )

    def _identity(self) -> dict[str, str]:
        return {"method": self.method, "scope": self.scope, "source": self.source}

    def _require_run(self) -> str:
        if self._run_id is None:
            raise RuntimeError("HTTP priority observation run has not begun")
        return self._run_id

    def _session_closed(self, session: _HttpPriorityBurstObservationSession) -> None:
        self._sessions.discard(session)


class _HttpPriorityBurstObservationSession:
    def __init__(
        self,
        *,
        probe: HttpPriorityBurstObservationProbe,
        run_id: str,
        sample_token: str,
        baseline: str,
        case: PriorityBurstCase,
    ) -> None:
        self._probe = probe
        self._run_id = run_id
        self._sample_token = sample_token
        self._baseline = baseline
        self._case = case
        self._submitted_at: float | None = None
        self._finished = False
        self._closed = False

    async def operation_submitted(self, monotonic_seconds: float) -> None:
        self._ensure_open()
        if self._submitted_at is not None:
            raise RuntimeError("priority observation operation has already been submitted")
        if not math.isfinite(monotonic_seconds) or monotonic_seconds < 0:
            raise ValueError("monotonic_seconds must be finite and non-negative")
        await self._probe._request(
            "operation_submitted",
            {
                "run_id": self._run_id,
                "sample_token": self._sample_token,
                "baseline": self._baseline,
                "sample_id": self._case.sample_id,
                "client_monotonic_seconds": monotonic_seconds,
            },
        )
        self._submitted_at = monotonic_seconds

    async def finish(self, result: PriorityBurstResult) -> PriorityBurstObservationResult:
        self._ensure_open()
        if self._submitted_at is None:
            raise RuntimeError("priority observation operation has not been submitted")
        if self._finished:
            raise RuntimeError("priority observation sample has already finished")
        response = await self._probe._request(
            "finish_sample",
            {
                "run_id": self._run_id,
                "sample_token": self._sample_token,
                "baseline": self._baseline,
                "sample_id": self._case.sample_id,
                "result": asdict(result),
            },
        )
        _validate_identity(
            response,
            baseline=self._baseline,
            sample_id=self._case.sample_id,
            method=self._probe.method,
            scope=self._probe.scope,
            source=self._probe.source,
        )
        observation = PriorityBurstObservationResult(
            queued_after_seconds=_required_non_negative_number(
                response, "queued_after_seconds", "finish_sample response"
            ),
            backend_started_after_seconds=_required_non_negative_number(
                response, "backend_started_after_seconds", "finish_sample response"
            ),
            backend_completed_after_seconds=_required_non_negative_number(
                response, "backend_completed_after_seconds", "finish_sample response"
            ),
            observed_backend_priority=_optional_int(
                response, "observed_backend_priority", "finish_sample response"
            ),
            queue_depth_at_submit=_required_non_negative_int(
                response, "queue_depth_at_submit", "finish_sample response"
            ),
            continuously_runnable=_required_bool(
                response, "continuously_runnable", "finish_sample response"
            ),
        )
        if observation.backend_started_after_seconds < observation.queued_after_seconds:
            raise ValueError("finish_sample response backend start precedes queue observation")
        if observation.backend_completed_after_seconds < observation.backend_started_after_seconds:
            raise ValueError("finish_sample response backend completion precedes backend start")
        self._finished = True
        return observation

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await self._probe._request(
                "close_sample",
                {
                    "run_id": self._run_id,
                    "sample_token": self._sample_token,
                    "baseline": self._baseline,
                    "sample_id": self._case.sample_id,
                    "finished": self._finished,
                },
            )
        finally:
            self._closed = True
            self._probe._session_closed(self)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("priority observation sample is closed")


async def _post_json_async(
    config: HttpPriorityBurstObservationConfig,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    return await asyncio.to_thread(_post_json, config, payload)


def _post_json(
    config: HttpPriorityBurstObservationConfig,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    headers = {"Accept": "application/json", "Content-Type": "application/json", **config.headers}
    if config.api_key is not None:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(
        config.endpoint,
        data=json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:  # noqa: S310
        raw = response.read()
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("priority observation sidecar response must be UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise TypeError("priority observation sidecar response must be a JSON object")
    return cast(dict[str, object], decoded)


def _validate_identity(
    response: Mapping[str, object],
    *,
    baseline: str,
    sample_id: str,
    method: str,
    scope: str,
    source: str,
) -> None:
    expected = {
        "baseline": baseline,
        "sample_id": sample_id,
        "method": method,
        "scope": scope,
        "source": source,
    }
    for name, value in expected.items():
        if response.get(name) != value:
            raise ValueError(f"finish_sample response {name} does not match the active observation sample")


def _required_string(value: Mapping[str, object], field: str, location: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{location}.{field} must be a non-empty string")
    return result


def _required_bool(value: Mapping[str, object], field: str, location: str) -> bool:
    result = value.get(field)
    if not isinstance(result, bool):
        raise ValueError(f"{location}.{field} must be a boolean")
    return result


def _required_non_negative_number(value: Mapping[str, object], field: str, location: str) -> float:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise ValueError(f"{location}.{field} must be a non-negative finite number")
    numeric = float(result)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{location}.{field} must be a non-negative finite number")
    return numeric


def _required_non_negative_int(value: Mapping[str, object], field: str, location: str) -> int:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError(f"{location}.{field} must be a non-negative integer")
    return result


def _optional_int(value: Mapping[str, object], field: str, location: str) -> int | None:
    result = value.get(field)
    if result is None:
        return None
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{location}.{field} must be an integer or null")
    return result


def _validate_header(value: object, location: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or ":" in value
        or "\r" in value
        or "\n" in value
        or value != value.strip()
    ):
        raise ValueError(f"{location} must be a non-empty HTTP header name")
