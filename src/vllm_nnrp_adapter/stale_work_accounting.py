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

from .adoption_evidence import GPU_ACCOUNTING_METHODS, GPU_ACCOUNTING_SCOPES
from .stale_work_workload import (
    StaleWorkAccountingResult,
    StaleWorkCase,
    StaleWorkResult,
)

_SCHEMA_VERSION = "nnrp-stale-work-accounting/v1"


@dataclass(frozen=True)
class HttpAccountingProbeConfig:
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
        if self.method not in GPU_ACCOUNTING_METHODS:
            raise ValueError(f"method must be one of {', '.join(GPU_ACCOUNTING_METHODS)}")
        if self.scope not in GPU_ACCOUNTING_SCOPES:
            raise ValueError(f"scope must be one of {', '.join(GPU_ACCOUNTING_SCOPES)}")
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


class HttpStaleWorkAccountingProbe:
    """Independent stale-work accounting backed by a deployment-owned HTTP sidecar."""

    def __init__(self, config: HttpAccountingProbeConfig) -> None:
        self._config = config
        self.method = config.method
        self.scope = config.scope
        self.source = config.source
        self._run_id: str | None = None
        self._sessions: set[_HttpStaleWorkAccountingSession] = set()

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[StaleWorkCase],
    ) -> None:
        if self._run_id is not None or self._sessions:
            raise RuntimeError("HTTP accounting probe run is already active")
        response = await self._request(
            "begin_run",
            {
                "accounting": self._accounting_identity(),
                "workload": deepcopy(dict(workload)),
                "schedule": [_case_payload(case) for case in schedule],
            },
        )
        self._run_id = _required_string(response, "run_id", "begin_run response")

    async def start_sample(
        self,
        baseline: str,
        case: StaleWorkCase,
    ) -> _HttpStaleWorkAccountingSession:
        run_id = self._require_run()
        response = await self._request(
            "start_sample",
            {
                "run_id": run_id,
                "baseline": baseline,
                "sample": _case_payload(case),
            },
        )
        session = _HttpStaleWorkAccountingSession(
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
            raise RuntimeError("HTTP accounting probe cannot end while sample sessions are active")
        await self._request("end_run", {"run_id": run_id})
        self._run_id = None

    async def _request(self, action: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        request_payload = {
            "schema_version": _SCHEMA_VERSION,
            "action": action,
            **payload,
        }
        return await _post_json_async(self._config, request_payload)

    def _accounting_identity(self) -> dict[str, str]:
        return {
            "method": self.method,
            "scope": self.scope,
            "source": self.source,
        }

    def _require_run(self) -> str:
        if self._run_id is None:
            raise RuntimeError("HTTP accounting probe run has not begun")
        return self._run_id

    def _session_closed(self, session: _HttpStaleWorkAccountingSession) -> None:
        self._sessions.discard(session)


class _HttpStaleWorkAccountingSession:
    def __init__(
        self,
        *,
        probe: HttpStaleWorkAccountingProbe,
        run_id: str,
        sample_token: str,
        baseline: str,
        case: StaleWorkCase,
    ) -> None:
        self._probe = probe
        self._run_id = run_id
        self._sample_token = sample_token
        self._baseline = baseline
        self._case = case
        self._operation_started_at: float | None = None
        self._finished = False
        self._closed = False

    async def operation_started(self, monotonic_seconds: float) -> None:
        self._ensure_open()
        if self._operation_started_at is not None:
            raise RuntimeError("accounting sample operation has already started")
        if not math.isfinite(monotonic_seconds) or monotonic_seconds < 0:
            raise ValueError("monotonic_seconds must be finite and non-negative")
        await self._probe._request(
            "operation_started",
            {
                "run_id": self._run_id,
                "sample_token": self._sample_token,
                "baseline": self._baseline,
                "sample_id": self._case.sample_id,
                "client_monotonic_seconds": monotonic_seconds,
            },
        )
        self._operation_started_at = monotonic_seconds

    async def finish(
        self,
        result: StaleWorkResult,
        *,
        control_kind: str | None,
        control_dispatched: bool,
    ) -> StaleWorkAccountingResult:
        self._ensure_open()
        if self._operation_started_at is None:
            raise RuntimeError("accounting sample operation has not started")
        if self._finished:
            raise RuntimeError("accounting sample has already finished")
        response = await self._probe._request(
            "finish_sample",
            {
                "run_id": self._run_id,
                "sample_token": self._sample_token,
                "baseline": self._baseline,
                "sample_id": self._case.sample_id,
                "control_kind": control_kind,
                "control_dispatched": control_dispatched,
                "result": asdict(result),
            },
        )
        _validate_observation_identity(
            response,
            baseline=self._baseline,
            sample_id=self._case.sample_id,
            method=self._probe.method,
            scope=self._probe.scope,
            source=self._probe.source,
        )
        accounting = StaleWorkAccountingResult(
            control_accepted=_required_bool(response, "control_accepted", "finish_sample response"),
            control_accepted_after_seconds=_optional_non_negative_number(
                response,
                "control_accepted_after_seconds",
                "finish_sample response",
            ),
            backend_stopped_after_seconds=_required_non_negative_number(
                response,
                "backend_stopped_after_seconds",
                "finish_sample response",
            ),
            gpu_seconds=_required_non_negative_number(response, "gpu_seconds", "finish_sample response"),
        )
        self._finished = True
        return accounting

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
            raise RuntimeError("accounting sample is closed")


async def _post_json_async(
    config: HttpAccountingProbeConfig,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    return await asyncio.to_thread(_post_json, config, payload)


def _post_json(
    config: HttpAccountingProbeConfig,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        **config.headers,
    }
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
        raise ValueError("accounting sidecar response must be UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise TypeError("accounting sidecar response must be a JSON object")
    return cast(dict[str, object], decoded)


def _case_payload(case: StaleWorkCase) -> dict[str, object]:
    return asdict(case)


def _validate_observation_identity(
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
            raise ValueError(f"finish_sample response {name} does not match the active accounting sample")


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


def _optional_non_negative_number(
    value: Mapping[str, object],
    field: str,
    location: str,
) -> float | None:
    if value.get(field) is None:
        return None
    return _required_non_negative_number(value, field, location)


def _validate_header(value: str, location: str) -> None:
    if not isinstance(value, str) or not value or ":" in value or "\r" in value or "\n" in value:
        raise ValueError(f"{location} must be a valid HTTP header name")
