from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, cast

from .benchmark import synthetic_prompt
from .stale_work_workload import StaleWorkCase, StaleWorkResult

_CONTROL_KINDS = frozenset(("cancel", "abort", "deadline", "supersede"))


class _HttpResponse(Protocol):
    def __iter__(self) -> Iterator[bytes]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class OpenAiHttpSseDriverConfig:
    endpoint: str
    api_key: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    sample_id_header: str = "X-NNRP-Benchmark-Sample-Id"

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute http:// or https:// URL")
        if self.api_key is not None and not self.api_key:
            raise ValueError("api_key must be non-empty when provided")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        _validate_header(self.sample_id_header, "sample_id_header")
        copied_headers: dict[str, str] = {}
        for name, value in self.headers.items():
            _validate_header(name, "headers name")
            if not isinstance(value, str) or not value or "\r" in value or "\n" in value:
                raise ValueError("headers values must be non-empty single-line strings")
            copied_headers[name] = value
        object.__setattr__(self, "headers", MappingProxyType(copied_headers))


class RawOpenAiHttpSseDriver:
    """Raw OpenAI HTTP/SSE baseline whose only control is client disconnect."""

    baseline = "raw_openai_http_sse"

    def __init__(self, config: OpenAiHttpSseDriverConfig) -> None:
        self._config = config
        self._active: set[_RawOpenAiHttpSseOperation] = set()
        self._begun = False

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[StaleWorkCase],
    ) -> None:
        if self._begun or self._active:
            raise RuntimeError("raw HTTP/SSE driver run is already active")
        if not schedule or len(schedule) != workload.get("sample_count"):
            raise ValueError("raw HTTP/SSE driver requires the complete workload schedule")
        self._begun = True

    async def warmup(self, case: StaleWorkCase) -> None:
        operation = await self.start(case)
        try:
            result = await operation.wait()
            if result.terminal_outcome != "completed":
                raise RuntimeError("raw HTTP/SSE warmup request did not complete")
        finally:
            await operation.close()

    async def start(self, case: StaleWorkCase) -> _RawOpenAiHttpSseOperation:
        if not self._begun:
            raise RuntimeError("raw HTTP/SSE driver has not begun a run")
        operation = _RawOpenAiHttpSseOperation(self._config, case, self._active.discard)
        self._active.add(operation)
        return operation

    async def end_run(self) -> None:
        operations = tuple(self._active)
        if operations:
            await asyncio.gather(*(operation.close() for operation in operations))
        self._begun = False


class _RawOpenAiHttpSseOperation:
    def __init__(
        self,
        config: OpenAiHttpSseDriverConfig,
        case: StaleWorkCase,
        release: Callable[[_RawOpenAiHttpSseOperation], None],
    ) -> None:
        self._config = config
        self._case = case
        self._release = release
        self._state_lock = threading.Lock()
        self._wait_lock = asyncio.Lock()
        self._response: _HttpResponse | None = None
        self._control_kind: str | None = None
        self._finished = False
        self._closed = False
        self._result: StaleWorkResult | None = None

    async def apply_control(self, control_kind: str) -> bool:
        if control_kind not in _CONTROL_KINDS:
            raise ValueError(f"unsupported stale-work control kind: {control_kind}")
        with self._state_lock:
            if self._finished or self._closed or self._control_kind is not None:
                return False
            self._control_kind = control_kind
            response = self._response
        if response is not None:
            await asyncio.to_thread(_close_response, response)
        return True

    async def wait(self) -> StaleWorkResult:
        async with self._wait_lock:
            if self._result is None:
                self._result = await asyncio.to_thread(self._run_sync)
            return self._result

    async def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            response = self._response
        if response is not None:
            await asyncio.to_thread(_close_response, response)
        self._release(self)

    def _run_sync(self) -> StaleWorkResult:
        request = urllib.request.Request(
            self._config.endpoint,
            data=_request_body(self._case),
            headers=_request_headers(self._config, self._case),
            method="POST",
        )
        response: _HttpResponse | None = None
        stream_completed = False
        late_result_count = 0
        try:
            response = _urlopen(request, self._config.timeout_seconds)
            with self._state_lock:
                self._response = response
                close_immediately = self._control_kind is not None or self._closed
            if close_immediately:
                _close_response(response)
            for raw_line in response:
                data = _sse_data(raw_line)
                if data is None:
                    continue
                if data == "[DONE]":
                    stream_completed = True
                    break
                parsed = json.loads(data)
                if not isinstance(parsed, Mapping):
                    raise ValueError("OpenAI SSE data must decode to an object")
                with self._state_lock:
                    if self._control_kind is not None:
                        late_result_count += 1
        except (OSError, urllib.error.URLError, UnicodeDecodeError, ValueError):
            terminal_outcome = "cancelled" if self._control_requested() else "failed"
        else:
            terminal_outcome = "completed" if stream_completed or not self._control_requested() else "cancelled"
        finally:
            if response is not None:
                _close_response(response)
            with self._state_lock:
                self._response = None
                self._finished = True

        return StaleWorkResult(
            terminal_outcome=terminal_outcome,
            useful_result_weight=0.0 if self._case.is_stale else float(terminal_outcome == "completed"),
            late_result_count=late_result_count,
        )

    def _control_requested(self) -> bool:
        with self._state_lock:
            return self._control_kind is not None


def _request_body(case: StaleWorkCase) -> bytes:
    return json.dumps(
        {
            "model": case.model,
            "messages": [{"role": "user", "content": synthetic_prompt(case.prompt_tokens)}],
            "stream": True,
            "max_tokens": case.max_completion_tokens,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _request_headers(config: OpenAiHttpSseDriverConfig, case: StaleWorkCase) -> dict[str, str]:
    headers = dict(config.headers)
    headers["Content-Type"] = "application/json"
    headers[config.sample_id_header] = case.sample_id
    if config.api_key is not None:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def _sse_data(raw_line: bytes) -> str | None:
    line = raw_line.decode("utf-8").strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[5:].strip()
    return data or None


def _urlopen(request: urllib.request.Request, timeout_seconds: float) -> _HttpResponse:
    return cast(_HttpResponse, urllib.request.urlopen(request, timeout=timeout_seconds))


def _close_response(response: _HttpResponse) -> None:
    try:
        response.close()
    except OSError:
        pass


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
