from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Protocol, cast

from .benchmark import synthetic_prompt
from .openai_http_sse import OpenAiHttpSseDriverConfig
from .priority_burst_workload import PriorityBurstCase, PriorityBurstResult


class _HttpResponse(Protocol):
    def __iter__(self) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class RawPriorityHttpSseDriver:
    """OpenAI HTTP/SSE baseline that deliberately omits vLLM priority."""

    baseline = "raw_openai_http_sse"

    def __init__(self, config: OpenAiHttpSseDriverConfig) -> None:
        self._config = config
        self._active: set[_PriorityHttpSseOperation] = set()
        self._begun = False

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[PriorityBurstCase],
    ) -> None:
        if self._begun or self._active:
            raise RuntimeError(f"{self.baseline} driver run is already active")
        if not schedule or len(schedule) != workload.get("sample_count"):
            raise ValueError(f"{self.baseline} driver requires the complete workload schedule")
        self._begun = True

    async def warmup(self, case: PriorityBurstCase) -> None:
        operation = await self.start(case)
        try:
            result = await operation.wait()
            if result.terminal_outcome != "completed":
                raise RuntimeError(f"{self.baseline} warmup request did not complete")
        finally:
            await operation.close()

    async def start(self, case: PriorityBurstCase) -> _PriorityHttpSseOperation:
        return self._start(case, include_priority=False)

    async def end_run(self) -> None:
        operations = tuple(self._active)
        if operations:
            await asyncio.gather(*(operation.close() for operation in operations))
        self._begun = False

    def _start(self, case: PriorityBurstCase, *, include_priority: bool) -> _PriorityHttpSseOperation:
        if not self._begun:
            raise RuntimeError(f"{self.baseline} driver run has not begun")
        operation = _PriorityHttpSseOperation(
            self._config,
            case,
            include_priority=include_priority,
            release=self._active.discard,
        )
        self._active.add(operation)
        return operation


class OrchestratedPriorityHttpSseDriver(RawPriorityHttpSseDriver):
    """HTTP/SSE baseline that applies vLLM priority atomically at admission."""

    baseline = "orchestrated_http_sse"

    async def start(self, case: PriorityBurstCase) -> _PriorityHttpSseOperation:
        return self._start(case, include_priority=True)


class _PriorityHttpSseOperation:
    def __init__(
        self,
        config: OpenAiHttpSseDriverConfig,
        case: PriorityBurstCase,
        *,
        include_priority: bool,
        release: Callable[[_PriorityHttpSseOperation], None],
    ) -> None:
        self._config = config
        self._case = case
        self._include_priority = include_priority
        self._release = release
        self._state_lock = threading.Lock()
        self._wait_lock = asyncio.Lock()
        self._response: _HttpResponse | None = None
        self._closed = False
        self._result: PriorityBurstResult | None = None

    async def wait(self) -> PriorityBurstResult:
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

    def _run_sync(self) -> PriorityBurstResult:
        request = urllib.request.Request(
            self._config.endpoint,
            data=_request_body(self._case, include_priority=self._include_priority),
            headers=_request_headers(self._config, self._case),
            method="POST",
        )
        response: _HttpResponse | None = None
        stream_completed = False
        try:
            response = _urlopen(request, self._config.timeout_seconds)
            with self._state_lock:
                self._response = response
                close_immediately = self._closed
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
        except TimeoutError:
            outcome = "timed_out"
        except urllib.error.URLError as error:
            outcome = "timed_out" if isinstance(error.reason, TimeoutError) else "failed"
        except (UnicodeDecodeError, ValueError, OSError):
            outcome = "failed"
        else:
            outcome = "completed" if stream_completed else "failed"
        finally:
            if response is not None:
                _close_response(response)
            with self._state_lock:
                self._response = None
        return PriorityBurstResult(outcome)


def _request_body(case: PriorityBurstCase, *, include_priority: bool) -> bytes:
    body: dict[str, object] = {
        "model": case.model,
        "messages": [{"role": "user", "content": synthetic_prompt(case.prompt_tokens)}],
        "stream": True,
        "max_tokens": case.max_completion_tokens,
        "stream_options": {"include_usage": True},
    }
    if include_priority:
        body["priority"] = case.backend_priority
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def _request_headers(
    config: OpenAiHttpSseDriverConfig,
    case: PriorityBurstCase,
) -> dict[str, str]:
    headers = dict(config.headers)
    headers["Accept"] = "text/event-stream"
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
