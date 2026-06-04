from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from nnrp.client import TypedPayload
from nnrp.core import HeaderFlags, ResultClass, ResultFlags

from .adapter import OpenAiNnrpAdapter

TERMINAL_EVENT_TYPES = frozenset({"response.completed", "response.error", "response.cancelled"})
PROFILE_EVENT_MEDIA_TYPE = "application/vnd.nnrp.openai-compatible.event+json"


class NnrpResultSession(Protocol):
    async def send_result(
        self,
        *,
        frame_id: int,
        typed_payloads: tuple[TypedPayload, ...] = (),
        result_flags: ResultFlags = ResultFlags.NONE,
        result_class: ResultClass = ResultClass.COMPLETE,
        status_code: int = 0,
        flags: HeaderFlags = HeaderFlags.NONE,
        view_id: int = 0,
        route_id: int = 0,
        trace_id: int = 0,
        active_profile_id: int = 0,
        inference_ms: int = 0,
        queue_ms: int = 0,
        server_total_ms: int = 0,
    ) -> int:
        pass


@dataclass(frozen=True, slots=True)
class NnrpFrameContext:
    frame_id: int
    view_id: int = 0
    route_id: int = 0
    trace_id: int = 0
    active_profile_id: int = 0


@dataclass(frozen=True, slots=True)
class EmittedNnrpResult:
    event: Mapping[str, Any]
    stream_id: int
    terminal: bool
    status_code: int


async def emit_openai_profile_results(
    adapter: OpenAiNnrpAdapter,
    session: NnrpResultSession,
    request: Mapping[str, Any],
    *,
    frame: NnrpFrameContext,
) -> list[EmittedNnrpResult]:
    emitted: list[EmittedNnrpResult] = []
    async for event in adapter.handle_request(request):
        emitted.append(await emit_profile_event(session, event, frame=frame))
    return emitted


async def emit_profile_event(
    session: NnrpResultSession,
    event: Mapping[str, Any],
    *,
    frame: NnrpFrameContext,
) -> EmittedNnrpResult:
    terminal = is_terminal_profile_event(event)
    status_code = status_code_for_profile_event(event)
    stream_id = await session.send_result(
        frame_id=frame.frame_id,
        typed_payloads=(TypedPayload.structured_event(encode_profile_event(event)),),
        result_flags=ResultFlags.NONE if terminal else ResultFlags.PARTIAL,
        result_class=ResultClass.COMPLETE if terminal else ResultClass.PARTIAL,
        status_code=status_code,
        flags=HeaderFlags.NONE,
        view_id=frame.view_id,
        route_id=frame.route_id,
        trace_id=frame.trace_id,
        active_profile_id=frame.active_profile_id,
    )
    return EmittedNnrpResult(event=event, stream_id=stream_id, terminal=terminal, status_code=status_code)


def encode_profile_event(event: Mapping[str, Any]) -> bytes:
    envelope = {
        "media_type": PROFILE_EVENT_MEDIA_TYPE,
        "event": dict(event),
    }
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_profile_event(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("profile event payload must decode to a JSON object")
    if value.get("media_type") != PROFILE_EVENT_MEDIA_TYPE:
        raise ValueError("profile event payload media_type is not the OpenAI-compatible event media type")
    event = value.get("event")
    if not isinstance(event, dict):
        raise ValueError("profile event payload must carry an event object")
    return event


def is_terminal_profile_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") in TERMINAL_EVENT_TYPES


def status_code_for_profile_event(event: Mapping[str, Any]) -> int:
    event_type = event.get("type")
    if event_type == "response.error":
        error = event.get("error")
        code = error.get("code") if isinstance(error, Mapping) else None
        if code == "request_timeout":
            return 504
        if code in {"backend_overload", "scheduler_rejected"}:
            return 503
        return 500
    if event_type == "response.cancelled":
        return 499
    return 200
