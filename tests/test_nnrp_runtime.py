import asyncio
import json
import socket
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import pytest
from nnrp.adapters import create_tcp_server_configuration, serve_tcp
from nnrp.client import SubmitRequest, TypedPayload
from nnrp.client.transport import connect_client_session
from nnrp.core import HeaderFlags, ResultClass, ResultFlags, TransportId
from nnrp.server import accept_server_session

from vllm_nnrp_adapter import OpenAiNnrpAdapter
from vllm_nnrp_adapter.nnrp_runtime import (
    NnrpFrameContext,
    decode_profile_event,
    emit_openai_profile_results,
    emit_profile_event,
    encode_profile_event,
    is_terminal_profile_event,
    status_code_for_profile_event,
)
from vllm_nnrp_adapter.profile import CHAT_COMPLETIONS_CREATE, OPENAI_COMPATIBLE_SCHEMA_VERSION


class StreamingBackend:
    def create_chat_completion(self, body: Mapping[str, Any]) -> AsyncIterator[Mapping[str, Any]]:
        async def chunks() -> AsyncIterator[Mapping[str, Any]]:
            yield {"choices": [{"index": 0, "delta": {"content": body["messages"][0]["content"]}}]}
            yield {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        return chunks()


class ErrorBackend:
    def create_chat_completion(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        raise RuntimeError("backend overload: queue is full")


@dataclass
class SentResult:
    frame_id: int
    typed_payloads: tuple[TypedPayload, ...]
    result_flags: ResultFlags
    result_class: ResultClass
    status_code: int
    flags: HeaderFlags
    view_id: int
    route_id: int
    trace_id: int
    active_profile_id: int


class RecordingResultSession:
    def __init__(self) -> None:
        self.sent: list[SentResult] = []

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
        assert inference_ms == 0
        assert queue_ms == 0
        assert server_total_ms == 0
        self.sent.append(
            SentResult(
                frame_id=frame_id,
                typed_payloads=typed_payloads,
                result_flags=result_flags,
                result_class=result_class,
                status_code=status_code,
                flags=flags,
                view_id=view_id,
                route_id=route_id,
                trace_id=trace_id,
                active_profile_id=active_profile_id,
            )
        )
        return len(self.sent) * 4 + 2


@pytest.mark.asyncio
async def test_emit_profile_event_writes_structured_result_push_shape() -> None:
    session = RecordingResultSession()
    event = {"type": "response.output_text.delta", "delta": "hello", "index": 0}

    emitted = await emit_profile_event(
        session,
        event,
        frame=NnrpFrameContext(frame_id=17, view_id=2, route_id=3, trace_id=4, active_profile_id=5),
    )

    assert emitted.stream_id == 6
    assert emitted.terminal is False
    assert emitted.status_code == 200
    assert session.sent[0].frame_id == 17
    assert session.sent[0].result_flags is ResultFlags.PARTIAL
    assert session.sent[0].result_class is ResultClass.PARTIAL
    assert session.sent[0].flags is HeaderFlags.NONE
    assert session.sent[0].view_id == 2
    assert session.sent[0].route_id == 3
    assert session.sent[0].trace_id == 4
    assert session.sent[0].active_profile_id == 5
    assert decode_profile_event(session.sent[0].typed_payloads[0].payload) == event


@pytest.mark.asyncio
async def test_emit_openai_profile_results_marks_error_terminal() -> None:
    session = RecordingResultSession()
    emitted = await emit_openai_profile_results(
        OpenAiNnrpAdapter(ErrorBackend()),
        session,
        _chat_request(stream=False),
        frame=NnrpFrameContext(frame_id=23),
    )

    assert len(emitted) == 1
    assert emitted[0].terminal is True
    assert emitted[0].status_code == 503
    assert session.sent[0].result_class is ResultClass.COMPLETE
    assert session.sent[0].flags is HeaderFlags.NONE
    assert decode_profile_event(session.sent[0].typed_payloads[0].payload)["error"]["code"] == "backend_overload"


@pytest.mark.asyncio
async def test_emit_openai_profile_results_streams_through_real_nnrp_tcp_session() -> None:
    host = "127.0.0.1"
    port = _reserve_port()
    server_done = asyncio.Event()
    adapter = OpenAiNnrpAdapter(StreamingBackend())

    async with serve_tcp(host, port, configuration=create_tcp_server_configuration()) as listener:

        async def run_server() -> None:
            session = await accept_server_session(listener, session_id=77, active_model_name="llama")
            try:
                submit = await session.receive_submit(timeout=5.0)
                payload = json.loads(submit.request.typed_payloads[0].payload.decode("utf-8"))
                await emit_openai_profile_results(
                    adapter,
                    session,
                    payload,
                    frame=NnrpFrameContext(frame_id=submit.request.frame_id, trace_id=submit.packet.header.trace_id),
                )
                await asyncio.sleep(0.05)
            finally:
                await session.close()
                server_done.set()

        server_task = asyncio.create_task(run_server())
        try:
            async with connect_client_session(
                host,
                tcp_port=port,
                requested_model="llama",
                selected_transport_id=TransportId.TCP,
            ) as client:
                await client.send_submit(
                    SubmitRequest(
                        frame_id=31,
                        typed_payloads=(TypedPayload.structured_event(json.dumps(_chat_request()).encode("utf-8")),),
                        trace_id=99,
                    )
                )

                first = await client.receive_result(timeout=5.0)
                second = await client.receive_result(timeout=5.0)
                third = await client.receive_result(timeout=5.0)

                assert first.packet.header.frame_id == 31
                assert first.packet.header.trace_id == 99
                assert first.metadata.result_class is ResultClass.PARTIAL
                assert decode_profile_event(first.structured_events[0])["delta"] == "hello"
                assert second.metadata.result_class is ResultClass.PARTIAL
                assert decode_profile_event(second.structured_events[0])["usage"]["total_tokens"] == 2
                assert third.metadata.result_class is ResultClass.COMPLETE
                assert third.packet.header.flags is HeaderFlags.NONE
                assert decode_profile_event(third.structured_events[0])["type"] == "response.completed"
        finally:
            await server_task
            await asyncio.wait_for(server_done.wait(), timeout=5.0)


def test_profile_event_payload_roundtrip_and_terminal_status_codes() -> None:
    completed = {"type": "response.completed", "body": {"id": "ok"}}
    timeout_error = {"type": "response.error", "error": {"code": "request_timeout"}}
    cancelled = {"type": "response.cancelled", "reason": "client_cancelled"}

    assert decode_profile_event(encode_profile_event(completed)) == completed
    assert is_terminal_profile_event(completed) is True
    assert status_code_for_profile_event(completed) == 200
    assert status_code_for_profile_event(timeout_error) == 504
    assert status_code_for_profile_event(cancelled) == 499

    with pytest.raises(ValueError, match="media_type"):
        decode_profile_event(b'{"media_type":"wrong","event":{}}')
    with pytest.raises(ValueError, match="event object"):
        decode_profile_event(b'{"media_type":"application/vnd.nnrp.openai-compatible.event+json","event":[]}')


def _chat_request(*, stream: bool = True) -> dict[str, Any]:
    return {
        "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "operation": CHAT_COMPLETIONS_CREATE,
        "body": {
            "model": "llama",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
        },
    }


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
