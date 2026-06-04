from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest

from vllm_nnrp_adapter.adapter import OpenAiNnrpAdapter
from vllm_nnrp_adapter.embedded import (
    EmbeddedTcpServerConfig,
    create_embedded_openai_adapter,
    load_serving_chat_factory,
    run_embedded_tcp_server,
    serve_embedded_tcp_listener,
)


class FakeBackend:
    def create_chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        return {"choices": [{"message": {"content": body["model"]}}]}


class FakeServingChat:
    def create_chat_completion(self, request: object, raw_request: object | None = None) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}}]}


@dataclass
class FakeSession:
    closed: bool = False
    requests: int = 0

    async def receive_submit(self, timeout: float | None = None) -> object:
        self.requests += 1
        raise RuntimeError("stop before result emission")

    async def close(self) -> None:
        self.closed = True


class FakeListener:
    pass


def make_serving_chat() -> FakeServingChat:
    return FakeServingChat()


def test_create_embedded_openai_adapter_wraps_vllm_serving_object() -> None:
    adapter = create_embedded_openai_adapter(FakeServingChat())

    assert isinstance(adapter, OpenAiNnrpAdapter)


def test_load_serving_chat_factory_loads_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("fake_serving_module")
    module.make_serving_chat = make_serving_chat
    monkeypatch.setitem(sys.modules, "fake_serving_module", module)

    assert load_serving_chat_factory("fake_serving_module:make_serving_chat")() is not None


def test_load_serving_chat_factory_rejects_bad_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="module.path:factory"):
        load_serving_chat_factory("missing_separator")

    module = ModuleType("fake_non_callable_module")
    module.value = object()
    monkeypatch.setitem(sys.modules, "fake_non_callable_module", module)

    with pytest.raises(TypeError, match="not callable"):
        load_serving_chat_factory("fake_non_callable_module:value")


@pytest.mark.asyncio
async def test_serve_embedded_tcp_listener_accepts_and_closes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession()
    calls: list[tuple[object, int | None, str, float]] = []

    async def fake_accept_server_session(
        listener: object,
        *,
        session_id: int | None = None,
        active_model_name: str = "",
        timeout: float = 10.0,
    ) -> FakeSession:
        calls.append((listener, session_id, active_model_name, timeout))
        return session

    async def fake_serve_openai_profile_session(
        adapter: OpenAiNnrpAdapter,
        accepted_session: FakeSession,
        *,
        max_requests: int | None = None,
        receive_timeout: float | None = None,
        close_on_exit: bool = False,
    ) -> int:
        assert isinstance(adapter, OpenAiNnrpAdapter)
        assert accepted_session is session
        assert max_requests == 3
        assert receive_timeout == 0.5
        assert close_on_exit is True
        await accepted_session.close()
        return 3

    monkeypatch.setattr("vllm_nnrp_adapter.embedded.accept_server_session", fake_accept_server_session)
    monkeypatch.setattr("vllm_nnrp_adapter.embedded.serve_openai_profile_session", fake_serve_openai_profile_session)

    listener = FakeListener()
    handled = await serve_embedded_tcp_listener(
        OpenAiNnrpAdapter(FakeBackend()),
        listener,
        config=EmbeddedTcpServerConfig(
            active_model_name="llama",
            session_id=17,
            accept_timeout=1.5,
            receive_timeout=0.5,
            max_sessions=1,
            max_requests_per_session=3,
        ),
    )

    assert handled == 1
    assert session.closed is True
    assert calls == [(listener, 17, "llama", 1.5)]


@pytest.mark.asyncio
async def test_run_embedded_tcp_server_opens_tcp_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeTcpContext:
        async def __aenter__(self) -> FakeListener:
            return FakeListener()

        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            calls["closed"] = True

    def fake_create_tcp_server_configuration(*, idle_timeout: float | None = None, no_delay: bool = True) -> object:
        calls["configuration"] = (idle_timeout, no_delay)
        return object()

    def fake_serve_tcp(host: str, port: int, *, configuration: object | None = None) -> FakeTcpContext:
        calls["serve_tcp"] = (host, port, configuration is not None)
        return FakeTcpContext()

    async def fake_serve_embedded_tcp_listener(
        adapter: OpenAiNnrpAdapter,
        listener: FakeListener,
        *,
        config: EmbeddedTcpServerConfig,
    ) -> int:
        assert isinstance(adapter, OpenAiNnrpAdapter)
        assert isinstance(listener, FakeListener)
        assert config.host == "0.0.0.0"
        return 2

    monkeypatch.setattr(
        "vllm_nnrp_adapter.embedded.create_tcp_server_configuration",
        fake_create_tcp_server_configuration,
    )
    monkeypatch.setattr("vllm_nnrp_adapter.embedded.serve_tcp", fake_serve_tcp)
    monkeypatch.setattr("vllm_nnrp_adapter.embedded.serve_embedded_tcp_listener", fake_serve_embedded_tcp_listener)

    handled = await run_embedded_tcp_server(
        FakeServingChat(),
        config=EmbeddedTcpServerConfig(host="0.0.0.0", port=7767, idle_timeout=5.0, no_delay=False),
    )

    assert handled == 2
    assert calls["configuration"] == (5.0, False)
    assert calls["serve_tcp"] == ("0.0.0.0", 7767, True)
    assert calls["closed"] is True
