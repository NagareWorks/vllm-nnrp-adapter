from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from nnrp.adapters import create_tcp_server_configuration, serve_tcp
from nnrp.server import accept_server_session

from .adapter import OpenAiNnrpAdapter
from .nnrp_runtime import serve_openai_profile_session
from .vllm_factory import create_vllm_backend


@dataclass(frozen=True, slots=True)
class EmbeddedTcpServerConfig:
    host: str = "127.0.0.1"
    port: int = 7766
    active_model_name: str = ""
    session_id: int | None = None
    accept_timeout: float = 10.0
    receive_timeout: float | None = None
    max_sessions: int | None = None
    max_requests_per_session: int | None = None
    idle_timeout: float | None = None
    no_delay: bool = True


DEFAULT_EMBEDDED_TCP_SERVER_CONFIG = EmbeddedTcpServerConfig()


def create_embedded_openai_adapter(serving_chat: object) -> OpenAiNnrpAdapter:
    return OpenAiNnrpAdapter(create_vllm_backend(serving_chat))


async def run_embedded_tcp_server(
    serving_chat: object,
    *,
    config: EmbeddedTcpServerConfig = DEFAULT_EMBEDDED_TCP_SERVER_CONFIG,
) -> int:
    adapter = create_embedded_openai_adapter(serving_chat)
    server_configuration = create_tcp_server_configuration(
        idle_timeout=config.idle_timeout,
        no_delay=config.no_delay,
    )
    async with serve_tcp(config.host, config.port, configuration=server_configuration) as listener:
        return await serve_embedded_tcp_listener(adapter, listener, config=config)


async def serve_embedded_tcp_listener(
    adapter: OpenAiNnrpAdapter,
    listener: object,
    *,
    config: EmbeddedTcpServerConfig = DEFAULT_EMBEDDED_TCP_SERVER_CONFIG,
) -> int:
    handled_sessions = 0
    while config.max_sessions is None or handled_sessions < config.max_sessions:
        session = await accept_server_session(
            listener,
            session_id=config.session_id,
            active_model_name=config.active_model_name,
            timeout=config.accept_timeout,
        )
        await serve_openai_profile_session(
            adapter,
            session,
            max_requests=config.max_requests_per_session,
            receive_timeout=config.receive_timeout,
            close_on_exit=True,
        )
        handled_sessions += 1
    return handled_sessions


def run_embedded_tcp_server_sync(
    serving_factory_spec: str,
    *,
    config: EmbeddedTcpServerConfig = DEFAULT_EMBEDDED_TCP_SERVER_CONFIG,
) -> int:
    serving_chat = load_serving_chat_factory(serving_factory_spec)()
    return asyncio.run(run_embedded_tcp_server(serving_chat, config=config))


def load_serving_chat_factory(factory_spec: str) -> Callable[[], object]:
    module_name, separator, symbol_name = factory_spec.partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError("serving factory spec must use 'module.path:factory'")

    module = importlib.import_module(module_name)
    factory = getattr(module, symbol_name)
    if not callable(factory):
        raise TypeError(f"vLLM serving factory is not callable: {factory_spec}")
    return cast(Callable[[], object], factory)
