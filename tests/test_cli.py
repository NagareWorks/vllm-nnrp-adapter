from __future__ import annotations

import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nnrp import TransportPolicy

from vllm_nnrp_adapter import cli
from vllm_nnrp_adapter.adapter import OpenAiNnrpAdapter


def test_serve_cli_builds_provider_neutral_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    certificate = tmp_path / "server.der"
    private_key = tmp_path / "server-key.der"
    certificate.write_bytes(b"certificate")
    private_key.write_bytes(b"private-key")
    captured: list[tuple[str, cli.NnrpServerConfig]] = []

    def fake_run(coroutine: Coroutine[Any, Any, None]) -> None:
        coroutine.close()

    async def completed() -> None:
        return None

    def fake_serve_backend(backend_spec: str, config: cli.NnrpServerConfig) -> Coroutine[Any, Any, None]:
        captured.append((backend_spec, config))
        return completed()

    monkeypatch.setattr(cli.asyncio, "run", fake_run)
    monkeypatch.setattr(cli, "_serve_backend", fake_serve_backend)

    result = cli.main(
        [
            "serve",
            "--backend",
            "host.runtime:make_backend",
            "--endpoint",
            "nnrp://runtime.local/vllm",
            "--provider-route",
            "tcp=tcp://0.0.0.0:7766",
            "--provider-route",
            "quic=quic://0.0.0.0:7767",
            "--provider-route",
            "ipc=npipe://nnrp-vllm",
            "--provider-route",
            "websocket=wss://0.0.0.0:7768/nnrp",
            "--provider-certificate",
            f"quic={certificate}",
            "--provider-certificate",
            f"websocket={certificate}",
            "--provider-private-key",
            f"quic={private_key}",
            "--provider-private-key",
            f"websocket={private_key}",
            "--transport-policy",
            "prefer_quic",
        ]
    )

    assert result == 0
    assert len(captured) == 1
    backend_spec, config = captured[0]
    routes = config.provider_routes
    assert backend_spec == "host.runtime:make_backend"
    assert config.endpoint == "nnrp://runtime.local/vllm"
    assert config.transport_policy is TransportPolicy.PREFER_QUIC
    assert set(routes) == {"tcp", "quic", "ipc", "websocket"}
    assert routes["ipc"].provider_endpoint == "npipe://nnrp-vllm"
    assert routes["quic"].security is not None
    assert routes["quic"].security.certificate_der == b"certificate"
    assert routes["websocket"].security is not None


@pytest.mark.parametrize(
    ("routes", "certificates", "keys", "message"),
    [
        (["udp=udp://host:1"], [], [], "NAME=VALUE"),
        (["tcp=tcp://host:1", "tcp=tcp://host:2"], [], [], "duplicate"),
        (["quic=quic://host:1"], ["quic=certificate.der"], [], "both certificate and private key"),
        ([], ["quic=certificate.der"], ["quic=key.der"], "matching --provider-route"),
    ],
)
def test_provider_routes_reject_invalid_or_incomplete_configuration(
    routes: list[str],
    certificates: list[str],
    keys: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cli._provider_routes(routes, certificates=certificates, private_keys=keys)


@pytest.mark.asyncio
async def test_serve_backend_loads_factory_and_enters_native_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = object()
    config = cli.NnrpServerConfig(endpoint="nnrp://runtime.local/vllm")
    captured: list[tuple[OpenAiNnrpAdapter, cli.NnrpServerConfig]] = []

    async def fake_load_backend(spec: str) -> object:
        assert spec == "host.runtime:make_backend"
        return backend

    async def fake_serve(adapter: OpenAiNnrpAdapter, *, config: cli.NnrpServerConfig) -> None:
        captured.append((adapter, config))

    monkeypatch.setattr(cli, "load_backend_async", fake_load_backend)
    monkeypatch.setattr(cli, "serve", fake_serve)

    await cli._serve_backend("host.runtime:make_backend", config)

    assert len(captured) == 1
    adapter, captured_config = captured[0]
    assert isinstance(adapter, OpenAiNnrpAdapter)
    assert adapter._backend is backend
    assert captured_config is config


@pytest.mark.asyncio
async def test_wire_target_publishes_bound_tcp_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = object()
    ready_output = tmp_path / "target.json"

    async def fake_load_backend(spec: str) -> object:
        assert spec == "mock"
        return backend

    async def fake_serve_with_ready(
        adapter: OpenAiNnrpAdapter,
        *,
        config: cli.NnrpServerConfig,
        on_ready: Any,
    ) -> None:
        assert adapter._backend is backend
        assert config.endpoint == "nnrp://wire-target.local/vllm"
        assert config.provider_routes["tcp"].provider_endpoint == "tcp://127.0.0.1:0"
        on_ready({"tcp": SimpleNamespace(address="127.0.0.1:39123")})

    monkeypatch.setattr(cli, "load_backend_async", fake_load_backend)
    monkeypatch.setattr(cli, "_serve_with_ready", fake_serve_with_ready)

    await cli._serve_wire_target("mock", ready_output, "0.1.0")

    manifest = json.loads(ready_output.read_text(encoding="utf-8"))
    assert manifest["target_name"] == "vllm-nnrp-adapter"
    assert manifest["protocol_version"] == "nnrp-1-preview4"
    assert manifest["suite_version"] == "0.1.0"
    assert manifest["wire_conformance"]["transports"] == [
        {"name": "tcp", "endpoint": "127.0.0.1:39123", "tls": False}
    ]
    assert manifest["wire_conformance"]["capabilities"] == ["profile.openai-compatible.level1.wire"]
