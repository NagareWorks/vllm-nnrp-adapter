from __future__ import annotations

import argparse
import json
import os
from collections.abc import Coroutine
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nnrp import TransportPolicy

from vllm_nnrp_adapter import cli
from vllm_nnrp_adapter.adapter import OpenAiNnrpAdapter
from vllm_nnrp_adapter.observability import PrometheusObservationSink

LOCAL_IPC_ENDPOINT = "npipe://nnrp-vllm" if os.name == "nt" else "unix:///tmp/nnrp-vllm.sock"


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
            f"ipc={LOCAL_IPC_ENDPOINT}",
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
    assert routes["ipc"].provider_endpoint == LOCAL_IPC_ENDPOINT
    assert routes["quic"].security is not None
    assert routes["quic"].security.certificate_der == b"certificate"
    assert routes["websocket"].security is not None
    assert len(config.observation_sinks) == 1


def test_serve_cli_enables_metrics_only_with_explicit_port(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics_sink = SimpleNamespace(
        observe_server_startup=lambda _observation: None,
        observe_operation=lambda _observation: None,
    )
    captured: list[cli.NnrpServerConfig] = []

    @contextmanager
    def fake_metrics_sink(host: str, port: int | None) -> Any:
        assert host == "127.0.0.2"
        assert port == 9464
        yield metrics_sink

    async def completed() -> None:
        return None

    def fake_run(coroutine: Coroutine[Any, Any, None]) -> None:
        coroutine.close()

    def fake_serve_backend(_backend_spec: str, config: cli.NnrpServerConfig) -> Coroutine[Any, Any, None]:
        captured.append(config)
        return completed()

    monkeypatch.setattr(cli, "_standalone_metrics_sink", fake_metrics_sink)
    monkeypatch.setattr(cli, "_serve_backend", fake_serve_backend)
    monkeypatch.setattr(cli.asyncio, "run", fake_run)

    assert cli.main(
        [
            "serve",
            "--backend",
            "host.runtime:make_backend",
            "--endpoint",
            "nnrp://runtime.local/vllm",
            "--metrics-host",
            "127.0.0.2",
            "--metrics-port",
            "9464",
        ]
    ) == 0
    assert len(captured) == 1
    assert captured[0].observation_sinks[1] is metrics_sink


def test_standalone_metrics_sink_owns_http_server_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    import prometheus_client

    server = SimpleNamespace(shutdown_calls=0, close_calls=0)
    thread = SimpleNamespace(join_timeouts=[])

    def shutdown() -> None:
        server.shutdown_calls += 1

    def server_close() -> None:
        server.close_calls += 1

    def join(timeout: int) -> None:
        thread.join_timeouts.append(timeout)

    server.shutdown = shutdown
    server.server_close = server_close
    thread.join = join

    def fake_start_http_server(port: int, *, addr: str, registry: object) -> tuple[object, object]:
        assert port == 9464
        assert addr == "127.0.0.2"
        assert registry is not None
        return server, thread

    monkeypatch.setattr(prometheus_client, "start_http_server", fake_start_http_server)

    with cli._standalone_metrics_sink("127.0.0.2", 9464) as sink:
        assert isinstance(sink, PrometheusObservationSink)

    assert server.shutdown_calls == 1
    assert server.close_calls == 1
    assert thread.join_timeouts == [5]


@pytest.mark.parametrize("value", ["0", "65536", "not-a-port"])
def test_parse_port_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_port(value)


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
async def test_wire_target_publishes_bound_provider_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = object()
    ready_output = tmp_path / "target.json"
    observation_output = tmp_path / "observations.jsonl"

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
        assert config.transport_policy is TransportPolicy.AUTO
        assert config.provider_routes["tcp"].provider_endpoint == "tcp://127.0.0.1:0"
        assert config.provider_routes["quic"].provider_endpoint == "quic://127.0.0.1:0"
        assert config.provider_routes["quic"].security is not None
        assert config.provider_routes["quic"].security.certificate_der == b"certificate"
        assert config.provider_routes["quic"].security.private_key_pkcs8_der == b"private-key"
        assert config.provider_routes["websocket"].provider_endpoint == "ws://127.0.0.1:0/nnrp"
        ipc_endpoint = config.provider_routes["ipc"].provider_endpoint
        assert ipc_endpoint.startswith(("npipe://", "unix://"))
        assert config.transports is not None
        assert len(config.transports) == 4
        assert config.accept_timeout_ms == 1_000
        on_ready(
            {
                "tcp": SimpleNamespace(address="127.0.0.1:39123"),
                "quic": SimpleNamespace(address="127.0.0.1:39125"),
                "ipc": SimpleNamespace(uri=ipc_endpoint),
                "websocket": SimpleNamespace(uri="ws://127.0.0.1:39124/nnrp"),
            }
        )

    monkeypatch.setattr(cli, "load_backend_async", fake_load_backend)
    monkeypatch.setattr(cli, "_serve_with_ready", fake_serve_with_ready)

    await cli._serve_wire_target(
        "mock",
        ready_output,
        "0.1.0",
        observation_output=observation_output,
        certificate_der=b"certificate",
        private_key_pkcs8_der=b"private-key",
    )

    manifest = json.loads(ready_output.read_text(encoding="utf-8"))
    assert manifest["target_name"] == "vllm-nnrp-adapter"
    assert manifest["protocol_version"] == "nnrp-1-preview4"
    assert manifest["suite_version"] == "0.1.0"
    assert manifest["wire_conformance"]["transports"] == [
        {"name": "tcp", "endpoint": "127.0.0.1:39123", "tls": False},
        {
            "name": "quic",
            "endpoint": "127.0.0.1:39125",
            "tls": True,
            "security": {
                "server_name": "localhost",
                "trusted_certificate_der_path": "certs/server.der",
                "certificate_der_path": "certs/server.der",
                "private_key_pkcs8_der_path": "certs/server-key.der",
            },
        },
        {
            "name": "ipc",
            "endpoint": cli._wire_target_provider_routes(
                ready_output,
                certificate_der=b"certificate",
                private_key_pkcs8_der=b"private-key",
            )["ipc"].provider_endpoint,
            "tls": False,
        },
        {"name": "websocket", "endpoint": "ws://127.0.0.1:39124/nnrp", "tls": False},
    ]
    assert manifest["wire_conformance"]["capabilities"] == [
        "profile.openai-compatible.level1.wire",
        "control.cancel_abort",
        "control.capability_costs",
        "control.deadline_expire",
        "control.result_drop_reason",
        "control.trace_context",
    ]
    evidence = [json.loads(line) for line in observation_output.read_text(encoding="utf-8").splitlines()]
    assert evidence == []


def test_wire_evidence_sink_writes_machine_readable_records(tmp_path: Path) -> None:
    output = tmp_path / "wire" / "observations.jsonl"
    sink = cli._WireEvidenceSink(output)
    startup = SimpleNamespace(to_log_fields=lambda: {"transport_policy": "auto"})
    operation = SimpleNamespace(to_log_fields=lambda: {"operation_id": 7, "terminal_outcome": "completed"})

    sink.observe_server_startup(startup)  # type: ignore[arg-type]
    sink.observe_operation(operation)  # type: ignore[arg-type]

    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [
        {"record_type": "server_startup", "transport_policy": "auto"},
        {"operation_id": 7, "record_type": "operation", "terminal_outcome": "completed"},
    ]


def test_wire_target_endpoint_requires_every_bound_provider() -> None:
    with pytest.raises(RuntimeError, match="bound websocket endpoint"):
        cli._wire_target_endpoint({}, "websocket", attribute="uri")


def test_wire_target_provider_selection_is_canonical_and_rejects_invalid_sets(tmp_path: Path) -> None:
    routes = cli._wire_target_provider_routes(
        tmp_path / "target.json",
        certificate_der=b"certificate",
        private_key_pkcs8_der=b"private-key",
        provider_names=("websocket", "tcp"),
    )

    assert tuple(routes) == ("tcp", "websocket")
    with pytest.raises(ValueError, match="at least one provider"):
        cli._wire_target_provider_routes(
            tmp_path / "target.json",
            certificate_der=b"certificate",
            private_key_pkcs8_der=b"private-key",
            provider_names=(),
        )
    with pytest.raises(ValueError, match="must be unique"):
        cli._wire_target_provider_routes(
            tmp_path / "target.json",
            certificate_der=b"certificate",
            private_key_pkcs8_der=b"private-key",
            provider_names=("tcp", "tcp"),
        )


def test_wire_target_secure_websocket_route_and_manifest_use_route_local_tls(tmp_path: Path) -> None:
    routes = cli._wire_target_provider_routes(
        tmp_path / "target.json",
        certificate_der=b"certificate",
        private_key_pkcs8_der=b"private-key",
        provider_names=("websocket",),
        secure_websocket=True,
    )

    route = routes["websocket"]
    assert route.provider_endpoint == "wss://127.0.0.1:0/nnrp"
    assert route.security is not None
    assert route.security.certificate_der == b"certificate"
    assert route.security.private_key_pkcs8_der == b"private-key"
    assert cli._wire_target_manifest_transport(
        {"websocket": SimpleNamespace(uri="wss://127.0.0.1:39124/nnrp")},
        "websocket",
    ) == {
        "name": "websocket",
        "endpoint": "wss://127.0.0.1:39124/nnrp",
        "tls": True,
        "security": {
            "server_name": "127.0.0.1",
            "trusted_certificate_der_path": "certs/server.der",
            "certificate_der_path": "certs/server.der",
            "private_key_pkcs8_der_path": "certs/server-key.der",
        },
    }
def test_wire_target_posix_ipc_endpoint_stays_below_unix_socket_path_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deeply_nested = tmp_path.joinpath(*(f"policy-{index}" for index in range(20)), "target.json")
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: "/" + "nested/" * 20)

    endpoint = cli._wire_target_ipc_endpoint(deeply_nested, platform="posix")

    assert endpoint.startswith("unix:///tmp/nnrp-vllm-")
    assert len(os.fsencode(endpoint.removeprefix("unix://"))) < 100
    assert str(deeply_nested.parent) not in endpoint


def test_wire_target_removes_posix_ipc_socket(tmp_path: Path) -> None:
    socket_path = tmp_path / "nnrp.sock"
    socket_path.touch()

    cli._remove_wire_target_ipc_socket(
        {"ipc": cli.NativeServerProviderRoute(provider_endpoint=f"unix://{socket_path.as_posix()}")},
        platform="posix",
    )

    assert not socket_path.exists()
