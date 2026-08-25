from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from nnrp import (  # type: ignore[import-untyped]
    NativeTransportServerSecurity,
    TransportPolicy,
    load_native_transport_binding,
)
from nnrp.server import NativeServerProviderRoute  # type: ignore[import-untyped]

from .adapter import OpenAiNnrpAdapter
from .benchmark import BenchmarkConfig, run_benchmark_sync, run_comparison_benchmark_sync
from .conformance import MockChatCompletionBackend, load_backend_async, run_conformance_plan_sync
from .nnrp_runtime import NnrpServerConfig, _serve_with_ready, serve
from .observability import (
    ObservationSink,
    OperationObservation,
    PrometheusObservationSink,
    ServerStartupObservation,
    StructuredLogObservationSink,
)

_PROVIDER_NAMES = frozenset({"tcp", "quic", "ipc", "websocket"})
_TRANSPORT_POLICIES = {policy.name.lower(): policy for policy in TransportPolicy}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm-nnrp-adapter")
    subcommands = parser.add_subparsers(dest="command", required=True)

    conformance = subcommands.add_parser(
        "run-conformance-plan",
        help="Execute an OpenAI NNRP API conformance plan and write case results.",
    )
    conformance.add_argument("--plan", type=Path, required=True)
    conformance.add_argument("--output", type=Path, required=True)
    conformance.add_argument(
        "--backend",
        default="mock",
        help="Backend factory spec. Use 'mock' for smoke tests or 'module.path:factory' for a vLLM backend.",
    )

    benchmark = subcommands.add_parser(
        "run-benchmark",
        help="Measure OpenAI NNRP API adapter latency and throughput.",
    )
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument(
        "--backend",
        default="mock",
        help="Backend factory spec. Use 'mock' for smoke tests or 'module.path:factory' for a vLLM backend.",
    )
    benchmark.add_argument("--iterations", type=int, default=200)
    benchmark.add_argument("--warmup", type=int, default=20)
    benchmark.add_argument("--model", default="mock-model")
    benchmark.add_argument(
        "--comparison",
        action="store_true",
        help="Run the long-context OpenAI HTTP SSE versus NNRP direct comparison matrix.",
    )
    benchmark.add_argument(
        "--prompt-tokens",
        default="4096,8192,16384,20480",
        help="Comma-separated synthetic prompt token counts for comparison mode.",
    )
    benchmark.add_argument(
        "--concurrency",
        default="1,2,4",
        help="Comma-separated concurrency levels for comparison mode.",
    )
    benchmark.add_argument("--max-completion-tokens", type=int, default=128)
    benchmark.add_argument("--http-url", help="OpenAI-compatible HTTP chat completions URL for SSE comparison.")
    benchmark.add_argument("--http-api-key", help="Bearer token for the HTTP comparison endpoint.")
    benchmark.add_argument(
        "--markdown-output",
        type=Path,
        help="Write a generated combined comparison table beside the raw JSON evidence.",
    )

    server = subcommands.add_parser(
        "serve",
        help="Serve the OpenAI-compatible profile through installed NNRP transport providers.",
    )
    server.add_argument("--backend", required=True, help="Backend factory spec: module.path:factory_name.")
    server.add_argument("--endpoint", required=True, help="NNRP application endpoint using nnrp:// or nnrps://.")
    server.add_argument(
        "--provider-route",
        action="append",
        default=[],
        metavar="NAME=LOCATOR",
        help="Provider-local bind locator. Repeat for tcp, quic, ipc, or websocket.",
    )
    server.add_argument(
        "--provider-certificate",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="DER certificate for a secured provider route.",
    )
    server.add_argument(
        "--provider-private-key",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="PKCS#8 DER private key for a secured provider route.",
    )
    server.add_argument(
        "--transport-policy",
        choices=sorted(_TRANSPORT_POLICIES),
        default="auto",
    )
    server.add_argument("--max-active-sessions", type=int, default=8)
    server.add_argument("--max-operations-per-session", type=int, default=4)
    server.add_argument("--native-workers", type=int, default=9)
    server.add_argument(
        "--metrics-host",
        default="127.0.0.1",
        help="Bind host for the opt-in Prometheus HTTP endpoint.",
    )
    server.add_argument(
        "--metrics-port",
        type=_parse_port,
        help="Explicitly enable a Prometheus /metrics endpoint on this port.",
    )

    wire_target = subcommands.add_parser(
        "serve-wire-target",
        help="Serve the adapter as an external OpenAI-profile wire conformance target.",
    )
    wire_target.add_argument("--ready-output", type=Path, required=True)
    wire_target.add_argument("--observation-output", type=Path)
    wire_target.add_argument("--wire-certificate", type=Path, required=True)
    wire_target.add_argument("--wire-private-key", type=Path, required=True)
    wire_target.add_argument("--backend", default="mock")
    wire_target.add_argument("--suite-version", default="0.1.0")

    args = parser.parse_args(argv)
    if args.command == "run-conformance-plan":
        run_conformance_plan_sync(args.plan, args.output, args.backend)
        return 0
    if args.command == "run-benchmark":
        benchmark_config = BenchmarkConfig(
            iterations=args.iterations,
            warmup=args.warmup,
            model=args.model,
            prompt_tokens=_parse_int_tuple(args.prompt_tokens, option_name="--prompt-tokens"),
            concurrency=_parse_int_tuple(args.concurrency, option_name="--concurrency"),
            max_completion_tokens=args.max_completion_tokens,
            http_url=args.http_url,
            http_api_key=args.http_api_key,
        )
        if args.comparison:
            run_comparison_benchmark_sync(
                args.output,
                args.backend,
                benchmark_config,
                args.markdown_output,
            )
        else:
            if args.markdown_output is not None:
                parser.error("--markdown-output requires --comparison")
            run_benchmark_sync(args.output, args.backend, benchmark_config)
        return 0
    if args.command == "serve":
        try:
            with _standalone_metrics_sink(args.metrics_host, args.metrics_port) as metrics_sink:
                observation_sinks: tuple[ObservationSink, ...] = (StructuredLogObservationSink(),)
                if metrics_sink is not None:
                    observation_sinks += (metrics_sink,)
                server_config = NnrpServerConfig(
                    endpoint=args.endpoint,
                    provider_routes=_provider_routes(
                        args.provider_route,
                        certificates=args.provider_certificate,
                        private_keys=args.provider_private_key,
                    ),
                    transport_policy=_TRANSPORT_POLICIES[args.transport_policy],
                    max_active_sessions=args.max_active_sessions,
                    max_operations_per_session=args.max_operations_per_session,
                    native_worker_count=args.native_workers,
                    observation_sinks=observation_sinks,
                )
                asyncio.run(_serve_backend(args.backend, server_config))
        except KeyboardInterrupt:
            return 130
        return 0
    if args.command == "serve-wire-target":
        try:
            asyncio.run(
                _serve_wire_target(
                    args.backend,
                    args.ready_output,
                    args.suite_version,
                    observation_output=args.observation_output,
                    certificate_der=args.wire_certificate.read_bytes(),
                    private_key_pkcs8_der=args.wire_private_key.read_bytes(),
                )
            )
        except KeyboardInterrupt:
            return 130
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


def _parse_int_tuple(value: str, *, option_name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise SystemExit(f"{option_name} must be a comma-separated list of integers") from error
    if not parsed:
        raise SystemExit(f"{option_name} must not be empty")
    return parsed


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


@contextmanager
def _standalone_metrics_sink(host: str, port: int | None) -> Iterator[ObservationSink | None]:
    if port is None:
        yield None
        return
    try:
        from prometheus_client import CollectorRegistry, start_http_server
    except ImportError as error:
        raise RuntimeError(
            "--metrics-port requires the 'prometheus' optional dependency"
        ) from error

    registry = CollectorRegistry()
    sink = PrometheusObservationSink(registry)
    server, thread = start_http_server(port, addr=host, registry=registry)
    try:
        yield sink
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def _serve_backend(backend_spec: str, config: NnrpServerConfig) -> None:
    backend = await load_backend_async(backend_spec)
    await serve(OpenAiNnrpAdapter(backend), config=config)


async def _serve_wire_target(
    backend_spec: str,
    ready_output: Path,
    suite_version: str,
    *,
    observation_output: Path | None = None,
    certificate_der: bytes,
    private_key_pkcs8_der: bytes,
) -> None:
    backend = await load_backend_async(backend_spec)
    if isinstance(backend, MockChatCompletionBackend):
        backend = MockChatCompletionBackend(stream_inter_event_delay_s=0.5)
    provider_routes = _wire_target_provider_routes(
        ready_output,
        certificate_der=certificate_der,
        private_key_pkcs8_der=private_key_pkcs8_der,
    )
    observation_sinks: tuple[ObservationSink, ...] = (StructuredLogObservationSink(),)
    if observation_output is not None:
        observation_sinks += (_WireEvidenceSink(observation_output),)
    config = NnrpServerConfig(
        endpoint="nnrp://wire-target.local/vllm",
        provider_routes=provider_routes,
        transports=tuple(load_native_transport_binding(name) for name in provider_routes),
        accept_timeout_ms=1_000,
        max_active_sessions=1,
        max_operations_per_session=1,
        native_worker_count=2,
        observation_sinks=observation_sinks,
    )
    ready = asyncio.get_running_loop().create_future()

    def publish_ready(bound_endpoints: Mapping[str, object]) -> None:
        tcp = _wire_target_endpoint(bound_endpoints, "tcp", attribute="address")
        quic = _wire_target_endpoint(bound_endpoints, "quic", attribute="address")
        ipc = _wire_target_endpoint(bound_endpoints, "ipc", attribute="uri")
        websocket = _wire_target_endpoint(bound_endpoints, "websocket", attribute="uri")
        document = {
            "$schema": "https://github.com/NagareWorks/nnrp-conformance/schemas/wire-conformance-target.schema.json",
            "target_name": "vllm-nnrp-adapter",
            "protocol_version": "nnrp-1-preview4",
            "suite_version": suite_version,
            "wire_conformance": {
                "modes": ["suite_as_client"],
                "transports": [
                    {"name": "tcp", "endpoint": tcp, "tls": False},
                    {
                        "name": "quic",
                        "endpoint": quic,
                        "tls": True,
                        "security": {
                            "server_name": "localhost",
                            "trusted_certificate_der_path": "certs/server.der",
                            "certificate_der_path": "certs/server.der",
                            "private_key_pkcs8_der_path": "certs/server-key.der",
                        },
                    },
                    {"name": "ipc", "endpoint": ipc, "tls": False},
                    {"name": "websocket", "endpoint": websocket, "tls": False},
                ],
                "host_route_providers": [],
                "capabilities": [
                    "profile.openai-compatible.level1.wire",
                    "control.cancel_abort",
                    "control.deadline_expire",
                    "control.result_drop_reason",
                    "control.trace_context",
                ],
                "limits": {"max_frame_bytes": 67_108_864, "max_in_flight": 1},
            },
        }
        ready_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = ready_output.with_suffix(f"{ready_output.suffix}.tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        temporary.replace(ready_output)

    def capture_ready(bound_endpoints: Mapping[str, object]) -> None:
        if ready.done():
            raise RuntimeError("wire target readiness was published more than once")
        ready.set_result(dict(bound_endpoints))

    serve_task = asyncio.create_task(
        _serve_with_ready(
            OpenAiNnrpAdapter(backend),
            config=config,
            on_ready=capture_ready,
        )
    )
    try:
        await asyncio.wait((serve_task, ready), return_when=asyncio.FIRST_COMPLETED)
        if ready.done():
            publish_ready(ready.result())
        await serve_task
    finally:
        if not serve_task.done():
            serve_task.cancel()
            await asyncio.gather(serve_task, return_exceptions=True)


class _WireEvidenceSink:
    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def observe_server_startup(self, observation: ServerStartupObservation) -> None:
        self._append("server_startup", observation.to_log_fields())

    def observe_operation(self, observation: OperationObservation) -> None:
        self._append("operation", observation.to_log_fields())

    def _append(self, record_type: str, fields: Mapping[str, object]) -> None:
        record = {"record_type": record_type, **fields}
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)


def _wire_target_provider_routes(
    ready_output: Path,
    *,
    certificate_der: bytes,
    private_key_pkcs8_der: bytes,
) -> Mapping[str, NativeServerProviderRoute]:
    if os.name == "nt":
        ipc_endpoint = f"npipe://nnrp-vllm-wire-{os.getpid()}"
    else:
        socket_path = (ready_output.parent.resolve() / "nnrp-vllm-wire.sock").as_posix()
        ipc_endpoint = f"unix://{socket_path}"
    return {
        "tcp": NativeServerProviderRoute(provider_endpoint="tcp://127.0.0.1:0"),
        "quic": NativeServerProviderRoute(
            provider_endpoint="quic://127.0.0.1:0",
            security=NativeTransportServerSecurity(
                certificate_der=certificate_der,
                private_key_pkcs8_der=private_key_pkcs8_der,
            ),
        ),
        "ipc": NativeServerProviderRoute(provider_endpoint=ipc_endpoint),
        "websocket": NativeServerProviderRoute(provider_endpoint="ws://127.0.0.1:0/nnrp"),
    }


def _wire_target_endpoint(bound_endpoints: Mapping[str, object], name: str, *, attribute: str) -> str:
    endpoint = bound_endpoints.get(name)
    value = getattr(endpoint, attribute, None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"wire target did not expose its bound {name} endpoint")
    return value


def _provider_routes(
    route_values: Sequence[str],
    *,
    certificates: Sequence[str],
    private_keys: Sequence[str],
) -> Mapping[str, NativeServerProviderRoute]:
    locators = _named_values(route_values, option_name="--provider-route")
    certificate_paths = _named_values(certificates, option_name="--provider-certificate")
    private_key_paths = _named_values(private_keys, option_name="--provider-private-key")
    security_names = set(certificate_paths) | set(private_key_paths)
    if security_names - set(locators):
        names = ", ".join(sorted(security_names - set(locators)))
        raise ValueError(f"security material requires a matching --provider-route: {names}")
    if set(certificate_paths) != set(private_key_paths):
        raise ValueError("each secured provider route requires both certificate and private key")

    routes: dict[str, NativeServerProviderRoute] = {}
    for name, locator in locators.items():
        security = None
        if name in certificate_paths:
            security = NativeTransportServerSecurity(
                certificate_der=Path(certificate_paths[name]).read_bytes(),
                private_key_pkcs8_der=Path(private_key_paths[name]).read_bytes(),
            )
        routes[name] = NativeServerProviderRoute(provider_endpoint=locator, security=security)
    return routes


def _named_values(values: Sequence[str], *, option_name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or name not in _PROVIDER_NAMES or not item:
            raise ValueError(f"{option_name} must use NAME=VALUE with tcp, quic, ipc, or websocket")
        if name in parsed:
            raise ValueError(f"duplicate {option_name} for {name}")
        parsed[name] = item
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
