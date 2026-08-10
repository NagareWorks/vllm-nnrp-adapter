from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path

from nnrp import NativeTransportServerSecurity, TransportPolicy  # type: ignore[import-untyped]
from nnrp.server import NativeServerProviderRoute  # type: ignore[import-untyped]

from .adapter import OpenAiNnrpAdapter
from .benchmark import BenchmarkConfig, run_benchmark_sync, run_comparison_benchmark_sync
from .conformance import load_backend_async, run_conformance_plan_sync
from .nnrp_runtime import NnrpServerConfig, serve

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
        runner = run_comparison_benchmark_sync if args.comparison else run_benchmark_sync
        runner(
            args.output,
            args.backend,
            benchmark_config,
        )
        return 0
    if args.command == "serve":
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
        )
        try:
            asyncio.run(_serve_backend(args.backend, server_config))
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


async def _serve_backend(backend_spec: str, config: NnrpServerConfig) -> None:
    backend = await load_backend_async(backend_spec)
    await serve(OpenAiNnrpAdapter(backend), config=config)


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
