from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .benchmark import BenchmarkConfig, run_benchmark_sync
from .conformance import run_conformance_plan_sync
from .embedded import EmbeddedTcpServerConfig, run_embedded_tcp_server_sync


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

    serve_tcp = subcommands.add_parser(
        "serve-tcp",
        help="Run the embedded OpenAI NNRP API profile server beside a vLLM serving object.",
    )
    serve_tcp.add_argument(
        "--serving-factory",
        required=True,
        help="Factory spec returning a vLLM OpenAIServingChat-compatible object, as 'module.path:factory'.",
    )
    serve_tcp.add_argument("--host", default="127.0.0.1")
    serve_tcp.add_argument("--port", type=int, default=7766)
    serve_tcp.add_argument("--active-model-name", default="")
    serve_tcp.add_argument("--session-id", type=int)
    serve_tcp.add_argument("--accept-timeout", type=float, default=10.0)
    serve_tcp.add_argument("--receive-timeout", type=float)
    serve_tcp.add_argument("--max-sessions", type=int)
    serve_tcp.add_argument("--max-requests-per-session", type=int)
    serve_tcp.add_argument("--idle-timeout", type=float)
    serve_tcp.add_argument("--no-delay", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(argv)
    if args.command == "run-conformance-plan":
        run_conformance_plan_sync(args.plan, args.output, args.backend)
        return 0
    if args.command == "run-benchmark":
        run_benchmark_sync(
            args.output,
            args.backend,
            BenchmarkConfig(iterations=args.iterations, warmup=args.warmup, model=args.model),
        )
        return 0
    if args.command == "serve-tcp":
        run_embedded_tcp_server_sync(
            args.serving_factory,
            config=EmbeddedTcpServerConfig(
                host=args.host,
                port=args.port,
                active_model_name=args.active_model_name,
                session_id=args.session_id,
                accept_timeout=args.accept_timeout,
                receive_timeout=args.receive_timeout,
                max_sessions=args.max_sessions,
                max_requests_per_session=args.max_requests_per_session,
                idle_timeout=args.idle_timeout,
                no_delay=args.no_delay,
            ),
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
