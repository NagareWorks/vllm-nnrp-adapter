from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .benchmark import BenchmarkConfig, run_benchmark_sync, run_comparison_benchmark_sync
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
        config = BenchmarkConfig(
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
            config,
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


def _parse_int_tuple(value: str, *, option_name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise SystemExit(f"{option_name} must be a comma-separated list of integers") from error
    if not parsed:
        raise SystemExit(f"{option_name} must not be empty")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
