# vLLM NNRP Adapter Usage

This document records the Preview4 native-role runtime shape for host installs, vLLM containers,
adapter startup, and OpenAI API profile conformance.

## Host Installation

Use the normal package when you only need mock conformance, benchmark smoke, or adapter unit tests:

```bash
python -m pip install vllm-nnrp-adapter
```

Install the vLLM extra only on a host or container that can actually import and run vLLM:

```bash
python -m pip install "vllm-nnrp-adapter[vllm]"
```

The extra permits `vllm>=0.18.0,<0.27` to resolve, while production startup accepts only the named,
feature-probed families in the [generated compatibility table](vllm-compatibility.md). An untested
minor line inside the installation interval fails at startup instead of falling through to an
accidental import or method match.

For local development:

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy
pytest -q
```

## vLLM Container Installation

In a vLLM image, install the adapter after the base vLLM stack is present:

```dockerfile
FROM vllm/vllm-openai:latest

RUN python -m pip install --no-cache-dir "vllm-nnrp-adapter[vllm]"
```

If the image already pins vLLM, prefer installing the adapter without pulling a second vLLM version:

```dockerfile
RUN python -m pip install --no-cache-dir --no-deps vllm-nnrp-adapter
```

Then verify the adapter import boundary:

```bash
python - <<'PY'
from vllm_nnrp_adapter import OpenAiNnrpAdapter, create_vllm_backend
print(OpenAiNnrpAdapter, create_vllm_backend)
PY
```

## Backend Factory

The adapter CLI accepts `--backend module.path:factory_name`. The factory must return an object compatible with the adapter backend protocol, or it can return a wrapped vLLM serving object:

```python
from vllm_nnrp_adapter import create_vllm_backend


def make_backend():
    serving_chat = build_or_retrieve_openai_serving_chat()
    return create_vllm_backend(serving_chat)
```

The vLLM wrapper converts profile request bodies into `ChatCompletionRequest` at runtime and calls `create_chat_completion`.
For streaming requests, production uses the engine client directly and rejects any path that would
return rendered SSE. Complex streaming features without an implemented direct binding fail
explicitly and are absent from the capability manifest. `HttpSseSmokeBackend` exists only for
comparison tests and is not exported or selected by the production factory.

## Native NNRP Server

Run the profile server beside vLLM with one application endpoint and one or more provider-local
routes. The installed `nnrp-py` provider packages own listener discovery, carrier behavior, and
native artifacts.

```bash
vllm-nnrp-adapter serve \
  --backend my_vllm_app.serving:make_backend \
  --endpoint nnrp://runtime.example/vllm \
  --provider-route tcp=tcp://0.0.0.0:7766 \
  --provider-route ipc=unix:///run/nnrp-vllm.sock \
  --transport-policy auto
```

For in-process integration, construct typed routes and call the same provider-neutral runtime:

```python
import asyncio

from nnrp import TransportPolicy
from nnrp.server import NativeServerProviderRoute
from vllm_nnrp_adapter import NnrpServerConfig, OpenAiNnrpAdapter, serve


async def start_nnrp_profile_server(backend):
    await serve(
        OpenAiNnrpAdapter(backend),
        config=NnrpServerConfig(
            endpoint="nnrp://runtime.example/vllm",
            provider_routes={
                "tcp": NativeServerProviderRoute(provider_endpoint="tcp://0.0.0.0:7766"),
            },
            transport_policy=TransportPolicy.FORCE_TCP,
        ),
    )


asyncio.create_task(start_nnrp_profile_server(backend))
```

The application endpoint stays `nnrp://` or `nnrps://`. Carrier locators such as `tcp://`,
`quic://`, `unix://`, `npipe://`, `ws://`, and `wss://` appear only inside provider routes.
By default, the runtime discovers installed transport providers. Embedders that already own a
provider registry can pass its `NativeTransportBinding` sequence through
`NnrpServerConfig(transports=...)`; that explicit sequence is forwarded unchanged to
`listen_native_server`.

The native role receives operation bodies directly. Ordered non-terminal profile events are sent as
`PARTIAL_RESULT`; exactly one `response.completed`, `response.error`, or `response.cancelled` event
completes the operation through terminal `RESULT_PUSH`.

## Conformance

Run against the shared OpenAI API profile plan:

```bash
vllm-nnrp-adapter run-conformance-plan \
  --plan artifacts/api-profile-execution-plan.json \
  --output artifacts/api-profile-results.json \
  --backend module.path:make_backend
```

For CI without vLLM:

```bash
vllm-nnrp-adapter run-conformance-plan \
  --plan tests/fixtures/api-profile-execution-plan.json \
  --output artifacts/api-profile-results.json \
  --backend mock
```

## Benchmark

```bash
vllm-nnrp-adapter run-benchmark \
  --output artifacts/openai-profile-benchmark.json \
  --backend module.path:make_backend \
  --iterations 200 \
  --warmup 20
```

The report includes non-streaming roundtrip latency, streaming event latency and throughput, and cancellation latency. Mock reports and HTTP-relay reports are smoke checks, not release-readiness evidence for the NNRP transport path.

Run the release-readiness comparison matrix with the engine-direct NNRP path and, when an OpenAI HTTP endpoint is available, the HTTP SSE baseline:

```bash
vllm-nnrp-adapter run-benchmark \
  --comparison \
  --output artifacts/openai-nnrp-comparison.json \
  --backend module.path:make_backend \
  --model example-model \
  --prompt-tokens 4096,8192,16384,20480 \
  --concurrency 1,2,4 \
  --max-completion-tokens 128 \
  --http-url http://127.0.0.1:8000/v1/chat/completions
```

The comparison report records TTFT, TPOT, RTT, output-token throughput, request throughput, error rate, and sampled error families for each prompt-size/concurrency pair. The NNRP path uses the in-process engine-direct adapter path; the HTTP path consumes OpenAI-compatible SSE chunks from the configured endpoint.

The first recorded release-readiness baseline is
[openai-nnrp-direct-vllm-0.18.1-t4-2026-06-04](benchmarks/openai-nnrp-direct-vllm-0.18.1-t4-2026-06-04.md).
It should be treated as the current NNRP direct-path baseline. HTTP/SSE-to-NNRP relay data is useful only as smoke
evidence and should not be used to justify the optimized runtime path.

## Request Envelope

```json
{
  "schema_version": "openai-compatible/1",
  "operation": "chat.completions.create",
  "body": {
    "model": "example-model",
    "messages": [
      {
        "role": "user",
        "content": "Say hello."
      }
    ],
    "stream": true
  },
  "nnrp": {
    "timeout_ms": 30000,
    "diagnostics": true,
    "cancel_after_events": 1
  }
}
```

`timeout_ms` bounds backend await and stream-next latency. `diagnostics` emits a `response.diagnostics` event. `cancel_after_events` is used by conformance and local testing to model caller-driven cancellation.
