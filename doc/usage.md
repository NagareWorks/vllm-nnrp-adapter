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

### Provider Routes And Binding Policy

Each route is keyed by one canonical provider name: `tcp`, `quic`, `ipc`, or `websocket`. TCP and
QUIC can derive a bind authority from the application endpoint when their route locator is omitted.
IPC and WebSocket require an explicit provider-local locator. Use `unix:///path` for Unix-domain IPC,
`npipe://name` for Windows named pipes, and `ws://` or `wss://` for WebSocket.

`AUTO` and `PREFER_*` create one logical server by atomically binding every installed provider that
is allowed, resolved, platform-compatible, and security-compatible. A preference changes
deterministic ordering; it does not suppress the other eligible listeners. `FORCE_*` restricts the
logical server to one provider. If any listener required by the selected policy fails, startup rolls
back the complete listener set and the adapter does not publish readiness.

Omitting a route does not disable an installed provider. For an in-process host that needs an
authoritative provider registry, pass exactly the bindings it owns:

```python
from nnrp import TransportPolicy, load_native_transport_binding

config = NnrpServerConfig(
    endpoint="nnrp://runtime.example/vllm",
    provider_routes={
        "tcp": NativeServerProviderRoute(provider_endpoint="tcp://0.0.0.0:7766"),
    },
    transports=(load_native_transport_binding("tcp"),),
    transport_policy=TransportPolicy.FORCE_TCP,
)
```

### Route Security

Security material is route-local. The server object contains DER certificate bytes and PKCS#8 DER
private-key bytes. QUIC always requires it. Native `wss://` requires it, while `ws://` and IPC reject
it. Supplying it to TCP enables the provider's TLS mode. An `nnrps://` application endpoint admits
only carriers that satisfy authenticated encryption; plain TCP, IPC, and `ws://` remain in provider
diagnostics but are not opened.

```bash
vllm-nnrp-adapter serve \
  --backend my_vllm_app.serving:make_backend \
  --endpoint nnrps://runtime.example/vllm \
  --provider-route quic=quic://0.0.0.0:7767 \
  --provider-certificate quic=/run/nnrp/server.der \
  --provider-private-key quic=/run/nnrp/server-key.der \
  --provider-route websocket=wss://0.0.0.0:7768/nnrp \
  --provider-certificate websocket=/run/nnrp/server.der \
  --provider-private-key websocket=/run/nnrp/server-key.der \
  --transport-policy prefer_quic
```

The native role receives operation bodies directly. Ordered non-terminal profile events are sent as
`PARTIAL_RESULT`; exactly one `response.completed`, `response.error`, or `response.cancelled` event
completes the operation through terminal `RESULT_PUSH`.

## Observability

The default server configuration writes immutable startup and terminal-operation records as
structured JSON logs. To export metrics, install the optional dependency and register the adapter
collector into the Prometheus registry already owned by the deployment:

```bash
python -m pip install "vllm-nnrp-adapter[prometheus]"
```

```python
from prometheus_client import REGISTRY
from vllm_nnrp_adapter import (
    NnrpServerConfig,
    PrometheusObservationSink,
    StructuredLogObservationSink,
)

config = NnrpServerConfig(
    endpoint="nnrp://runtime.example/vllm",
    observation_sinks=(
        StructuredLogObservationSink(),
        PrometheusObservationSink(REGISTRY),
    ),
)
```

The adapter does not start an HTTP `/metrics` server. The host remains responsible for exposing its
existing registry. Metric labels are restricted to bounded protocol/runtime categories; request
identifiers, prompts, generated text, model ids, and arbitrary metadata never become labels. A sink
failure is logged and isolated from request serving and from the remaining sinks.

Each terminal `OperationObservation` carries the selected transport, native connection/session
handle identities, wire session/operation/frame/route/view/trace/profile identities, backend and
model identity, token and byte counts, terminal diagnostics, and the `PROGRESS` stage timeline.
Queue, admission, preprocessing, first-event, inter-event, and terminal latencies are derived from
that same immutable record.

`TRACE_CONTEXT` remains runtime metadata and never enters the OpenAI request body. For supported
vLLM engine-direct bindings, an eligible session context becomes the operation default and is
forwarded as a W3C `traceparent`; an operation-scoped context can replace it until backend dispatch.
Later updates remain observable but do not mutate the in-flight request. Opaque trace attributes
are counted for diagnostics and are never copied into headers or logs.

The `serve` command does not open a metrics port by default. To expose the adapter collector from
the serving process, install the `prometheus` extra and opt in with an explicit bind address:

```powershell
python -m pip install ".[prometheus]"
vllm-nnrp-adapter serve `
  --backend host.runtime:make_backend `
  --endpoint nnrp://runtime.local/vllm `
  --metrics-host 127.0.0.1 `
  --metrics-port 9464
```

The endpoint is stopped with the adapter process. Deployments that already own a Prometheus
registry should continue to register `PrometheusObservationSink` directly instead.

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
  --markdown-output artifacts/openai-nnrp-comparison.md \
  --backend module.path:make_backend \
  --model example-model \
  --prompt-tokens 4096,8192,16384,20480 \
  --concurrency 1,2,4 \
  --max-completion-tokens 128 \
  --http-url http://127.0.0.1:8000/v1/chat/completions
```

The JSON report retains the workload manifest, raw samples, confidence intervals, TTFT, TPOT, RTT, cancellation latency,
output-token throughput, request throughput, and error rate for each prompt-size/concurrency pair. `--markdown-output`
generates the combined comparison table from that same report so published numbers cannot drift from the raw evidence. The
NNRP path uses the in-process engine-direct adapter path; the HTTP path consumes OpenAI-compatible SSE chunks from the
configured endpoint. Model-dominated chat parity is compatibility evidence, not the adapter's primary performance claim.

The first recorded release-readiness baseline is
[openai-nnrp-direct-vllm-0.18.1-t4-2026-06-04](benchmarks/openai-nnrp-direct-vllm-0.18.1-t4-2026-06-04.md).
It should be treated as the current NNRP direct-path baseline. HTTP/SSE-to-NNRP relay data is useful only as smoke
evidence and should not be used to justify the optimized runtime path.

## Level 1 JSON And Typed-Control Boundary

The `openai-compatible/1` Level 1 profile uses JSON only for the application request and profile
events. NNRP session, scheduling, cancellation, flow-control, observability, object, and cache
semantics remain protocol messages with typed binary metadata. They are not extension fields in an
OpenAI request body.

| Phase | NNRP message | Body representation |
| --- | --- | --- |
| Request | `FRAME_SUBMIT` | Exactly one 24-byte typed-payload descriptor followed by one UTF-8 JSON `STRUCTURED_EVENT` payload. The descriptor uses `profile_id=0`, `schema_id=0`, `schema_version=0`, and `stream_semantics=SNAPSHOT`; the submit metadata uses `payload_kind_bitmap=0x10` and `payload_frame_count=1`. |
| Streaming event | `PARTIAL_RESULT` | One raw UTF-8 JSON event object. It has no data-plane prelude, typed-payload descriptor, event concatenation, or SSE delimiter. |
| Terminal profile event | `RESULT_PUSH` | Exactly one typed `STRUCTURED_EVENT` snapshot with the same descriptor as the request. This carries `response.completed`, `response.error`, or `response.cancelled`. |
| Empty success | `RESULT_PUSH` | Zero payload frames and an empty body. |
| Runtime coordination | Dedicated Preview4 control messages | Typed metadata and, where defined by NNRP, a binary body or diagnostic tail. These messages never become profile JSON fields. |

The adapter currently applies typed `CANCEL`, `ABORT`, `SUPERSEDE`, `DEADLINE`, `EXPIRE_AT`, and
`PRIORITY_UPDATE` operation controls. It also handles typed capability negotiation, trace context,
backpressure, and credit updates. Progress, recoverable errors, retry hints, and result-drop reasons
are emitted through their dedicated NNRP messages. Runtime objects and cache references stay on the
typed runtime-object and cache-reference paths; the Level 1 JSON envelope is not a fallback encoding
for them.

### Request Envelope

The top-level object accepts only the following fields. Unknown fields are rejected rather than
silently forwarded.

| Field | Requirement | Meaning |
| --- | --- | --- |
| `schema_version` | Required | Must equal `openai-compatible/1`. |
| `operation` | Required | Must equal `chat.completions.create` for Level 1. |
| `request_id` | Optional | Caller correlation id; the adapter derives one from native operation identity when omitted. |
| `body` | Required | OpenAI-compatible chat-completions request object. |
| `nnrp` | Optional | Adapter policy object; accepts only `timeout_ms` and `diagnostics`. |

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
    "diagnostics": true
  }
}
```

`timeout_ms` bounds backend await and stream-next latency. `diagnostics` emits a
`response.diagnostics` event. Cancellation is not an envelope policy: native callers send the
frozen `CANCEL` or `ABORT` control message, while direct Python callers cancel or close their async
iterator. Fields such as deadlines, priorities, route hints, trace context, object descriptors, or
cache references are invalid at the envelope top level and must use their frozen typed protocol
messages.
