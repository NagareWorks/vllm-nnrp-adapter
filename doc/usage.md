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

The report includes profile validation, profile-event mapping, adapter-side runtime-control handling after native event
delivery, non-streaming roundtrip latency, streaming event latency and throughput, and cancellation latency. The
runtime-control scenarios cover typed priority/deadline decoding and application, cancellation dispatch, and registry
cleanup; they do not include carrier or FFI delivery time. Mock reports and HTTP-relay reports are smoke checks, not
release-readiness evidence for the NNRP transport path.

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
Evidence writing rejects endpoint URLs, IP addresses, user-directory paths, bearer/API tokens, and UUID-style machine or
request identifiers. Use public-safe model aliases and environment labels in reports intended for publication.

### Stale-Work Adoption Workload

The stale-work runner executes raw OpenAI HTTP/SSE, equivalently orchestrated HTTP/SSE, and direct
NNRP as separate runs in a seeded random order. All three drivers receive the same immutable sample
ids, arrival offsets, and cancellation, abort, deadline, or supersession schedule. The runner limits
in-flight requests but does not estimate GPU use. A separate accounting probe must observe
server-side control acceptance, backend stop timing, and GPU seconds from the accounting source
named in the manifest.

```bash
vllm-nnrp-adapter run-stale-workload \
  --manifest artifacts/stale-work-30.json \
  --raw-output artifacts/stale-work-30-raw.json \
  --report-output artifacts/stale-work-30-report.json \
  --outcome-output artifacts/stale-work-30-outcome.json \
  --driver raw_openai_http_sse=deployment.benchmarks:make_raw_http_driver \
  --driver orchestrated_http_sse=deployment.benchmarks:make_orchestrated_http_driver \
  --driver direct_nnrp=deployment.benchmarks:make_direct_nnrp_driver \
  --accounting-probe deployment.benchmarks:make_cuda_accounting_probe
```

The `vllm_nnrp_adapter.stale_work_drivers` module provides the raw, equivalently orchestrated, and
direct-NNRP driver implementations. A zero-argument factory supplies deployment endpoints,
credentials, provider routes, transport policy, controllers, and timeouts without placing those
values in benchmark evidence. The direct driver opens one native NNRP connection and one
multiplexed session for the run, submits the frozen OpenAI NNRP typed-payload envelope, and uses the
runner-owned sample id as `request_id` for server-side accounting correlation.
The raw driver sends an OpenAI-compatible streaming chat request and closes the live response for
every scheduled control. That close is only client cancellation: abort, deadline, and supersession
are deliberately not reported as equivalent server controls.

`OrchestratedHttpSseDriver` sends the same HTTP/SSE request but delegates actual control dispatch to
a deployment-owned `OrchestratedHttpController`. This boundary is required because vLLM does not
expose one stable, version-independent HTTP control endpoint for cancellation, abort, deadline, and
supersession. The controller receives an `OrchestratedHttpControl` containing the runner sample id,
control kind, absolute deadline when applicable, and the replacement sample id for supersession.
It implements async `begin_run`, `dispatch`, and `end_run`; `dispatch` returns only whether the
deployment sent the control. The driver keeps the original response open to observe late results
and submits a real `<sample-id>:replacement` HTTP request for supersession. Controller code must not
report server acceptance or GPU accounting through that boolean.

The direct driver dispatches native `CANCEL`, `ABORT`, `DEADLINE`, and `SUPERSEDE` controls on the
same session as the request. Supersession also submits a real replacement request whose correlation
id is `<sample-id>:replacement`; it does not model replacement as a local flag. A single event pump
routes concurrent progress, partial-result, terminal-result, and lifecycle events by operation and
frame identity. Terminal outcomes come from received lifecycle or `RESULT_DROP_REASON` evidence,
while post-dispatch progress, partial results, and successful result pushes are retained as late
result observations. The driver's boolean control result still means dispatch only. Server
acceptance, backend stop time, and GPU attribution remain exclusively owned by the independent
accounting probe.

The manifest is a JSON object with `scenario: "stale_work"`, a `stale_work_ratio` of `0.1`, `0.3`,
or `0.5`, the installed adapter version and lowercase Git revision, public-safe model/engine/GPU
labels, fixed arrival and cancellation schedule names,
`arrival_interval_seconds`, `control_delay_seconds`, prompt and completion token limits, warmup and sample counts,
`max_in_flight`, and a seeded `gpu_accounting` declaration. Acceptance thresholds are evaluated
only for exact scheduled-batch CUDA attribution or non-overlapping dedicated-device active time.
Per-request inference intervals remain a named proxy and cannot substantiate GPU-second claims.

Each zero-argument driver factory returns an object with a canonical `baseline` and async
`begin_run`, `warmup`, `start`, and `end_run` methods. `start` returns a live operation with
`apply_control`, `wait`, and `close`. The runner owns sample identity, arrival and control timing,
bounded concurrency, and baseline order. Operations return `StaleWorkResult` with terminal outcome,
useful-result weight, and the number of result events observed after control dispatch; they cannot
overwrite runner-owned identity, timing, or GPU evidence. `apply_control` reports only whether the
client dispatched the control. It is not a server acknowledgement. The aggregate report classifies
post-dispatch events as late results only when the independent accounting probe confirms that the
server accepted the control. This preserves raw HTTP observations without claiming that a client
disconnect was accepted as cancellation, abort, expiry, or supersession.

The zero-argument accounting-probe factory declares `method`, `scope`, and `source` values that must
exactly match `gpu_accounting` in the manifest. It creates one accounting session per sample. That
session receives the runner's monotonic operation-start timestamp and independently reports whether
the server accepted a dispatched control, when the backend stopped, and attributed GPU seconds.
Request duration, HTTP disconnect, and successful NNRP frame submission are not accepted as GPU-stop
proxies. Without this probe the command refuses to run. A raw baseline may complete stale work when
it cannot dispatch or the server does not accept the scheduled control; stale identity still comes
from the shared schedule. Changed schedules, invalid terminal semantics, probe declaration drift,
and sensitive evidence fail the run before either output is published.

For deployments that expose request-correlated CUDA accounting through a sidecar, the adapter
provides `HttpStaleWorkAccountingProbe` and `HttpAccountingProbeConfig` in
`vllm_nnrp_adapter.stale_work_accounting`. This is benchmark infrastructure, not an NNRP control
surface. The factory can configure the sidecar endpoint and credentials without putting either in
the committed manifest:

```python
from vllm_nnrp_adapter.stale_work_accounting import (
    HttpAccountingProbeConfig,
    HttpStaleWorkAccountingProbe,
)


def make_cuda_accounting_probe() -> HttpStaleWorkAccountingProbe:
    return HttpStaleWorkAccountingProbe(
        HttpAccountingProbeConfig(
            endpoint="https://accounting.internal.example/v1/stale-work",
            method="cuda_event_attribution",
            scope="scheduled_batch",
            source="deployment-cuda-events",
            api_key="read-from-deployment-secret-store",
        )
    )
```

The sidecar accepts JSON objects with schema version `nnrp-stale-work-accounting/v1` and an `action`
of `begin_run`, `start_sample`, `operation_started`, `finish_sample`, `close_sample`, or `end_run`.
The adapter supplies the immutable workload schedule and correlation identity. `finish_sample` must
return the active `baseline`, `sample_id`, `method`, `scope`, and `source`, plus
`control_accepted`, optional `control_accepted_after_seconds`, `backend_stopped_after_seconds`, and
`gpu_seconds`. All durations are non-negative offsets from that sample's operation start. The probe
rejects mismatched identity or accounting declarations, so telemetry from another request, run, or
measurement source cannot silently enter the evidence artifact. The sidecar must derive acceptance
and backend stop from server-side observation; it must not infer them from client disconnect or the
adapter's control-dispatch result.

The raw and aggregate files are written atomically only after all three baselines validate. The
outcome file is always written: successful runs record the randomized baseline order and sample
count, while failed runs record only the safe phase, baseline/sample identity when known, and error
type. Exception text is deliberately excluded so local paths, endpoints, and request content cannot
leak into committed failure evidence.

## Priority-Burst Adoption Workload

`run-priority-burst-workload` measures whether urgent requests receive useful scheduler treatment
while an existing normal-priority queue remains runnable. It executes raw OpenAI HTTP/SSE,
HTTP/SSE with equivalent external orchestration, and direct NNRP in seeded order against one fixed
arrival schedule. Every baseline receives the same request ids, prompt/output limits, normal
traffic, and contiguous urgent burst.

```bash
vllm-nnrp-adapter run-priority-burst-workload \
  --manifest artifacts/priority-burst.json \
  --raw-output artifacts/priority-burst-raw.json \
  --report-output artifacts/priority-burst-report.json \
  --outcome-output artifacts/priority-burst-outcome.json \
  --driver raw_openai_http_sse=deployment.benchmarks:make_raw_priority_driver \
  --driver orchestrated_http_sse=deployment.benchmarks:make_orchestrated_priority_driver \
  --driver direct_nnrp=deployment.benchmarks:make_direct_priority_driver \
  --observation-probe deployment.benchmarks:make_scheduler_probe
```

The manifest uses `scenario: "priority_burst"`,
`arrival_schedule: "fixed_interval_contiguous_burst"`, and
`priority_application: "pre_backend_dispatch"`. It fixes `burst_start_ordinal`, `burst_size`,
`normal_priority`, `urgent_priority`, `minimum_queue_depth`, total samples, bounded client
concurrency, warmup, seed, model, engine, GPU class, and request sizes. vLLM schedules lower
integer values before higher values, so `urgent_priority` must be lower than `normal_priority`.
Normal traffic is required on both sides of the burst; a manifest cannot manufacture an urgent-only
comparison.

Drivers implement async `begin_run`, `warmup`, `start`, and `end_run`; each live operation implements
`wait` and `close`. The raw driver must not inject priority. The equivalently orchestrated and direct
NNRP drivers must apply the case's requested backend priority before vLLM dispatch. A live
`PRIORITY_UPDATE` after dispatch is a separate conditional capability and is not silently used as a
substitute for admission priority.

The scheduler observation probe is independent of all three drivers and declares a `method`,
`scope`, and public-safe `source` that exactly match the manifest. Per sample it reports queue,
backend-start, backend-completion, observed-priority, queue-depth, and continuously-runnable state.
Only dedicated-engine vLLM scheduler traces or engine request events can evaluate acceptance.
Shared-engine measurements and request-duration proxies are retained as diagnostics but cannot
substantiate the priority claim. Every urgent sample must observe the declared minimum queue depth,
and direct NNRP must expose the requested backend priority.

The aggregate report publishes urgent p50/p95/p99 latency, normal latency, completed-request
throughput, continuously-runnable normal-request starvation, queue-saturation evidence, and
priority evidence. The Preview4 hypothesis passes only when direct NNRP reduces urgent p95 by at
least 30% versus raw HTTP/SSE, leaves no runnable normal request incomplete after drain, and loses
no more than 5% total throughput. Failed and non-evaluable hypotheses remain in the report.

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
