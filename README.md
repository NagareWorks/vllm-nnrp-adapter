<p align="center">
  <img src="assets/nnrp-readme-banner.svg" alt="NNRP - Neural Network Runtime Protocol" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/NagareWorks/vllm-nnrp-adapter/actions"><img alt="CI" src="https://img.shields.io/badge/CI-python-22c55e"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white"></a>
  <a href="https://nagareworks.github.io/nnrp-doc/"><img alt="Docs" src="https://img.shields.io/badge/docs-nnrp--doc-38bdf8"></a>
  <a href="https://github.com/NagareWorks/vllm-nnrp-adapter/blob/main/LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-64748b"></a>
</p>

# vllm-nnrp-adapter

vLLM adapter for serving the frozen NNRP OpenAI-compatible API profile.

This repository binds vLLM's OpenAI-compatible serving surface to NNRP sessions, result streams, cancellation, diagnostics, and conformance recipes. It is not a second OpenAI HTTP server and it does not redefine vLLM scheduling policy. The adapter owns the translation boundary between OpenAI-compatible request/response objects and NNRP profile events.

## Scope

The first implementation slice targets NNRP `openai-compatible/1` Level 1:

1. `chat.completions.create` request envelopes.
2. Streaming text deltas.
3. Non-streaming completion bodies.
4. Cancellation and timeout mapping.
5. OpenAI-compatible error bodies.
6. Usage summary events.
7. Tool-call event pass-through when vLLM and the selected model expose tool-call data.
8. Capability document generation for SDK feature probes and conformance selection.

Level 2 `responses.create` and Level 3 model discovery / embeddings are explicit follow-up work, not hidden behavior in the Level 1 adapter.

## Supported vLLM Line

The optional dependency accepts `vllm>=0.18.0,<0.27` for installation. This is not blanket support
for every minor release in that interval. Runtime binding is limited to the feature-probed `0.18.x`,
`0.22.x`, and `0.26.x` families, anchored at `0.18.1`, `0.22.1`, and `0.26.0`. See the
[generated compatibility table](doc/vllm-compatibility.md).

Production streaming uses the in-process engine-direct path. It does not parse vLLM's rendered SSE
output or relay through an HTTP server. Tool-call streaming is therefore omitted from the published
capability manifest until a version-bound engine-direct parser is implemented and tested. The
explicit `HttpSseSmokeBackend` remains internal comparison tooling and is never auto-selected.

## Layout

- `src/vllm_nnrp_adapter/profile.py`: frozen profile constants, request envelope validation, event builders, and capability document helpers.
- `src/vllm_nnrp_adapter/adapter.py`: profile-level async request handler that maps backend responses to NNRP profile events.
- `src/vllm_nnrp_adapter/nnrp_contract.py`: Preview4 `nnrp-py` version and native-role contract validation.
- `src/vllm_nnrp_adapter/vllm_backend.py`: vLLM serving-object wrapper and method probing.
- `src/vllm_nnrp_adapter/vllm_compat.py`: named compatibility registry and startup validation.
- `src/vllm_nnrp_adapter/http_sse_smoke.py`: explicit HTTP/SSE comparison backend outside production selection.
- `src/vllm_nnrp_adapter/conformance.py`: OpenAI NNRP API conformance plan executor and result writer.
- `conformance/openai-api-capabilities.json`: Level 1 capability declaration consumed by `nnrp-conformance`.
- `tests/`: profile and adapter mapping tests that do not require a GPU runtime.
- `doc/todo/v1-preview4/`: Preview4 implementation checklist for the adapter.

## Development

```powershell
python -m pip install -e .[dev]
ruff check .
pytest -q
python -m build
```

Install the vLLM extra only in an environment that can actually host vLLM:

```powershell
python -m pip install -e .[dev,vllm]
```

See [doc/usage.md](doc/usage.md) for host installation, vLLM container installation, backend factory setup, conformance, benchmark, and request-envelope examples.

## Adapter Shape

```python
from vllm_nnrp_adapter import OpenAiNnrpAdapter, OpenAiNnrpCapabilityDocument

capabilities = OpenAiNnrpCapabilityDocument.level1(models=("example-model",))
adapter = OpenAiNnrpAdapter(backend=my_vllm_backend, capabilities=capabilities)

events = [
    event
    async for event in adapter.handle_request(
        {
            "schema_version": "openai-compatible/1",
            "operation": "chat.completions.create",
            "body": {
                "model": "example-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        }
    )
]
```

The adapter consumes JSON-compatible request envelopes and emits JSON-compatible profile events.
Normal streaming events are partial results; `response.completed`, `response.error`, and
`response.cancelled` are terminal profile events.

## Native Server Quick Start

Start the adapter with an NNRP application endpoint and explicit provider-local routes:

```bash
vllm-nnrp-adapter serve \
  --backend my_vllm_app.serving:make_backend \
  --endpoint nnrp://runtime.example/vllm \
  --provider-route tcp=tcp://0.0.0.0:7766 \
  --provider-route ipc=unix:///run/nnrp-vllm.sock \
  --transport-policy auto
```

The adapter does not implement carriers or package native provider artifacts. Installed `nnrp-py`
transport distributions own TCP, QUIC, IPC, and WebSocket listeners; the adapter passes typed
routes and policy to the native server role.

## Conformance Smoke

`nnrp-conformance` owns the OpenAI NNRP API profile recipes, plan, and result schemas. Generate the
Preview4 plan from the suite, run it through the adapter, and return the results to the independent
validator:

```powershell
cargo run --manifest-path ../nnrp-conformance/Cargo.toml -p nnrp-conformance-runner -- `
  api-profile-plan `
  --protocol ../nnrp-conformance/protocol/nnrp-1-preview4/manifest.json `
  --profile ../nnrp-conformance/profiles/openai-compatible/1/manifest.json `
  --capabilities conformance/openai-api-capabilities.json `
  --output artifacts/api-profile-plan.json `
  --results-path artifacts/api-profile-results.json `
  --evidence-dir artifacts/api-profile-evidence

vllm-nnrp-adapter run-conformance-plan `
  --plan artifacts/api-profile-plan.json `
  --output artifacts/api-profile-results.json `
  --backend mock

cargo run --manifest-path ../nnrp-conformance/Cargo.toml -p nnrp-conformance-runner -- `
  validate-api-profile-results `
  --plan artifacts/api-profile-plan.json `
  --results artifacts/api-profile-results.json
```

CI also reruns the same Level 1 plan with non-critical extensions removed. The selected case set and
all terminal outcomes must remain identical, so the diagnostics extension cannot become a hidden
baseline dependency. Each execution writes one evidence document per selected recipe.

Use `--backend module.path:factory_name` when running against a real vLLM serving object. The factory must return an object that exposes the adapter backend protocol.

For an existing vLLM `OpenAIServingChat` object, wrap it with the built-in request factory:

```python
from vllm_nnrp_adapter import create_vllm_backend

backend = create_vllm_backend(serving_chat)
```

The factory converts profile request bodies into vLLM `ChatCompletionRequest` objects at runtime. vLLM remains an optional dependency, so normal CI and conformance smoke tests can still run without a GPU serving stack.

## Benchmark Smoke

The adapter includes a small profile-level benchmark runner for the same backend boundary used by conformance. The default mock backend is useful for CI and regression checks; a real vLLM backend factory can be supplied with the same `module.path:factory_name` syntax.

```powershell
vllm-nnrp-adapter run-benchmark `
  --output artifacts/openai-profile-benchmark.json `
  --backend mock `
  --iterations 200 `
  --warmup 20
```

The report records non-streaming roundtrip p50/p95 latency, streaming event p50/p95 latency and event throughput, plus cancellation latency. Mock benchmark reports are adapter-shape smoke tests only; release-readiness numbers must use the in-process vLLM NNRP path.

The first release-readiness baseline is recorded in
[doc/benchmarks/openai-nnrp-direct-vllm-0.18.1-t4-2026-06-04.md](doc/benchmarks/openai-nnrp-direct-vllm-0.18.1-t4-2026-06-04.md).
It exercises vLLM `0.18.1` on a Tesla T4 with 4K/8K/16K/20K prompts and concurrency 1/2/4 through the engine-direct
NNRP profile path and the OpenAI-compatible HTTP/SSE endpoint. HTTP/SSE-to-NNRP relay measurements remain smoke evidence
only.

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `unsupported_operation` | The request operation is not part of `openai-compatible/1` Level 1. | Use `chat.completions.create` for the current profile level. |
| `request_timeout` | `nnrp.timeout_ms` expired before vLLM returned or streamed the next event. | Raise the timeout, reduce queue pressure, or inspect the vLLM engine logs. |
| `backend_overload` | vLLM reported overload, rate limit, or too many queued requests. | Treat it as scheduler pressure and retry according to the caller policy. |
| `scheduler_rejected` | The vLLM scheduler rejected the request path. | Check model capacity, admission policy, and active queue depth. |
| `backend_cancelled` | vLLM reported an abort/cancel path. | Check whether the NNRP caller cancelled the frame or whether vLLM aborted internally. |
| `vLLM compatibility check failed` | The installed version has no named binding, or its serving object is missing a required feature. | Use a family listed in the generated compatibility table and pass its `OpenAIServingChat` object to `create_vllm_backend`. |

Set `nnrp.diagnostics=true` in the request envelope to emit a `response.diagnostics` event with selected model, operation, and backend family before the request result stream.

## Contributors

<a href="https://github.com/NagareWorks/vllm-nnrp-adapter/graphs/contributors" title="Open the contributors graph for individual GitHub profiles and IDs.">
  <img src="https://contrib.rocks/image?repo=NagareWorks/vllm-nnrp-adapter" alt="Contributors" />
</a>

The avatar wall above updates automatically from the repository contributor list.
