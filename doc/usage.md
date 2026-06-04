# vLLM NNRP Adapter Usage

This document records the preview3 runtime shape for host installs, vLLM containers, adapter startup, and OpenAI API profile conformance.

## Host Installation

Use the normal package when you only need mock conformance, benchmark smoke, or adapter unit tests:

```bash
python -m pip install vllm-nnrp-adapter
```

Install the vLLM extra only on a host or container that can actually import and run vLLM:

```bash
python -m pip install "vllm-nnrp-adapter[vllm]"
```

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

The report includes non-streaming roundtrip latency, streaming event latency and throughput, and cancellation latency.

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
