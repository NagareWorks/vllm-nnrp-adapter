# OpenAI NNRP API Level 1 Baseline

This baseline records the release-readiness checks for the first `openai-compatible/1` Level 1 adapter slice.

## Mock Profile Gate

Command:

```bash
vllm-nnrp-adapter run-benchmark \
  --output artifacts/openai-profile-benchmark.json \
  --backend mock \
  --iterations 200 \
  --warmup 20
```

Result summary from the local release gate:

| Scenario | p50 us | p95 us | Notes |
|---|---:|---:|---|
| `chat.non_streaming.roundtrip` | 8.3 | 14.8 | Adapter-only request envelope and completion mapping. |
| `chat.streaming.event_latency` | 4.3 | 15.9 | Adapter-only event mapping, 600 total events, 113509.53 events/s. |
| `chat.streaming.cancellation_latency` | 43.5 | 80.0 | Adapter cancellation policy path. |

## vLLM 0.18.1 Lower-Bound Smoke

Environment:

| Field | Value |
|---|---|
| Host | `ubuntu@106.52.245.226` |
| GPU | NVIDIA Tesla T4, 15 GiB |
| Python | 3.11.15 |
| vLLM | 0.18.1 |
| Model | `/home/ubuntu/models/Qwen2.5-3B-Instruct` |
| Runtime flags | `dtype=float16`, `max_model_len=1024`, `gpu_memory_utilization=0.65`, `enforce_eager=True` |

Smoke result:

- vLLM loaded the Qwen2.5-3B model and generated a non-empty response.
- `OpenAiNnrpAdapter` emitted `response.diagnostics` followed by `response.completed` for a real vLLM-backed request.
- The adapter request factory resolved `vllm.entrypoints.openai.chat_completion.protocol:ChatCompletionRequest`.

Benchmark smoke result, `iterations=2`, `warmup=1`:

| Scenario | p50 us | p95 us | Notes |
|---|---:|---:|---|
| `chat.non_streaming.roundtrip` | 423376.809 | 433797.272 | Includes model generation on Tesla T4. |
| `chat.streaming.event_latency` | 8.922 | 425911.447 | First event includes model generation; subsequent mapped events are adapter-level. |
| `chat.streaming.cancellation_latency` | 423765.130 | 425710.422 | Includes one model generation before cancellation terminal event. |

## Current Stable vLLM Line

PyPI reported `vllm==0.22.0` as the current stable release during the baseline pass. A separate Python 3.11 environment installed `vllm==0.22.0`, imported the package, and constructed `ChatCompletionRequest` through the same request-factory path used by `0.18.1`.

The `0.22.0` check is a compatibility smoke for the OpenAI request type boundary. Full GPU benchmark numbers should be refreshed when a production-like vLLM 0.22.x runtime image is selected for the adapter release train.
