# OpenAI NNRP Direct vLLM Baseline

This benchmark records the first release-readiness baseline for the in-process OpenAI NNRP adapter path. It measures
the adapter path that calls vLLM's OpenAI serving object directly and streams NNRP profile events without routing through
the OpenAI HTTP/SSE server.

## Environment

| Field | Value |
| --- | --- |
| Date | 2026-06-04 |
| Host class | Single-GPU Linux benchmark host |
| GPU | NVIDIA Tesla T4, single GPU |
| vLLM | `0.18.1` |
| Adapter commit | `5e1251d` |
| Model | Huihui-MoE-0.8B-2E local fixture |
| Served model | `huihui` |
| Max model length | `21504` |
| Max completion tokens | `32` |
| Warmup | `1` request per scenario |
| Iterations | `4` requests per scenario |
| NNRP raw report | `doc/benchmarks/raw/openai-nnrp-direct-vllm-0.18.1-t4-2026-06-04.json` |
| HTTP raw report | `doc/benchmarks/raw/openai-http-sse-vllm-0.18.1-t4-2026-06-04.json` |

The vLLM engine reported a maximum KV-cache concurrency of `4.27x` for `21504` tokens per request on this T4 host. The
20K prompt with concurrency 4 therefore runs close to the host's long-context capacity limit.

## Results

The delta column reports `NNRP TTFT - HTTP TTFT`; negative values mean the NNRP path was faster for TTFT.

| Prompt tokens | Concurrency | NNRP success/error | HTTP success/error | NNRP TTFT p50 | HTTP TTFT p50 | NNRP TPOT p50 | HTTP TPOT p50 | NNRP RTT p50 | HTTP RTT p50 | NNRP tok/s | HTTP tok/s | TTFT delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 4 / 0 | 4 / 0 | 148.00 ms | 150.49 ms | 75.16 ms | 75.06 ms | 2478.36 ms | 2476.59 ms | 12.91 | 12.92 | -2.48 ms |
| 4096 | 2 | 4 / 0 | 4 / 0 | 289.25 ms | 290.84 ms | 95.63 ms | 95.53 ms | 4925.93 ms | 3214.31 ms | 15.82 | 20.06 | -1.59 ms |
| 4096 | 4 | 4 / 0 | 4 / 0 | 350.57 ms | 257.35 ms | 115.65 ms | 112.85 ms | 3935.71 ms | 3755.63 ms | 32.52 | 34.07 | +93.22 ms |
| 8192 | 1 | 4 / 0 | 4 / 0 | 158.91 ms | 157.53 ms | 77.25 ms | 77.18 ms | 2551.60 ms | 2549.93 ms | 12.36 | 12.37 | +1.39 ms |
| 8192 | 2 | 4 / 0 | 4 / 0 | 303.95 ms | 306.71 ms | 99.77 ms | 99.68 ms | 3340.13 ms | 3339.76 ms | 19.13 | 19.14 | -2.76 ms |
| 8192 | 4 | 4 / 0 | 4 / 0 | 395.60 ms | 400.92 ms | 119.57 ms | 119.42 ms | 4102.39 ms | 4102.95 ms | 31.20 | 31.19 | -5.32 ms |
| 16384 | 1 | 4 / 0 | 4 / 0 | 224.31 ms | 227.09 ms | 81.04 ms | 81.04 ms | 1844.98 ms | 1844.84 ms | 8.82 | 8.82 | -2.78 ms |
| 16384 | 2 | 4 / 0 | 4 / 0 | 241.25 ms | 239.99 ms | 145.53 ms | 145.96 ms | 2858.90 ms | 2860.08 ms | 13.19 | 13.18 | +1.26 ms |
| 16384 | 4 | 4 / 0 | 4 / 0 | 585.34 ms | 614.83 ms | 200.66 ms | 200.96 ms | 3796.02 ms | 3830.57 ms | 17.91 | 17.75 | -29.50 ms |
| 20480 | 1 | 4 / 0 | 4 / 0 | 266.26 ms | 266.51 ms | 166.21 ms | 166.32 ms | 2759.11 ms | 2761.05 ms | 5.80 | 5.79 | -0.25 ms |
| 20480 | 2 | 4 / 0 | 4 / 0 | 469.51 ms | 469.91 ms | 209.79 ms | 209.83 ms | 3616.32 ms | 3614.57 ms | 9.12 | 9.12 | -0.40 ms |
| 20480 | 4 | 4 / 0 | 4 / 0 | 634.65 ms | 636.42 ms | 220.72 ms | 220.85 ms | 3834.35 ms | 3837.74 ms | 10.46 | 10.45 | -1.78 ms |

## Notes

- The NNRP columns exercise the in-process engine-direct adapter path. The HTTP columns exercise vLLM's
  OpenAI-compatible SSE endpoint under the same model, GPU, max length, completion length, warmup, and iteration settings.
- The benchmark uses synthetic long prompts generated with a single-token unit for this model family. Earlier
  `benchmark token` prompt generation overstated long-context size for this tokenizer and is not used as baseline data.
- The HTTP endpoint was measured in a separate run so only one vLLM engine was resident on the GPU at a time.
- Historical HTTP/SSE-to-NNRP relay measurements are smoke evidence only. They are not release-readiness evidence for the
  optimized NNRP runtime path.
