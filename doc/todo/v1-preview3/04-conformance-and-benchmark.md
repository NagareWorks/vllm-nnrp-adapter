# Conformance And Benchmark

- [x] Add unit tests for profile envelope validation and chunk-to-event mapping.
- [x] Add adapter runner command for `nnrp-conformance`.
- [x] Add conformance capability manifest for Level 1 chat baseline.
- [x] Add benchmark harness for streaming throughput, p50/p95 event latency, cancellation latency, and non-streaming roundtrip.
- [x] Add a true in-process benchmark path for `OpenAI HTTP SSE` versus `NNRP submit/result-push`.
- [x] Measure 4K, 8K, 16K, and 20K input prompts with controlled concurrency 1, 2, and 4.
- [x] Report TTFT, TPOT, RTT, output-token throughput, cancellation latency, and per-request error rate for both paths.
- [ ] Run the in-process benchmark against vLLM `0.18.1` and smoke the current stable vLLM line.
- [ ] Record release-readiness baseline results from the in-process NNRP path before release.
- [ ] Mark HTTP/SSE-to-NNRP relay benchmark data as non-release smoke evidence only.
