# vLLM NNRP Adapter Value Model

## Product Hypothesis

The adapter exists to make vLLM a governable NNRP runtime node. It is not a claim that changing the
application carrier makes token generation faster. The primary value is the ability to stop useless
work, protect urgent work, bound producers, transfer large runtime payloads efficiently, explain
drops, and contain vLLM version changes inside a tested binding.

Model-dominated chat latency remains a compatibility and regression check. It is not sufficient
evidence for or against the runtime-orchestration value of NNRP.

## Comparison Baselines

Every adoption claim uses all three baselines:

1. **Raw OpenAI HTTP/SSE**: normal vLLM serving behavior without equivalent orchestration.
2. **HTTP/SSE plus equivalent orchestration**: the same cancellation, deadline, priority,
   backpressure, and recovery policy implemented outside NNRP.
3. **Direct NNRP**: the native NNRP server and in-process vLLM binding without an HTTP/SSE relay.

The first comparison shows end-to-end operational improvement. The second isolates whether NNRP
provides a simpler or more efficient mechanism than a custom HTTP control layer.

## Value Metrics

### Deadline-Weighted Useful Goodput

`sum(useful_result_weight before deadline) / wall_clock_seconds`

A completed result has zero useful weight after cancellation, supersession, or expiry. Workloads may
assign higher weights to urgent requests, but the same weights must be used for every baseline.

### Wasted Compute Ratio

`gpu_seconds spent on cancelled, expired, superseded, or dropped work / total gpu_seconds`

GPU accounting must measure the point at which backend work actually stops, not only when the
adapter acknowledges a control frame.

### Cancellation Effect Latency

Time from the adapter accepting `CANCEL` or `ABORT` to the backend ceasing generation for the target
operation. Report p50, p95, p99, and the late-result rate.

### Transport Amplification

`bytes transferred across the provider / useful application payload bytes`

Include framing, metadata, retries, and retransmitted application data. Report request and result
directions separately.

### Copy Amplification

`bytes copied or materialized by the adapter / useful application payload bytes`

When exact byte-copy instrumentation is unavailable, report allocated bytes, peak resident memory,
and serialization CPU as named proxies rather than presenting them as exact copy counts.

### Semantic Defect Rate

`requests with a missing, duplicate, late, misrouted, or incorrectly terminated event / requests`

Conformance failures, duplicate terminal results, ignored mandatory controls, and unexplained result
drops all count as defects.

### Upgrade Containment

Record application files changed, adapter binding files changed, fixture files changed, conformance
failures, and elapsed time to restore service when moving between advertised vLLM anchors.

## Experiment Matrix

| Scenario | Controlled variable | Primary evidence |
| --- | --- | --- |
| Stale work | 10%, 30%, and 50% cancellation, expiry, or supersession | Wasted compute, cancellation effect latency, useful goodput |
| Priority burst | Urgent arrivals during saturated queues | Priority p95/p99, starvation, total throughput |
| Pressure recovery | Producer overrun and credit exhaustion | Queue bound, recovery latency, drops, diagnostics |
| Heavy payload | 1 MiB, 16 MiB, and 64 MiB request/result objects | Transport/copy amplification, CPU, peak memory |
| Failure recovery | Disconnect, provider failure, retry, restart | Duplicate work, recovery time, terminal uniqueness |
| Version upgrade | Legacy, transition, and current vLLM anchors | Application churn, binding churn, conformance failures |

Each run fixes the model, engine, GPU, prompt/output limits, arrival schedule, cancellation schedule,
warmup, random seed, and sample count. Compared runs are randomized or interleaved and report raw
samples plus confidence intervals.

## Preview4 Acceptance Hypotheses

These thresholds are falsifiable release hypotheses, not guaranteed marketing claims:

- A control-free workload demonstrates throughput equivalence only when the complete two-sided 90%
  confidence interval for relative NNRP throughput lies inside `[-3%, +3%]` versus equivalent
  HTTP/SSE.
- At 30% stale work, NNRP reduces wasted GPU seconds by at least 40% against raw HTTP/SSE and is no
  more than 3% worse than HTTP/SSE with equivalent orchestration.
- Fewer than 0.1% of operations produce a late result after accepted cancellation, abort, or expiry.
- A priority burst reduces urgent-request p95 latency by at least 30%, leaves no continuously
  runnable normal operation incomplete after the queue drains, and regresses total throughput by no
  more than 5%.
- The workload manifest chooses serialization CPU or peak adapter memory as its primary
  heavy-payload metric before execution. That metric improves by at least 30% without increasing
  semantic defects.
- Moving between advertised vLLM anchors requires no application-code changes; version-specific
  changes stay inside adapter bindings and fixtures.

A failed hypothesis remains in the published evidence with its failure mode. It cannot be replaced
by an unrelated throughput result.

A threshold becomes a mandatory release gate only after the capability ledger classifies the
capability as core. Results for conditional capabilities decide whether they should become core;
they do not force an implementation merely because the protocol can express the behavior.

## Anti-Overdesign Gate

An advertised capability must have all four of the following:

1. A concrete vLLM or adapter mechanism that applies the requested behavior.
2. An observable backend or wire effect proving that the mechanism ran.
3. An independent scenario that does not rely on the same mock used by the implementation test.
4. A quantitative acceptance threshold and reproducible evidence artifact.

Capabilities missing any item remain optional or experimental. They are excluded from release
claims and mandatory release gates. This rule prevents a large control surface from being mistaken
for delivered operational value.
