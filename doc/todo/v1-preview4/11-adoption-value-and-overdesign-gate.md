# 11 - Adoption Value And Overdesign Gate

## Evidence Contract

- [x] Use `doc/value-model.md` as the source of truth for value metrics, comparison baselines,
  scenarios, and acceptance hypotheses.
- [x] Compare raw OpenAI HTTP/SSE, HTTP/SSE with equivalent orchestration, and direct NNRP rather
  than attributing orchestration benefits to the carrier alone.
- [x] Keep model, engine, GPU, arrival schedule, cancellation schedule, prompt/output limits,
  warmup, and random seed identical across compared runs.
- [x] Persist machine-readable workload manifests, raw samples, aggregate results, confidence
  intervals, adapter/version metadata, and failure records.
- [x] Exclude host IPs, usernames, tokens, request content, and machine identifiers from committed
  evidence.

## Governable-Runtime Scenarios

- [ ] Run stale-work ratios of 10%, 30%, and 50% with cancellation, abort, deadline expiry, and
  superseding requests.
- [ ] Measure useful GPU seconds, wasted GPU seconds, cancellation effect latency, late-result rate,
  and deadline-weighted useful goodput.
  The runner includes a correlated HTTP accounting-sidecar bridge with strict run, sample,
  accounting-source, and lifecycle validation. This item remains open until real CUDA-attributed
  10%, 30%, and 50% workload evidence is recorded.
- [ ] Run priority-burst scenarios under queue saturation and measure high-priority p95/p99,
  low-priority starvation, throughput, and scheduler fairness.
  The adapter now provides a deterministic three-baseline runner, independent scheduler-observation
  contract, strict aggregation, and acceptance evaluator. This item remains open until a dedicated
  vLLM/GPU run records queue-saturation and observed-priority evidence.
- [ ] Run backpressure and credit-exhaustion scenarios and measure bounded queue depth, recovery
  latency, producer overrun, and dropped-result diagnostics.
- [ ] Run 1 MiB, 16 MiB, and 64 MiB payload scenarios and measure transport amplification, copy
  amplification, serialization CPU, peak memory, and delivered useful bytes.
- [ ] Run disconnect, provider failure, retry, and process-restart scenarios and measure duplicate
  execution, recovery time, terminal-result uniqueness, and result-drop reasons.
- [ ] Upgrade between compatibility anchors and measure application-code changes, adapter-only
  changes, conformance failures, and time to restore service.

## Acceptance Hypotheses

- [ ] Apply a capability threshold as a mandatory release gate only after the capability ledger
  classifies that capability as core; keep conditional capability results as classification
  evidence until then.
- [ ] Validate control-free throughput equivalence by keeping the complete two-sided 90% confidence
  interval for relative NNRP throughput inside `[-3%, +3%]` versus equivalent HTTP/SSE.
- [ ] Validate that a 30% stale-work workload reduces wasted GPU seconds by at least 40% against raw
  HTTP/SSE and stays within a 3% non-inferiority margin of HTTP/SSE with equivalent orchestration.
- [ ] Validate a late-result rate below 0.1% after accepted cancellation, abort, or deadline expiry.
- [ ] Validate at least a 30% reduction in high-priority p95 latency under the priority-burst
  scenario, with no continuously runnable lower-priority operation left incomplete when the queue
  drains and no more than 5% total-throughput regression.
- [ ] Choose serialization CPU or peak adapter memory as the primary heavy-payload metric in the
  workload manifest, then validate at least a 30% reduction without increasing semantic defect
  rate.
- [ ] Validate that moving between advertised compatibility anchors requires no application-code
  changes and confines version-specific changes to adapter bindings and fixtures.
- [ ] Publish failed hypotheses with raw evidence; do not relabel a failed value experiment as a
  successful transport benchmark.

## Capability Ledger And Release Gate

- [ ] Maintain a ledger mapping every advertised capability to its NNRP surface, concrete vLLM
  mechanism, observable effect, benchmark scenario, metric, threshold, and evidence artifact.
- [x] Remove a capability from advertisements when its vLLM mechanism is absent, emulated only by a
  mock, or unable to produce the documented effect.
- [x] Keep optional or experimental capabilities out of mandatory release gates until their ledger
  entries are complete.
- [x] Fail release validation when a core capability lacks mechanism, observation, independent
  scenario, threshold, or reproducible evidence.
- [ ] Publish one concise adoption report that separates compatibility, operational control,
  resource efficiency, transport efficiency, and protocol overhead.
