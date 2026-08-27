# 10 - Benchmarks, Packaging, And Docs

## Benchmarks

- [ ] Keep this workstream focused on adapter overhead, regression detection, and reproducible
  technical evidence; keep adoption-value claims in `11-adoption-value-and-overdesign-gate.md`.
- [x] Measure profile validation and event mapping without vLLM to isolate adapter overhead.
- [ ] Measure native submit-to-admission, backend chunk-to-result, control-frame handling, and
  cancellation-to-abort latency.
  Adapter-side priority, deadline, cancellation dispatch, and registry cleanup are measured after
  native event delivery. The production server-operation path also measures submit delivery to
  backend dispatch and backend chunk to partial-result send with a synthetic native operation;
  native poll-to-delivery and actual FFI send completion remain required before this item is complete.
- [ ] Measure queue pressure, backpressure response, credit recovery, stale-result suppression, and
  object-reference overhead. Backpressure application, output-credit reservation, and blocked
  producer recovery after `CREDIT_UPDATE` are measured. Object declaration, reference validation,
  and reference resolution overhead are also measured; stale-result suppression remains.
- [ ] Re-run the 4K, 8K, 16K, and 20K prompt matrix at concurrency 1, 2, and 4.
- [ ] Compare direct NNRP and OpenAI HTTP/SSE on the same model, engine, GPU, arrival schedule,
  cancellation schedule, prompt, output limit, warmup, and concurrency.
- [ ] Record TTFT, TPOT, RTT, request throughput, output-token throughput, cancellation latency,
  adapter CPU, allocated bytes, error rate, and late-result count.
- [ ] Run the release matrix on vLLM `0.18.1`, `0.22.1`, and `0.26.0`.
- [x] Keep raw evidence and one combined comparison table under `doc/benchmarks` without host IPs,
  usernames, tokens, or machine ids.
- [x] Randomize or interleave compared runs, report sample counts and confidence intervals, and keep
  the benchmark input manifest beside each result.
- [x] State plainly that model-dominated chat parity validates compatibility; it is not the primary
  heavy-transport performance claim.

## Packaging And Workflows

- [x] Build wheel and sdist with the Preview4 `nnrp-py` dependency and updated vLLM range.
- [x] Verify wheel contents, `py.typed`, metadata, README, license, entrypoints, and clean install.
- [x] Keep vLLM optional for normal installation and CI.
- [ ] Add explicit workflow inputs for the three GPU smoke versions and evidence destination.
- [x] Make release reruns idempotent and keep publication opt-in.
- [x] Run the complete TODO, test, conformance, package, and documentation gate before a release tag.

## Documentation

- [x] Update README and usage examples from `serve-tcp` to provider-neutral native server startup.
- [x] Document application endpoints, provider routes, security, single-provider binding, and
  multi-provider server binding behavior.
- [ ] Document cancellation, abort, deadlines, priorities, budgets, progress, pressure, capability
  costs, route hints, traces, drops, recovery, objects, and cache behavior.
- [x] Document the exact Level 1 JSON boundary and the typed binary runtime-control boundary.
- [x] Update the English and Chinese `nnrp-doc` vLLM design pages to distinguish the
  `0.18.0..<0.27` installation band from the tested compatibility anchors.
- [x] Verify every documented adapter symbol against the installed package.
- [ ] Record evidence for every advertised compatibility anchor and every unsupported capability.
