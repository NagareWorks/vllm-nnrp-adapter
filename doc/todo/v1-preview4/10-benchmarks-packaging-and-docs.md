# 10 - Benchmarks, Packaging, And Docs

## Benchmarks

- [ ] Keep this workstream focused on adapter overhead, regression detection, and reproducible
  technical evidence; keep adoption-value claims in `11-adoption-value-and-overdesign-gate.md`.
- [ ] Measure profile validation and event mapping without vLLM to isolate adapter overhead.
- [ ] Measure native submit-to-admission, backend chunk-to-result, control-frame handling, and
  cancellation-to-abort latency.
- [ ] Measure queue pressure, backpressure response, credit recovery, stale-result suppression, and
  object-reference overhead.
- [ ] Re-run the 4K, 8K, 16K, and 20K prompt matrix at concurrency 1, 2, and 4.
- [ ] Compare direct NNRP and OpenAI HTTP/SSE on the same model, engine, GPU, arrival schedule,
  cancellation schedule, prompt, output limit, warmup, and concurrency.
- [ ] Record TTFT, TPOT, RTT, request throughput, output-token throughput, cancellation latency,
  adapter CPU, allocated bytes, error rate, and late-result count.
- [ ] Run the release matrix on vLLM `0.18.1`, `0.22.1`, and `0.26.0`.
- [ ] Keep raw evidence and one combined comparison table under `doc/benchmarks` without host IPs,
  usernames, tokens, or machine ids.
- [ ] Randomize or interleave compared runs, report sample counts and confidence intervals, and keep
  the benchmark input manifest beside each result.
- [ ] State plainly that model-dominated chat parity validates compatibility; it is not the primary
  heavy-transport performance claim.

## Packaging And Workflows

- [x] Build wheel and sdist with the Preview4 `nnrp-py` dependency and updated vLLM range.
- [ ] Verify wheel contents, `py.typed`, metadata, README, license, entrypoints, and clean install.
- [ ] Keep vLLM optional for normal installation and CI.
- [ ] Add explicit workflow inputs for the three GPU smoke versions and evidence destination.
- [ ] Make release reruns idempotent and keep publication opt-in.
- [ ] Run the complete TODO, test, conformance, package, and documentation gate before a release tag.

## Documentation

- [x] Update README and usage examples from `serve-tcp` to provider-neutral native server startup.
- [ ] Document application endpoints, provider routes, security, single-provider selection, and
  multi-provider probing.
- [ ] Document cancellation, abort, deadlines, priorities, budgets, progress, pressure, capability
  costs, route hints, traces, drops, recovery, objects, and cache behavior.
- [ ] Document the exact Level 1 JSON boundary and the typed binary runtime-control boundary.
- [ ] Update the English and Chinese `nnrp-doc` vLLM design pages to distinguish the
  `0.18.0..<0.27` installation band from the tested compatibility anchors.
- [ ] Verify every documented adapter symbol against the installed package.
- [ ] Record evidence for every advertised compatibility anchor and every unsupported capability.
