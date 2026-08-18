# 00 - Scope And Ownership

## Product Boundary

- [x] Keep the adapter positioned as vLLM compatibility for NNRP deployments, not as a claim that
  token generation becomes faster than OpenAI HTTP/SSE.
- [x] Keep vLLM responsible for model loading, tokenization, scheduling, batching, generation, and
  backend-specific limits.
- [x] Keep NNRP responsible for carrier selection, sessions, runtime control, flow control, typed
  payloads, object/cache references, result delivery, and wire diagnostics.
- [x] Keep `openai-compatible/1` responsible for JSON request envelopes, profile events, errors,
  usage, tool-call events, and capability documents.
- [x] Reject any implementation route that starts a second HTTP service or converts HTTP/SSE traffic
  back into NNRP for the production path.

## Module Ownership

- [x] Assign dependency and public export changes to `01-contract-and-dependency-adoption.md`.
- [x] Assign vLLM serving-object compatibility to `vllm_backend.py` and `vllm_factory.py` under
  `02-vllm-version-and-backend-bindings.md`.
- [ ] Assign native role/session orchestration to `nnrp_runtime.py` under
  `03-native-server-and-session-lifecycle.md`.
- [ ] Add a dedicated runtime-control module owned by `04-runtime-control-mapping.md`.
- [ ] Add a dedicated runtime-object/cache module owned by
  `05-runtime-objects-cache-and-payloads.md`.
- [ ] Assign CLI and provider composition to `embedded.py`, `cli.py`, and a provider-serving module
  under `06-transport-provider-serving.md`.
- [ ] Assign OpenAI envelope/event semantics to `profile.py` and `adapter.py` under
  `07-openai-profile-runtime.md`.
- [ ] Add a dedicated observation/export module owned by `08-observability-and-exporters.md`.
- [ ] Assign manifests, execution-plan handling, and E2E targets to
  `09-conformance-and-vllm-integration.md`.
- [ ] Assign benchmark, workflow, README, and final export integration to
  `10-benchmarks-packaging-and-docs.md`.
- [ ] Assign adoption experiments, value thresholds, and capability evidence to
  `11-adoption-value-and-overdesign-gate.md`.

## Delivery Tiers

- [ ] Treat native serving, direct vLLM binding, partial results, cancellation, abort, deadlines,
  backpressure, credits, traces, drop diagnostics, and independent wire E2E as the core delivery
  loop.
- [ ] Advertise priority updates, route hints, runtime objects, and cache references only when the
  adapter can name and test the concrete vLLM mechanism that applies them.
- [ ] Keep KV-cache identity integration, standalone metrics serving, and backend-specific scheduler
  extensions optional until each has an implemented mechanism and independent evidence.
- [x] Distinguish an installable vLLM version from a tested compatibility binding in every package,
  diagnostic, document, and release claim.

## Completion Discipline

- [ ] Keep every implementation item in one workstream before code is written; do not create hidden
  child tasks merely to close a parent checkbox.
- [ ] Keep parent checkboxes open until every listed child behavior and test is complete.
- [ ] Run the full local CI contract before each commit is pushed.
- [ ] Require remote CI and configured AI review before merging a pull request.
- [ ] Require every advertised capability to have a real backend mechanism, an observable effect,
  an independent benchmark scenario, and a quantitative acceptance threshold.
- [ ] Keep capabilities without that evidence out of release claims and mandatory release gates.
