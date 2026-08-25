# vLLM NNRP Adapter Preview4 Implementation Todo

Preview4 moves the adapter from a Preview3 TCP packet bridge to the frozen NNRP/1 Preview4 native
server, runtime-control, runtime-object, cache-reference, provider, and wire-conformance contracts.
The production path remains an in-process vLLM backend binding and never relays through the OpenAI
HTTP/SSE endpoint.

## Workstreams

- [ ] [00 - Scope and ownership](00-scope-and-ownership.md)
- [x] [01 - Contract and dependency adoption](01-contract-and-dependency-adoption.md)
- [ ] [02 - vLLM version and backend bindings](02-vllm-version-and-backend-bindings.md)
- [x] [03 - Native server and session lifecycle](03-native-server-and-session-lifecycle.md)
- [ ] [04 - Runtime control mapping](04-runtime-control-mapping.md)
- [ ] [05 - Runtime objects, cache, and payloads](05-runtime-objects-cache-and-payloads.md)
- [ ] [06 - Transport provider serving](06-transport-provider-serving.md)
- [ ] [07 - OpenAI profile runtime](07-openai-profile-runtime.md)
- [ ] [08 - Observability and exporters](08-observability-and-exporters.md)
- [ ] [09 - Conformance and vLLM integration](09-conformance-and-vllm-integration.md)
- [ ] [10 - Benchmarks, packaging, and docs](10-benchmarks-packaging-and-docs.md)
- [ ] [11 - Adoption value and overdesign gate](11-adoption-value-and-overdesign-gate.md)

## Frozen Delivery Rules

- The protocol baseline is NNRP/1 Preview4 and the API profile is `openai-compatible/1` Level 1.
- The adapter consumes `nnrp-py>=1.0.0rc4.post19,<1.0.0rc5`; it does not preserve Preview3 SDK entrypoints.
- The declared vLLM installation range is `>=0.18.0,<0.27`; it is not a blanket support claim.
  Tested compatibility anchors are `0.18.1`, `0.22.1`, and `0.26.0`, and runtime diagnostics use
  the same binding registry as the published compatibility table.
- OpenAI-compatible request and profile-event bodies remain JSON as frozen by the profile. Runtime
  control, object, cache, flow, and diagnostics semantics use typed NNRP frames, never JSON control
  envelopes.
- The production server uses the Rust-backed `nnrp-py` native role API. Packet transport helpers and
  HTTP/SSE relay paths are tooling-only smoke surfaces.
- TCP, QUIC, IPC, and WebSocket providers own their behavior and native artifacts. The adapter owns
  provider composition and vLLM integration, not provider implementations.
- Unsupported mandatory control semantics fail explicitly and affect advertised capabilities. They
  are never silently ignored.
- An advertised capability must map to a real vLLM mechanism, produce an observable effect, pass an
  independent benchmark scenario, and satisfy its quantitative acceptance threshold.
- Mock backends prove mapping logic only. Release evidence requires an independent wire peer and a
  real vLLM process on the named GPU matrix.
- A workstream is checked here only when every checkbox in its linked document is complete.
