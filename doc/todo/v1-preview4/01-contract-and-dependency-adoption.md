# 01 - Contract And Dependency Adoption

## Dependency Baseline

- [ ] Replace the Preview3 dependency with `nnrp-py>=1.0.0rc4.post14,<1.0.0rc5`.
- [ ] Remove imports of packet-only `serve_tcp`, `accept_server_session`, `ServerSession`, and
  Preview3 submit/result wrappers from the production adapter path.
- [ ] Import the frozen native server, operation, event, endpoint, transport-policy, provider-route,
  runtime-control, runtime-object, and cache-reference models from `nnrp-py`.
- [ ] Fail adapter startup with the installed `nnrp-py` version and required range when the binding
  does not expose the Preview4 native role contract.
- [ ] Keep the package free of its own Rust libraries; provider artifacts arrive through installed
  `nnrp-py` transport distributions.

## Public Surface

- [ ] Keep `OpenAiNnrpAdapter` as the profile mapper and `OpenAiNnrpCapabilityDocument` as the
  profile capability document.
- [ ] Keep `create_vllm_backend` as the explicit serving-object binding entrypoint.
- [ ] Replace TCP-specific production startup with one provider-neutral `serve` entrypoint accepting
  an `nnrp://` or `nnrps://` application endpoint and provider routes.
- [ ] Keep explicit test helpers separate from production exports.
- [ ] Export only application-facing typed APIs; do not export FFI handles, raw message codes, or
  generic frame-send functions.
- [ ] Remove Preview3 compatibility aliases instead of forwarding them to the Preview4 runtime.

## Contract Tests

- [ ] Add import tests against the installed Preview4 `nnrp-py` package rather than repository-local
  protocol doubles.
- [ ] Add public-signature tests for every exported adapter entrypoint.
- [ ] Add negative tests proving Preview3 runtime objects and packet sessions are rejected by the
  production entrypoint.
- [ ] Add dependency metadata tests that reject a release artifact carrying the old `rc3` range.
