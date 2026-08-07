# 06 - Transport Provider Serving

## Provider-Neutral Startup

- [ ] Replace `serve-tcp` with `serve` accepting one `nnrp://` or `nnrps://` application endpoint.
- [ ] Accept explicit TCP, QUIC, IPC, and WebSocket provider routes with provider-local locators and
  security material.
- [ ] Pass an explicit provider registry through to `listen_native_server` when supplied; otherwise
  use official installed-provider discovery.
- [ ] Select the only installed provider directly and probe only when multiple eligible providers
  are installed.
- [ ] Expose selected provider, bound provider endpoints, probe diagnostics, and active transport in
  startup and session observations.

## Provider Boundaries

- [ ] Require each available binding to own its carrier implementation, listener, role adoption, and
  transport-scoped native artifact.
- [ ] Keep adapter, client, server, and profile packages free of hidden provider artifacts.
- [ ] Treat unavailable bindings as diagnostics and never bind them.
- [ ] Keep `unix://`, `npipe://`, `ws://`, and `wss://` locators inside provider routes while public
  application endpoints remain `nnrp://` or `nnrps://`.
- [ ] Fail the logical server atomically when mandatory provider listeners cannot be established.

## Security And Validation

- [ ] Validate TCP and IPC route locality and platform constraints before opening native handles.
- [ ] Require QUIC and WSS security material according to the frozen endpoint contract.
- [ ] Reject WebSocket text messages; accept only NNRP binary runtime frames.
- [ ] Test single-provider TCP, QUIC, IPC, and WebSocket startup and active-session exchange.
- [ ] Test multi-provider auto, prefer, force, and probe policy with real provider evidence.
- [ ] Test missing artifacts, incompatible platform routes, listener rollback, handshake rejection,
  and active-provider diagnostics.
