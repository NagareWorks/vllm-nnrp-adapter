# 06 - Transport Provider Serving

## Provider-Neutral Startup

- [x] Replace `serve-tcp` with `serve` accepting one `nnrp://` or `nnrps://` application endpoint.
- [x] Accept explicit TCP, QUIC, IPC, and WebSocket provider routes with provider-local locators and
  security material.
- [x] Pass an explicit provider registry through to `listen_native_server` when supplied; otherwise
  use official installed-provider discovery.
- [x] Preserve every explicit provider binding for the official listener so Auto and Prefer can
  bind every eligible server route; never apply client-side single-provider selection or probing.
- [ ] Expose eligible providers and bound provider endpoints in startup observations, then record
  the active transport for each accepted session.

## Provider Boundaries

- [ ] Require each available binding to own its carrier implementation, listener, role adoption, and
  transport-scoped native artifact.
- [x] Keep adapter, client, server, and profile packages free of hidden provider artifacts.
- [ ] Treat unavailable bindings as diagnostics and never bind them.
- [x] Keep `unix://`, `npipe://`, `ws://`, and `wss://` locators inside provider routes while public
  application endpoints remain `nnrp://` or `nnrps://`.
- [ ] Fail the logical server atomically when mandatory provider listeners cannot be established.

## Security And Validation

- [ ] Validate TCP and IPC route locality and platform constraints before opening native handles.
- [ ] Require QUIC and WSS security material according to the frozen endpoint contract.
- [ ] Reject WebSocket text messages; accept only NNRP binary runtime frames.
- [ ] Test single-provider TCP, QUIC, IPC, and WebSocket startup and active-session exchange.
- [ ] Test multi-provider auto, prefer, and force binding policy with real provider evidence.
- [ ] Test missing artifacts, incompatible platform routes, listener rollback, handshake rejection,
  and active-provider diagnostics.
