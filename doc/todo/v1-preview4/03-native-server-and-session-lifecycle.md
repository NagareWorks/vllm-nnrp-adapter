# 03 - Native Server And Session Lifecycle

## Native Role Adoption

- [x] Replace packet listener/session orchestration with `listen_native_server` and
  `NativeRuntimeServerSession`.
- [x] Accept `NativeRuntimeServerOperation` values and decode the OpenAI request from the operation
  body without exposing native buffers.
- [x] Send every non-terminal OpenAI profile event through
  `NativeRuntimeServerSession.send_partial_result` with a monotonically increasing result sequence.
- [x] Send terminal success exactly once through `NativeRuntimeServerOperation.send_result`; never
  use `send_result` for an intermediate streaming event because it completes the native operation.
- [x] Poll wire and lifecycle events through `poll_event` or the coarse `poll_events` batch surface.
- [x] Run blocking native calls on a bounded adapter-owned worker path so the vLLM event loop remains
  responsive.
- [ ] Preserve connection, session, operation, frame, route, view, trace, and active-profile identity
  through the complete request lifetime.

## Operation State

- [x] Maintain one operation registry keyed by NNRP operation id and one backend request id per live
  operation.
- [x] Define explicit `accepted`, `queued`, `admitted`, `streaming`, `completed`, `cancelled`,
  `dropped`, and `failed` states.
- [x] Enforce legal state transitions and one terminal outcome.
- [x] Reject a second terminal send and any partial result emitted after the native operation has
  completed, cancelled, expired, superseded, dropped, or failed.
- [x] Reject duplicate operation ids without corrupting the existing operation.
- [x] Stop emitting profile events immediately after a terminal state.
- [ ] Release operation resources after success, error, cancellation, abort, expiration, disconnect,
  or server shutdown.

## Concurrency And Shutdown

- [x] Serve multiple accepted sessions and multiple operations per session concurrently.
- [x] Keep per-operation event order while allowing independent operations to progress.
- [ ] Bound accepted sessions, active operations, queued operations, and pending output events.
- [ ] Apply graceful shutdown in the order: stop listeners, stop admission, cancel operations, drain
  terminal diagnostics, close sessions, close the native server.
- [ ] Add lifecycle tests for peer disconnect, listener failure, backend failure, cancellation races,
  shutdown races, and restart after clean closure.
