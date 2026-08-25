# 09 - Conformance And vLLM Integration

## OpenAI API Profile Conformance

- [x] Update the capability manifest to the real Preview4 Level 1 surface.
- [x] Consume suite-generated API profile execution plans instead of maintaining a divergent local
  case list.
- [x] Pass streaming chat, non-streaming chat, invalid body, unsupported operation, usage, event
  order, tool calls when advertised, cancellation, backend error, and capability-document cases.
- [x] Validate extension declarations and ensure Level 1 success never depends on a vendor extension.
- [x] Store adapter plan, result, and evidence files as CI artifacts.

## Wire-Level Conformance

- [x] Start the real Rust-backed adapter server as an external wire target.
- [ ] Let `nnrp-conformance` act as client, server, and proxy where each scenario requires.
- [ ] Cover native handshake, submit/result order, cancellation/abort, priority/deadline, progress,
  pressure/credit, capability costs, route hints, trace context, result-drop reasons, objects,
  deltas, and cache references.
- [x] Assert that streaming profile events use ordered `PARTIAL_RESULT` frames and that exactly one
  terminal `RESULT_PUSH` or typed drop/error outcome closes each operation.
- [x] Run installed-provider scenarios for TCP, QUIC, IPC, and WebSocket.
- [ ] Validate missing, reordered, duplicate, unexpected, late-after-terminal, and malformed frames.
- [x] Require protocol-visible carrier evidence rather than adapter self-report.

## CI And GPU Integration

- [ ] Keep unit/profile tests, API profile conformance, wire E2E, vLLM contract matrix, and GPU smoke
  as separate CI jobs.
- [x] Run normal CI without importing vLLM or requiring a GPU.
- [ ] Run lower-bound `0.18.1`, transition `0.22.1`, and current `0.26.0` GPU smoke against the same
  Level 1 request/event assertions.
- [ ] Exercise cancellation through a real NNRP control frame and verify the vLLM engine abort id.
- [ ] Make every release gate fail on skipped mandatory behavior; mock-only passes are insufficient.
