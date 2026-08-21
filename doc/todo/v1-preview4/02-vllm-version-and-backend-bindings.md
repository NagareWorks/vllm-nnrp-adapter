# 02 - vLLM Version And Backend Bindings

## Installation Band And Compatibility Anchors

- [x] Extend the optional vLLM dependency to `vllm>=0.18.0,<0.27`.
- [x] Treat that dependency range as installation eligibility, not as a claim that every minor line
  has a tested compatibility binding.
- [x] Define explicit legacy (`0.18.1`), transition (`0.22.1`), and current (`0.26.0`)
  compatibility anchors.
- [x] Detect the installed vLLM version and required serving-object features at startup, then select
  a named compatibility binding only when both checks pass.
- [x] Add contract-shape fixtures for the three compatibility anchors and their named binding
  families.
- [ ] Add real GPU smoke jobs for `0.18.1`, `0.22.1`, and `0.26.0`.
- [x] Reject an untested or incompatible vLLM minor line with a diagnostic naming the detected
  version, missing feature, and tested compatibility anchors.
- [x] Generate the public compatibility table from the same binding registry used at runtime.

## Backend Calls

- [x] Bind the OpenAI serving-object path for each compatibility family without starting the HTTP
  server.
- [x] Bind the engine-direct generation path for each compatibility family that exposes the required
  request preprocessing and engine client APIs.
- [x] Keep HTTP/SSE parsing behind an explicit smoke-only backend and exclude it from production
  auto-selection.
- [x] Preserve request id, model id, LoRA selection, sampling parameters, multimodal inputs, usage,
  finish reasons, and tool-call deltas across each binding.
- [x] Normalize backend error objects without depending on rendered HTTP responses.
- [x] Provide one idempotent backend abort operation and record whether vLLM accepted it.
- [x] Close or cancel the underlying async generator exactly once on cancellation, timeout, abort,
  shutdown, or client disconnect.

## Compatibility Tests

- [x] Test method selection and request construction against version-specific fakes generated from
  the three compatibility anchors.
- [x] Test streaming, non-streaming, usage, tool calls, scheduler rejection, and backend cancellation
  in every compatibility family.
- [x] Test feature probing, exact unsupported-version diagnostics, and rejection of a version that
  falls inside the installation band but has no tested binding.
- [x] Test that the production backend never enters the HTTP/SSE parser.
- [x] Record the vLLM version, compatibility binding, model, engine configuration, and GPU in every
  integration result.
