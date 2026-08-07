# 02 - vLLM Version And Backend Bindings

## Installation Band And Compatibility Anchors

- [ ] Extend the optional vLLM dependency to `vllm>=0.18.0,<0.27`.
- [ ] Treat that dependency range as installation eligibility, not as a claim that every minor line
  has a tested compatibility binding.
- [ ] Define explicit legacy (`0.18.1`), transition (`0.22.1`), and current (`0.26.0`)
  compatibility anchors.
- [ ] Detect the installed vLLM version and required serving-object features at startup, then select
  a named compatibility binding only when both checks pass.
- [ ] Add contract-shape fixtures for the three compatibility anchors and their named binding
  families.
- [ ] Add real GPU smoke jobs for `0.18.1`, `0.22.1`, and `0.26.0`.
- [ ] Reject an untested or incompatible vLLM minor line with a diagnostic naming the detected
  version, missing feature, and tested compatibility anchors.
- [ ] Generate the public compatibility table from the same binding registry used at runtime.

## Backend Calls

- [ ] Bind the OpenAI serving-object path for each compatibility family without starting the HTTP
  server.
- [ ] Bind the engine-direct generation path for each compatibility family that exposes the required
  request preprocessing and engine client APIs.
- [ ] Keep HTTP/SSE parsing behind an explicit smoke-only backend and exclude it from production
  auto-selection.
- [ ] Preserve request id, model id, LoRA selection, sampling parameters, multimodal inputs, usage,
  finish reasons, and tool-call deltas across each binding.
- [ ] Normalize backend error objects without depending on rendered HTTP responses.
- [ ] Provide one idempotent backend abort operation and record whether vLLM accepted it.
- [ ] Close or cancel the underlying async generator exactly once on cancellation, timeout, abort,
  shutdown, or client disconnect.

## Compatibility Tests

- [ ] Test method selection and request construction against version-specific fakes generated from
  the three compatibility anchors.
- [ ] Test streaming, non-streaming, usage, tool calls, scheduler rejection, and backend cancellation
  in every compatibility family.
- [ ] Test feature probing, exact unsupported-version diagnostics, and rejection of a version that
  falls inside the installation band but has no tested binding.
- [ ] Test that the production backend never enters the HTTP/SSE parser.
- [ ] Record the vLLM version, compatibility binding, model, engine configuration, and GPU in every
  integration result.
