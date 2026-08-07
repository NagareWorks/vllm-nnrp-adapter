# 02 - vLLM Version And Backend Bindings

## Supported Version Band

- [ ] Extend the optional vLLM dependency to `vllm>=0.18.0,<0.27`.
- [ ] Detect the installed vLLM version at startup and select an explicit compatibility binding.
- [ ] Keep `0.18.0` in the declared band and `0.18.1` as the lower-bound GPU target.
- [ ] Add contract-shape fixtures for `0.18.0`, `0.18.1`, `0.19.1`, `0.20.2`, `0.21.0`, `0.22.1`,
  `0.23.0`, `0.24.0`, `0.25.1`, and `0.26.0`.
- [ ] Add real GPU smoke jobs for `0.18.1`, `0.22.1`, and `0.26.0`.
- [ ] Reject an unknown vLLM minor line with a diagnostic naming the detected version and supported
  compatibility bindings.

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
  the named vLLM releases.
- [ ] Test streaming, non-streaming, usage, tool calls, scheduler rejection, and backend cancellation
  in every compatibility family.
- [ ] Test that the production backend never enters the HTTP/SSE parser.
- [ ] Record the vLLM version, compatibility binding, model, engine configuration, and GPU in every
  integration result.
