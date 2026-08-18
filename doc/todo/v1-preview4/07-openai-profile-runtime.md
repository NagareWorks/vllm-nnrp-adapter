# 07 - OpenAI Profile Runtime

## Level 1 Request Contract

- [ ] Validate `schema_version`, `operation`, optional `request_id`, `body`, and optional `nnrp`
  policy exactly as frozen for `openai-compatible/1`.
- [ ] Accept only `chat.completions.create` for Level 1.
- [ ] Preserve OpenAI-compatible model, messages, temperature, top-p, token limit, tools,
  tool-choice, metadata, stream, and supported multimodal fields.
- [ ] Keep NNRP timeout, diagnostics, cache, transport, and cancellation policy out of the OpenAI
  request body passed to vLLM.
- [ ] Reject Level 2 and Level 3 operations until their complete behavior and conformance manifests
  are implemented.

## Result Event Contract

- [ ] Emit ordered `response.output_text.delta` events for streaming text as `PARTIAL_RESULT`
  payloads, preserving one monotonic result sequence per operation.
- [ ] Emit `response.tool_call.started`, `.delta`, `.completed`, and `.error` with stable tool-call
  identity when advertised; non-terminal tool events also use `PARTIAL_RESULT`.
- [ ] Emit `response.usage` without inventing absent token counts and preserve its order relative to
  text and tool events.
- [ ] Emit at most one `response.completed`, `response.error`, or `response.cancelled` terminal
  profile event when applicable, then produce exactly one matching NNRP terminal outcome.
- [ ] Preserve the final streaming or non-streaming OpenAI-compatible response body, when available,
  in the single terminal `RESULT_PUSH` sent by `NativeRuntimeServerOperation.send_result`; never
  invent a final body when vLLM provides only ordered deltas and terminal metadata.
- [ ] Keep optional original OpenAI chunks ignorable and exclude them from baseline client
  requirements.

## Errors And Capabilities

- [ ] Map invalid request, unsupported operation, model error, scheduler rejection, overload,
  timeout, cancellation, and backend failure to frozen profile error bodies and NNRP terminal state.
- [ ] Keep application errors distinct from transport, protocol, and adapter-internal failures.
- [x] Generate the Level 1 capability document from real backend and adapter behavior.
- [ ] Advertise streaming, non-streaming, cancellation, usage, tool calls, diagnostics, limits, and
  extensions only when tested.
- [ ] Validate provider-specific extensions as declared, non-critical, and ignorable for the Level 1
  baseline.
