# 07 - OpenAI Profile Runtime

## Level 1 Request Contract

- [x] Validate `schema_version`, `operation`, optional `request_id`, `body`, and optional `nnrp`
  policy exactly as frozen for `openai-compatible/1`.
- [x] Accept only `chat.completions.create` for Level 1.
- [x] Preserve OpenAI-compatible model, messages, temperature, top-p, token limit, tools,
  tool-choice, metadata, stream, and supported multimodal fields.
- [x] Keep NNRP timeout, diagnostics, cache, transport, and cancellation policy out of the OpenAI
  request body passed to vLLM.
- [x] Reject Level 2 and Level 3 operations until their complete behavior and conformance manifests
  are implemented.

## Result Event Contract

- [x] Emit ordered `response.output_text.delta` events for streaming text as `PARTIAL_RESULT`
  payloads, preserving one monotonic result sequence per operation.
- [x] Emit `response.tool_call.started`, `.delta`, `.completed`, and `.error` with stable tool-call
  identity when advertised; non-terminal tool events also use `PARTIAL_RESULT`.
- [x] Emit `response.usage` without inventing absent token counts and preserve its order relative to
  text and tool events.
- [x] Emit at most one `response.completed`, `response.error`, or `response.cancelled` terminal
  profile event when applicable, then produce exactly one matching NNRP terminal outcome.
- [x] Preserve the final streaming or non-streaming OpenAI-compatible response body, when available,
  in the single terminal `RESULT_PUSH` sent by `NativeRuntimeServerOperation.send_result`; never
  invent a final body when vLLM provides only ordered deltas and terminal metadata.
- [x] Keep optional original OpenAI chunks ignorable and exclude them from baseline client
  requirements.

## Errors And Capabilities

- [x] Map invalid request, unsupported operation, model error, scheduler rejection, overload,
  timeout, cancellation, and backend failure to frozen profile error bodies and NNRP terminal state.
- [x] Keep application errors distinct from transport, protocol, and adapter-internal failures.
- [x] Generate the Level 1 capability document from real backend and adapter behavior.
- [x] Advertise streaming, non-streaming, cancellation, usage, tool calls, diagnostics, limits, and
  extensions only when tested.
- [x] Validate provider-specific extensions as declared, non-critical, and ignorable for the Level 1
  baseline.
