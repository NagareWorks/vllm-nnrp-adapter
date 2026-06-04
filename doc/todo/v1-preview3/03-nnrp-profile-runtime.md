# NNRP Profile Runtime

- [x] Add request envelope validation for `schema_version`, `operation`, and chat body shape.
- [x] Add profile event builders for text deltas, usage, completion, tool-call deltas, and errors.
- [x] Connect the profile adapter to a real NNRP server/session integration from `nnrp-py`.
- [x] Add result-push emission for streaming events through NNRP frames.
- [x] Add an embedded vLLM-process NNRP session loop for submit/result-push without an intermediate HTTP request.
- [x] Preserve frame ownership, trace id, route id, and view id through the embedded vLLM request lifecycle.
- [x] Add timeout and cancellation policy plumbing through the profile runtime.
- [ ] Map NNRP timeout/cancel signals onto vLLM request abort and stream closure.
- [x] Add diagnostics fields for selected model, operation, backend family, and backend error family.
