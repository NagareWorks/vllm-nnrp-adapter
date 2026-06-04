# NNRP Profile Runtime

- [x] Add request envelope validation for `schema_version`, `operation`, and chat body shape.
- [x] Add profile event builders for text deltas, usage, completion, tool-call deltas, and errors.
- [ ] Connect the profile adapter to a real NNRP server/session integration from `nnrp-py`.
- [ ] Add result-push emission for streaming events through NNRP frames.
- [x] Add timeout and cancellation policy plumbing through the profile runtime.
- [x] Add diagnostics fields for selected model, operation, backend family, and backend error family.
