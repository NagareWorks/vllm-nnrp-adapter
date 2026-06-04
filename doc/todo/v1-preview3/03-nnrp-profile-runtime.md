# NNRP Profile Runtime

- [x] Add request envelope validation for `schema_version`, `operation`, and chat body shape.
- [x] Add profile event builders for text deltas, usage, completion, tool-call deltas, and errors.
- [ ] Connect the profile adapter to a real NNRP server/session integration from `nnrp-py`.
- [ ] Add result-push emission for streaming events through NNRP frames.
- [ ] Add timeout and cancellation policy plumbing through the NNRP runtime.
- [ ] Add diagnostics fields for queue delay, selected model, selected transport, and backend error family.

