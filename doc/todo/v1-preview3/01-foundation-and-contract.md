# Foundation And Contract

- [x] Freeze the repository around the NNRP `openai-compatible/1` profile instead of a local HTTP clone.
- [x] Declare the vLLM support floor as `>=0.18.0` with `0.18.1` as the first lower-bound CI target.
- [x] Define Level 1 as the first implementation slice: `chat.completions.create`, streaming, non-streaming, cancellation, errors, usage, tool-call pass-through, and capability document.
- [ ] Add repository-scoped conformance capability metadata once the adapter runner command is implemented.
- [ ] Add profile compatibility fixtures that mirror the shared conformance recipe names.

