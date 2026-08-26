# 05 - Runtime Objects, Cache, And Payloads

## Typed Payload Boundary

- [x] Keep the `openai-compatible/1` envelope and profile events JSON because that encoding is part
  of the frozen profile.
- [x] Carry runtime control, object, cache, and diagnostics metadata through typed Preview4 frames,
  never through ad hoc JSON fields inside the OpenAI body.
- [x] Decode structured-event submit payloads with bounded size and depth checks.
- [x] Preserve opaque, image, audio, video, document, tool-result, and tensor object references
  without base64-expanding them into the request envelope.
- [x] Reject a request that references an unavailable, expired, or incompatible object before vLLM
  admission.

## Object Lifecycle

- [x] Track `OBJECT_DECLARE`, `OBJECT_REF`, `OBJECT_RELEASE`, `OBJECT_PATCH`, and `OBJECT_DELTA` by
  session and operation ownership.
- [ ] Validate object kind, version, offset, length, memory-location hint, ownership hint, and lease.
- [ ] Map supported multimodal object references into vLLM multimodal request inputs.
- [ ] Map tool-result and document-chunk references into the selected OpenAI-compatible body field
  without changing profile semantics.
- [ ] Release borrowed or transferred objects on terminal operation state according to the frozen
  ownership and release rules.
- [x] Stop and diagnose operations whose referenced object is invalidated mid-flight.

## Cache References

- [x] Keep cache policy opt-in and explicit; do not perform implicit lookups.
- [ ] Resolve `CACHE_REFERENCE` against adapter-visible object/cache providers and emit `CACHE_MISS`
  with the frozen miss reason on failure.
- [ ] Apply `CACHE_INVALIDATE` to adapter-owned references and active dependent operations.
- [x] Keep NNRP object/cache identity separate from vLLM KV-cache internals.
- [ ] Advertise KV/prefix-cache integration only after the selected vLLM binding supplies a real,
  tested identity and invalidation mapping.
- [ ] Test lease expiry, version mismatch, schema mismatch, permission denial, producer loss, and
  invalidation races.
