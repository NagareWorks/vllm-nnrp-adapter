# 08 - Observability And Exporters

## Observation Record

- [x] Add one immutable operation observation record shared by diagnostics, metrics, logs, and
  benchmarks.
- [x] Record model, operation, vLLM version, backend binding, selected transport, connection,
  session, operation, frame, route, view, trace, and profile identities.
- [x] Record queue delay, admission latency, preprocessing latency, time to first event, inter-event
  latency, terminal latency, output event count, token usage, and bytes where available.
- [x] Record cancellation source, backend abort acceptance, error family, drop reason, retry hint,
  pressure state, and terminal outcome.
- [x] Derive stage transitions from the same record used for `TRACE_CONTEXT` and `PROGRESS`.

## Export Boundary

- [x] Expose an observation sink protocol and a structured-log sink.
- [x] Provide an optional Prometheus collector that registers into an existing registry.
- [x] Do not bind an HTTP `/metrics` server by default.
- [x] Keep standalone metrics serving behind an explicit deployment command and port.
- [x] Use stable metric names and bounded labels; never put request ids, prompts, generated text, or
  arbitrary model metadata in labels.
- [x] Keep diagnostic and metrics values consistent by deriving both from the same terminal record.

## Validation

- [x] Test successful, failed, cancelled, expired, superseded, dropped, and disconnected records.
- [x] Test concurrent operations for identity isolation and one terminal observation each.
- [x] Test exporters with no vLLM or GPU dependency.
- [x] Include observation evidence in wire E2E artifacts.
- [ ] Include observation evidence in GPU benchmark artifacts.
