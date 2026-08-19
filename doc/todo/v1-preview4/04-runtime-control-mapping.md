# 04 - Runtime Control Mapping

## Cancellation And Freshness

- [x] Map `CANCEL` to cooperative backend abort, stream closure, `response.cancelled`, and a typed
  terminal drop reason when no final profile event can be delivered.
- [x] Map `ABORT` to immediate backend abort and suppress every later backend chunk.
- [x] Map `SUPERSEDE` to atomic old-operation cancellation and new-operation admission.
- [x] Map `DEADLINE` and `EXPIRE_AT` to absolute operation timers and discard stale backend output.
- [x] Preserve control sequence ordering and reject stale or duplicate updates.
- [ ] Record cancellation source, reason code, backend acceptance, and terminal state.

## Scheduling And Budgets

- [ ] Apply `PRIORITY_UPDATE` before vLLM admission and through a supported live scheduler hook.
- [ ] Return a typed recoverable error for a mandatory live reprioritization that the selected vLLM
  binding cannot honor.
- [ ] Apply `BUDGET_UPDATE` to token, compute, memory, and bandwidth limits without mutating the
  OpenAI request body after validation.
- [ ] Abort or degrade an operation that exceeds a hard budget and identify the exceeded budget.
- [ ] Map `ROUTE_HINT` and `EXECUTION_HINT` into adapter admission metadata and vLLM routing inputs.
- [ ] Reject an unsupported must-honor route or execution hint; ignore only a documented best-effort
  hint and record that outcome.

## Progress, Pressure, And Capabilities

- [x] Emit `PROGRESS` with frozen stage codes for queued, admitted, preprocessing, executing,
  producing-partial, finalizing, completed, dropped, and failed transitions.
- [ ] Preserve ordered OpenAI profile events in the NNRP result stream; do not replace the frozen
  profile mapping with JSON control messages.
- [ ] Map adapter queue and output pressure to `BACKPRESSURE` and `CREDIT_UPDATE`.
- [ ] Stop reading or emitting beyond the effective credit window.
- [ ] Answer `CAPABILITY_NEGOTIATION` with supported controls, cost model, preference rank, byte
  limit, unit limit, backend family, and vLLM version.
- [ ] Emit `DEGRADE_PROFILE` only for a capability downgrade permitted by the request flags.

## Diagnostics And Recovery

- [ ] Propagate `TRACE_CONTEXT` into operation observations and backend trace headers where the vLLM
  binding supports them.
- [ ] Emit `RESULT_DROP_REASON` for deadline, supersede, peer cancellation, backpressure, capability
  mismatch, budget, object invalidation, and transport closure outcomes.
- [ ] Emit `ERROR_RECOVERABLE` and `RETRY_AFTER` for transient admission, queue, and backend errors.
- [ ] Test every control frame with the frozen metadata model and reserved-value validation.
- [ ] Test unsupported mandatory semantics explicitly; no control frame may disappear without an
  observation or response.
