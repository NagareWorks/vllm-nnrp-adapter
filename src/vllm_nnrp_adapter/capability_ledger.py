from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nnrp import PREVIEW4_CAPABILITY_TOKENS  # type: ignore[import-untyped]


class CapabilityClassification(StrEnum):
    CORE = "core"
    CONDITIONAL = "conditional"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityEvidence:
    token: str
    classification: CapabilityClassification
    mechanism: str
    observable_effect: str
    benchmark_metric: str
    acceptance_threshold: str | None = None
    independent_scenario: str | None = None
    evidence_artifact: str | None = None
    backend_requirement: str | None = None
    advertised_by_default: bool = True

    @property
    def has_runtime_mechanism(self) -> bool:
        return bool(self.mechanism and self.observable_effect)

    @property
    def release_gate_ready(self) -> bool:
        return all(
            (
                self.has_runtime_mechanism,
                self.acceptance_threshold,
                self.independent_scenario,
                self.evidence_artifact,
            )
        )


_WIRE_LEVEL1_SCENARIO = "wire.profile.openai-compatible.level1"
_WIRE_LEVEL1_EVIDENCE = "artifacts/wire-e2e/results.json"

RUNTIME_CAPABILITY_LEDGER = (
    RuntimeCapabilityEvidence(
        token="payload.typed",
        classification=CapabilityClassification.CORE,
        mechanism="Decode typed FRAME_SUBMIT payload frames before OpenAI profile validation.",
        observable_effect="The wire target observes a typed request and typed terminal result without an SSE relay.",
        benchmark_metric="semantic defect rate",
        acceptance_threshold="zero malformed, missing, duplicate, or incorrectly terminated frames",
        independent_scenario=_WIRE_LEVEL1_SCENARIO,
        evidence_artifact=_WIRE_LEVEL1_EVIDENCE,
    ),
    RuntimeCapabilityEvidence(
        token="control.cancel_abort",
        classification=CapabilityClassification.CORE,
        mechanism="Dispatch CANCEL and ABORT to the live backend stream abort path.",
        observable_effect="Backend generation stops and the operation reaches one typed terminal outcome.",
        benchmark_metric="cancellation effect latency and late-result rate",
        acceptance_threshold="late-result rate below 0.1% after accepted cancellation or abort",
    ),
    RuntimeCapabilityEvidence(
        token="control.supersede",
        classification=CapabilityClassification.EXPERIMENTAL,
        mechanism="Atomically admit the replacement operation and terminate the obsolete operation.",
        observable_effect="The obsolete operation emits SUPERSEDED while the replacement remains live.",
        benchmark_metric="wasted GPU seconds and late-result rate",
    ),
    RuntimeCapabilityEvidence(
        token="control.priority_update",
        classification=CapabilityClassification.CONDITIONAL,
        mechanism="Apply admission priority and invoke the selected vLLM binding's live scheduler hook.",
        observable_effect="The backend request priority changes or a typed must-honor rejection is emitted.",
        benchmark_metric="high-priority p95 latency, starvation, and throughput",
        acceptance_threshold="at least 30% lower high-priority p95 with no starvation and at most 5% throughput loss",
        backend_requirement="selected vLLM binding exposes a live runtime-priority hook",
        advertised_by_default=False,
    ),
    RuntimeCapabilityEvidence(
        token="control.deadline_expire",
        classification=CapabilityClassification.CORE,
        mechanism="Arm absolute operation deadlines and abort backend work when they expire.",
        observable_effect="Expired work emits DEADLINE_EXPIRED and later backend chunks are suppressed.",
        benchmark_metric="wasted GPU seconds and late-result rate",
        acceptance_threshold="late-result rate below 0.1% after accepted expiry",
    ),
    RuntimeCapabilityEvidence(
        token="control.progress_partial",
        classification=CapabilityClassification.CORE,
        mechanism="Map ordered OpenAI profile deltas to PARTIAL_RESULT and lifecycle stages to PROGRESS.",
        observable_effect="The wire target observes ordered partial frames followed by exactly one terminal result.",
        benchmark_metric="semantic defect rate",
        acceptance_threshold="zero reordered, duplicate, late, or missing profile events",
        independent_scenario=_WIRE_LEVEL1_SCENARIO,
        evidence_artifact=_WIRE_LEVEL1_EVIDENCE,
    ),
    RuntimeCapabilityEvidence(
        token="control.credit_backpressure",
        classification=CapabilityClassification.CORE,
        mechanism="Apply peer credit windows before pulling and emitting backend output.",
        observable_effect="Output pauses at exhausted credit and resumes after CREDIT_UPDATE.",
        benchmark_metric="bounded queue depth, producer overrun, and recovery latency",
        acceptance_threshold="zero output beyond the active credit window and zero producer overruns",
    ),
    RuntimeCapabilityEvidence(
        token="control.capability_costs",
        classification=CapabilityClassification.CORE,
        mechanism="Intersect canonical capability offers with the adapter's mechanism-backed ledger.",
        observable_effect="The peer receives a canonical accepted subset or a typed capability mismatch.",
        benchmark_metric="semantic defect rate",
        acceptance_threshold="zero accepted tokens without a mechanism-backed ledger entry",
    ),
    RuntimeCapabilityEvidence(
        token="control.trace_context",
        classification=CapabilityClassification.CORE,
        mechanism="Correlate session or operation trace metadata and pass supported headers to vLLM.",
        observable_effect="Operation observations and supported backend requests retain the same trace identity.",
        benchmark_metric="trace correlation defect rate",
        acceptance_threshold="zero lost or mismatched trace identities on accepted trace updates",
    ),
    RuntimeCapabilityEvidence(
        token="control.result_drop_reason",
        classification=CapabilityClassification.CORE,
        mechanism="Emit typed drop terminals for cancellation, expiry, supersession, and transport closure.",
        observable_effect="Every implemented stale-result path reports one protocol-visible reason code.",
        benchmark_metric="unexplained-drop and duplicate-terminal rates",
        acceptance_threshold="zero unexplained drops and zero duplicate terminal outcomes",
    ),
    RuntimeCapabilityEvidence(
        token="control.degrade_profile",
        classification=CapabilityClassification.CONDITIONAL,
        mechanism="Return an accepted capability subset only when the offer permits degradation.",
        observable_effect="The peer receives DEGRADE_PROFILE instead of a silent capability downgrade.",
        benchmark_metric="capability-negotiation semantic defect rate",
    ),
    RuntimeCapabilityEvidence(
        token="control.recoverable_error",
        classification=CapabilityClassification.EXPERIMENTAL,
        mechanism="Map transient admission, queue, and backend failures to typed retry guidance.",
        observable_effect="The peer receives ERROR_RECOVERABLE and RETRY_AFTER with bounded diagnostics.",
        benchmark_metric="recovery latency and duplicate execution rate",
    ),
)

_LEDGER_BY_TOKEN = {entry.token: entry for entry in RUNTIME_CAPABILITY_LEDGER}

if len(_LEDGER_BY_TOKEN) != len(RUNTIME_CAPABILITY_LEDGER):
    raise RuntimeError("runtime capability ledger contains duplicate tokens")
if unknown := set(_LEDGER_BY_TOKEN).difference(PREVIEW4_CAPABILITY_TOKENS):
    raise RuntimeError(f"runtime capability ledger contains unknown Preview4 tokens: {sorted(unknown)!r}")
if incomplete := sorted(entry.token for entry in RUNTIME_CAPABILITY_LEDGER if not entry.has_runtime_mechanism):
    raise RuntimeError(f"runtime capability ledger advertises entries without mechanisms: {incomplete!r}")


def supported_runtime_capabilities(*, supports_runtime_priority: bool) -> frozenset[str]:
    supported = {
        entry.token
        for entry in RUNTIME_CAPABILITY_LEDGER
        if entry.advertised_by_default and entry.has_runtime_mechanism
    }
    if supports_runtime_priority:
        supported.add("control.priority_update")
    return frozenset(supported)
