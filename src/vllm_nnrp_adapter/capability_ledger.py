from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nnrp import PREVIEW4_CAPABILITY_TOKENS  # type: ignore[import-untyped]

from .profile import VLLM_DIAGNOSTICS_EXTENSION


class CapabilityClassification(StrEnum):
    CORE = "core"
    CONDITIONAL = "conditional"
    EXPERIMENTAL = "experimental"


class ProfileAdvertisementSurface(StrEnum):
    OPERATION = "operation"
    BASELINE_EVENT = "baseline-event"
    EXTENSION = "extension"
    RUNTIME_CONTROL = "runtime-control"


@dataclass(frozen=True, slots=True)
class ProfileCapabilityEvidence:
    name: str
    surface: ProfileAdvertisementSurface
    mechanism: str
    observable_effect: str
    test_scenario: str
    operation_field: str | None = None
    extension: dict[str, object] | None = None
    backend_requirement: str | None = None
    advertised_by_default: bool = True

    @property
    def has_evidence(self) -> bool:
        return bool(self.mechanism and self.observable_effect and self.test_scenario)


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
                self.benchmark_metric,
                self.acceptance_threshold,
                self.independent_scenario,
                self.evidence_artifact,
            )
        )


_WIRE_LEVEL1_SCENARIO = "wire.profile.openai-compatible.level1"
_WIRE_LEVEL1_EVIDENCE = "artifacts/wire-e2e/results.json"

OPENAI_PROFILE_CAPABILITY_LEDGER = (
    ProfileCapabilityEvidence(
        name="streaming",
        surface=ProfileAdvertisementSurface.OPERATION,
        operation_field="streaming",
        mechanism="Map the backend async iterator to ordered profile events and NNRP partial results.",
        observable_effect="The client receives ordered deltas followed by exactly one terminal result.",
        test_scenario="test_adapter_maps_streaming_chat_chunks",
    ),
    ProfileCapabilityEvidence(
        name="non_streaming",
        surface=ProfileAdvertisementSurface.OPERATION,
        operation_field="non_streaming",
        mechanism="Map a backend completion object to one profile completion and one terminal result.",
        observable_effect="The client receives the final OpenAI-compatible completion body without an SSE relay.",
        test_scenario="test_adapter_maps_non_streaming_chat_body",
    ),
    ProfileCapabilityEvidence(
        name="cancellation",
        surface=ProfileAdvertisementSurface.OPERATION,
        operation_field="cancellation",
        mechanism="Propagate NNRP cancellation to the active backend request and close its stream.",
        observable_effect="The operation emits one cancellation outcome and suppresses late backend output.",
        test_scenario="test_native_cancel_aborts_vllm_engine_with_derived_request_id",
    ),
    ProfileCapabilityEvidence(
        name="usage",
        surface=ProfileAdvertisementSurface.BASELINE_EVENT,
        mechanism="Preserve backend usage data as an ordered response.usage event.",
        observable_effect="Usage is emitted only when supplied by the backend and token counts are never invented.",
        test_scenario="test_engine_direct_stream_emits_usage_after_terminal_delta_without_recounting_prompt",
    ),
    ProfileCapabilityEvidence(
        name="tool_calls",
        surface=ProfileAdvertisementSurface.OPERATION,
        operation_field="tool_calls",
        mechanism="Preserve backend tool-call identity and emit the complete tool-call event lifecycle.",
        observable_effect="Tool calls are advertised only for a backend binding that exposes tool-call support.",
        test_scenario="test_engine_direct_tool_calls_follow_each_vllm_parser_family",
        backend_requirement="backend.supports_tool_calls is true",
        advertised_by_default=False,
    ),
    ProfileCapabilityEvidence(
        name=VLLM_DIAGNOSTICS_EXTENSION,
        surface=ProfileAdvertisementSurface.EXTENSION,
        mechanism="Emit bounded adapter diagnostics only when the request opts in.",
        observable_effect="Level 1 clients may ignore diagnostics while opted-in clients receive typed fields.",
        test_scenario="test_adapter_emits_request_diagnostics_when_requested",
        extension={
            "name": VLLM_DIAGNOSTICS_EXTENSION,
            "critical": False,
            "description": (
                "Adapter may include NNRP diagnostics without changing Level 1 baseline pass/fail."
            ),
        },
    ),
    ProfileCapabilityEvidence(
        name="limits",
        surface=ProfileAdvertisementSurface.RUNTIME_CONTROL,
        mechanism=(
            "Keep profile limits out of the API capability document and negotiate them through typed runtime control."
        ),
        observable_effect=(
            "The adapter emits no profile-level limit claim while enforceable budget handling is unavailable."
        ),
        test_scenario="test_profile_capability_ledger_keeps_limits_out_of_profile_document",
        advertised_by_default=False,
    ),
)

_PROFILE_LEDGER_BY_NAME = {entry.name: entry for entry in OPENAI_PROFILE_CAPABILITY_LEDGER}
if len(_PROFILE_LEDGER_BY_NAME) != len(OPENAI_PROFILE_CAPABILITY_LEDGER):
    raise RuntimeError("OpenAI profile capability ledger contains duplicate entries")
if incomplete_profile_evidence := sorted(
    entry.name for entry in OPENAI_PROFILE_CAPABILITY_LEDGER if not entry.has_evidence
):
    raise RuntimeError(
        f"OpenAI profile capability ledger contains entries without evidence: {incomplete_profile_evidence!r}"
    )


def openai_profile_operation_capabilities(*, supports_tool_calls: bool) -> dict[str, bool]:
    capabilities: dict[str, bool] = {}
    for entry in OPENAI_PROFILE_CAPABILITY_LEDGER:
        if entry.operation_field is None:
            continue
        supported = entry.advertised_by_default
        if entry.name == "tool_calls":
            supported = supports_tool_calls
        capabilities[entry.operation_field] = supported
    return capabilities


def openai_profile_extensions() -> tuple[dict[str, object], ...]:
    return tuple(
        dict(entry.extension)
        for entry in OPENAI_PROFILE_CAPABILITY_LEDGER
        if entry.advertised_by_default and entry.extension is not None
    )

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
        independent_scenario="wire.control.cancel-abort.client",
        evidence_artifact=_WIRE_LEVEL1_EVIDENCE,
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
        independent_scenario="wire.control.deadline-before-submit.client",
        evidence_artifact=_WIRE_LEVEL1_EVIDENCE,
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
        independent_scenario="wire.control.credit-backpressure.client",
        evidence_artifact=_WIRE_LEVEL1_EVIDENCE,
    ),
    RuntimeCapabilityEvidence(
        token="control.capability_costs",
        classification=CapabilityClassification.CORE,
        mechanism="Intersect canonical capability offers with the adapter's mechanism-backed ledger.",
        observable_effect="The peer receives a canonical accepted subset or a typed capability mismatch.",
        benchmark_metric="semantic defect rate",
        acceptance_threshold="zero accepted tokens without a mechanism-backed ledger entry",
        independent_scenario="wire.control.capability-negotiation.client",
        evidence_artifact=_WIRE_LEVEL1_EVIDENCE,
    ),
    RuntimeCapabilityEvidence(
        token="control.trace_context",
        classification=CapabilityClassification.CORE,
        mechanism="Correlate session or operation trace metadata and pass supported headers to vLLM.",
        observable_effect="Operation observations and supported backend requests retain the same trace identity.",
        benchmark_metric="trace correlation defect rate",
        acceptance_threshold="zero lost or mismatched trace identities on accepted trace updates",
        independent_scenario="wire.control.cancel-abort.client",
        evidence_artifact=_WIRE_LEVEL1_EVIDENCE,
    ),
    RuntimeCapabilityEvidence(
        token="control.result_drop_reason",
        classification=CapabilityClassification.CORE,
        mechanism="Emit typed drop terminals for cancellation, expiry, supersession, and transport closure.",
        observable_effect="Every implemented stale-result path reports one protocol-visible reason code.",
        benchmark_metric="unexplained-drop and duplicate-terminal rates",
        acceptance_threshold="zero unexplained drops and zero duplicate terminal outcomes",
        independent_scenario="wire.control.cancel-abort.client",
        evidence_artifact=_WIRE_LEVEL1_EVIDENCE,
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
