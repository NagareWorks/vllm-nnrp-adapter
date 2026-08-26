from __future__ import annotations

import ast
from pathlib import Path

from nnrp import PREVIEW4_CAPABILITY_TOKENS

from vllm_nnrp_adapter.capability_ledger import (
    OPENAI_PROFILE_CAPABILITY_LEDGER,
    RUNTIME_CAPABILITY_LEDGER,
    CapabilityClassification,
    ProfileAdvertisementSurface,
    RuntimeCapabilityEvidence,
    openai_profile_extensions,
    openai_profile_operation_capabilities,
    supported_runtime_capabilities,
)


def test_runtime_capability_ledger_uses_only_frozen_tokens_and_complete_mechanisms() -> None:
    tokens = [entry.token for entry in RUNTIME_CAPABILITY_LEDGER]

    assert len(tokens) == len(set(tokens))
    assert set(tokens).issubset(PREVIEW4_CAPABILITY_TOKENS)
    assert all(entry.has_runtime_mechanism for entry in RUNTIME_CAPABILITY_LEDGER)


def test_runtime_release_gate_requires_a_benchmark_metric() -> None:
    entry = RuntimeCapabilityEvidence(
        token="payload.typed",
        classification=CapabilityClassification.CORE,
        mechanism="mechanism",
        observable_effect="effect",
        benchmark_metric="",
        acceptance_threshold="threshold",
        independent_scenario="scenario",
        evidence_artifact="artifact.json",
    )

    assert entry.release_gate_ready is False


def test_runtime_capability_advertisement_is_derived_from_the_ledger() -> None:
    default_tokens = supported_runtime_capabilities(supports_runtime_priority=False)
    priority_tokens = supported_runtime_capabilities(supports_runtime_priority=True)

    assert "control.priority_update" not in default_tokens
    assert priority_tokens == default_tokens | {"control.priority_update"}
    assert default_tokens == {
        entry.token for entry in RUNTIME_CAPABILITY_LEDGER if entry.advertised_by_default
    }


def test_core_capabilities_expose_thresholds_without_claiming_missing_evidence() -> None:
    core_entries = [
        entry for entry in RUNTIME_CAPABILITY_LEDGER if entry.classification is CapabilityClassification.CORE
    ]

    assert core_entries
    assert all(entry.benchmark_metric for entry in core_entries)
    assert all(entry.acceptance_threshold for entry in core_entries)
    assert all(entry.advertised_by_default for entry in core_entries)
    assert all(entry.release_gate_ready for entry in core_entries)
    assert {entry.token for entry in core_entries if entry.release_gate_ready} == {
        "payload.typed",
        "control.cancel_abort",
        "control.capability_costs",
        "control.deadline_expire",
        "control.progress_partial",
        "control.trace_context",
        "control.result_drop_reason",
    }


def test_conditional_backend_capabilities_are_not_advertised_unconditionally() -> None:
    priority = next(entry for entry in RUNTIME_CAPABILITY_LEDGER if entry.token == "control.priority_update")

    assert priority.classification is CapabilityClassification.CONDITIONAL
    assert priority.backend_requirement is not None
    assert priority.advertised_by_default is False
    assert priority.release_gate_ready is False


def test_unproven_runtime_capabilities_are_not_core_release_claims() -> None:
    unproven = {entry.token: entry for entry in RUNTIME_CAPABILITY_LEDGER if not entry.release_gate_ready}

    assert unproven
    assert all(entry.classification is not CapabilityClassification.CORE for entry in unproven.values())


def test_runtime_advertisement_excludes_unimplemented_claims() -> None:
    advertised = supported_runtime_capabilities(supports_runtime_priority=True)

    assert advertised.isdisjoint(
        {
            "control.budget_update",
            "control.route_execution_hint",
            "object.lifecycle",
            "cache.reference",
        }
    )


def test_profile_capability_ledger_has_evidence_for_every_release_claim() -> None:
    assert {entry.name for entry in OPENAI_PROFILE_CAPABILITY_LEDGER} == {
        "streaming",
        "non_streaming",
        "cancellation",
        "usage",
        "tool_calls",
        "vllm.diagnostics",
        "limits",
    }
    assert all(entry.has_evidence for entry in OPENAI_PROFILE_CAPABILITY_LEDGER)

    test_names = {
        node.name
        for path in Path(__file__).parent.glob("test_*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    }
    assert {entry.test_scenario for entry in OPENAI_PROFILE_CAPABILITY_LEDGER}.issubset(test_names)


def test_profile_operation_advertisement_is_derived_from_evidence_and_backend_support() -> None:
    assert openai_profile_operation_capabilities(supports_tool_calls=False) == {
        "streaming": True,
        "non_streaming": True,
        "tool_calls": False,
        "cancellation": True,
    }
    assert openai_profile_operation_capabilities(supports_tool_calls=True)["tool_calls"] is True


def test_profile_extensions_are_declared_non_critical_and_evidence_backed() -> None:
    assert openai_profile_extensions() == (
        {
            "name": "vllm.diagnostics",
            "critical": False,
            "description": "Adapter may include NNRP diagnostics without changing Level 1 baseline pass/fail.",
        },
    )


def test_profile_capability_ledger_keeps_limits_out_of_profile_document() -> None:
    limits = next(entry for entry in OPENAI_PROFILE_CAPABILITY_LEDGER if entry.name == "limits")

    assert limits.surface is ProfileAdvertisementSurface.RUNTIME_CONTROL
    assert limits.advertised_by_default is False
    assert limits.operation_field is None
    assert limits.extension is None
