from __future__ import annotations

from nnrp import PREVIEW4_CAPABILITY_TOKENS

from vllm_nnrp_adapter.capability_ledger import (
    RUNTIME_CAPABILITY_LEDGER,
    CapabilityClassification,
    supported_runtime_capabilities,
)


def test_runtime_capability_ledger_uses_only_frozen_tokens_and_complete_mechanisms() -> None:
    tokens = [entry.token for entry in RUNTIME_CAPABILITY_LEDGER]

    assert len(tokens) == len(set(tokens))
    assert set(tokens).issubset(PREVIEW4_CAPABILITY_TOKENS)
    assert all(entry.has_runtime_mechanism for entry in RUNTIME_CAPABILITY_LEDGER)


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
    assert {entry.token for entry in core_entries if entry.release_gate_ready} == {
        "payload.typed",
        "control.progress_partial",
    }


def test_conditional_backend_capabilities_are_not_advertised_unconditionally() -> None:
    priority = next(entry for entry in RUNTIME_CAPABILITY_LEDGER if entry.token == "control.priority_update")

    assert priority.classification is CapabilityClassification.CONDITIONAL
    assert priority.backend_requirement is not None
    assert priority.advertised_by_default is False
