from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509 import DNSName, IPAddress

from scripts.run_wire_e2e import _validate_observation_evidence, _write_self_signed_certificate


def test_wire_observation_gate_accepts_profile_and_runtime_control_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "observations.jsonl"
    records = [
        {"record_type": "server_startup"},
        _operation(901, "tcp", "completed"),
        _operation(901, "quic", "completed"),
        _operation(901, "ipc", "completed"),
        _operation(901, "websocket", "completed"),
        _operation(101, "tcp", "cancelled", cancelled=True),
        _operation(151, "tcp", "completed"),
        _operation(101, "ipc", "cancelled", cancelled=True),
    ]
    evidence.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")

    _validate_observation_evidence(evidence)


def test_wire_observation_gate_rejects_cancel_without_an_in_flight_partial(tmp_path: Path) -> None:
    evidence = tmp_path / "observations.jsonl"
    records = [
        {"record_type": "server_startup"},
        _operation(901, "tcp", "completed"),
        _operation(901, "quic", "completed"),
        _operation(901, "ipc", "completed"),
        _operation(901, "websocket", "completed"),
        _operation(101, "tcp", "cancelled", cancelled=True, output_event_count=0),
        _operation(151, "tcp", "completed"),
        _operation(101, "ipc", "cancelled", cancelled=True),
    ]
    evidence.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")

    with pytest.raises(RuntimeError, match="in-flight partial result"):
        _validate_observation_evidence(evidence)


def test_wire_observation_gate_rejects_completed_operation_with_disconnect_metadata(tmp_path: Path) -> None:
    evidence = tmp_path / "observations.jsonl"
    completed_with_disconnect = _operation(901, "tcp", "completed")
    completed_with_disconnect.update(
        {
            "cancellation_kind": "peer_disconnect",
            "cancellation_source": "client",
            "drop_reason": "transport_closed",
        }
    )
    records = [
        {"record_type": "server_startup"},
        completed_with_disconnect,
        _operation(901, "quic", "completed"),
        _operation(901, "ipc", "completed"),
        _operation(901, "websocket", "completed"),
        _operation(101, "tcp", "cancelled", cancelled=True),
        _operation(151, "tcp", "completed"),
        _operation(101, "ipc", "cancelled", cancelled=True),
    ]
    evidence.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")

    with pytest.raises(RuntimeError, match="late control event"):
        _validate_observation_evidence(evidence)


def test_wire_certificate_is_self_signed_for_local_quic_endpoint(tmp_path: Path) -> None:
    certificate_path, private_key_path = _write_self_signed_certificate(tmp_path / "certs")

    certificate = x509.load_der_x509_certificate(certificate_path.read_bytes())
    private_key = serialization.load_der_private_key(private_key_path.read_bytes(), password=None)
    subject_alt_name = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value

    assert certificate.issuer == certificate.subject
    assert subject_alt_name.get_values_for_type(DNSName) == ["localhost"]
    assert [str(value) for value in subject_alt_name.get_values_for_type(IPAddress)] == ["127.0.0.1"]
    assert private_key.key_size == 2048


def _operation(
    operation_id: int,
    transport: str,
    outcome: str,
    *,
    cancelled: bool = False,
    output_event_count: int = 1,
) -> dict[str, object]:
    record: dict[str, object] = {
        "record_type": "operation",
        "operation_id": operation_id,
        "selected_transport": transport,
        "terminal_outcome": outcome,
        "stage_transitions": [{"stage_name": "executing"}],
        "output_event_count": output_event_count,
    }
    if cancelled:
        record.update(
            {
                "cancellation_kind": "cancel",
                "cancellation_source": "client",
                "drop_reason": "peer_cancelled",
            }
        )
    return record
