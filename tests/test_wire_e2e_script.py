from __future__ import annotations

import json
import socket
from collections import Counter
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509 import DNSName, IPAddress

from scripts.run_wire_e2e import (
    _masked_websocket_text_frame,
    _read_websocket_rejection,
    _validate_observation_evidence,
    _validate_websocket_upgrade,
    _websocket_endpoint,
    _write_self_signed_certificate,
)


def test_wire_observation_gate_accepts_profile_and_runtime_control_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "observations.jsonl"
    records = [
        _startup("auto", "tcp", "quic", "ipc", "websocket"),
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
        _startup("auto", "tcp", "quic", "ipc", "websocket"),
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
        _startup("auto", "tcp", "quic", "ipc", "websocket"),
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


def test_wire_observation_gate_accepts_verified_terminal_delivery_disconnect_race(tmp_path: Path) -> None:
    evidence = tmp_path / "observations.jsonl"
    websocket_delivery = _operation(901, "websocket", "dropped")
    websocket_delivery.update(
        {
            "cancellation_kind": "peer_disconnect",
            "cancellation_source": "client",
            "drop_reason": None,
            "stage_transitions": [{"stage_name": "executing"}, {"stage_name": "completed"}],
        }
    )
    records = [
        _startup("auto", "tcp", "quic", "ipc", "websocket"),
        _operation(901, "tcp", "completed"),
        _operation(901, "quic", "completed"),
        _operation(901, "ipc", "completed"),
        websocket_delivery,
        _operation(101, "tcp", "cancelled", cancelled=True),
        _operation(151, "tcp", "completed"),
        _operation(101, "ipc", "cancelled", cancelled=True),
    ]
    evidence.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")

    _validate_observation_evidence(evidence)


def test_wire_observation_gate_rejects_disconnect_race_without_completed_stage(tmp_path: Path) -> None:
    evidence = tmp_path / "observations.jsonl"
    websocket_delivery = _operation(901, "websocket", "dropped")
    websocket_delivery.update(
        {
            "cancellation_kind": "peer_disconnect",
            "cancellation_source": "client",
            "drop_reason": None,
        }
    )
    records = [
        _startup("auto", "tcp", "quic", "ipc", "websocket"),
        _operation(901, "tcp", "completed"),
        _operation(901, "quic", "completed"),
        _operation(901, "ipc", "completed"),
        websocket_delivery,
        _operation(101, "tcp", "cancelled", cancelled=True),
        _operation(151, "tcp", "completed"),
        _operation(101, "ipc", "cancelled", cancelled=True),
    ]
    evidence.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")

    with pytest.raises(RuntimeError, match="completed-stage disconnect evidence"):
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


def test_websocket_text_probe_uses_a_masked_text_frame() -> None:
    payload = b"not-nnrp"

    frame = _masked_websocket_text_frame(payload)

    assert frame[0] == 0x81
    assert frame[1] == 0x80 | len(payload)
    mask = frame[2:6]
    assert bytes(value ^ mask[index % 4] for index, value in enumerate(frame[6:])) == payload


def test_websocket_upgrade_and_manifest_endpoint_validation(tmp_path: Path) -> None:
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    response = (
        b"HTTP/1.1 101 Switching Protocols\r\n"
        b"Upgrade: websocket\r\n"
        b"Connection: Upgrade\r\n"
        b"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n"
    )
    manifest = tmp_path / "target.json"
    manifest.write_text(
        json.dumps(
            {
                "wire_conformance": {
                    "transports": [{"name": "websocket", "endpoint": "ws://127.0.0.1:7768/nnrp"}]
                }
            }
        ),
        encoding="utf-8",
    )

    _validate_websocket_upgrade(response, key)

    assert _websocket_endpoint(manifest) == "ws://127.0.0.1:7768/nnrp"
    with pytest.raises(RuntimeError, match="invalid Sec-WebSocket-Accept"):
        _validate_websocket_upgrade(response.replace(b"s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", b"invalid"), key)


def test_websocket_text_probe_requires_a_close_outcome() -> None:
    with socket.socket() as connection:
        assert _read_websocket_rejection(connection, b"\x88\x00") == "close-frame"
        with pytest.raises(RuntimeError, match="opcode 2"):
            _read_websocket_rejection(connection, b"\x82\x00")


def test_wire_observation_gate_accepts_force_tcp_provider_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "observations.jsonl"
    records = [
        _startup("force_tcp", "tcp"),
        _operation(901, "tcp", "completed"),
        _operation(101, "tcp", "cancelled", cancelled=True),
        _operation(151, "tcp", "completed"),
    ]
    evidence.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")

    _validate_observation_evidence(
        evidence,
        expected_operations=Counter(
            {
                (901, "tcp", "completed"): 1,
                (101, "tcp", "cancelled"): 1,
                (151, "tcp", "completed"): 1,
            }
        ),
        expected_policy="force_tcp",
        expected_providers=frozenset({"tcp"}),
    )


def _startup(policy: str, *providers: str) -> dict[str, object]:
    return {
        "record_type": "server_startup",
        "transport_policy": policy,
        "eligible_providers": list(providers),
        "bound_provider_endpoints": {provider: f"{provider}://bound" for provider in providers},
    }


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
