from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

_ALL_PROVIDERS = ("tcp", "quic", "ipc", "websocket")
_ALL_OPERATIONS = Counter(
    {
        (901, "tcp", "completed"): 1,
        (901, "quic", "completed"): 1,
        (901, "ipc", "completed"): 1,
        (901, "websocket", "completed"): 1,
        (101, "tcp", "cancelled"): 1,
        (151, "tcp", "completed"): 1,
        (101, "ipc", "cancelled"): 1,
    }
)
_FORCE_TCP_OPERATIONS = Counter(
    {
        (901, "tcp", "completed"): 1,
        (101, "tcp", "cancelled"): 1,
        (151, "tcp", "completed"): 1,
    }
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the adapter against the independent NNRP wire suite.")
    parser.add_argument("--conformance-root", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    args = parser.parse_args()

    conformance_root = args.conformance_root.resolve()
    artifacts = args.artifact_directory.resolve()
    _run_policy_case(
        conformance_root,
        artifacts,
        transport_policy="auto",
        providers=_ALL_PROVIDERS,
        expected_operations=_ALL_OPERATIONS,
    )
    _run_policy_case(
        conformance_root,
        artifacts / "policies" / "prefer-quic",
        transport_policy="prefer_quic",
        providers=_ALL_PROVIDERS,
        expected_operations=_ALL_OPERATIONS,
    )
    _run_policy_case(
        conformance_root,
        artifacts / "policies" / "force-tcp",
        transport_policy="force_tcp",
        providers=("tcp",),
        expected_operations=_FORCE_TCP_OPERATIONS,
    )
    return 0


def _run_policy_case(
    conformance_root: Path,
    artifacts: Path,
    *,
    transport_policy: str,
    providers: tuple[str, ...],
    expected_operations: Counter[tuple[int, str, str]],
) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    target_manifest = artifacts / "target.json"
    plan = artifacts / "plan.json"
    results = artifacts / "results.json"
    evidence = artifacts / "evidence"
    target_log = artifacts / "target.log"
    observation_evidence = artifacts / "observations.jsonl"
    for path in (target_manifest, plan, results, target_log, observation_evidence):
        path.unlink(missing_ok=True)

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    certificate_path, private_key_path = _write_self_signed_certificate(artifacts / "certs")
    target_command = [
        sys.executable,
        "-m",
        "vllm_nnrp_adapter.cli",
        "serve-wire-target",
        "--ready-output",
        str(target_manifest),
        "--observation-output",
        str(observation_evidence),
        "--wire-certificate",
        str(certificate_path),
        "--wire-private-key",
        str(private_key_path),
        "--transport-policy",
        transport_policy,
    ]
    for provider in providers:
        target_command.extend(("--provider", provider))
    with target_log.open("w", encoding="utf-8") as log:
        target = subprocess.Popen(
            target_command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        )
        try:
            _wait_for_ready(target, target_manifest)
            if "websocket" in providers:
                rejection = _assert_websocket_text_rejected(_websocket_endpoint(target_manifest))
                (artifacts / "websocket-text-rejection.json").write_text(
                    json.dumps(rejection, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            _run_runner(
                conformance_root,
                [
                    "wire-plan",
                    "--suite",
                    str(conformance_root / "wire-conformance/nnrp-1-preview4/manifest.json"),
                    "--target",
                    str(target_manifest),
                    "--scenarios",
                    str(conformance_root / "wire-conformance/nnrp-1-preview4/cases/openai-compatible-e2e.json"),
                    "--scenarios",
                    str(conformance_root / "wire-conformance/nnrp-1-preview4/cases/runtime-control-e2e.json"),
                    "--output",
                    str(plan),
                    "--results-path",
                    str(results),
                    "--evidence-dir",
                    str(evidence),
                ],
                environment,
            )
            _run_runner(
                conformance_root,
                ["wire-run", "--plan", str(plan), "--target", str(target_manifest), "--output", str(results)],
                environment,
            )
            _run_runner(
                conformance_root,
                ["validate-wire-results", "--plan", str(plan), "--results", str(results)],
                environment,
            )
        finally:
            _stop_target(target)
    _validate_observation_evidence(
        observation_evidence,
        expected_operations=expected_operations,
        expected_policy=transport_policy,
        expected_providers=frozenset(providers),
    )


def _websocket_endpoint(target_manifest: Path) -> str:
    target = json.loads(target_manifest.read_text(encoding="utf-8"))
    transports = target.get("wire_conformance", {}).get("transports", [])
    endpoints = [item.get("endpoint") for item in transports if item.get("name") == "websocket"]
    if len(endpoints) != 1 or not isinstance(endpoints[0], str):
        raise RuntimeError("wire target manifest does not contain exactly one WebSocket endpoint")
    return endpoints[0]


def _assert_websocket_text_rejected(endpoint: str) -> dict[str, object]:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "ws" or parsed.hostname is None or parsed.port is None:
        raise RuntimeError("text-frame E2E requires a plain ws:// loopback endpoint")
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    host_header = parsed.hostname if parsed.port == 80 else f"{parsed.hostname}:{parsed.port}"
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host_header}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    ).encode("ascii")
    with socket.create_connection((parsed.hostname, parsed.port), timeout=5) as connection:
        connection.settimeout(5)
        connection.sendall(request)
        response, remainder = _read_http_upgrade_response(connection)
        _validate_websocket_upgrade(response, key)
        connection.sendall(_masked_websocket_text_frame(b"not-nnrp"))
        rejection = _read_websocket_rejection(connection, remainder)
    return {"endpoint": endpoint, "handshake_status": 101, "rejection": rejection}


def _read_http_upgrade_response(connection: socket.socket) -> tuple[bytes, bytes]:
    response = bytearray()
    while b"\r\n\r\n" not in response:
        chunk = connection.recv(4_096)
        if not chunk:
            raise RuntimeError("WebSocket carrier closed before completing the HTTP upgrade")
        response.extend(chunk)
        if len(response) > 65_536:
            raise RuntimeError("WebSocket HTTP upgrade response exceeded 64 KiB")
    header, remainder = bytes(response).split(b"\r\n\r\n", 1)
    return header, remainder


def _validate_websocket_upgrade(response: bytes, key: str) -> None:
    lines = response.decode("ascii").split("\r\n")
    if not lines or not lines[0].startswith("HTTP/1.1 101 "):
        raise RuntimeError(f"WebSocket carrier rejected the HTTP upgrade: {lines[0] if lines else ''}")
    headers = {
        name.strip().lower(): value.strip()
        for line in lines[1:]
        if ":" in line
        for name, value in (line.split(":", 1),)
    }
    expected_accept = base64.b64encode(
        hashlib.sha1(f"{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11".encode("ascii")).digest()
    ).decode("ascii")
    if headers.get("sec-websocket-accept") != expected_accept:
        raise RuntimeError("WebSocket carrier returned an invalid Sec-WebSocket-Accept value")


def _masked_websocket_text_frame(payload: bytes) -> bytes:
    if len(payload) >= 126:
        raise ValueError("text-frame E2E payload must use the compact WebSocket length encoding")
    mask = os.urandom(4)
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return bytes((0x81, 0x80 | len(payload))) + mask + masked


def _read_websocket_rejection(connection: socket.socket, remainder: bytes) -> str:
    data = remainder
    try:
        while not data:
            chunk = connection.recv(4_096)
            if not chunk:
                return "connection-closed"
            data += chunk
    except (ConnectionResetError, ConnectionAbortedError):
        return "connection-reset"
    except TimeoutError as error:
        raise RuntimeError("WebSocket carrier left a text-frame connection open") from error
    if data[0] & 0x0F != 0x08:
        raise RuntimeError(f"WebSocket carrier replied to a text frame with opcode {data[0] & 0x0F}")
    return "close-frame"


def _wait_for_ready(target: subprocess.Popen[str], target_manifest: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if target_manifest.is_file():
            return
        return_code = target.poll()
        if return_code is not None:
            raise RuntimeError(f"wire target exited before readiness with code {return_code}")
        time.sleep(0.05)
    raise TimeoutError("wire target did not publish readiness within 30 seconds")


def _run_runner(conformance_root: Path, arguments: list[str], environment: dict[str, str]) -> None:
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(conformance_root / "Cargo.toml"),
            "-p",
            "nnrp-conformance-runner",
            "--",
            *arguments,
        ],
        check=True,
        env=environment,
    )


def _stop_target(target: subprocess.Popen[str]) -> None:
    if target.poll() is not None:
        return
    target.terminate()
    try:
        target.wait(timeout=10)
    except subprocess.TimeoutExpired:
        target.kill()
        target.wait(timeout=10)


def _validate_observation_evidence(
    path: Path,
    *,
    expected_operations: Counter[tuple[int, str, str]] = _ALL_OPERATIONS,
    expected_policy: str = "auto",
    expected_providers: frozenset[str] = frozenset(_ALL_PROVIDERS),
) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    startup_records = [record for record in records if record.get("record_type") == "server_startup"]
    operation_records = [record for record in records if record.get("record_type") == "operation"]
    if len(startup_records) != 1:
        raise RuntimeError(f"wire target emitted {len(startup_records)} startup observation records")
    if len(operation_records) != sum(expected_operations.values()):
        raise RuntimeError(f"wire target emitted {len(operation_records)} operation observation records")
    startup = startup_records[0]
    if startup.get("transport_policy") != expected_policy:
        raise RuntimeError("wire target startup observation carries the wrong transport policy")
    if set(startup.get("eligible_providers", ())) != expected_providers:
        raise RuntimeError("wire target startup observation carries the wrong eligible provider set")
    bound_providers = startup.get("bound_provider_endpoints")
    if not isinstance(bound_providers, dict) or set(bound_providers) != expected_providers:
        raise RuntimeError("wire target startup observation carries the wrong bound provider set")
    observed_operations = Counter(
        (
            record.get("operation_id"),
            record.get("selected_transport"),
            _normalized_operation_outcome(record),
        )
        for record in operation_records
    )
    if observed_operations != expected_operations:
        raise RuntimeError(
            "wire observations do not match the exercised profile and runtime-control operations: "
            f"{observed_operations}"
        )
    for record in operation_records:
        if not isinstance(record.get("operation_id"), int) or record["operation_id"] <= 0:
            raise RuntimeError("wire observation did not preserve operation identity")
        if not isinstance(record.get("stage_transitions"), list) or not record["stage_transitions"]:
            raise RuntimeError("wire observation did not preserve the PROGRESS stage timeline")
        terminal_outcome = record.get("terminal_outcome")
        if terminal_outcome == "cancelled":
            if (
                record.get("cancellation_kind") != "cancel"
                or record.get("cancellation_source") != "client"
                or record.get("drop_reason") != "peer_cancelled"
            ):
                raise RuntimeError("wire cancellation observation did not preserve control and drop evidence")
            if not isinstance(record.get("output_event_count"), int) or record["output_event_count"] < 1:
                raise RuntimeError("wire cancellation observation did not preserve its in-flight partial result")
        elif terminal_outcome == "dropped":
            if (
                record.get("operation_id") != 901
                or record.get("cancellation_kind") != "peer_disconnect"
                or record.get("cancellation_source") != "client"
                or record.get("drop_reason") is not None
                or not any(
                    transition.get("stage_name") == "completed"
                    for transition in record["stage_transitions"]
                    if isinstance(transition, dict)
                )
            ):
                raise RuntimeError("wire terminal-delivery race lacks completed-stage disconnect evidence")
        elif any(
            record.get(field) is not None
            for field in ("cancellation_kind", "cancellation_source", "drop_reason")
        ):
            raise RuntimeError("wire completion observation was reclassified by a late control event")


def _normalized_operation_outcome(record: dict[str, object]) -> object:
    if (
        record.get("operation_id") == 901
        and record.get("terminal_outcome") == "dropped"
        and record.get("cancellation_kind") == "peer_disconnect"
        and record.get("cancellation_source") == "client"
    ):
        return "completed"
    return record.get("terminal_outcome")


def _write_self_signed_certificate(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    private_key = rsa.generate_private_key(public_exponent=65_537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NNRP conformance"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )
    certificate_path = directory / "server.der"
    private_key_path = directory / "server-key.der"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.DER))
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return certificate_path, private_key_path


if __name__ == "__main__":
    raise SystemExit(main())
