from __future__ import annotations

import inspect
import tomllib
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
from nnrp.core import MessageType
from nnrp.runtime import (
    BudgetMetadata,
    CapabilityMetadata,
    ControlRequestMetadata,
    PartialResultMetadata,
    PressureMetadata,
    ProgressMetadata,
    RecoverableErrorMetadata,
    ResultDropReasonMetadata,
    RetryAfterMetadata,
    RouteHintMetadata,
    RuntimeRole,
    SchedulingMetadata,
    SupersedeMetadata,
    TraceContextMetadata,
    decode_runtime_control_metadata,
    encode_runtime_control_metadata,
)
from packaging.specifiers import SpecifierSet

import vllm_nnrp_adapter
from vllm_nnrp_adapter import NNRP_PY_REQUIRED_RANGE, NnrpRuntimeContractError
from vllm_nnrp_adapter.nnrp_contract import validate_nnrp_runtime_contract
from vllm_nnrp_adapter.vllm_factory import create_vllm_backend

EXPECTED_PUBLIC_API = {
    "OPENAI_COMPATIBLE_PROFILE",
    "OPENAI_COMPATIBLE_SCHEMA_VERSION",
    "NNRP_PY_REQUIRED_RANGE",
    "BenchmarkConfig",
    "NnrpServeStatistics",
    "NnrpServerConfig",
    "NnrpRuntimeContractError",
    "ObservationSink",
    "OperationIdentity",
    "OperationObservation",
    "OperationStageTransition",
    "OpenAiNnrpAdapter",
    "OpenAiNnrpCapabilityDocument",
    "OpenAiNnrpError",
    "OpenAiNnrpRequest",
    "ProfileEvent",
    "PrometheusObservationSink",
    "ServerStartupObservation",
    "StructuredLogObservationSink",
    "VllmBackend",
    "build_cancelled_event",
    "build_diagnostics_event",
    "build_error_event",
    "build_text_delta_event",
    "build_usage_event",
    "create_chat_completion_request",
    "create_vllm_backend",
    "run_benchmark",
    "serve",
    "validate_nnrp_runtime_contract",
}

EXPECTED_ENTRYPOINT_SIGNATURES = {
    "BenchmarkConfig": "(iterations: 'int' = 200, warmup: 'int' = 20, model: 'str' = 'mock-model', "
    "prompt_tokens: 'tuple[int, ...]' = (4096, 8192, 16384, 20480), "
    "concurrency: 'tuple[int, ...]' = (1, 2, 4), max_completion_tokens: 'int' = 128, "
    "http_url: 'str | None' = None, http_api_key: 'str | None' = None) -> None",
    "OpenAiNnrpAdapter": "(backend: 'ChatCompletionBackend', *, "
    "capabilities: 'OpenAiNnrpCapabilityDocument | None' = None) -> 'None'",
    "OpenAiNnrpCapabilityDocument": "(compatibility_levels: 'tuple[int, ...]', "
    "operations: 'tuple[dict[str, Any], ...]', models: 'tuple[dict[str, Any], ...]' = (), "
    "extensions: 'tuple[dict[str, Any], ...]' = ()) -> None",
    "OpenAiNnrpError": "(error_type: 'str', code: 'str', message: 'str') -> 'None'",
    "NnrpServeStatistics": "(accepted_sessions: 'int', accepted_operations: 'int', partial_results: 'int', "
    "terminal_results: 'int') -> None",
    "NnrpServerConfig": "(endpoint: 'str', provider_routes: 'Mapping[str, NativeServerProviderRoute]' = <factory>, "
    "transports: 'Sequence[NativeTransportBinding] | None' = None, "
    "transport_policy: 'TransportPolicy' = <TransportPolicy.AUTO: 0>, "
    "session_options: 'NativeServerSessionOptions' = <factory>, "
    "accept_timeout_ms: 'int' = 100, receive_timeout_ms: 'int' = 100, max_active_sessions: 'int' = 8, "
    "max_operations_per_session: 'int' = 4, native_worker_count: 'int' = 9, "
    "observation_sinks: 'Sequence[ObservationSink]' = <factory>) -> None",
    "VllmBackend": "(serving_chat: 'object', *, request_factory: 'object | None' = None, "
    "prefer_engine_direct: 'bool' = True) -> 'None'",
    "build_cancelled_event": "(reason: 'str' = 'client_cancelled') -> 'dict[str, Any]'",
    "build_diagnostics_event": "(fields: 'Mapping[str, Any]') -> 'dict[str, Any]'",
    "build_error_event": "(error_type: 'str', code: 'str', message: 'str', *, "
    "diagnostics: 'Mapping[str, Any] | None' = None) -> 'dict[str, Any]'",
    "build_text_delta_event": "(delta: 'str', *, index: 'int' = 0, "
    "openai_chunk: 'Mapping[str, Any] | None' = None) -> 'dict[str, Any]'",
    "build_usage_event": "(usage: 'Mapping[str, Any]') -> 'dict[str, Any]'",
    "create_chat_completion_request": "(body: 'Mapping[str, Any]') -> 'object'",
    "create_vllm_backend": "(serving_chat: 'object') -> 'VllmBackend'",
    "run_benchmark": "(*, backend: 'ChatCompletionBackend', config: 'BenchmarkConfig') -> 'dict[str, Any]'",
    "serve": "(adapter: 'OpenAiNnrpAdapter', *, config: 'NnrpServerConfig', "
    "stop_event: 'asyncio.Event | None' = None) -> 'NnrpServeStatistics'",
    "validate_nnrp_runtime_contract": "(*, installed_version: 'str | None' = None) -> 'str'",
}


def test_installed_preview4_contract_is_available() -> None:
    installed_version = validate_nnrp_runtime_contract()

    assert installed_version == version("nnrp-py")
    assert installed_version in SpecifierSet(NNRP_PY_REQUIRED_RANGE)


def test_contract_rejects_preview3_before_binding_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_preview3() -> str:
        raise NnrpRuntimeContractError(
            "installed nnrp-py version 1.0.0rc3.post5 does not satisfy " + NNRP_PY_REQUIRED_RANGE
        )

    monkeypatch.setattr("vllm_nnrp_adapter.vllm_factory.validate_nnrp_runtime_contract", reject_preview3)

    with pytest.raises(NnrpRuntimeContractError, match="rc3.post5"):
        create_vllm_backend(object())


def test_contract_rejects_missing_preview4_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_contract.import_module",
        lambda module_name: SimpleNamespace() if module_name == "nnrp.server" else __import__(module_name),
    )

    with pytest.raises(NnrpRuntimeContractError, match="nnrp.server.NativeServer"):
        validate_nnrp_runtime_contract(installed_version="1.0.0rc4.post20")


def test_contract_rejects_old_synchronous_native_role_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import nnrp

    class OldNativeRuntimeServerOperation:
        def send_partial_result(self) -> None:
            pass

        def send_result(self) -> None:
            pass

        def send_result_drop(self) -> None:
            pass

    old_nnrp = SimpleNamespace(
        **{
            name: getattr(nnrp, name)
            for name in (
                "PREVIEW4_CAPABILITY_TOKENS",
                "NativeRuntimeServerSession",
                "NativeTransportBinding",
                "NativeTransportEndpoint",
                "NnrpEndpoint",
                "TransportPolicy",
            )
        },
        NativeRuntimeServerOperation=OldNativeRuntimeServerOperation,
    )
    real_import = __import__
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_contract.import_module",
        lambda module_name: old_nnrp if module_name == "nnrp" else real_import(module_name, fromlist=["*"]),
    )

    with pytest.raises(NnrpRuntimeContractError, match="incompatible Preview4 native role contract"):
        validate_nnrp_runtime_contract(installed_version="1.0.0rc4.post20")


def test_contract_rejects_private_transport_binding_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import import_module

    server_module = import_module("nnrp.server")

    def old_listen(options: object, *, _transports: object = None) -> None:
        pass

    old_server_module = SimpleNamespace(
        **{
            name: getattr(server_module, name)
            for name in (
                "NativeServer",
                "NativeServerAcceptOptions",
                "NativeServerBootstrapOptions",
                "NativeServerProviderRoute",
                "NativeServerSessionOptions",
            )
        },
        listen_native_server=old_listen,
    )
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_contract.import_module",
        lambda module_name: old_server_module if module_name == "nnrp.server" else import_module(module_name),
    )

    with pytest.raises(NnrpRuntimeContractError, match="public transports keyword"):
        validate_nnrp_runtime_contract(installed_version="1.0.0rc4.post20")


def test_contract_rejects_transport_binding_without_role_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    import nnrp

    class OldNativeTransportBinding:
        async def listen(self) -> None:
            pass

    old_nnrp = SimpleNamespace(
        **{
            name: getattr(nnrp, name)
            for name in (
                "PREVIEW4_CAPABILITY_TOKENS",
                "NativeRuntimeServerOperation",
                "NativeRuntimeServerSession",
                "NativeTransportEndpoint",
                "NnrpEndpoint",
                "TransportPolicy",
            )
        },
        NativeTransportBinding=OldNativeTransportBinding,
    )
    real_import = __import__
    monkeypatch.setattr(
        "vllm_nnrp_adapter.nnrp_contract.import_module",
        lambda module_name: old_nnrp if module_name == "nnrp" else real_import(module_name, fromlist=["*"]),
    )

    with pytest.raises(NnrpRuntimeContractError, match="NativeTransportBinding.adopt_server"):
        validate_nnrp_runtime_contract(installed_version="1.0.0rc4.post20")


def test_contract_rejects_versions_outside_preview4_range() -> None:
    for incompatible_version in ("1.0.0rc3.post5", "1.0.0rc4.post17", "1.0.0rc5"):
        with pytest.raises(NnrpRuntimeContractError, match=incompatible_version):
            validate_nnrp_runtime_contract(installed_version=incompatible_version)


def test_contract_rejects_frozen_control_metadata_shape_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    import nnrp.runtime as runtime_module

    class DriftedControlRequestMetadata:
        STRUCT = SimpleNamespace(size=31)

    monkeypatch.setattr(runtime_module, "ControlRequestMetadata", DriftedControlRequestMetadata)

    with pytest.raises(NnrpRuntimeContractError, match="metadata drift: ControlRequestMetadata"):
        validate_nnrp_runtime_contract(installed_version="1.0.0rc4.post20")


@pytest.mark.parametrize(
    ("message_type", "metadata", "tail"),
    (
        (MessageType.CANCEL, ControlRequestMetadata(1, 1, 0, RuntimeRole.CLIENT, 0, 1), b"c"),
        (MessageType.ABORT, ControlRequestMetadata(1, 2, 0, RuntimeRole.CLIENT, 0, 1), b"a"),
        (MessageType.PRIORITY_UPDATE, SchedulingMetadata(1, 3, 1, -1, 0, 0), b""),
        (MessageType.DEADLINE, SchedulingMetadata(1, 4, 0, 0, 1, 0), b""),
        (MessageType.EXPIRE_AT, SchedulingMetadata(1, 5, 0, 0, 1, 0), b""),
        (MessageType.SUPERSEDE, SupersedeMetadata(1, 2, 6, 0, 0, 1), b"s"),
        (MessageType.BUDGET_UPDATE, BudgetMetadata(1, 2, 3, 4, 5, 1), b""),
        (MessageType.PROGRESS, ProgressMetadata(1, 7, 1, 5000, 0, 1), b"p"),
        (MessageType.PARTIAL_RESULT, PartialResultMetadata(1, 8, 0, 8, 1, 0), b"r"),
        (MessageType.BACKPRESSURE, PressureMetadata(1, 1, 1, 1, 0, 2), b""),
        (MessageType.CREDIT_UPDATE, PressureMetadata(1, 2, 0, 0, 0, 2), b""),
        (MessageType.CAPABILITY_NEGOTIATION, CapabilityMetadata(0, 1, 1, 1, 0, 0, 1, 0), b"x"),
        (MessageType.DEGRADE_PROFILE, CapabilityMetadata(0, 1, 1, 1, 0, 0, 1, 2), b"x"),
        (MessageType.ROUTE_HINT, RouteHintMetadata(1, 1, 3, 0, 0, 1, 2), b"h"),
        (MessageType.EXECUTION_HINT, RouteHintMetadata(1, 1, 3, 0, 0, 1, 1), b"h"),
        (MessageType.TRACE_CONTEXT, TraceContextMetadata(1, 2, 0, 1, 0, 1), b"t"),
        (
            MessageType.RESULT_DROP_REASON,
            ResultDropReasonMetadata(1, 1, 1, RuntimeRole.RUNTIME, 0, 1),
            b"d",
        ),
        (
            MessageType.ERROR_RECOVERABLE,
            RecoverableErrorMetadata(1, 1, 0, RuntimeRole.RUNTIME, 0, 0, 1, 1, 0, 1),
            b"e",
        ),
        (MessageType.RETRY_AFTER, RetryAfterMetadata(1, 1, 10, 1, 0, RuntimeRole.RUNTIME, 0, 1), b"r"),
    ),
)
def test_frozen_preview4_control_metadata_round_trips(
    message_type: MessageType,
    metadata: object,
    tail: bytes,
) -> None:
    payload = encode_runtime_control_metadata(message_type, metadata, tail=tail)  # type: ignore[arg-type]
    decoded = decode_runtime_control_metadata(message_type, payload)

    assert decoded.metadata == metadata
    assert decoded.tail == tail


@pytest.mark.parametrize(
    "metadata",
    (
        ControlRequestMetadata(1, 1, 0, RuntimeRole.CLIENT, 0x04, 0),
        SchedulingMetadata(1, 1, 0, 0, 0, 0x04),
        SupersedeMetadata(1, 2, 1, 0, 0x02, 0),
        BudgetMetadata(1, 0, 0, 0, 0, 0x04),
        PartialResultMetadata(1, 1, 0, 1, 0, 0x04),
        PressureMetadata(1, 0, 0, 0, 0, 0x04),
        CapabilityMetadata(0, 0, 0, 0, 0, 0, 0, 0x04),
        RouteHintMetadata(1, 0, 0, 0, 0, 0, 0x04),
        TraceContextMetadata(1, 1, 0, 0, 0x04, 0),
        ResultDropReasonMetadata(1, 1, 0, RuntimeRole.RUNTIME, 0x04, 0),
        RecoverableErrorMetadata(0, 0, 0, RuntimeRole.RUNTIME, 0x04, 0, 0, 0, 0, 0),
        RetryAfterMetadata(1, 1, 0, 0, 0, RuntimeRole.RUNTIME, 0x04, 0),
    ),
)
def test_frozen_preview4_control_metadata_rejects_reserved_flag_bits(metadata: object) -> None:
    with pytest.raises(ValueError, match="reserved bits set"):
        metadata.pack()  # type: ignore[attr-defined]


def test_frozen_preview4_progress_rejects_reserved_percent_range() -> None:
    with pytest.raises(ValueError, match="percent_x100"):
        ProgressMetadata(1, 1, 1, 10_001, 0, 0).pack()


@pytest.mark.parametrize(
    "metadata",
    (
        ControlRequestMetadata(1, 1, 0, RuntimeRole.CLIENT, 0, 0),
        PressureMetadata(1, 0, 0, 0, 0, 2),
        ResultDropReasonMetadata(1, 1, 0, RuntimeRole.RUNTIME, 0, 0),
    ),
)
def test_frozen_preview4_control_metadata_rejects_nonzero_reserved_fields(metadata: object) -> None:
    payload = bytearray(metadata.pack())  # type: ignore[attr-defined]
    payload[-1] = 1

    with pytest.raises(ValueError, match="reserved must be zero"):
        type(metadata).unpack(payload)


def test_public_surface_contains_only_application_facing_entrypoints() -> None:
    assert set(vllm_nnrp_adapter.__all__) == EXPECTED_PUBLIC_API
    assert "NnrpFrameContext" not in vllm_nnrp_adapter.__all__
    assert "serve_openai_profile_session" not in vllm_nnrp_adapter.__all__
    assert "run_embedded_tcp_server" not in vllm_nnrp_adapter.__all__

    actual_signatures = {
        name: str(inspect.signature(getattr(vllm_nnrp_adapter, name))) for name in EXPECTED_ENTRYPOINT_SIGNATURES
    }
    assert actual_signatures == EXPECTED_ENTRYPOINT_SIGNATURES


def test_project_dependency_metadata_requires_preview4_without_native_payloads() -> None:
    project_root = Path(__file__).parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    assert "nnrp-py>=1.0.0rc4.post20,<1.0.0rc5" in dependencies
    assert all("rc3" not in dependency for dependency in dependencies)
    assert metadata["project"]["optional-dependencies"]["vllm"] == ["vllm>=0.18.0,<0.27"]
    assert metadata["project"]["optional-dependencies"]["prometheus"] == ["prometheus-client>=0.21,<1"]

    native_suffixes = {".dll", ".dylib", ".so", ".wasm"}
    packaged_native_files = [
        path
        for path in (project_root / "src" / "vllm_nnrp_adapter").rglob("*")
        if path.is_file() and path.suffix.lower() in native_suffixes
    ]
    assert packaged_native_files == []
