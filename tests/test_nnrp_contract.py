from __future__ import annotations

import inspect
import tomllib
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    "NnrpRuntimeContractError",
    "OpenAiNnrpAdapter",
    "OpenAiNnrpCapabilityDocument",
    "OpenAiNnrpError",
    "OpenAiNnrpRequest",
    "ProfileEvent",
    "VllmBackend",
    "build_cancelled_event",
    "build_diagnostics_event",
    "build_error_event",
    "build_text_delta_event",
    "build_usage_event",
    "create_chat_completion_request",
    "create_vllm_backend",
    "run_benchmark",
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
    "operations: 'tuple[dict[str, Any], ...]', models: 'tuple[dict[str, Any], ...]' = ()) -> None",
    "OpenAiNnrpError": "(error_type: 'str', code: 'str', message: 'str') -> 'None'",
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
        validate_nnrp_runtime_contract(installed_version="1.0.0rc4.post14")


def test_contract_rejects_versions_outside_preview4_range() -> None:
    for incompatible_version in ("1.0.0rc3.post5", "1.0.0rc5"):
        with pytest.raises(NnrpRuntimeContractError, match=incompatible_version):
            validate_nnrp_runtime_contract(installed_version=incompatible_version)


def test_public_surface_contains_only_application_facing_entrypoints() -> None:
    assert set(vllm_nnrp_adapter.__all__) == EXPECTED_PUBLIC_API
    assert "NnrpFrameContext" not in vllm_nnrp_adapter.__all__
    assert "serve_openai_profile_session" not in vllm_nnrp_adapter.__all__
    assert "run_embedded_tcp_server" not in vllm_nnrp_adapter.__all__

    actual_signatures = {
        name: str(inspect.signature(getattr(vllm_nnrp_adapter, name)))
        for name in EXPECTED_ENTRYPOINT_SIGNATURES
    }
    assert actual_signatures == EXPECTED_ENTRYPOINT_SIGNATURES


def test_project_dependency_metadata_requires_preview4_without_native_payloads() -> None:
    project_root = Path(__file__).parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    assert "nnrp-py>=1.0.0rc4.post14,<1.0.0rc5" in dependencies
    assert all("rc3" not in dependency for dependency in dependencies)

    native_suffixes = {".dll", ".dylib", ".so", ".wasm"}
    packaged_native_files = [
        path
        for path in (project_root / "src" / "vllm_nnrp_adapter").rglob("*")
        if path.is_file() and path.suffix.lower() in native_suffixes
    ]
    assert packaged_native_files == []
