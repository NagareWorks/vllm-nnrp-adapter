from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from inspect import Parameter, iscoroutinefunction, signature

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

NNRP_PY_DISTRIBUTION = "nnrp-py"
NNRP_PY_REQUIRED_RANGE = ">=1.0.0rc4.post19,<1.0.0rc5"
_NNRP_PY_REQUIRED_SPECIFIER = SpecifierSet(NNRP_PY_REQUIRED_RANGE)

_REQUIRED_NNRP_SYMBOLS = {
    "nnrp": (
        "PREVIEW4_CAPABILITY_TOKENS",
        "NativeRuntimeServerOperation",
        "NativeRuntimeServerSession",
        "NativeTransportBinding",
        "NativeTransportEndpoint",
        "NnrpEndpoint",
        "TransportPolicy",
    ),
    "nnrp.runtime": (
        "CacheReferenceMetadata",
        "CapabilityMetadata",
        "ControlRequestMetadata",
        "NativeRuntimeEvent",
        "ObjectDescriptorMetadata",
        "PartialResultMetadata",
    ),
    "nnrp.server": (
        "NativeServer",
        "NativeServerAcceptOptions",
        "NativeServerBootstrapOptions",
        "NativeServerProviderRoute",
        "NativeServerSessionOptions",
        "listen_native_server",
    ),
}

_REQUIRED_NNRP_METHODS = {
    "nnrp.NativeTransportBinding": {
        "listen": True,
        "adopt_server": False,
    },
    "nnrp.NativeRuntimeServerSession": {
        "degrade_profile": False,
        "negotiate_capabilities": False,
        "poll_events": False,
    },
    "nnrp.NativeRuntimeServerOperation": {
        "send_partial_result": True,
        "send_progress": True,
        "send_result": True,
        "send_result_drop": True,
    },
    "nnrp.server.NativeServer": {
        "accept": True,
    },
}


class NnrpRuntimeContractError(RuntimeError):
    """Raised when the installed NNRP SDK cannot satisfy the adapter contract."""


def validate_nnrp_runtime_contract(*, installed_version: str | None = None) -> str:
    resolved_version = installed_version or _installed_nnrp_version()
    try:
        parsed_version = Version(resolved_version)
    except InvalidVersion as error:
        raise NnrpRuntimeContractError(
            f"installed {NNRP_PY_DISTRIBUTION} version {resolved_version!r} is invalid; "
            f"required range is {NNRP_PY_REQUIRED_RANGE}"
        ) from error

    if parsed_version not in _NNRP_PY_REQUIRED_SPECIFIER:
        raise NnrpRuntimeContractError(
            f"installed {NNRP_PY_DISTRIBUTION} version {resolved_version} does not satisfy "
            f"{NNRP_PY_REQUIRED_RANGE}"
        )

    missing_symbols: list[str] = []
    for module_name, symbol_names in _REQUIRED_NNRP_SYMBOLS.items():
        module = import_module(module_name)
        missing_symbols.extend(
            f"{module_name}.{symbol_name}" for symbol_name in symbol_names if not hasattr(module, symbol_name)
        )
    if missing_symbols:
        raise NnrpRuntimeContractError(
            f"installed {NNRP_PY_DISTRIBUTION} version {resolved_version} does not expose the Preview4 native "
            f"role contract; missing: {', '.join(missing_symbols)}"
        )
    listen_parameters = signature(import_module("nnrp.server").listen_native_server).parameters
    transports_parameter = listen_parameters.get("transports")
    if transports_parameter is None or transports_parameter.kind is not Parameter.KEYWORD_ONLY:
        raise NnrpRuntimeContractError(
            f"installed {NNRP_PY_DISTRIBUTION} version {resolved_version} exposes an incompatible Preview4 native "
            "role contract; listen_native_server must expose the public transports keyword"
        )
    incompatible_methods: list[str] = []
    for owner_path, method_contracts in _REQUIRED_NNRP_METHODS.items():
        module_name, owner_name = owner_path.rsplit(".", 1)
        owner = getattr(import_module(module_name), owner_name)
        for method_name, async_required in method_contracts.items():
            method = getattr(owner, method_name, None)
            if not callable(method) or iscoroutinefunction(method) is not async_required:
                mode = "async" if async_required else "synchronous"
                incompatible_methods.append(f"{owner_path}.{method_name} ({mode})")
    if incompatible_methods:
        raise NnrpRuntimeContractError(
            f"installed {NNRP_PY_DISTRIBUTION} version {resolved_version} exposes an incompatible Preview4 native "
            f"role contract; required methods: {', '.join(incompatible_methods)}"
        )
    return resolved_version


def _installed_nnrp_version() -> str:
    try:
        return version(NNRP_PY_DISTRIBUTION)
    except PackageNotFoundError as error:
        raise NnrpRuntimeContractError(
            f"{NNRP_PY_DISTRIBUTION} is not installed; required range is {NNRP_PY_REQUIRED_RANGE}"
        ) from error
