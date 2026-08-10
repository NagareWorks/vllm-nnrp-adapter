from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

NNRP_PY_DISTRIBUTION = "nnrp-py"
NNRP_PY_REQUIRED_RANGE = ">=1.0.0rc4.post14,<1.0.0rc5"
_NNRP_PY_REQUIRED_SPECIFIER = SpecifierSet(NNRP_PY_REQUIRED_RANGE)

_REQUIRED_NNRP_SYMBOLS = {
    "nnrp": (
        "NativeRuntimeServerOperation",
        "NativeRuntimeServerSession",
        "NativeTransportEndpoint",
        "NnrpEndpoint",
        "TransportPolicy",
    ),
    "nnrp.runtime": (
        "CacheReferenceMetadata",
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
    return resolved_version


def _installed_nnrp_version() -> str:
    try:
        return version(NNRP_PY_DISTRIBUTION)
    except PackageNotFoundError as error:
        raise NnrpRuntimeContractError(
            f"{NNRP_PY_DISTRIBUTION} is not installed; required range is {NNRP_PY_REQUIRED_RANGE}"
        ) from error
