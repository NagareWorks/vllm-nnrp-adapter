from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from inspect import Parameter, iscoroutinefunction, signature

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

NNRP_PY_DISTRIBUTION = "nnrp-py"
NNRP_PY_REQUIRED_RANGE = ">=1.0.0rc4.post20,<1.0.0rc5"
_NNRP_PY_REQUIRED_SPECIFIER = SpecifierSet(NNRP_PY_REQUIRED_RANGE)

_REQUIRED_NNRP_SYMBOLS = {
    "nnrp.core": ("MessageType",),
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
        "BudgetMetadata",
        "CacheReferenceMetadata",
        "CapabilityMetadata",
        "ControlRequestMetadata",
        "RecoverableErrorMetadata",
        "NativeRuntimeEvent",
        "ObjectDescriptorMetadata",
        "PartialResultMetadata",
        "PressureMetadata",
        "ProgressMetadata",
        "ResultDropReasonMetadata",
        "RetryAfterMetadata",
        "RouteHintMetadata",
        "SchedulingMetadata",
        "SupersedeMetadata",
        "TraceContextMetadata",
        "decode_runtime_control_metadata",
        "encode_runtime_control_metadata",
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

_REQUIRED_CONTROL_MESSAGE_TYPES = (
    "CANCEL",
    "ABORT",
    "PRIORITY_UPDATE",
    "DEADLINE",
    "EXPIRE_AT",
    "SUPERSEDE",
    "BUDGET_UPDATE",
    "PROGRESS",
    "PARTIAL_RESULT",
    "BACKPRESSURE",
    "CREDIT_UPDATE",
    "CAPABILITY_NEGOTIATION",
    "DEGRADE_PROFILE",
    "ROUTE_HINT",
    "EXECUTION_HINT",
    "TRACE_CONTEXT",
    "RESULT_DROP_REASON",
    "ERROR_RECOVERABLE",
    "RETRY_AFTER",
)

_REQUIRED_CONTROL_METADATA_CONTRACTS = {
    "ControlRequestMetadata": (
        ("operation_id", "control_sequence", "reason_code", "source_role", "flags", "diagnostic_bytes"),
        32,
    ),
    "SchedulingMetadata": (
        ("operation_id", "control_sequence", "priority_class", "priority_delta", "deadline_unix_ms", "flags"),
        32,
    ),
    "SupersedeMetadata": (
        (
            "old_operation_id",
            "new_operation_id",
            "control_sequence",
            "drop_reason_code",
            "flags",
            "diagnostic_bytes",
        ),
        32,
    ),
    "BudgetMetadata": (
        (
            "operation_id",
            "compute_budget_units",
            "memory_budget_bytes",
            "bandwidth_budget_bytes",
            "token_budget",
            "flags",
        ),
        40,
    ),
    "ProgressMetadata": (
        ("operation_id", "progress_sequence", "stage_code", "percent_x100", "object_id", "body_bytes"),
        32,
    ),
    "PartialResultMetadata": (
        ("operation_id", "result_sequence", "object_id", "delta_sequence", "body_bytes", "flags"),
        40,
    ),
    "PressureMetadata": (
        ("scope_id", "credit_window", "pressure_level", "pressure_reason", "retry_after_ms", "flags"),
        32,
    ),
    "CapabilityMetadata": (
        (
            "profile_id",
            "capability_count",
            "cost_model_id",
            "preference_rank",
            "limit_bytes",
            "limit_units",
            "body_bytes",
            "flags",
        ),
        32,
    ),
    "RouteHintMetadata": (
        ("operation_id", "route_id", "executor_class", "affinity_class", "deadline_unix_ms", "body_bytes", "flags"),
        32,
    ),
    "TraceContextMetadata": (
        ("trace_id", "span_id", "parent_span_id", "stage_code", "flags", "body_bytes"),
        32,
    ),
    "ResultDropReasonMetadata": (
        ("operation_id", "result_sequence", "drop_reason_code", "source_role", "flags", "diagnostic_bytes"),
        32,
    ),
    "RecoverableErrorMetadata": (
        (
            "error_code",
            "error_scope",
            "recovery_action",
            "source_role",
            "flags",
            "retry_after_ms",
            "related_session_id",
            "related_frame_id",
            "related_view_id",
            "diagnostic_bytes",
        ),
        32,
    ),
    "RetryAfterMetadata": (
        (
            "scope_id",
            "control_sequence",
            "retry_after_ms",
            "jitter_ms",
            "reason_code",
            "source_role",
            "flags",
            "diagnostic_bytes",
        ),
        32,
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
        "send_backpressure": False,
        "send_credit_update": False,
        "send_recoverable_error": False,
        "send_retry_after": False,
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
            f"installed {NNRP_PY_DISTRIBUTION} version {resolved_version} does not satisfy {NNRP_PY_REQUIRED_RANGE}"
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
    _validate_runtime_control_metadata_contract(resolved_version)
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


def _validate_runtime_control_metadata_contract(resolved_version: str) -> None:
    core_module = import_module("nnrp.core")
    runtime_module = import_module("nnrp.runtime")
    message_type = core_module.MessageType
    missing_message_types = [name for name in _REQUIRED_CONTROL_MESSAGE_TYPES if not hasattr(message_type, name)]
    incompatible_metadata: list[str] = []
    for name, (expected_fields, expected_size) in _REQUIRED_CONTROL_METADATA_CONTRACTS.items():
        metadata_type = getattr(runtime_module, name)
        actual_fields = tuple(signature(metadata_type).parameters)
        struct = getattr(metadata_type, "STRUCT", None)
        actual_size = getattr(struct, "size", None)
        if actual_fields != expected_fields or actual_size != expected_size:
            incompatible_metadata.append(f"{name}(fields={actual_fields!r}, size={actual_size!r})")
    if missing_message_types or incompatible_metadata:
        details = []
        if missing_message_types:
            details.append(f"missing message types: {', '.join(missing_message_types)}")
        if incompatible_metadata:
            details.append(f"metadata drift: {', '.join(incompatible_metadata)}")
        raise NnrpRuntimeContractError(
            f"installed {NNRP_PY_DISTRIBUTION} version {resolved_version} exposes an incompatible Preview4 "
            f"runtime-control contract; {'; '.join(details)}"
        )


def _installed_nnrp_version() -> str:
    try:
        return version(NNRP_PY_DISTRIBUTION)
    except PackageNotFoundError as error:
        raise NnrpRuntimeContractError(
            f"{NNRP_PY_DISTRIBUTION} is not installed; required range is {NNRP_PY_REQUIRED_RANGE}"
        ) from error
