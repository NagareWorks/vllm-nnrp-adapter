from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict, cast

OPENAI_COMPATIBLE_PROFILE = "openai-compatible"
OPENAI_COMPATIBLE_SCHEMA_VERSION = "openai-compatible/1"
CHAT_COMPLETIONS_CREATE = "chat.completions.create"


class OpenAiNnrpError(Exception):
    def __init__(self, error_type: str, code: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.code = code
        self.message = message

    def to_event(self) -> dict[str, Any]:
        return build_error_event(error_type=self.error_type, code=self.code, message=self.message)


class OpenAiNnrpPolicy(TypedDict, total=False):
    timeout_ms: int
    diagnostics: bool
    cancel_after_events: int


class OpenAiNnrpRequest(TypedDict):
    schema_version: str
    operation: str
    body: dict[str, Any]
    request_id: NotRequired[str]
    nnrp: NotRequired[OpenAiNnrpPolicy]


class ProfileEvent(TypedDict):
    type: str


@dataclass(frozen=True)
class OpenAiNnrpCapabilityDocument:
    compatibility_levels: tuple[int, ...]
    operations: tuple[dict[str, Any], ...]
    models: tuple[dict[str, Any], ...] = ()
    extensions: tuple[dict[str, Any], ...] = ()

    @classmethod
    def level1(cls, models: tuple[str, ...] = ()) -> OpenAiNnrpCapabilityDocument:
        return cls(
            compatibility_levels=(1,),
            operations=(
                {
                    "name": CHAT_COMPLETIONS_CREATE,
                    "streaming": True,
                    "non_streaming": True,
                    "tool_calls": True,
                    "cancellation": True,
                },
            ),
            models=tuple({"id": model, "owned_by": "adapter"} for model in models),
            extensions=(
                {
                    "name": "diagnostics",
                    "critical": False,
                    "description": (
                        "Adapter may include NNRP diagnostics without changing Level 1 baseline pass/fail."
                    ),
                },
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": OPENAI_COMPATIBLE_PROFILE,
            "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
            "compatibility_levels": list(self.compatibility_levels),
            "operations": list(self.operations),
            "models": list(self.models),
            "extensions": list(self.extensions),
        }

    def supports_operation(self, operation: str) -> bool:
        return any(item.get("name") == operation for item in self.operations)


def validate_request(request: Mapping[str, Any], capabilities: OpenAiNnrpCapabilityDocument) -> OpenAiNnrpRequest:
    schema_version = request.get("schema_version")
    if schema_version != OPENAI_COMPATIBLE_SCHEMA_VERSION:
        raise OpenAiNnrpError(
            "invalid_request_error",
            "unsupported_schema_version",
            f"Expected schema_version {OPENAI_COMPATIBLE_SCHEMA_VERSION!r}.",
        )

    operation = request.get("operation")
    if not isinstance(operation, str) or not capabilities.supports_operation(operation):
        raise OpenAiNnrpError("invalid_request_error", "unsupported_operation", "Unsupported profile operation.")

    body = request.get("body")
    if not isinstance(body, dict):
        raise OpenAiNnrpError("invalid_request_error", "invalid_body", "Request body must be a JSON object.")

    if operation == CHAT_COMPLETIONS_CREATE:
        _validate_chat_body(body)

    validated: OpenAiNnrpRequest = {
        "schema_version": schema_version,
        "operation": operation,
        "body": body,
    }
    if "request_id" in request:
        request_id = request["request_id"]
        if not isinstance(request_id, str):
            raise OpenAiNnrpError(
                "invalid_request_error",
                "invalid_request_id",
                "request_id must be a string when provided.",
            )
        validated["request_id"] = request_id
    if "nnrp" in request:
        policy = request["nnrp"]
        if not isinstance(policy, dict):
            raise OpenAiNnrpError(
                "invalid_request_error",
                "invalid_nnrp_policy",
                "nnrp must be a JSON object when provided.",
            )
        _validate_nnrp_policy(policy)
        validated["nnrp"] = cast(OpenAiNnrpPolicy, policy)
    return validated


def _validate_chat_body(body: Mapping[str, Any]) -> None:
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise OpenAiNnrpError("invalid_request_error", "missing_model", "Chat completion body must include model.")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise OpenAiNnrpError(
            "invalid_request_error",
            "missing_messages",
            "Chat completion body must include messages.",
        )


def _validate_nnrp_policy(policy: Mapping[str, Any]) -> None:
    timeout_ms = policy.get("timeout_ms")
    if "timeout_ms" in policy and (type(timeout_ms) is not int or timeout_ms < 0):
        raise OpenAiNnrpError(
            "invalid_request_error",
            "invalid_nnrp_policy",
            "nnrp.timeout_ms must be a non-negative integer when provided.",
        )

    diagnostics = policy.get("diagnostics")
    if "diagnostics" in policy and not isinstance(diagnostics, bool):
        raise OpenAiNnrpError(
            "invalid_request_error",
            "invalid_nnrp_policy",
            "nnrp.diagnostics must be a boolean when provided.",
        )

    cancel_after_events = policy.get("cancel_after_events")
    if "cancel_after_events" in policy and (
        type(cancel_after_events) is not int or cancel_after_events < 0
    ):
        raise OpenAiNnrpError(
            "invalid_request_error",
            "invalid_nnrp_policy",
            "nnrp.cancel_after_events must be a non-negative integer when provided.",
        )


def build_text_delta_event(
    delta: str,
    *,
    index: int = 0,
    openai_chunk: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"type": "response.output_text.delta", "index": index, "delta": delta}
    if openai_chunk is not None:
        event["openai_chunk"] = dict(openai_chunk)
    return event


def build_tool_call_started_event(
    *,
    index: int,
    item_id: str,
    call_id: str,
    name: str,
    openai_chunk: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "response.tool_call.started",
        "index": index,
        "item_id": item_id,
        "call_id": call_id,
        "name": name,
    }
    if openai_chunk is not None:
        event["openai_chunk"] = dict(openai_chunk)
    return event


def build_tool_call_delta_event(
    arguments_delta: str,
    *,
    index: int,
    item_id: str,
    call_id: str,
    openai_chunk: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "response.tool_call.delta",
        "index": index,
        "item_id": item_id,
        "call_id": call_id,
        "arguments_delta": arguments_delta,
    }
    if openai_chunk is not None:
        event["openai_chunk"] = dict(openai_chunk)
    return event


def build_tool_call_completed_event(
    *,
    index: int,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
    openai_chunk: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "response.tool_call.completed",
        "index": index,
        "item_id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }
    if openai_chunk is not None:
        event["openai_chunk"] = dict(openai_chunk)
    return event


def build_tool_call_error_event(
    *,
    index: int,
    item_id: str,
    call_id: str,
    error_type: str,
    code: str,
    message: str,
    openai_chunk: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "response.tool_call.error",
        "index": index,
        "item_id": item_id,
        "call_id": call_id,
        "error": {
            "type": error_type,
            "code": code,
            "message": message,
        },
    }
    if openai_chunk is not None:
        event["openai_chunk"] = dict(openai_chunk)
    return event


def build_usage_event(usage: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "response.usage", "usage": dict(usage)}


def build_diagnostics_event(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "response.diagnostics", "diagnostics": dict(fields)}


def build_completed_event(body: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "response.completed", "body": dict(body)}


def build_error_event(
    error_type: str,
    code: str,
    message: str,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "response.error",
        "error": {
            "type": error_type,
            "code": code,
            "message": message,
        },
    }
    if diagnostics is not None:
        event["diagnostics"] = dict(diagnostics)
    return event


def build_cancelled_event(reason: str = "client_cancelled") -> dict[str, Any]:
    return {"type": "response.cancelled", "reason": reason}
