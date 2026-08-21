import json
from pathlib import Path

import pytest

from vllm_nnrp_adapter import OpenAiNnrpAdapter
from vllm_nnrp_adapter.profile import (
    CHAT_COMPLETIONS_CREATE,
    OPENAI_COMPATIBLE_SCHEMA_VERSION,
    OpenAiNnrpCapabilityDocument,
    OpenAiNnrpError,
    validate_request,
)


def test_level1_capability_document_shape() -> None:
    document = OpenAiNnrpCapabilityDocument.level1(models=("llama",)).to_dict()

    assert document["profile"] == "openai-compatible"
    assert document["schema_version"] == OPENAI_COMPATIBLE_SCHEMA_VERSION
    assert document["compatibility_levels"] == [1]
    assert document["operations"][0]["name"] == CHAT_COMPLETIONS_CREATE
    assert document["operations"][0]["cancellation"] is True
    assert document["models"] == [{"id": "llama", "owned_by": "adapter"}]


def test_release_capability_manifest_advertises_tested_tool_call_lifecycle() -> None:
    class ProductionCapabilityBackend:
        supports_tool_calls = True

        def create_chat_completion(self, body):
            return {"model": body["model"], "choices": []}

    manifest = json.loads(
        (Path(__file__).parents[1] / "conformance" / "openai-api-capabilities.json").read_text(encoding="utf-8")
    )
    adapter = OpenAiNnrpAdapter(ProductionCapabilityBackend())

    assert manifest["operations"][0]["tool_calls"] is True
    assert manifest["operations"] == list(adapter.capabilities.operations)


def test_validate_chat_request_preserves_body_and_policy() -> None:
    capabilities = OpenAiNnrpCapabilityDocument.level1()

    request = validate_request(
        {
            "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
            "operation": CHAT_COMPLETIONS_CREATE,
            "request_id": "req-1",
            "body": {
                "model": "llama",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            "nnrp": {"timeout_ms": 10, "diagnostics": True},
        },
        capabilities,
    )

    assert request["request_id"] == "req-1"
    assert request["body"]["model"] == "llama"
    assert request["nnrp"]["timeout_ms"] == 10


def test_validate_rejects_unknown_schema() -> None:
    with pytest.raises(OpenAiNnrpError) as error:
        validate_request(
            {
                "schema_version": "other",
                "operation": CHAT_COMPLETIONS_CREATE,
                "body": {"model": "llama", "messages": [{"role": "user", "content": "hello"}]},
            },
            OpenAiNnrpCapabilityDocument.level1(),
        )

    assert error.value.code == "unsupported_schema_version"


def test_validate_rejects_missing_messages() -> None:
    with pytest.raises(OpenAiNnrpError) as error:
        validate_request(
            {
                "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
                "operation": CHAT_COMPLETIONS_CREATE,
                "body": {"model": "llama"},
            },
            OpenAiNnrpCapabilityDocument.level1(),
        )

    assert error.value.code == "missing_messages"


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    (
        ("request_id", 7, "invalid_request_id"),
        ("nnrp", [], "invalid_nnrp_policy"),
    ),
)
def test_validate_rejects_invalid_optional_envelope_fields(
    field: str,
    value: object,
    expected_code: str,
) -> None:
    envelope = {
        "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "operation": CHAT_COMPLETIONS_CREATE,
        "body": {"model": "llama", "messages": [{"role": "user", "content": "hello"}]},
        field: value,
    }

    with pytest.raises(OpenAiNnrpError) as error:
        validate_request(envelope, OpenAiNnrpCapabilityDocument.level1())

    assert error.value.code == expected_code


@pytest.mark.parametrize(
    ("policy", "message"),
    (
        ({"timeout_ms": -1}, "timeout_ms"),
        ({"timeout_ms": True}, "timeout_ms"),
        ({"diagnostics": "yes"}, "diagnostics"),
        ({"cancel_after_events": -1}, "cancel_after_events"),
        ({"cancel_after_events": False}, "cancel_after_events"),
    ),
)
def test_validate_rejects_invalid_known_nnrp_policy_values(
    policy: dict[str, object],
    message: str,
) -> None:
    envelope = {
        "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
        "operation": CHAT_COMPLETIONS_CREATE,
        "body": {"model": "llama", "messages": [{"role": "user", "content": "hello"}]},
        "nnrp": policy,
    }

    with pytest.raises(OpenAiNnrpError, match=message) as error:
        validate_request(envelope, OpenAiNnrpCapabilityDocument.level1())

    assert error.value.code == "invalid_nnrp_policy"


@pytest.mark.parametrize("operation", ("responses.create", "models.list", "embeddings.create"))
def test_level1_rejects_level2_and_level3_operations(operation: str) -> None:
    with pytest.raises(OpenAiNnrpError) as error:
        validate_request(
            {
                "schema_version": OPENAI_COMPATIBLE_SCHEMA_VERSION,
                "operation": operation,
                "body": {"model": "llama", "messages": [{"role": "user", "content": "hello"}]},
            },
            OpenAiNnrpCapabilityDocument.level1(),
        )

    assert error.value.code == "unsupported_operation"
