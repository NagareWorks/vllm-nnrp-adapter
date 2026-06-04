import pytest

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
