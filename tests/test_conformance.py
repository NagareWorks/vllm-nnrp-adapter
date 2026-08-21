import json
from pathlib import Path

import pytest

from vllm_nnrp_adapter.cli import main
from vllm_nnrp_adapter.conformance import MockChatCompletionBackend, load_backend, run_conformance_plan

FIXTURE_PLAN = Path(__file__).parent / "fixtures" / "api-profile-execution-plan.json"


def make_backend() -> MockChatCompletionBackend:
    return MockChatCompletionBackend()


@pytest.mark.asyncio
async def test_conformance_runner_executes_plan_with_mock_backend() -> None:
    plan = json.loads(FIXTURE_PLAN.read_text(encoding="utf-8"))

    report = await run_conformance_plan(plan, backend=MockChatCompletionBackend())

    assert report["profile"] == "openai-compatible"
    assert report["schema_version"] == "openai-compatible/1"
    assert report["adapter"] == "vllm-nnrp-adapter"
    assert len(report["results"]) == 8
    assert {result["outcome"] for result in report["results"]} == {"passed"}

    terminals = {result["id"]: result["terminal"] for result in report["results"]}
    assert terminals["openai-compatible.chat.streaming-text"] == "success"
    assert terminals["openai-compatible.chat.non-streaming"] == "success"
    assert terminals["openai-compatible.chat.invalid-body"] == "error"
    assert terminals["openai-compatible.chat.unsupported-operation"] == "error"
    assert terminals["openai-compatible.chat.usage"] == "success"
    assert terminals["openai-compatible.chat.tool-call-delta"] == "success"
    assert terminals["openai-compatible.chat.cancellation"] == "cancelled"
    assert terminals["openai-compatible.chat.backend-error"] == "error"
    tool_result = next(
        result for result in report["results"] if result["id"] == "openai-compatible.chat.tool-call-delta"
    )
    assert [event["type"] for event in tool_result["events"] if event["type"].startswith("response.tool_call")] == [
        "response.tool_call.started",
        "response.tool_call.delta",
        "response.tool_call.completed",
    ]


def test_cli_writes_conformance_result_file(tmp_path: Path) -> None:
    output = tmp_path / "api-profile-results.json"

    exit_code = main(
        [
            "run-conformance-plan",
            "--plan",
            str(FIXTURE_PLAN),
            "--output",
            str(output),
            "--backend",
            "mock",
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["results"]) == 8
    assert report["results"][1]["events"][0]["type"] == "response.completed"


def test_load_backend_accepts_mock_and_module_factory() -> None:
    assert isinstance(load_backend("mock"), MockChatCompletionBackend)
    assert isinstance(load_backend(f"{__name__}:make_backend"), MockChatCompletionBackend)


def test_load_backend_rejects_bad_specs() -> None:
    with pytest.raises(ValueError):
        load_backend("missing_separator")

    with pytest.raises(TypeError):
        load_backend(f"{__name__}:FIXTURE_PLAN")


@pytest.mark.asyncio
async def test_conformance_runner_preserves_nnrp_policy() -> None:
    plan = json.loads(FIXTURE_PLAN.read_text(encoding="utf-8"))
    plan["cases"] = [plan["cases"][0]]
    plan["cases"][0]["request"]["nnrp"] = {"diagnostics": True}

    report = await run_conformance_plan(plan, backend=MockChatCompletionBackend())

    assert report["results"][0]["outcome"] == "passed"


@pytest.mark.asyncio
async def test_conformance_runner_rejects_bad_plan_shape() -> None:
    with pytest.raises(TypeError, match="cases must be a JSON array"):
        await run_conformance_plan(
            {
                "profile": "openai-compatible",
                "schema_version": "openai-compatible/1",
                "adapter": "vllm-nnrp-adapter",
                "cases": {},
            },
            backend=MockChatCompletionBackend(),
        )


@pytest.mark.asyncio
async def test_conformance_runner_fails_when_expected_field_is_absent() -> None:
    plan = json.loads(FIXTURE_PLAN.read_text(encoding="utf-8"))
    plan["cases"] = [case for case in plan["cases"] if case["id"] == "openai-compatible.chat.backend-error"]
    plan["cases"][0]["expect"]["events"][0]["fields"]["error"]["code"] = "different"

    report = await run_conformance_plan(plan, backend=MockChatCompletionBackend())

    assert report["results"][0]["outcome"] == "failed"
    assert "response.error count mismatch" in report["results"][0]["message"]
