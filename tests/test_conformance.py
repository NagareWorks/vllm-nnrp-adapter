import json
from pathlib import Path

import pytest

from vllm_nnrp_adapter.cli import main
from vllm_nnrp_adapter.conformance import MockChatCompletionBackend, load_backend, run_conformance_plan

FIXTURE_PLAN = Path(__file__).parent / "fixtures" / "api-profile-execution-plan.json"


def make_backend() -> MockChatCompletionBackend:
    return MockChatCompletionBackend()


def test_mock_backend_rejects_negative_stream_delay() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        MockChatCompletionBackend(stream_inter_event_delay_s=-0.001)


@pytest.mark.asyncio
async def test_mock_backend_honors_stream_inter_event_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_delays: list[float] = []

    async def capture_delay(delay: float) -> None:
        observed_delays.append(delay)

    monkeypatch.setattr("vllm_nnrp_adapter.conformance.asyncio.sleep", capture_delay)
    stream = MockChatCompletionBackend(stream_inter_event_delay_s=0.5).create_chat_completion({"stream": True})
    assert hasattr(stream, "__aiter__")
    iterator = stream.__aiter__()

    await anext(iterator)
    await anext(iterator)

    assert observed_delays == [0.5]


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
    plan = json.loads(FIXTURE_PLAN.read_text(encoding="utf-8"))
    evidence_directory = tmp_path / "evidence"
    plan["artifacts"]["evidence_dir"] = str(evidence_directory)
    plan_path = tmp_path / "api-profile-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "api-profile-results.json"

    exit_code = main(
        [
            "run-conformance-plan",
            "--plan",
            str(plan_path),
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
    evidence_files = sorted(evidence_directory.glob("*.json"))
    assert len(evidence_files) == 8
    evidence = json.loads(evidence_files[0].read_text(encoding="utf-8"))
    assert evidence["case"]["id"] == report["results"][0]["id"]


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


@pytest.mark.asyncio
async def test_level1_baseline_reports_but_never_requires_original_openai_chunks() -> None:
    plan = json.loads(FIXTURE_PLAN.read_text(encoding="utf-8"))
    plan["cases"] = [case for case in plan["cases"] if case["id"] == "openai-compatible.chat.streaming-text"]

    report = await run_conformance_plan(plan, backend=MockChatCompletionBackend())

    result = report["results"][0]
    text_event = next(event for event in result["events"] if event["type"] == "response.output_text.delta")
    assert text_event["openai_chunk"]["object"] == "chat.completion.chunk"
    assert result["outcome"] == "passed"

    plan["cases"][0]["expect"]["events"][0]["fields"] = {"openai_chunk": {"object": "chat.completion.chunk"}}
    rejected = await run_conformance_plan(plan, backend=MockChatCompletionBackend())

    assert rejected["results"][0]["outcome"] == "failed"
    assert "response.output_text.delta count mismatch" in rejected["results"][0]["message"]
