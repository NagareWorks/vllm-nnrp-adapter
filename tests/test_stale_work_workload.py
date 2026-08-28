from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_nnrp_adapter.cli import main
from vllm_nnrp_adapter.stale_work_workload import (
    STALE_WORK_BASELINES,
    StaleWorkCase,
    StaleWorkResult,
    build_stale_work_schedule,
    run_stale_workload,
)

_OUTCOME_BY_CONTROL = {
    "cancel": "cancelled",
    "abort": "aborted",
    "deadline": "expired",
    "supersede": "superseded",
}


def _manifest(*, ratio: float, sample_count: int = 100, random_seed: int = 17) -> dict[str, object]:
    return {
        "scenario": "stale_work",
        "stale_work_ratio": ratio,
        "adapter_version": "0.1.0",
        "adapter_revision": "abcdef0",
        "model": "public-test-model",
        "engine": "vllm-0.26",
        "gpu": "test-gpu-class",
        "arrival_schedule": "seeded-fixed-interval",
        "arrival_interval_seconds": 0.025,
        "cancellation_schedule": "seeded-stale-selection",
        "control_delay_seconds": 0.001,
        "prompt_tokens": 4096,
        "max_completion_tokens": 128,
        "warmup": 2,
        "random_seed": random_seed,
        "sample_count": sample_count,
        "max_in_flight": 4,
        "gpu_accounting": {
            "method": "device_active_time",
            "scope": "dedicated_device",
            "source": "test-device-counter",
        },
    }


@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5])
def test_schedule_is_deterministic_balanced_and_matches_ratio(ratio: float) -> None:
    manifest = _manifest(ratio=ratio)

    first = build_stale_work_schedule(manifest)
    second = build_stale_work_schedule(manifest)
    stale = [case for case in first if case.is_stale]

    assert first == second
    assert len(first) == 100
    assert len(stale) == int(100 * ratio)
    assert {case.control_kind for case in stale} == {"cancel", "abort", "deadline", "supersede"}
    assert first[3].scheduled_offset_seconds == pytest.approx(0.075)
    assert [case.sample_id for case in first] == [f"sample-{index:06d}" for index in range(100)]


def test_schedule_changes_stale_selection_with_seed_but_not_arrival_order() -> None:
    first = build_stale_work_schedule(_manifest(ratio=0.3, random_seed=1))
    second = build_stale_work_schedule(_manifest(ratio=0.3, random_seed=2))

    assert [case.ordinal for case in first] == [case.ordinal for case in second]
    assert [case.is_stale for case in first] != [case.is_stale for case in second]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (_manifest(ratio=0.1, sample_count=11), "represent stale_work_ratio exactly"),
        (_manifest(ratio=0.1, sample_count=10), "at least one cancel, abort, deadline, and supersede"),
    ],
)
def test_schedule_rejects_unrepresentable_or_incomplete_control_mix(
    manifest: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_stale_work_schedule(manifest)


class _FakeDriver:
    def __init__(self, baseline: str, *, stale_gpu_seconds: float) -> None:
        self.baseline = baseline
        self.stale_gpu_seconds = stale_gpu_seconds
        self.warmup_cases: list[StaleWorkCase] = []
        self.executed_cases: list[StaleWorkCase] = []
        self.max_active = 0
        self.active = 0
        self.ended = False

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[StaleWorkCase],
    ) -> None:
        assert workload["scenario"] == "stale_work"
        assert len(schedule) == workload["sample_count"]

    async def warmup(self, case: StaleWorkCase) -> None:
        self.warmup_cases.append(case)

    async def start(self, case: StaleWorkCase) -> _FakeOperation:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.executed_cases.append(case)
        return _FakeOperation(self, case)

    async def end_run(self) -> None:
        self.ended = True


class _FakeOperation:
    def __init__(self, driver: _FakeDriver, case: StaleWorkCase) -> None:
        self.driver = driver
        self.case = case
        self.accepted_control: str | None = None
        self.closed = False

    async def apply_control(self, control_kind: str) -> bool:
        accepted = self.driver.baseline != "raw_openai_http_sse"
        if accepted:
            self.accepted_control = control_kind
        return accepted

    async def wait(self) -> StaleWorkResult:
        await asyncio.sleep(0.05)
        terminal_outcome = (
            _OUTCOME_BY_CONTROL[self.accepted_control]
            if self.accepted_control is not None
            else "completed"
        )
        return StaleWorkResult(
            terminal_outcome=terminal_outcome,
            backend_stopped_after_seconds=0.05,
            gpu_seconds=self.driver.stale_gpu_seconds if self.case.is_stale else 0.5,
            useful_result_weight=0.0 if self.case.is_stale else 1.0,
            late_result_count=0,
        )

    async def close(self) -> None:
        if not self.closed:
            self.driver.active -= 1
            self.closed = True


def _executor_manifest() -> dict[str, object]:
    manifest = _manifest(ratio=0.5, sample_count=10)
    manifest["arrival_interval_seconds"] = 0.0
    manifest["max_in_flight"] = 2
    manifest["gpu_accounting"] = {
        "method": "cuda_event_attribution",
        "scope": "scheduled_batch",
        "source": "test-cuda-events",
    }
    return manifest


def _drivers() -> list[_FakeDriver]:
    return [
        _FakeDriver("raw_openai_http_sse", stale_gpu_seconds=0.4),
        _FakeDriver("orchestrated_http_sse", stale_gpu_seconds=0.2),
        _FakeDriver("direct_nnrp", stale_gpu_seconds=0.1),
    ]


@pytest.mark.asyncio
async def test_executor_runs_identical_randomized_baselines_with_bounded_concurrency() -> None:
    drivers = _drivers()

    evidence, report = await run_stale_workload(_executor_manifest(), drivers=drivers)

    execution_order = [run["baseline"] for run in evidence["runs"]]
    assert execution_order == report["baseline_execution_order"]
    assert set(execution_order) == set(STALE_WORK_BASELINES)
    signatures = [
        [(case.sample_id, case.control_kind) for case in driver.executed_cases]
        for driver in drivers
    ]
    assert signatures[1:] == signatures[:1] * 2
    assert all(len(driver.warmup_cases) == 2 for driver in drivers)
    assert all(driver.max_active == 2 for driver in drivers)
    assert all(driver.ended for driver in drivers)
    raw = next(run for run in report["runs"] if run["baseline"] == "raw_openai_http_sse")
    assert raw["stale_sample_count"] == 5
    assert raw["wasted_gpu_seconds"] == pytest.approx(2.0)
    assert raw["semantic_defect_count"] == 0
    assert all(sample["terminal_outcome"] == "completed" for sample in raw["samples"])
    for run in report["runs"]:
        for sample in run["samples"]:
            assert sample["operation_started_at_seconds"] >= sample["scheduled_offset_seconds"]
            if sample["control_kind"] is None:
                assert sample["control_scheduled_at_seconds"] is None
                assert sample["control_issued_at_seconds"] is None
            else:
                assert sample["control_scheduled_at_seconds"] == pytest.approx(
                    sample["operation_started_at_seconds"] + 0.001
                )
                assert sample["control_issued_at_seconds"] >= sample["control_scheduled_at_seconds"]
                if sample["control_accepted"]:
                    assert sample["control_accepted_at_seconds"] >= sample["control_issued_at_seconds"]


@pytest.mark.asyncio
async def test_executor_requires_exact_driver_set() -> None:
    with pytest.raises(ValueError, match="exactly the three stale-work baselines"):
        await run_stale_workload(_executor_manifest(), drivers=_drivers()[:2])


@pytest.mark.asyncio
async def test_executor_requires_manifest_to_name_installed_adapter_version() -> None:
    manifest = _executor_manifest()
    manifest["adapter_version"] = "9.9.9"

    with pytest.raises(ValueError, match="must match the installed"):
        await run_stale_workload(manifest, drivers=_drivers())


class _BadSampleDriver(_FakeDriver):
    async def start(self, case: StaleWorkCase) -> _BadResultOperation:
        operation = await super().start(case)
        return _BadResultOperation(operation)


class _SensitiveSampleDriver(_FakeDriver):
    async def start(self, case: StaleWorkCase) -> _SensitiveResultOperation:
        operation = await super().start(case)
        return _SensitiveResultOperation(operation)


class _BadResultOperation:
    def __init__(self, operation: _FakeOperation) -> None:
        self.operation = operation

    async def apply_control(self, control_kind: str) -> bool:
        return await self.operation.apply_control(control_kind)

    async def wait(self) -> Mapping[str, object]:
        return {"sample_id": "wrong-sample"}

    async def close(self) -> None:
        await self.operation.close()


class _SensitiveResultOperation(_BadResultOperation):
    async def wait(self) -> StaleWorkResult:
        result = await self.operation.wait()
        return replace(result, terminal_outcome=r"C:\Users\operator\private-run")


@pytest.mark.asyncio
async def test_executor_rejects_driver_schedule_drift_and_still_ends_run() -> None:
    drivers = _drivers()
    bad_driver = _BadSampleDriver("direct_nnrp", stale_gpu_seconds=0.1)
    drivers[-1] = bad_driver

    with pytest.raises(RuntimeError, match="sample_validation"):
        await run_stale_workload(_executor_manifest(), drivers=drivers)

    assert bad_driver.ended is True


@pytest.mark.asyncio
async def test_executor_rejects_sensitive_driver_evidence() -> None:
    drivers = _drivers()
    drivers[-1] = _SensitiveSampleDriver("direct_nnrp", stale_gpu_seconds=0.1)

    with pytest.raises(ValueError, match="Windows user path"):
        await run_stale_workload(_executor_manifest(), drivers=drivers)


_CLI_DRIVERS: list[_FakeDriver] = []


def _make_cli_driver(baseline: str, stale_gpu_seconds: float) -> _FakeDriver:
    driver = _FakeDriver(baseline, stale_gpu_seconds=stale_gpu_seconds)
    _CLI_DRIVERS.append(driver)
    return driver


def make_raw_driver() -> _FakeDriver:
    return _make_cli_driver("raw_openai_http_sse", 0.4)


async def make_orchestrated_driver() -> _FakeDriver:
    return _make_cli_driver("orchestrated_http_sse", 0.2)


def make_direct_driver() -> _FakeDriver:
    return _make_cli_driver("direct_nnrp", 0.1)


def make_bad_direct_driver() -> _FakeDriver:
    driver = _BadSampleDriver("direct_nnrp", stale_gpu_seconds=0.1)
    _CLI_DRIVERS.append(driver)
    return driver


def test_cli_loads_three_driver_factories_and_writes_raw_and_aggregate_evidence(tmp_path: Path) -> None:
    _CLI_DRIVERS.clear()
    manifest_path = tmp_path / "manifest.json"
    raw_output = tmp_path / "raw.json"
    report_output = tmp_path / "report.json"
    outcome_output = tmp_path / "outcome.json"
    manifest_path.write_text(json.dumps(_executor_manifest()), encoding="utf-8")

    exit_code = main(
        [
            "run-stale-workload",
            "--manifest",
            str(manifest_path),
            "--raw-output",
            str(raw_output),
            "--report-output",
            str(report_output),
            "--outcome-output",
            str(outcome_output),
            "--driver",
            f"raw_openai_http_sse={__name__}:make_raw_driver",
            "--driver",
            f"orchestrated_http_sse={__name__}:make_orchestrated_driver",
            "--driver",
            f"direct_nnrp={__name__}:make_direct_driver",
        ]
    )

    assert exit_code == 0
    raw = json.loads(raw_output.read_text(encoding="utf-8"))
    report = json.loads(report_output.read_text(encoding="utf-8"))
    outcome = json.loads(outcome_output.read_text(encoding="utf-8"))
    assert report["baseline_execution_order"] == [run["baseline"] for run in raw["runs"]]
    assert outcome == {
        "schema_version": "nnrp-adoption-run-outcome/v1",
        "status": "passed",
        "baseline_execution_order": report["baseline_execution_order"],
        "sample_count": 10,
    }
    assert {driver.baseline for driver in _CLI_DRIVERS} == set(STALE_WORK_BASELINES)
    assert all(driver.ended for driver in _CLI_DRIVERS)


def test_cli_rejects_same_raw_and_report_output(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "evidence.json"
    manifest_path.write_text(json.dumps(_executor_manifest()), encoding="utf-8")

    with pytest.raises(ValueError, match="must be different paths"):
        main(
            [
                "run-stale-workload",
                "--manifest",
                str(manifest_path),
                "--raw-output",
                str(output_path),
                "--report-output",
                str(output_path),
                "--outcome-output",
                str(tmp_path / "outcome.json"),
                "--driver",
                f"raw_openai_http_sse={__name__}:make_raw_driver",
            ]
        )


@pytest.mark.parametrize(
    "driver_arg",
    [
        "unknown=tests.test_stale_work_workload:make_raw_driver",
        "raw_openai_http_sse",
    ],
)
def test_cli_rejects_invalid_driver_assignment(driver_arg: str, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_executor_manifest()), encoding="utf-8")

    with pytest.raises(ValueError, match="--driver must use"):
        main(
            [
                "run-stale-workload",
                "--manifest",
                str(manifest_path),
                "--raw-output",
                str(tmp_path / "raw.json"),
                "--report-output",
                str(tmp_path / "report.json"),
                "--outcome-output",
                str(tmp_path / "outcome.json"),
                "--driver",
                driver_arg,
            ]
        )


def test_cli_persists_safe_failure_outcome_without_partial_evidence(tmp_path: Path) -> None:
    _CLI_DRIVERS.clear()
    manifest_path = tmp_path / "manifest.json"
    raw_output = tmp_path / "raw.json"
    report_output = tmp_path / "report.json"
    outcome_output = tmp_path / "outcome.json"
    manifest_path.write_text(json.dumps(_executor_manifest()), encoding="utf-8")

    with pytest.raises(RuntimeError, match="sample_validation"):
        main(
            [
                "run-stale-workload",
                "--manifest",
                str(manifest_path),
                "--raw-output",
                str(raw_output),
                "--report-output",
                str(report_output),
                "--outcome-output",
                str(outcome_output),
                "--driver",
                f"raw_openai_http_sse={__name__}:make_raw_driver",
                "--driver",
                f"orchestrated_http_sse={__name__}:make_orchestrated_driver",
                "--driver",
                f"direct_nnrp={__name__}:make_bad_direct_driver",
            ]
        )

    outcome = json.loads(outcome_output.read_text(encoding="utf-8"))
    assert outcome["status"] == "failed"
    assert outcome["phase"] == "sample_validation"
    assert outcome["baseline"] == "direct_nnrp"
    assert outcome["sample_id"].startswith("sample-")
    assert outcome["error_type"] == "StaleWorkExecutionError"
    assert not raw_output.exists()
    assert not report_output.exists()
