from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from vllm_nnrp_adapter.cli import main
from vllm_nnrp_adapter.priority_burst_workload import (
    PRIORITY_BURST_BASELINES,
    PriorityBurstCase,
    PriorityBurstObservationResult,
    PriorityBurstResult,
    aggregate_priority_burst_evidence,
    build_priority_burst_schedule,
    load_priority_burst_driver_async,
    load_priority_burst_observation_probe_async,
    run_priority_burst_workload,
    run_priority_burst_workload_file_sync,
)


def _manifest() -> dict[str, object]:
    return {
        "scenario": "priority_burst",
        "adapter_version": "0.1.0",
        "adapter_revision": "abcdef0",
        "model": "public-test-model",
        "engine": "vllm-0.26",
        "gpu": "test-gpu-class",
        "arrival_schedule": "fixed_interval_contiguous_burst",
        "priority_application": "pre_backend_dispatch",
        "arrival_interval_seconds": 0.0,
        "prompt_tokens": 4096,
        "max_completion_tokens": 128,
        "warmup": 1,
        "random_seed": 17,
        "sample_count": 8,
        "max_in_flight": 4,
        "burst_start_ordinal": 3,
        "burst_size": 2,
        "normal_priority": 0,
        "urgent_priority": -5,
        "minimum_queue_depth": 2,
        "scheduler_accounting": {
            "method": "vllm_scheduler_trace",
            "scope": "dedicated_engine",
            "source": "test-scheduler-events",
        },
    }


def test_schedule_is_deterministic_and_preserves_contiguous_burst() -> None:
    first = build_priority_burst_schedule(_manifest())
    second = build_priority_burst_schedule(_manifest())

    assert first == second
    assert [case.traffic_class for case in first] == [
        "normal",
        "normal",
        "normal",
        "urgent",
        "urgent",
        "normal",
        "normal",
        "normal",
    ]
    assert [case.backend_priority for case in first] == [0, 0, 0, -5, -5, 0, 0, 0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("urgent_priority", 0, "lower than normal_priority"),
        ("burst_start_ordinal", 0, "must be positive"),
        ("burst_size", 5, "leave normal traffic"),
        ("minimum_queue_depth", 4, "lower than max_in_flight"),
    ],
)
def test_manifest_rejects_invalid_priority_or_saturation_shape(
    field: str,
    value: object,
    message: str,
) -> None:
    manifest = _manifest()
    manifest[field] = value

    with pytest.raises(ValueError, match=message):
        build_priority_burst_schedule(manifest)


class _FakeDriver:
    def __init__(self, baseline: str) -> None:
        self.baseline = baseline
        self.active = 0
        self.max_active = 0
        self.cases: list[PriorityBurstCase] = []
        self.ended = False

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[PriorityBurstCase],
    ) -> None:
        assert workload["scenario"] == "priority_burst"
        assert len(schedule) == workload["sample_count"]

    async def warmup(self, case: PriorityBurstCase) -> None:
        assert not case.is_urgent

    async def start(self, case: PriorityBurstCase) -> _FakeOperation:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.cases.append(case)
        return _FakeOperation(self)

    async def end_run(self) -> None:
        self.ended = True


class _FakeOperation:
    def __init__(self, driver: _FakeDriver) -> None:
        self.driver = driver
        self.closed = False

    async def wait(self) -> PriorityBurstResult:
        await asyncio.sleep(0.005)
        return PriorityBurstResult("completed")

    async def close(self) -> None:
        if not self.closed:
            self.driver.active -= 1
            self.closed = True


class _FakeProbe:
    method = "vllm_scheduler_trace"
    scope = "dedicated_engine"
    source = "test-scheduler-events"

    def __init__(self) -> None:
        self.started = False
        self.ended = False

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[PriorityBurstCase],
    ) -> None:
        assert workload["scenario"] == "priority_burst"
        assert len(schedule) == workload["sample_count"]
        self.started = True

    async def start_sample(self, baseline: str, case: PriorityBurstCase) -> _FakeProbeSession:
        assert self.started
        return _FakeProbeSession(baseline, case)

    async def end_run(self) -> None:
        self.ended = True


class _FakeProbeSession:
    def __init__(self, baseline: str, case: PriorityBurstCase) -> None:
        self.baseline = baseline
        self.case = case
        self.submitted: float | None = None

    async def operation_submitted(self, monotonic_seconds: float) -> None:
        self.submitted = monotonic_seconds

    async def finish(self, result: PriorityBurstResult) -> PriorityBurstObservationResult:
        assert self.submitted is not None
        assert result.terminal_outcome == "completed"
        base = 0.003
        if self.case.is_urgent:
            base = 0.001 if self.baseline == "direct_nnrp" else 0.003
        return PriorityBurstObservationResult(
            queued_after_seconds=0.0,
            backend_started_after_seconds=base,
            backend_completed_after_seconds=base + 0.001,
            observed_backend_priority=(
                None if self.baseline == "raw_openai_http_sse" else self.case.backend_priority
            ),
            queue_depth_at_submit=2 if self.case.is_urgent else 0,
            continuously_runnable=True,
        )

    async def close(self) -> None:
        return None


def _drivers() -> list[_FakeDriver]:
    return [_FakeDriver(baseline) for baseline in PRIORITY_BURST_BASELINES]


@pytest.mark.asyncio
async def test_executor_runs_same_three_baseline_schedule_and_uses_independent_probe() -> None:
    drivers = _drivers()
    probe = _FakeProbe()

    evidence, report = await run_priority_burst_workload(
        _manifest(),
        drivers=drivers,
        observation_probe=probe,
    )

    assert set(run["baseline"] for run in evidence["runs"]) == set(PRIORITY_BURST_BASELINES)
    signatures = [[(case.sample_id, case.traffic_class) for case in driver.cases] for driver in drivers]
    assert signatures[1:] == signatures[:1] * 2
    assert all(driver.max_active == 4 for driver in drivers)
    assert all(driver.ended for driver in drivers)
    assert probe.ended is True
    assert report["acceptance"]["evaluated"] is True
    assert report["acceptance"]["hypotheses"]["no_normal_starvation"] == "pass"


@pytest.mark.asyncio
async def test_executor_rejects_missing_baseline_and_probe_identity_drift() -> None:
    with pytest.raises(ValueError, match="exactly the three priority-burst baselines"):
        await run_priority_burst_workload(
            _manifest(),
            drivers=_drivers()[:2],
            observation_probe=_FakeProbe(),
        )

    probe = _FakeProbe()
    probe.source = "wrong-source"
    with pytest.raises(ValueError, match="observation_probe.source must match"):
        await run_priority_burst_workload(
            _manifest(),
            drivers=_drivers(),
            observation_probe=probe,
        )


def _evidence() -> dict[str, Any]:
    workload = _manifest()
    samples = []
    for case in build_priority_burst_schedule(workload):
        samples.append(
            {
                "sample_id": case.sample_id,
                "ordinal": case.ordinal,
                "scheduled_offset_seconds": 0.0,
                "submitted_at_seconds": 0.0,
                "backend_queued_at_seconds": 0.0,
                "backend_started_at_seconds": 0.001,
                "backend_completed_at_seconds": 0.002,
                "terminal_at_seconds": 0.004 if case.is_urgent else 0.005,
                "traffic_class": case.traffic_class,
                "requested_backend_priority": case.backend_priority,
                "observed_backend_priority": case.backend_priority,
                "queue_depth_at_submit": 2 if case.is_urgent else 0,
                "continuously_runnable": True,
                "terminal_outcome": "completed",
            }
        )
    runs = []
    for baseline in PRIORITY_BURST_BASELINES:
        baseline_samples = deepcopy(samples)
        if baseline == "raw_openai_http_sse":
            for sample in baseline_samples:
                sample["observed_backend_priority"] = None
        runs.append(
            {
                "baseline": baseline,
                "wall_clock_seconds": 0.05 if baseline != "direct_nnrp" else 0.04,
                "samples": baseline_samples,
            }
        )
    return {
        "schema_version": 1,
        "workload": workload,
        "provenance": {
            "adapter_distribution": "vllm-nnrp-adapter",
            "adapter_version": "0.1.0",
            "adapter_revision": "abcdef0",
        },
        "runs": runs,
    }


def test_aggregator_calculates_priority_acceptance_and_starvation() -> None:
    evidence = _evidence()
    raw = evidence["runs"][0]
    assert isinstance(raw, dict)
    for sample in raw["samples"]:
        if sample["traffic_class"] == "urgent":
            sample["terminal_at_seconds"] = 0.010
            sample["observed_backend_priority"] = None

    report = aggregate_priority_burst_evidence(evidence)

    assert report["acceptance"]["evaluated"] is True
    assert report["acceptance"]["urgent_p95_reduction_vs_raw"] == pytest.approx(0.6)
    assert report["acceptance"]["hypotheses"] == {
        "urgent_p95_reduction_at_least_30_percent": "pass",
        "no_normal_starvation": "pass",
        "throughput_regression_at_most_5_percent": "pass",
    }


def test_aggregator_refuses_proxy_accounting_and_incomplete_priority_evidence() -> None:
    proxy = _evidence()
    proxy["workload"]["scheduler_accounting"] = {
        "method": "request_interval_proxy",
        "scope": "shared_engine",
        "source": "request-events",
    }
    report = aggregate_priority_burst_evidence(proxy)
    assert report["acceptance"]["evaluated"] is False

    incomplete = _evidence()
    direct = incomplete["runs"][2]
    direct["samples"][3]["observed_backend_priority"] = None
    report = aggregate_priority_burst_evidence(incomplete)
    assert report["acceptance"]["evaluated"] is False
    assert set(report["acceptance"]["hypotheses"].values()) == {"not_evaluable"}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["runs"].pop(), "three comparison baselines"),
        (
            lambda value: value["runs"][0]["samples"][0].update({"traffic_class": "urgent"}),
            "differs from workload burst schedule",
        ),
        (
            lambda value: value["runs"][0]["samples"][0].update({"backend_started_at_seconds": -1}),
            "must be non-negative",
        ),
        (
            lambda value: value["runs"][0]["samples"][1].update(
                {"ordinal": 0, "sample_id": "sample-000000"}
            ),
            "ordinal must be 1",
        ),
        (
            lambda value: value["runs"][0]["samples"][0].update({"terminal_at_seconds": 0.1}),
            "exceeds wall_clock_seconds",
        ),
        (
            lambda value: value["provenance"].update({"adapter_revision": "1234567"}),
            "must match",
        ),
    ],
)
def test_aggregator_rejects_incomplete_or_inconsistent_evidence(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    evidence = _evidence()
    mutate(evidence)

    with pytest.raises(ValueError, match=message):
        aggregate_priority_burst_evidence(evidence)


def test_aggregator_rejects_sensitive_public_metadata() -> None:
    evidence = _evidence()
    evidence["workload"]["gpu"] = r"C:\Users\operator\gpu-node"

    with pytest.raises(ValueError, match="prohibited machine"):
        aggregate_priority_burst_evidence(evidence)


def test_aggregator_requires_raw_baseline_to_remain_unprioritized() -> None:
    evidence = _evidence()
    raw = evidence["runs"][0]
    for sample in raw["samples"]:
        if sample["traffic_class"] == "urgent":
            sample["observed_backend_priority"] = sample["requested_backend_priority"]

    report = aggregate_priority_burst_evidence(evidence)

    assert report["runs"][0]["urgent_priority_observed"] is False
    assert report["acceptance"]["evaluated"] is False


def test_aggregate_cli_writes_priority_report(tmp_path: Path) -> None:
    input_path = tmp_path / "priority-raw.json"
    output_path = tmp_path / "priority-report.json"
    input_path.write_text(json.dumps(_evidence()), encoding="utf-8")

    assert main(
        [
            "aggregate-priority-burst-evidence",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["scenario"] == "priority_burst"


def _install_factory_module(name: str, *, failing_baseline: str | None = None) -> None:
    module = ModuleType(name)

    def driver_factory(baseline: str) -> Callable[[], _FakeDriver]:
        def factory() -> _FakeDriver:
            if baseline == failing_baseline:
                return _FailingDriver(baseline)
            return _FakeDriver(baseline)

        return factory

    module.make_raw = driver_factory("raw_openai_http_sse")
    module.make_orchestrated = driver_factory("orchestrated_http_sse")
    module.make_direct = driver_factory("direct_nnrp")
    module.make_probe = _FakeProbe
    module.not_callable = "invalid"
    sys.modules[name] = module


class _FailingDriver(_FakeDriver):
    async def start(self, case: PriorityBurstCase) -> _FakeOperation:
        raise RuntimeError(f"cannot start {case.sample_id}")


def test_run_cli_loads_factories_and_writes_priority_artifacts(tmp_path: Path) -> None:
    module_name = "priority_burst_test_factories"
    _install_factory_module(module_name)
    manifest_path = tmp_path / "manifest.json"
    raw_path = tmp_path / "raw.json"
    report_path = tmp_path / "report.json"
    outcome_path = tmp_path / "outcome.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    try:
        exit_code = main(
            [
                "run-priority-burst-workload",
                "--manifest",
                str(manifest_path),
                "--raw-output",
                str(raw_path),
                "--report-output",
                str(report_path),
                "--outcome-output",
                str(outcome_path),
                "--driver",
                f"raw_openai_http_sse={module_name}:make_raw",
                "--driver",
                f"orchestrated_http_sse={module_name}:make_orchestrated",
                "--driver",
                f"direct_nnrp={module_name}:make_direct",
                "--observation-probe",
                f"{module_name}:make_probe",
            ]
        )
    finally:
        sys.modules.pop(module_name, None)

    assert exit_code == 0
    assert json.loads(raw_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["scenario"] == "priority_burst"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["status"] == "completed"
    assert outcome["sample_count"] == 8
    assert set(outcome["baseline_execution_order"]) == set(PRIORITY_BURST_BASELINES)


def test_file_runner_writes_safe_failure_outcome(tmp_path: Path) -> None:
    module_name = "priority_burst_failing_factories"
    _install_factory_module(module_name, failing_baseline="direct_nnrp")
    manifest_path = tmp_path / "manifest.json"
    outcome_path = tmp_path / "outcome.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    try:
        with pytest.raises(Exception, match="priority-burst execution failed"):
            run_priority_burst_workload_file_sync(
                manifest_path,
                tmp_path / "raw.json",
                tmp_path / "report.json",
                outcome_path,
                driver_specs={
                    "raw_openai_http_sse": f"{module_name}:make_raw",
                    "orchestrated_http_sse": f"{module_name}:make_orchestrated",
                    "direct_nnrp": f"{module_name}:make_direct",
                },
                observation_probe_spec=f"{module_name}:make_probe",
            )
    finally:
        sys.modules.pop(module_name, None)

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome == {
        "baseline": "direct_nnrp",
        "error_type": "PriorityBurstExecutionError",
        "phase": "start",
        "sample_id": "sample-000000",
        "scenario": "priority_burst",
        "schema_version": 1,
        "status": "failed",
    }


@pytest.mark.asyncio
async def test_factory_loader_rejects_invalid_specs_and_non_callable_factory() -> None:
    with pytest.raises(ValueError, match="module.path:factory_name"):
        await load_priority_burst_driver_async("missing-separator", expected_baseline="direct_nnrp")

    module_name = "priority_burst_invalid_factory"
    _install_factory_module(module_name)
    try:
        with pytest.raises(TypeError, match="factory is not callable"):
            await load_priority_burst_observation_probe_async(f"{module_name}:not_callable")
    finally:
        sys.modules.pop(module_name, None)
