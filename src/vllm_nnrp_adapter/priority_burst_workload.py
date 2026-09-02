from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import math
import random
import statistics
import tempfile
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, cast

PRIORITY_BURST_BASELINES = (
    "raw_openai_http_sse",
    "orchestrated_http_sse",
    "direct_nnrp",
)
_TRAFFIC_CLASSES = ("normal", "urgent")
_TERMINAL_OUTCOMES = ("completed", "failed", "timed_out")
_SCHEDULER_METHODS = ("vllm_scheduler_trace", "engine_request_events", "request_interval_proxy")
_SCHEDULER_SCOPES = ("dedicated_engine", "shared_engine")
_ACCEPTANCE_ELIGIBLE_ACCOUNTING = {
    ("vllm_scheduler_trace", "dedicated_engine"),
    ("engine_request_events", "dedicated_engine"),
}


@dataclass(frozen=True)
class PriorityBurstCase:
    sample_id: str
    ordinal: int
    scheduled_offset_seconds: float
    model: str
    prompt_tokens: int
    max_completion_tokens: int
    traffic_class: str
    backend_priority: int

    @property
    def is_urgent(self) -> bool:
        return self.traffic_class == "urgent"


@dataclass(frozen=True)
class PriorityBurstResult:
    terminal_outcome: str


@dataclass(frozen=True)
class PriorityBurstObservationResult:
    queued_after_seconds: float
    backend_started_after_seconds: float
    backend_completed_after_seconds: float
    observed_backend_priority: int | None
    queue_depth_at_submit: int
    continuously_runnable: bool


class PriorityBurstOperation(Protocol):
    async def wait(self) -> PriorityBurstResult: ...

    async def close(self) -> None: ...


class PriorityBurstObservationSession(Protocol):
    async def operation_submitted(self, monotonic_seconds: float) -> None: ...

    async def finish(self, result: PriorityBurstResult) -> PriorityBurstObservationResult: ...

    async def close(self) -> None: ...


class PriorityBurstObservationProbe(Protocol):
    method: str
    scope: str
    source: str

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[PriorityBurstCase],
    ) -> None: ...

    async def start_sample(
        self,
        baseline: str,
        case: PriorityBurstCase,
    ) -> PriorityBurstObservationSession: ...

    async def end_run(self) -> None: ...


class PriorityBurstDriver(Protocol):
    baseline: str

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[PriorityBurstCase],
    ) -> None: ...

    async def warmup(self, case: PriorityBurstCase) -> None: ...

    async def start(self, case: PriorityBurstCase) -> PriorityBurstOperation: ...

    async def end_run(self) -> None: ...


class PriorityBurstExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        phase: str,
        baseline: str | None = None,
        sample_id: str | None = None,
    ) -> None:
        super().__init__(f"priority-burst execution failed during {phase}")
        self.phase = phase
        self.baseline = baseline
        self.sample_id = sample_id


def normalize_priority_burst_manifest(value: object) -> dict[str, Any]:
    manifest = _mapping(value, "workload")
    scenario = _required_string(manifest, "scenario", "workload")
    if scenario != "priority_burst":
        raise ValueError("workload.scenario must be 'priority_burst'")
    normal_priority = _required_int(manifest, "normal_priority", "workload")
    urgent_priority = _required_int(manifest, "urgent_priority", "workload")
    if urgent_priority >= normal_priority:
        raise ValueError("workload.urgent_priority must be lower than normal_priority for vLLM")
    sample_count = _required_positive_int(manifest, "sample_count", "workload")
    burst_start = _required_positive_int(manifest, "burst_start_ordinal", "workload")
    burst_size = _required_positive_int(manifest, "burst_size", "workload")
    if burst_start + burst_size >= sample_count:
        raise ValueError("workload priority burst must leave normal traffic before and after the burst")
    max_in_flight = _required_positive_int(manifest, "max_in_flight", "workload")
    saturation_depth = _required_positive_int(manifest, "minimum_queue_depth", "workload")
    if saturation_depth >= max_in_flight:
        raise ValueError("workload.minimum_queue_depth must be lower than max_in_flight")
    accounting = _scheduler_accounting(manifest.get("scheduler_accounting"))
    normalized = {
        "scenario": scenario,
        "adapter_version": _required_string(manifest, "adapter_version", "workload"),
        "adapter_revision": _required_git_revision(manifest, "adapter_revision", "workload"),
        "model": _public_label(manifest, "model", "workload"),
        "engine": _public_label(manifest, "engine", "workload"),
        "gpu": _public_label(manifest, "gpu", "workload"),
        "arrival_schedule": _required_choice(
            manifest,
            "arrival_schedule",
            ("fixed_interval_contiguous_burst",),
            "workload",
        ),
        "priority_application": _required_choice(
            manifest,
            "priority_application",
            ("pre_backend_dispatch",),
            "workload",
        ),
        "arrival_interval_seconds": _required_non_negative_number(
            manifest,
            "arrival_interval_seconds",
            "workload",
        ),
        "prompt_tokens": _required_positive_int(manifest, "prompt_tokens", "workload"),
        "max_completion_tokens": _required_positive_int(
            manifest,
            "max_completion_tokens",
            "workload",
        ),
        "warmup": _required_non_negative_int(manifest, "warmup", "workload"),
        "random_seed": _required_non_negative_int(manifest, "random_seed", "workload"),
        "sample_count": sample_count,
        "max_in_flight": max_in_flight,
        "burst_start_ordinal": burst_start,
        "burst_size": burst_size,
        "normal_priority": normal_priority,
        "urgent_priority": urgent_priority,
        "minimum_queue_depth": saturation_depth,
        "scheduler_accounting": accounting,
    }
    _reject_sensitive_strings(normalized, "workload")
    return normalized


def build_priority_burst_schedule(workload: object) -> tuple[PriorityBurstCase, ...]:
    manifest = normalize_priority_burst_manifest(workload)
    burst_start = cast(int, manifest["burst_start_ordinal"])
    burst_end = burst_start + cast(int, manifest["burst_size"])
    interval = cast(float, manifest["arrival_interval_seconds"])
    return tuple(
        PriorityBurstCase(
            sample_id=f"sample-{ordinal:06d}",
            ordinal=ordinal,
            scheduled_offset_seconds=ordinal * interval,
            model=cast(str, manifest["model"]),
            prompt_tokens=cast(int, manifest["prompt_tokens"]),
            max_completion_tokens=cast(int, manifest["max_completion_tokens"]),
            traffic_class="urgent" if burst_start <= ordinal < burst_end else "normal",
            backend_priority=(
                cast(int, manifest["urgent_priority"])
                if burst_start <= ordinal < burst_end
                else cast(int, manifest["normal_priority"])
            ),
        )
        for ordinal in range(cast(int, manifest["sample_count"]))
    )


async def load_priority_burst_driver_async(
    spec: str,
    *,
    expected_baseline: str,
) -> PriorityBurstDriver:
    driver = await _load_factory(spec, kind="priority-burst driver")
    _validate_driver(driver, expected_baseline=expected_baseline, location=spec)
    return cast(PriorityBurstDriver, driver)


async def load_priority_burst_observation_probe_async(spec: str) -> PriorityBurstObservationProbe:
    probe = await _load_factory(spec, kind="priority-burst observation probe")
    _validate_probe(probe, location=spec)
    return cast(PriorityBurstObservationProbe, probe)


async def run_priority_burst_workload(
    workload: object,
    *,
    drivers: Sequence[PriorityBurstDriver],
    observation_probe: PriorityBurstObservationProbe,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = normalize_priority_burst_manifest(workload)
    installed_version = _distribution_version("vllm-nnrp-adapter")
    if manifest["adapter_version"] != installed_version:
        raise ValueError(
            "workload.adapter_version must match the installed vllm-nnrp-adapter distribution "
            f"({installed_version})"
        )
    schedule = build_priority_burst_schedule(manifest)
    drivers_by_baseline: dict[str, PriorityBurstDriver] = {}
    for index, driver in enumerate(drivers):
        baseline = getattr(driver, "baseline", None)
        _validate_driver(driver, expected_baseline=baseline, location=f"drivers[{index}]")
        if baseline in drivers_by_baseline:
            raise ValueError(f"drivers contains duplicate baseline {baseline!r}")
        drivers_by_baseline[cast(str, baseline)] = driver
    missing = sorted(set(PRIORITY_BURST_BASELINES) - drivers_by_baseline.keys())
    extra = sorted(drivers_by_baseline.keys() - set(PRIORITY_BURST_BASELINES))
    if missing or extra:
        raise ValueError(
            "drivers must contain exactly the three priority-burst baselines; "
            f"missing={missing}, extra={extra}"
        )
    _validate_probe(observation_probe, location="observation_probe")
    accounting = cast(Mapping[str, object], manifest["scheduler_accounting"])
    for field in ("method", "scope", "source"):
        if getattr(observation_probe, field) != accounting[field]:
            raise ValueError(
                f"observation_probe.{field} must match workload.scheduler_accounting.{field}"
            )

    execution_order = list(PRIORITY_BURST_BASELINES)
    random.Random(f"priority-baseline-order:{manifest['random_seed']}").shuffle(execution_order)
    try:
        await observation_probe.begin_run(deepcopy(manifest), schedule)
    except Exception as error:
        raise PriorityBurstExecutionError(phase="observation_begin_run") from error
    try:
        runs = [
            await _run_driver(manifest, schedule, drivers_by_baseline[baseline], observation_probe)
            for baseline in execution_order
        ]
    finally:
        try:
            await observation_probe.end_run()
        except Exception as error:
            raise PriorityBurstExecutionError(phase="observation_end_run") from error
    evidence = {
        "schema_version": 1,
        "workload": deepcopy(manifest),
        "provenance": {
            "adapter_distribution": "vllm-nnrp-adapter",
            "adapter_version": manifest["adapter_version"],
            "adapter_revision": manifest["adapter_revision"],
        },
        "runs": runs,
    }
    report = aggregate_priority_burst_evidence(evidence)
    report["baseline_execution_order"] = execution_order
    return evidence, report


def aggregate_priority_burst_evidence(value: object) -> dict[str, Any]:
    evidence = _mapping(value, "evidence")
    if evidence.get("schema_version") != 1:
        raise ValueError("evidence.schema_version must be 1")
    workload = normalize_priority_burst_manifest(evidence.get("workload"))
    provenance = _mapping(evidence.get("provenance"), "evidence.provenance")
    if _required_string(provenance, "adapter_distribution", "evidence.provenance") != "vllm-nnrp-adapter":
        raise ValueError("evidence.provenance.adapter_distribution must be 'vllm-nnrp-adapter'")
    for field in ("adapter_version", "adapter_revision"):
        if _required_string(provenance, field, "evidence.provenance") != workload[field]:
            raise ValueError(f"evidence.provenance.{field} must match evidence.workload.{field}")
    run_values = _sequence(evidence.get("runs"), "evidence.runs")
    if len(run_values) != len(PRIORITY_BURST_BASELINES):
        raise ValueError("evidence.runs must contain the three comparison baselines")
    summaries: dict[str, dict[str, Any]] = {}
    for index, run_value in enumerate(run_values):
        summary = _aggregate_run(run_value, workload, location=f"evidence.runs[{index}]")
        baseline = cast(str, summary["baseline"])
        if baseline in summaries:
            raise ValueError(f"evidence.runs contains duplicate baseline {baseline!r}")
        summaries[baseline] = summary
    if set(summaries) != set(PRIORITY_BURST_BASELINES):
        raise ValueError("evidence.runs must contain each comparison baseline exactly once")
    _validate_schedule_identity(summaries)
    ordered = [summaries[baseline] for baseline in PRIORITY_BURST_BASELINES]
    report = {
        "schema_version": 1,
        "scenario": "priority_burst",
        "workload": workload,
        "provenance": dict(provenance),
        "runs": ordered,
        "acceptance": _acceptance(workload, summaries),
    }
    _reject_sensitive_strings(report, "report")
    return report


def aggregate_priority_burst_evidence_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    report = aggregate_priority_burst_evidence(json.loads(input_path.read_text(encoding="utf-8")))
    _atomic_write_json(output_path, report)
    return report


def run_priority_burst_workload_file_sync(
    manifest_path: Path,
    raw_output_path: Path,
    report_output_path: Path,
    outcome_output_path: Path,
    *,
    driver_specs: Mapping[str, str],
    observation_probe_spec: str,
) -> dict[str, Any]:
    if set(driver_specs) != set(PRIORITY_BURST_BASELINES):
        raise ValueError("driver_specs must contain each priority-burst baseline exactly once")

    async def execute() -> tuple[dict[str, Any], dict[str, Any]]:
        drivers = [
            await load_priority_burst_driver_async(driver_specs[baseline], expected_baseline=baseline)
            for baseline in PRIORITY_BURST_BASELINES
        ]
        probe = await load_priority_burst_observation_probe_async(observation_probe_spec)
        return await run_priority_burst_workload(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            drivers=drivers,
            observation_probe=probe,
        )

    try:
        evidence, report = asyncio.run(execute())
    except BaseException as error:
        outcome = {
            "schema_version": 1,
            "scenario": "priority_burst",
            "status": "failed",
            "error_type": type(error).__name__,
            "phase": getattr(error, "phase", "execution"),
            "baseline": getattr(error, "baseline", None),
            "sample_id": getattr(error, "sample_id", None),
        }
        _atomic_write_json(outcome_output_path, outcome)
        raise
    _atomic_write_json(raw_output_path, evidence)
    _atomic_write_json(report_output_path, report)
    outcome = {
        "schema_version": 1,
        "scenario": "priority_burst",
        "status": "completed",
        "baseline_execution_order": report["baseline_execution_order"],
        "sample_count": report["workload"]["sample_count"],
        "acceptance": report["acceptance"],
    }
    _atomic_write_json(outcome_output_path, outcome)
    return report


async def _run_driver(
    manifest: Mapping[str, Any],
    schedule: Sequence[PriorityBurstCase],
    driver: PriorityBurstDriver,
    probe: PriorityBurstObservationProbe,
) -> dict[str, Any]:
    baseline = driver.baseline
    driver_manifest = deepcopy(dict(manifest))
    try:
        await driver.begin_run(driver_manifest, schedule)
    except Exception as error:
        raise PriorityBurstExecutionError(phase="driver_begin_run", baseline=baseline) from error
    try:
        normal_case = next(case for case in schedule if not case.is_urgent)
        for index in range(cast(int, manifest["warmup"])):
            warmup_case = PriorityBurstCase(
                sample_id=f"warmup-{index:06d}",
                ordinal=-(index + 1),
                scheduled_offset_seconds=0.0,
                model=normal_case.model,
                prompt_tokens=normal_case.prompt_tokens,
                max_completion_tokens=normal_case.max_completion_tokens,
                traffic_class="normal",
                backend_priority=normal_case.backend_priority,
            )
            try:
                await driver.warmup(warmup_case)
            except Exception as error:
                raise PriorityBurstExecutionError(
                    phase="warmup",
                    baseline=baseline,
                    sample_id=warmup_case.sample_id,
                ) from error
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(cast(int, manifest["max_in_flight"]))
        samples = await asyncio.gather(
            *(
                _run_sample(driver, probe, case, started=started, semaphore=semaphore)
                for case in schedule
            )
        )
        wall_clock_seconds = max(time.perf_counter() - started, 1e-12)
    except BaseException:
        try:
            await driver.end_run()
        except Exception:
            pass
        raise
    else:
        try:
            await driver.end_run()
        except Exception as error:
            raise PriorityBurstExecutionError(phase="driver_end_run", baseline=baseline) from error
    if driver_manifest != manifest:
        raise PriorityBurstExecutionError(phase="manifest_validation", baseline=baseline)
    return {
        "baseline": baseline,
        "wall_clock_seconds": wall_clock_seconds,
        "samples": sorted(samples, key=lambda sample: cast(int, sample["ordinal"])),
    }


async def _run_sample(
    driver: PriorityBurstDriver,
    probe: PriorityBurstObservationProbe,
    case: PriorityBurstCase,
    *,
    started: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    delay = started + case.scheduled_offset_seconds - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    async with semaphore:
        try:
            observation = await probe.start_sample(driver.baseline, case)
        except Exception as error:
            raise PriorityBurstExecutionError(
                phase="observation_start_sample",
                baseline=driver.baseline,
                sample_id=case.sample_id,
            ) from error
        _validate_observation_session(observation, driver.baseline, case.sample_id)
        operation: PriorityBurstOperation | None = None
        try:
            try:
                operation = await driver.start(case)
            except Exception as error:
                raise PriorityBurstExecutionError(
                    phase="start",
                    baseline=driver.baseline,
                    sample_id=case.sample_id,
                ) from error
            _validate_operation(operation, driver.baseline, case.sample_id)
            submitted = time.perf_counter()
            try:
                await observation.operation_submitted(submitted)
            except Exception as error:
                raise PriorityBurstExecutionError(
                    phase="observation_operation_submitted",
                    baseline=driver.baseline,
                    sample_id=case.sample_id,
                ) from error
            try:
                result = await operation.wait()
            except Exception as error:
                raise PriorityBurstExecutionError(
                    phase="wait",
                    baseline=driver.baseline,
                    sample_id=case.sample_id,
                ) from error
            terminal = time.perf_counter()
            _validate_result(result, driver.baseline, case.sample_id)
            try:
                observed = await observation.finish(result)
            except Exception as error:
                raise PriorityBurstExecutionError(
                    phase="observation_finish",
                    baseline=driver.baseline,
                    sample_id=case.sample_id,
                ) from error
            _validate_observation(observed, driver.baseline, case.sample_id)
            sample = {
                "sample_id": case.sample_id,
                "ordinal": case.ordinal,
                "scheduled_offset_seconds": case.scheduled_offset_seconds,
                "submitted_at_seconds": submitted - started,
                "backend_queued_at_seconds": submitted - started + observed.queued_after_seconds,
                "backend_started_at_seconds": submitted - started + observed.backend_started_after_seconds,
                "backend_completed_at_seconds": submitted - started + observed.backend_completed_after_seconds,
                "terminal_at_seconds": terminal - started,
                "traffic_class": case.traffic_class,
                "requested_backend_priority": case.backend_priority,
                "observed_backend_priority": observed.observed_backend_priority,
                "queue_depth_at_submit": observed.queue_depth_at_submit,
                "continuously_runnable": observed.continuously_runnable,
                "terminal_outcome": result.terminal_outcome,
            }
        except BaseException:
            if operation is not None:
                try:
                    await operation.close()
                except Exception:
                    pass
            try:
                await observation.close()
            except Exception:
                pass
            raise
        else:
            assert operation is not None
            try:
                await operation.close()
            except Exception as error:
                raise PriorityBurstExecutionError(
                    phase="close",
                    baseline=driver.baseline,
                    sample_id=case.sample_id,
                ) from error
            try:
                await observation.close()
            except Exception as error:
                raise PriorityBurstExecutionError(
                    phase="observation_close",
                    baseline=driver.baseline,
                    sample_id=case.sample_id,
                ) from error
            return sample


def _aggregate_run(value: object, workload: Mapping[str, Any], *, location: str) -> dict[str, Any]:
    run = _mapping(value, location)
    baseline = _required_choice(run, "baseline", PRIORITY_BURST_BASELINES, location)
    wall_clock = _required_positive_number(run, "wall_clock_seconds", location)
    values = _sequence(run.get("samples"), f"{location}.samples")
    if len(values) != workload["sample_count"]:
        raise ValueError(f"{location}.samples must contain workload.sample_count entries")
    samples = []
    for index, item in enumerate(values):
        sample = _normalize_sample(item, workload, location=f"{location}.samples[{index}]")
        if sample["ordinal"] != index:
            raise ValueError(f"{location}.samples[{index}].ordinal must be {index}")
        if sample["terminal_at_seconds"] > wall_clock:
            raise ValueError(f"{location}.samples[{index}].terminal_at_seconds exceeds wall_clock_seconds")
        samples.append(sample)
    urgent = [sample for sample in samples if sample["traffic_class"] == "urgent"]
    normal = [sample for sample in samples if sample["traffic_class"] == "normal"]
    completed = [sample for sample in samples if sample["terminal_outcome"] == "completed"]
    urgent_completed = [sample for sample in urgent if sample["terminal_outcome"] == "completed"]
    normal_completed = [sample for sample in normal if sample["terminal_outcome"] == "completed"]
    starved = [
        sample
        for sample in normal
        if sample["continuously_runnable"] and sample["terminal_outcome"] != "completed"
    ]
    urgent_latency = [
        cast(float, sample["terminal_at_seconds"]) - cast(float, sample["scheduled_offset_seconds"])
        for sample in urgent_completed
    ]
    normal_latency = [
        cast(float, sample["terminal_at_seconds"]) - cast(float, sample["scheduled_offset_seconds"])
        for sample in normal_completed
    ]
    urgent_saturated = all(
        cast(int, sample["queue_depth_at_submit"]) >= cast(int, workload["minimum_queue_depth"])
        for sample in urgent
    )
    if baseline == "raw_openai_http_sse":
        priority_observed = all(
            sample["observed_backend_priority"] in (None, workload["normal_priority"])
            for sample in urgent
        )
    else:
        priority_observed = all(
            sample["observed_backend_priority"] == sample["requested_backend_priority"]
            for sample in samples
        )
    return {
        "baseline": baseline,
        "wall_clock_seconds": wall_clock,
        "sample_count": len(samples),
        "completed_count": len(completed),
        "urgent_completed_count": len(urgent_completed),
        "normal_completed_count": len(normal_completed),
        "urgent_latency_seconds": _distribution(urgent_latency),
        "normal_latency_seconds": _distribution(normal_latency),
        "throughput_requests_per_second": len(completed) / wall_clock,
        "normal_starvation_count": len(starved),
        "urgent_queue_saturation_observed": urgent_saturated,
        "urgent_priority_observed": priority_observed,
        "semantic_defect_count": len(samples) - len(completed),
        "samples": samples,
    }


def _normalize_sample(value: object, workload: Mapping[str, Any], *, location: str) -> dict[str, Any]:
    sample = _mapping(value, location)
    ordinal = _required_non_negative_int(sample, "ordinal", location)
    expected_id = f"sample-{ordinal:06d}"
    if _required_string(sample, "sample_id", location) != expected_id:
        raise ValueError(f"{location}.sample_id must match ordinal")
    burst_start = cast(int, workload["burst_start_ordinal"])
    burst_end = burst_start + cast(int, workload["burst_size"])
    expected_class = "urgent" if burst_start <= ordinal < burst_end else "normal"
    traffic_class = _required_choice(sample, "traffic_class", _TRAFFIC_CLASSES, location)
    if traffic_class != expected_class:
        raise ValueError(f"{location}.traffic_class differs from workload burst schedule")
    expected_priority = cast(int, workload[f"{traffic_class}_priority"])
    if _required_int(sample, "requested_backend_priority", location) != expected_priority:
        raise ValueError(f"{location}.requested_backend_priority differs from workload")
    scheduled = _required_non_negative_number(sample, "scheduled_offset_seconds", location)
    expected_offset = ordinal * cast(float, workload["arrival_interval_seconds"])
    if not math.isclose(scheduled, expected_offset, abs_tol=1e-12):
        raise ValueError(f"{location}.scheduled_offset_seconds differs from workload schedule")
    submitted = _required_non_negative_number(sample, "submitted_at_seconds", location)
    queued = _required_non_negative_number(sample, "backend_queued_at_seconds", location)
    backend_started = _required_non_negative_number(sample, "backend_started_at_seconds", location)
    backend_completed = _required_non_negative_number(sample, "backend_completed_at_seconds", location)
    terminal = _required_non_negative_number(sample, "terminal_at_seconds", location)
    if not scheduled <= submitted <= queued <= backend_started <= backend_completed <= terminal:
        raise ValueError(f"{location} lifecycle timestamps must be monotonic")
    observed_priority = sample.get("observed_backend_priority")
    if observed_priority is not None and (
        isinstance(observed_priority, bool) or not isinstance(observed_priority, int)
    ):
        raise ValueError(f"{location}.observed_backend_priority must be an integer or null")
    return {
        "sample_id": expected_id,
        "ordinal": ordinal,
        "scheduled_offset_seconds": scheduled,
        "submitted_at_seconds": submitted,
        "backend_queued_at_seconds": queued,
        "backend_started_at_seconds": backend_started,
        "backend_completed_at_seconds": backend_completed,
        "terminal_at_seconds": terminal,
        "traffic_class": traffic_class,
        "requested_backend_priority": expected_priority,
        "observed_backend_priority": observed_priority,
        "queue_depth_at_submit": _required_non_negative_int(sample, "queue_depth_at_submit", location),
        "continuously_runnable": _required_bool(sample, "continuously_runnable", location),
        "terminal_outcome": _required_choice(sample, "terminal_outcome", _TERMINAL_OUTCOMES, location),
    }


def _acceptance(workload: Mapping[str, Any], runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    accounting = cast(Mapping[str, object], workload["scheduler_accounting"])
    if accounting["acceptance_eligible"] is not True:
        return {
            "evaluated": False,
            "reason": "scheduler accounting is a shared-engine or request-interval proxy",
        }
    raw = runs["raw_openai_http_sse"]
    direct = runs["direct_nnrp"]
    raw_p95 = _distribution_value(raw["urgent_latency_seconds"], "p95_seconds")
    direct_p95 = _distribution_value(direct["urgent_latency_seconds"], "p95_seconds")
    raw_throughput = cast(float, raw["throughput_requests_per_second"])
    direct_throughput = cast(float, direct["throughput_requests_per_second"])
    latency_reduction = None if raw_p95 in (None, 0.0) or direct_p95 is None else (raw_p95 - direct_p95) / raw_p95
    throughput_regression = (
        None if raw_throughput == 0 else (raw_throughput - direct_throughput) / raw_throughput
    )
    evidence_complete = all(
        cast(int, run["urgent_completed_count"]) == workload["burst_size"]
        and run["urgent_queue_saturation_observed"] is True
        and run["urgent_priority_observed"] is True
        for run in runs.values()
    )
    return {
        "evaluated": evidence_complete,
        "reason": None if evidence_complete else "priority or queue-saturation evidence is incomplete",
        "urgent_p95_reduction_vs_raw": latency_reduction,
        "throughput_regression_vs_raw": throughput_regression,
        "direct_normal_starvation_count": direct["normal_starvation_count"],
        "hypotheses": {
            "urgent_p95_reduction_at_least_30_percent": _threshold_status(latency_reduction, minimum=0.30),
            "no_normal_starvation": "pass" if direct["normal_starvation_count"] == 0 else "fail",
            "throughput_regression_at_most_5_percent": _threshold_status(throughput_regression, maximum=0.05),
        } if evidence_complete else {
            "urgent_p95_reduction_at_least_30_percent": "not_evaluable",
            "no_normal_starvation": "not_evaluable",
            "throughput_regression_at_most_5_percent": "not_evaluable",
        },
    }


async def _load_factory(spec: str, *, kind: str) -> object:
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError(f"{kind} spec must be 'module.path:factory_name'")
    factory = getattr(importlib.import_module(module_name), factory_name)
    if not callable(factory):
        raise TypeError(f"{kind} factory is not callable: {spec}")
    value = factory()
    return await value if inspect.isawaitable(value) else value


def _validate_driver(value: object, *, expected_baseline: object, location: str) -> None:
    baseline = getattr(value, "baseline", None)
    if baseline not in PRIORITY_BURST_BASELINES or baseline != expected_baseline:
        raise ValueError(f"{location}.baseline must match one priority-burst baseline")
    for method in ("begin_run", "warmup", "start", "end_run"):
        if not callable(getattr(value, method, None)):
            raise TypeError(f"{location} must define callable {method}()")


def _validate_probe(value: object, *, location: str) -> None:
    if getattr(value, "method", None) not in _SCHEDULER_METHODS:
        raise ValueError(f"{location}.method is not a supported scheduler accounting method")
    if getattr(value, "scope", None) not in _SCHEDULER_SCOPES:
        raise ValueError(f"{location}.scope is not a supported scheduler accounting scope")
    source = getattr(value, "source", None)
    if not isinstance(source, str) or not source:
        raise ValueError(f"{location}.source must be non-empty")
    for method in ("begin_run", "start_sample", "end_run"):
        if not callable(getattr(value, method, None)):
            raise TypeError(f"{location} must define callable {method}()")


def _validate_result(value: object, baseline: str, sample_id: str) -> None:
    if not isinstance(value, PriorityBurstResult) or value.terminal_outcome not in _TERMINAL_OUTCOMES:
        raise PriorityBurstExecutionError(phase="result_validation", baseline=baseline, sample_id=sample_id)


def _validate_operation(value: object, baseline: str, sample_id: str) -> None:
    if any(not callable(getattr(value, method, None)) for method in ("wait", "close")):
        raise PriorityBurstExecutionError(phase="operation_validation", baseline=baseline, sample_id=sample_id)


def _validate_observation_session(value: object, baseline: str, sample_id: str) -> None:
    if any(
        not callable(getattr(value, method, None))
        for method in ("operation_submitted", "finish", "close")
    ):
        raise PriorityBurstExecutionError(
            phase="observation_session_validation",
            baseline=baseline,
            sample_id=sample_id,
        )


def _validate_observation(value: object, baseline: str, sample_id: str) -> None:
    if not isinstance(value, PriorityBurstObservationResult):
        raise PriorityBurstExecutionError(phase="observation_validation", baseline=baseline, sample_id=sample_id)
    if not (
        0 <= value.queued_after_seconds <= value.backend_started_after_seconds <= value.backend_completed_after_seconds
    ):
        raise PriorityBurstExecutionError(phase="observation_validation", baseline=baseline, sample_id=sample_id)
    if value.queue_depth_at_submit < 0:
        raise PriorityBurstExecutionError(phase="observation_validation", baseline=baseline, sample_id=sample_id)


def _scheduler_accounting(value: object) -> dict[str, Any]:
    accounting = _mapping(value, "workload.scheduler_accounting")
    method = _required_choice(accounting, "method", _SCHEDULER_METHODS, "workload.scheduler_accounting")
    scope = _required_choice(accounting, "scope", _SCHEDULER_SCOPES, "workload.scheduler_accounting")
    return {
        "method": method,
        "scope": scope,
        "source": _public_label(accounting, "source", "workload.scheduler_accounting"),
        "acceptance_eligible": (method, scope) in _ACCEPTANCE_ELIGIBLE_ACCOUNTING,
    }


def _validate_schedule_identity(runs: Mapping[str, Mapping[str, Any]]) -> None:
    signatures = []
    for baseline in PRIORITY_BURST_BASELINES:
        samples = cast(Sequence[Mapping[str, object]], runs[baseline]["samples"])
        signatures.append(
            tuple(
                (sample["sample_id"], sample["traffic_class"], sample["scheduled_offset_seconds"])
                for sample in samples
            )
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("evidence.runs must use the same priority-burst schedule across baselines")


def _distribution(samples: Sequence[float]) -> dict[str, Any] | None:
    if not samples:
        return None
    ordered = sorted(samples)
    mean = statistics.fmean(ordered)
    margin = 0.0 if len(ordered) == 1 else 1.6448536269514722 * statistics.stdev(ordered) / math.sqrt(len(ordered))
    return {
        "sample_count": len(ordered),
        "p50_seconds": _percentile(ordered, 0.50),
        "p95_seconds": _percentile(ordered, 0.95),
        "p99_seconds": _percentile(ordered, 0.99),
        "mean_90ci_seconds": {
            "mean_seconds": mean,
            "lower_seconds": max(0.0, mean - margin),
            "upper_seconds": mean + margin,
        },
    }


def _distribution_value(value: object, key: str) -> float | None:
    if not isinstance(value, Mapping):
        return None
    item = value.get(key)
    return float(item) if isinstance(item, int | float) and not isinstance(item, bool) else None


def _percentile(samples: Sequence[float], percentile: float) -> float:
    index = min(len(samples) - 1, max(0, int(round((len(samples) - 1) * percentile))))
    return samples[index]


def _threshold_status(value: object, *, minimum: float | None = None, maximum: float | None = None) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "not_evaluable"
    return "pass" if (minimum is None or value >= minimum) and (maximum is None or value <= maximum) else "fail"


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _sequence(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{location} must be an array")
    return value


def _required_string(value: Mapping[str, object], key: str, location: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{location}.{key} must be a non-empty string")
    return item


def _required_git_revision(value: Mapping[str, object], key: str, location: str) -> str:
    item = _required_string(value, key, location)
    if not 7 <= len(item) <= 40 or any(character not in "0123456789abcdef" for character in item):
        raise ValueError(f"{location}.{key} must be a lowercase 7-40 character Git revision")
    return item


def _public_label(value: Mapping[str, object], key: str, location: str) -> str:
    item = _required_string(value, key, location)
    _reject_sensitive_strings(item, f"{location}.{key}")
    return item


def _required_choice(
    value: Mapping[str, object],
    key: str,
    choices: Sequence[str],
    location: str,
) -> str:
    item = _required_string(value, key, location)
    if item not in choices:
        raise ValueError(f"{location}.{key} must be one of {', '.join(choices)}")
    return item


def _required_int(value: Mapping[str, object], key: str, location: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"{location}.{key} must be an integer")
    return item


def _required_non_negative_int(value: Mapping[str, object], key: str, location: str) -> int:
    item = _required_int(value, key, location)
    if item < 0:
        raise ValueError(f"{location}.{key} must be non-negative")
    return item


def _required_positive_int(value: Mapping[str, object], key: str, location: str) -> int:
    item = _required_int(value, key, location)
    if item <= 0:
        raise ValueError(f"{location}.{key} must be positive")
    return item


def _required_number(value: Mapping[str, object], key: str, location: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(float(item)):
        raise ValueError(f"{location}.{key} must be a finite number")
    return float(item)


def _required_non_negative_number(value: Mapping[str, object], key: str, location: str) -> float:
    item = _required_number(value, key, location)
    if item < 0:
        raise ValueError(f"{location}.{key} must be non-negative")
    return item


def _required_positive_number(value: Mapping[str, object], key: str, location: str) -> float:
    item = _required_number(value, key, location)
    if item <= 0:
        raise ValueError(f"{location}.{key} must be positive")
    return item


def _required_bool(value: Mapping[str, object], key: str, location: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ValueError(f"{location}.{key} must be a boolean")
    return item


def _reject_sensitive_strings(value: object, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"token", "access_token", "secret", "api_key", "username", "host_ip"}:
                raise ValueError(f"{location}.{key} contains prohibited sensitive metadata")
            _reject_sensitive_strings(item, f"{location}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_sensitive_strings(item, f"{location}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if "c:\\users\\" in lowered or "/home/" in lowered or "bearer " in lowered:
            raise ValueError(f"{location} contains prohibited machine or credential data")


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "0.1.0"


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
