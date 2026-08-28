from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import math
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, cast

from .adoption_evidence import aggregate_stale_work_evidence, normalize_stale_work_manifest
from .benchmark import validate_benchmark_evidence

_CONTROL_KINDS = ("cancel", "abort", "deadline", "supersede")
STALE_WORK_BASELINES = (
    "raw_openai_http_sse",
    "orchestrated_http_sse",
    "direct_nnrp",
)


@dataclass(frozen=True)
class StaleWorkCase:
    sample_id: str
    ordinal: int
    scheduled_offset_seconds: float
    model: str
    prompt_tokens: int
    max_completion_tokens: int
    control_kind: str | None
    control_delay_seconds: float

    @property
    def is_stale(self) -> bool:
        return self.control_kind is not None


@dataclass(frozen=True)
class StaleWorkResult:
    terminal_outcome: str
    backend_stopped_after_seconds: float
    gpu_seconds: float
    useful_result_weight: float
    late_result_count: int


class StaleWorkOperation(Protocol):
    async def apply_control(self, control_kind: str) -> bool: ...

    async def wait(self) -> StaleWorkResult: ...

    async def close(self) -> None: ...


class StaleWorkDriver(Protocol):
    baseline: str

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[StaleWorkCase],
    ) -> None: ...

    async def warmup(self, case: StaleWorkCase) -> None: ...

    async def start(self, case: StaleWorkCase) -> StaleWorkOperation: ...

    async def end_run(self) -> None: ...


class StaleWorkExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        phase: str,
        baseline: str | None = None,
        sample_id: str | None = None,
    ) -> None:
        super().__init__(f"stale-work execution failed during {phase}")
        self.phase = phase
        self.baseline = baseline
        self.sample_id = sample_id


def build_stale_work_schedule(workload: object) -> tuple[StaleWorkCase, ...]:
    manifest = normalize_stale_work_manifest(workload)
    sample_count = int(manifest["sample_count"])
    stale_work_ratio = float(manifest["stale_work_ratio"])
    stale_count_float = sample_count * stale_work_ratio
    stale_count = round(stale_count_float)
    if not math.isclose(stale_count_float, stale_count, abs_tol=1e-12):
        raise ValueError("workload.sample_count must represent stale_work_ratio exactly")
    if stale_count < len(_CONTROL_KINDS):
        raise ValueError("workload must contain at least one cancel, abort, deadline, and supersede sample")

    random_seed = int(manifest["random_seed"])
    rng = random.Random(random_seed)
    stale_ordinals = set(rng.sample(range(sample_count), stale_count))
    controls = [_CONTROL_KINDS[index % len(_CONTROL_KINDS)] for index in range(stale_count)]
    rng.shuffle(controls)
    control_iterator = iter(controls)
    interval = float(manifest["arrival_interval_seconds"])
    model = str(manifest["model"])
    prompt_tokens = int(manifest["prompt_tokens"])
    max_completion_tokens = int(manifest["max_completion_tokens"])
    control_delay_seconds = float(manifest["control_delay_seconds"])

    return tuple(
        StaleWorkCase(
            sample_id=f"sample-{ordinal:06d}",
            ordinal=ordinal,
            scheduled_offset_seconds=ordinal * interval,
            model=model,
            prompt_tokens=prompt_tokens,
            max_completion_tokens=max_completion_tokens,
            control_kind=next(control_iterator) if ordinal in stale_ordinals else None,
            control_delay_seconds=control_delay_seconds,
        )
        for ordinal in range(sample_count)
    )


async def load_stale_work_driver_async(spec: str, *, expected_baseline: str) -> StaleWorkDriver:
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("stale-work driver spec must be 'module.path:factory_name'")

    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    if not callable(factory):
        raise TypeError(f"stale-work driver factory is not callable: {spec}")
    driver = factory()
    if inspect.isawaitable(driver):
        driver = await driver
    _validate_driver(driver, expected_baseline=expected_baseline, location=spec)
    return cast(StaleWorkDriver, driver)


async def run_stale_workload(
    workload: object,
    *,
    drivers: Sequence[StaleWorkDriver],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = normalize_stale_work_manifest(workload)
    installed_adapter_version = _distribution_version("vllm-nnrp-adapter")
    if manifest["adapter_version"] != installed_adapter_version:
        raise ValueError(
            "workload.adapter_version must match the installed vllm-nnrp-adapter distribution "
            f"({installed_adapter_version})"
        )
    schedule = build_stale_work_schedule(manifest)
    drivers_by_baseline: dict[str, StaleWorkDriver] = {}
    for index, driver in enumerate(drivers):
        baseline = getattr(driver, "baseline", None)
        _validate_driver(driver, expected_baseline=baseline, location=f"drivers[{index}]")
        if baseline in drivers_by_baseline:
            raise ValueError(f"drivers contains duplicate baseline {baseline!r}")
        drivers_by_baseline[cast(str, baseline)] = driver

    missing = sorted(set(STALE_WORK_BASELINES) - drivers_by_baseline.keys())
    extra = sorted(drivers_by_baseline.keys() - set(STALE_WORK_BASELINES))
    if missing or extra:
        raise ValueError(
            "drivers must contain exactly the three stale-work baselines; "
            f"missing={missing}, extra={extra}"
        )

    execution_order = list(STALE_WORK_BASELINES)
    random.Random(f"baseline-order:{manifest['random_seed']}").shuffle(execution_order)
    runs = [
        await _run_driver(manifest, schedule, drivers_by_baseline[baseline])
        for baseline in execution_order
    ]
    evidence = {
        "schema_version": "nnrp-adoption-evidence/v1",
        "workload": manifest,
        "provenance": {
            "adapter_distribution": "vllm-nnrp-adapter",
            "adapter_version": installed_adapter_version,
            "adapter_revision": manifest["adapter_revision"],
            "nnrp_sdk_version": _distribution_version("nnrp-py"),
        },
        "runs": runs,
    }
    validate_benchmark_evidence(evidence)
    report = aggregate_stale_work_evidence(evidence)
    return evidence, report


async def run_stale_workload_file(
    manifest_path: Path,
    raw_output_path: Path,
    report_output_path: Path,
    outcome_output_path: Path,
    *,
    driver_specs: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_output_paths(raw_output_path, report_output_path, outcome_output_path)
    try:
        workload = json.loads(manifest_path.read_text(encoding="utf-8"))
        drivers = [
            await load_stale_work_driver_async(spec, expected_baseline=baseline)
            for baseline, spec in driver_specs.items()
        ]
        evidence, report = await run_stale_workload(workload, drivers=drivers)
        _write_json(raw_output_path, evidence)
        _write_json(report_output_path, report)
    except Exception as error:
        outcome = _failure_outcome(error)
        validate_benchmark_evidence(outcome)
        _write_json(outcome_output_path, outcome)
        raise

    outcome = {
        "schema_version": "nnrp-adoption-run-outcome/v1",
        "status": "passed",
        "baseline_execution_order": report["baseline_execution_order"],
        "sample_count": report["workload"]["sample_count"],
    }
    validate_benchmark_evidence(outcome)
    _write_json(outcome_output_path, outcome)
    return evidence, report


def run_stale_workload_file_sync(
    manifest_path: Path,
    raw_output_path: Path,
    report_output_path: Path,
    outcome_output_path: Path,
    *,
    driver_specs: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return asyncio.run(
        run_stale_workload_file(
            manifest_path,
            raw_output_path,
            report_output_path,
            outcome_output_path,
            driver_specs=driver_specs,
        )
    )


async def _run_driver(
    manifest: Mapping[str, object],
    schedule: Sequence[StaleWorkCase],
    driver: StaleWorkDriver,
) -> dict[str, Any]:
    driver_manifest = deepcopy(dict(manifest))
    try:
        await driver.begin_run(driver_manifest, schedule)
    except Exception as error:
        raise StaleWorkExecutionError(phase="begin_run", baseline=driver.baseline) from error

    try:
        await _warmup_driver(driver, manifest, schedule)
        run_started_at = time.perf_counter()
        semaphore = asyncio.Semaphore(cast(int, manifest["max_in_flight"]))

        async def execute_case(case: StaleWorkCase) -> dict[str, object]:
            await _sleep_until(run_started_at + case.scheduled_offset_seconds)
            async with semaphore:
                operation_started_at = time.perf_counter()
                try:
                    operation = await driver.start(case)
                except Exception as error:
                    raise StaleWorkExecutionError(
                        phase="start",
                        baseline=driver.baseline,
                        sample_id=case.sample_id,
                    ) from error

                _validate_operation(operation, baseline=driver.baseline, sample_id=case.sample_id)
                result_task = asyncio.create_task(operation.wait())
                control_scheduled_at = (
                    operation_started_at - run_started_at + case.control_delay_seconds
                    if case.is_stale
                    else None
                )
                control_issued_at: float | None = None
                control_accepted = False
                control_accepted_at: float | None = None
                try:
                    if case.control_kind is not None:
                        await _sleep_until(operation_started_at + case.control_delay_seconds)
                        control_issued_at = time.perf_counter() - run_started_at
                        try:
                            control_accepted = await operation.apply_control(case.control_kind)
                        except Exception as error:
                            raise StaleWorkExecutionError(
                                phase="apply_control",
                                baseline=driver.baseline,
                                sample_id=case.sample_id,
                            ) from error
                        if not isinstance(control_accepted, bool):
                            raise StaleWorkExecutionError(
                                phase="sample_validation",
                                baseline=driver.baseline,
                                sample_id=case.sample_id,
                            )
                        if control_accepted:
                            control_accepted_at = time.perf_counter() - run_started_at
                    try:
                        result = await result_task
                    except Exception as error:
                        raise StaleWorkExecutionError(
                            phase="wait",
                            baseline=driver.baseline,
                            sample_id=case.sample_id,
                        ) from error
                    _validate_result(result, baseline=driver.baseline, sample_id=case.sample_id)
                except BaseException:
                    if not result_task.done():
                        result_task.cancel()
                        await asyncio.gather(result_task, return_exceptions=True)
                    try:
                        await operation.close()
                    except Exception:
                        pass
                    raise
                else:
                    try:
                        await operation.close()
                    except Exception as error:
                        raise StaleWorkExecutionError(
                            phase="close",
                            baseline=driver.baseline,
                            sample_id=case.sample_id,
                        ) from error

                operation_started_at_seconds = operation_started_at - run_started_at
                return {
                    "sample_id": case.sample_id,
                    "scheduled_offset_seconds": case.scheduled_offset_seconds,
                    "operation_started_at_seconds": operation_started_at_seconds,
                    "terminal_outcome": result.terminal_outcome,
                    "control_kind": case.control_kind,
                    "control_scheduled_at_seconds": control_scheduled_at,
                    "control_issued_at_seconds": control_issued_at,
                    "control_accepted": control_accepted,
                    "control_accepted_at_seconds": control_accepted_at,
                    "backend_stopped_at_seconds": (
                        operation_started_at_seconds + result.backend_stopped_after_seconds
                    ),
                    "gpu_seconds": result.gpu_seconds,
                    "useful_result_weight": result.useful_result_weight,
                    "late_result_count": result.late_result_count,
                }

        samples = await asyncio.gather(*(execute_case(case) for case in schedule))
        wall_clock_seconds = time.perf_counter() - run_started_at
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
            raise StaleWorkExecutionError(phase="end_run", baseline=driver.baseline) from error

    if driver_manifest != manifest:
        raise StaleWorkExecutionError(phase="manifest_validation", baseline=driver.baseline)
    return {
        "baseline": driver.baseline,
        "wall_clock_seconds": wall_clock_seconds,
        "samples": samples,
    }


async def _warmup_driver(
    driver: StaleWorkDriver,
    manifest: Mapping[str, object],
    schedule: Sequence[StaleWorkCase],
) -> None:
    template = next(case for case in schedule if not case.is_stale)
    for index in range(cast(int, manifest["warmup"])):
        case = StaleWorkCase(
            sample_id=f"warmup-{index:06d}",
            ordinal=-(index + 1),
            scheduled_offset_seconds=0.0,
            model=template.model,
            prompt_tokens=template.prompt_tokens,
            max_completion_tokens=template.max_completion_tokens,
            control_kind=None,
            control_delay_seconds=template.control_delay_seconds,
        )
        try:
            await driver.warmup(case)
        except Exception as error:
            raise StaleWorkExecutionError(
                phase="warmup",
                baseline=driver.baseline,
                sample_id=case.sample_id,
            ) from error


def _validate_driver(driver: object, *, expected_baseline: object, location: str) -> None:
    baseline = getattr(driver, "baseline", None)
    if not isinstance(baseline, str) or baseline not in STALE_WORK_BASELINES:
        raise ValueError(f"{location} driver baseline must be one of {', '.join(STALE_WORK_BASELINES)}")
    if baseline != expected_baseline:
        raise ValueError(f"{location} created baseline {baseline!r}; expected {expected_baseline!r}")
    for method_name in ("begin_run", "warmup", "start", "end_run"):
        if not callable(getattr(driver, method_name, None)):
            raise TypeError(f"{location} driver must define callable {method_name}()")


def _validate_operation(operation: object, *, baseline: str, sample_id: str) -> None:
    for method_name in ("apply_control", "wait", "close"):
        if not callable(getattr(operation, method_name, None)):
            raise StaleWorkExecutionError(
                phase="operation_validation",
                baseline=baseline,
                sample_id=sample_id,
            )


def _validate_result(result: object, *, baseline: str, sample_id: str) -> None:
    if (
        not isinstance(result, StaleWorkResult)
        or not isinstance(result.terminal_outcome, str)
        or not result.terminal_outcome
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in (
                result.backend_stopped_after_seconds,
                result.gpu_seconds,
                result.useful_result_weight,
            )
        )
        or isinstance(result.late_result_count, bool)
        or not isinstance(result.late_result_count, int)
        or result.late_result_count < 0
    ):
        raise StaleWorkExecutionError(
            phase="sample_validation",
            baseline=baseline,
            sample_id=sample_id,
        )


async def _sleep_until(target: float) -> None:
    while True:
        remaining = target - time.perf_counter()
        if remaining <= 0:
            return
        await asyncio.sleep(remaining)


def _failure_outcome(error: Exception) -> dict[str, object]:
    outcome: dict[str, object] = {
        "schema_version": "nnrp-adoption-run-outcome/v1",
        "status": "failed",
        "error_type": type(error).__name__,
        "phase": "setup_or_validation",
    }
    if isinstance(error, StaleWorkExecutionError):
        outcome["phase"] = error.phase
        if error.baseline is not None:
            outcome["baseline"] = error.baseline
        if error.sample_id is not None:
            outcome["sample_id"] = error.sample_id
    return outcome


def _validate_output_paths(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("raw-output, report-output, and outcome-output must be different paths")


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as error:
        raise RuntimeError(f"required distribution metadata is unavailable: {distribution}") from error


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(f"{json.dumps(value, indent=2, sort_keys=True)}\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
