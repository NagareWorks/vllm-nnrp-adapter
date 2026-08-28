from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import math
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
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

    @property
    def is_stale(self) -> bool:
        return self.control_kind is not None


class StaleWorkDriver(Protocol):
    baseline: str

    async def begin_run(
        self,
        workload: Mapping[str, object],
        schedule: Sequence[StaleWorkCase],
    ) -> None: ...

    async def warmup(self, case: StaleWorkCase) -> None: ...

    async def execute(self, case: StaleWorkCase) -> Mapping[str, object]: ...

    async def end_run(self) -> None: ...


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

    return tuple(
        StaleWorkCase(
            sample_id=f"sample-{ordinal:06d}",
            ordinal=ordinal,
            scheduled_offset_seconds=ordinal * interval,
            model=model,
            prompt_tokens=prompt_tokens,
            max_completion_tokens=max_completion_tokens,
            control_kind=next(control_iterator) if ordinal in stale_ordinals else None,
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
        "runs": runs,
    }
    validate_benchmark_evidence(evidence)
    report = aggregate_stale_work_evidence(evidence)
    return evidence, report


async def run_stale_workload_file(
    manifest_path: Path,
    raw_output_path: Path,
    report_output_path: Path,
    *,
    driver_specs: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if raw_output_path.resolve() == report_output_path.resolve():
        raise ValueError("raw-output and report-output must be different paths")
    workload = json.loads(manifest_path.read_text(encoding="utf-8"))
    drivers = [
        await load_stale_work_driver_async(spec, expected_baseline=baseline)
        for baseline, spec in driver_specs.items()
    ]
    evidence, report = await run_stale_workload(workload, drivers=drivers)
    _write_json(raw_output_path, evidence)
    _write_json(report_output_path, report)
    return evidence, report


def run_stale_workload_file_sync(
    manifest_path: Path,
    raw_output_path: Path,
    report_output_path: Path,
    *,
    driver_specs: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return asyncio.run(
        run_stale_workload_file(
            manifest_path,
            raw_output_path,
            report_output_path,
            driver_specs=driver_specs,
        )
    )


async def _run_driver(
    manifest: Mapping[str, object],
    schedule: Sequence[StaleWorkCase],
    driver: StaleWorkDriver,
) -> dict[str, Any]:
    driver_manifest = deepcopy(dict(manifest))
    began = False
    try:
        await driver.begin_run(driver_manifest, schedule)
        began = True
        await _warmup_driver(driver, manifest, schedule)
        loop = asyncio.get_running_loop()
        run_started_at = loop.time()
        semaphore = asyncio.Semaphore(cast(int, manifest["max_in_flight"]))

        async def execute_case(case: StaleWorkCase) -> dict[str, object]:
            delay = run_started_at + case.scheduled_offset_seconds - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            async with semaphore:
                sample = dict(await driver.execute(case))
            if sample.get("sample_id") != case.sample_id:
                raise ValueError(
                    f"{driver.baseline} driver returned sample_id {sample.get('sample_id')!r}; "
                    f"expected {case.sample_id!r}"
                )
            if sample.get("control_kind") != case.control_kind:
                raise ValueError(
                    f"{driver.baseline} driver changed control_kind for {case.sample_id!r}: "
                    f"expected {case.control_kind!r}, got {sample.get('control_kind')!r}"
                )
            return sample

        samples = await asyncio.gather(*(execute_case(case) for case in schedule))
        wall_clock_seconds = loop.time() - run_started_at
    finally:
        if began:
            await driver.end_run()

    if driver_manifest != manifest:
        raise ValueError(f"{driver.baseline} driver mutated the normalized workload manifest")
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
        await driver.warmup(
            StaleWorkCase(
                sample_id=f"warmup-{index:06d}",
                ordinal=-(index + 1),
                scheduled_offset_seconds=0.0,
                model=template.model,
                prompt_tokens=template.prompt_tokens,
                max_completion_tokens=template.max_completion_tokens,
                control_kind=None,
            )
        )


def _validate_driver(driver: object, *, expected_baseline: object, location: str) -> None:
    baseline = getattr(driver, "baseline", None)
    if not isinstance(baseline, str) or baseline not in STALE_WORK_BASELINES:
        raise ValueError(f"{location} driver baseline must be one of {', '.join(STALE_WORK_BASELINES)}")
    if baseline != expected_baseline:
        raise ValueError(f"{location} created baseline {baseline!r}; expected {expected_baseline!r}")
    for method_name in ("begin_run", "warmup", "execute", "end_run"):
        if not callable(getattr(driver, method_name, None)):
            raise TypeError(f"{location} driver must define callable {method_name}()")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(value, indent=2, sort_keys=True)}\n", encoding="utf-8")
