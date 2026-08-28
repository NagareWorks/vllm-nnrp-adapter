from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .benchmark import validate_benchmark_evidence

_SCHEMA_VERSION = "nnrp-adoption-evidence/v1"
_BASELINES = (
    "raw_openai_http_sse",
    "orchestrated_http_sse",
    "direct_nnrp",
)
_STALE_OUTCOMES = frozenset({"cancelled", "aborted", "expired", "superseded"})
_CONTROL_BY_OUTCOME = {
    "cancelled": "cancel",
    "aborted": "abort",
    "expired": "deadline",
    "superseded": "supersede",
}
_OUTCOME_BY_CONTROL = {control: outcome for outcome, control in _CONTROL_BY_OUTCOME.items()}
_CONTROL_KINDS = tuple(_OUTCOME_BY_CONTROL)
_STALE_WORK_RATIOS = (0.1, 0.3, 0.5)
_GPU_ACCOUNTING_METHODS = (
    "device_active_time",
    "cuda_event_attribution",
    "request_inference_interval_proxy",
)
_GPU_ACCOUNTING_SCOPES = ("dedicated_device", "scheduled_batch", "request")
_ACCEPTANCE_ELIGIBLE_GPU_ACCOUNTING = frozenset(
    {
        ("device_active_time", "dedicated_device"),
        ("cuda_event_attribution", "scheduled_batch"),
    }
)


def aggregate_stale_work_evidence_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    report = aggregate_stale_work_evidence(source)
    validate_benchmark_evidence(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return report


def aggregate_stale_work_evidence(source: object) -> dict[str, Any]:
    root = _mapping(source, "evidence")
    if root.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"evidence.schema_version must be {_SCHEMA_VERSION!r}")

    workload = _workload_manifest(root.get("workload"))
    runs = _sequence(root.get("runs"), "evidence.runs")
    summaries: dict[str, dict[str, Any]] = {}
    baseline_execution_order: list[str] = []
    for index, run_value in enumerate(runs):
        run = _mapping(run_value, f"evidence.runs[{index}]")
        baseline = _required_choice(run, "baseline", _BASELINES, f"evidence.runs[{index}]")
        if baseline in summaries:
            raise ValueError(f"evidence.runs contains duplicate baseline {baseline!r}")
        summaries[baseline] = _aggregate_run(run, workload, location=f"evidence.runs[{index}]")
        baseline_execution_order.append(baseline)

    missing = sorted(set(_BASELINES) - summaries.keys())
    extra = sorted(summaries.keys() - set(_BASELINES))
    if missing or extra:
        raise ValueError(
            "evidence.runs must contain exactly the three comparison baselines; "
            f"missing={missing}, extra={extra}"
        )

    _validate_schedule_identity(summaries)

    ordered_summaries = [summaries[baseline] for baseline in _BASELINES]
    report = {
        "benchmark_kind": "stale_work_adoption_evidence",
        "schema_version": _SCHEMA_VERSION,
        "workload": workload,
        "baseline_execution_order": baseline_execution_order,
        "runs": ordered_summaries,
        "acceptance": _acceptance(workload, summaries),
    }
    validate_benchmark_evidence(report)
    return report


def normalize_stale_work_manifest(value: object) -> dict[str, Any]:
    manifest = _mapping(value, "evidence.workload")
    if manifest.get("scenario") != "stale_work":
        raise ValueError("evidence.workload.scenario must be 'stale_work'")
    stale_work_ratio = _required_number(manifest, "stale_work_ratio", "evidence.workload")
    if not any(math.isclose(stale_work_ratio, expected, abs_tol=1e-12) for expected in _STALE_WORK_RATIOS):
        raise ValueError("evidence.workload.stale_work_ratio must be one of 0.1, 0.3, or 0.5")

    max_in_flight = _required_positive_int(manifest, "max_in_flight", "evidence.workload")
    normalized = {
        "scenario": "stale_work",
        "stale_work_ratio": stale_work_ratio,
        "model": _required_string(manifest, "model", "evidence.workload"),
        "engine": _required_string(manifest, "engine", "evidence.workload"),
        "gpu": _required_string(manifest, "gpu", "evidence.workload"),
        "arrival_schedule": _required_string(manifest, "arrival_schedule", "evidence.workload"),
        "arrival_interval_seconds": _required_non_negative_number(
            manifest,
            "arrival_interval_seconds",
            "evidence.workload",
        ),
        "cancellation_schedule": _required_string(manifest, "cancellation_schedule", "evidence.workload"),
        "prompt_tokens": _required_positive_int(manifest, "prompt_tokens", "evidence.workload"),
        "max_completion_tokens": _required_positive_int(
            manifest,
            "max_completion_tokens",
            "evidence.workload",
        ),
        "warmup": _required_non_negative_int(manifest, "warmup", "evidence.workload"),
        "random_seed": _required_non_negative_int(manifest, "random_seed", "evidence.workload"),
        "sample_count": _required_positive_int(manifest, "sample_count", "evidence.workload"),
        "max_in_flight": max_in_flight,
        "gpu_accounting": _gpu_accounting(manifest.get("gpu_accounting"), max_in_flight=max_in_flight),
    }
    validate_benchmark_evidence(normalized)
    return normalized


def _workload_manifest(value: object) -> dict[str, Any]:
    return normalize_stale_work_manifest(value)


def _gpu_accounting(value: object, *, max_in_flight: int) -> dict[str, Any]:
    accounting = _mapping(value, "evidence.workload.gpu_accounting")
    method = _required_choice(
        accounting,
        "method",
        _GPU_ACCOUNTING_METHODS,
        "evidence.workload.gpu_accounting",
    )
    scope = _required_choice(
        accounting,
        "scope",
        _GPU_ACCOUNTING_SCOPES,
        "evidence.workload.gpu_accounting",
    )
    acceptance_eligible = (method, scope) in _ACCEPTANCE_ELIGIBLE_GPU_ACCOUNTING
    if method == "device_active_time" and max_in_flight != 1:
        acceptance_eligible = False
    return {
        "method": method,
        "scope": scope,
        "source": _required_string(accounting, "source", "evidence.workload.gpu_accounting"),
        "acceptance_eligible": acceptance_eligible,
    }


def _aggregate_run(run: Mapping[str, object], workload: Mapping[str, object], *, location: str) -> dict[str, Any]:
    baseline = _required_choice(run, "baseline", _BASELINES, location)
    wall_clock_seconds = _required_positive_number(run, "wall_clock_seconds", location)
    samples = _sequence(run.get("samples"), f"{location}.samples")
    expected_count = cast(int, workload["sample_count"])
    if len(samples) != expected_count:
        raise ValueError(f"{location}.samples must contain workload.sample_count={expected_count} entries")

    normalized_samples: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for index, sample_value in enumerate(samples):
        sample = _normalize_sample(sample_value, location=f"{location}.samples[{index}]")
        sample_id = str(sample["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"{location}.samples contains duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        if float(sample["backend_stopped_at_seconds"]) > wall_clock_seconds:
            raise ValueError(f"{location}.samples[{index}].backend_stopped_at_seconds exceeds wall_clock_seconds")
        normalized_samples.append(sample)

    stale_samples = [sample for sample in normalized_samples if sample["control_kind"] is not None]
    observed_ratio = len(stale_samples) / len(normalized_samples)
    expected_ratio = cast(float, workload["stale_work_ratio"])
    if not math.isclose(observed_ratio, expected_ratio, abs_tol=1e-12):
        raise ValueError(
            f"{location}.samples stale-work ratio {observed_ratio:.6f} "
            f"does not match workload ratio {expected_ratio:.6f}"
        )

    total_gpu_seconds = sum(float(sample["gpu_seconds"]) for sample in normalized_samples)
    wasted_gpu_samples = [float(sample["gpu_seconds"]) for sample in stale_samples]
    wasted_gpu_seconds = sum(wasted_gpu_samples)
    useful_gpu_seconds = total_gpu_seconds - wasted_gpu_seconds
    accepted_samples = [sample for sample in stale_samples if sample["control_accepted"]]
    effect_latencies = [
        float(sample["backend_stopped_at_seconds"]) - float(sample["control_accepted_at_seconds"])
        for sample in accepted_samples
    ]
    late_result_operations = sum(int(sample["late_result_count"]) > 0 for sample in accepted_samples)
    useful_result_weight = sum(float(sample["useful_result_weight"]) for sample in normalized_samples)
    semantic_defects = sum(_is_semantic_defect(sample) for sample in normalized_samples)

    return {
        "baseline": baseline,
        "wall_clock_seconds": wall_clock_seconds,
        "sample_count": len(normalized_samples),
        "stale_sample_count": len(stale_samples),
        "accepted_control_count": len(accepted_samples),
        "total_gpu_seconds": total_gpu_seconds,
        "useful_gpu_seconds": useful_gpu_seconds,
        "wasted_gpu_seconds": wasted_gpu_seconds,
        "wasted_compute_ratio": wasted_gpu_seconds / total_gpu_seconds if total_gpu_seconds > 0 else 0.0,
        "deadline_weighted_useful_goodput": useful_result_weight / wall_clock_seconds,
        "cancellation_effect_latency_seconds": _distribution(effect_latencies),
        "late_result_operation_count": late_result_operations,
        "late_result_rate": late_result_operations / len(accepted_samples) if accepted_samples else None,
        "semantic_defect_count": semantic_defects,
        "semantic_defect_rate": semantic_defects / len(normalized_samples),
        "gpu_seconds_90ci": {
            "total": _mean_confidence_interval([float(sample["gpu_seconds"]) for sample in normalized_samples]),
            "wasted": _mean_confidence_interval(wasted_gpu_samples),
        },
        "samples": normalized_samples,
    }


def _normalize_sample(value: object, *, location: str) -> dict[str, Any]:
    sample = _mapping(value, location)
    terminal_outcome = _required_choice(
        sample,
        "terminal_outcome",
        ("completed", "failed", *sorted(_STALE_OUTCOMES)),
        location,
    )
    control_kind = sample.get("control_kind")
    if control_kind is not None and control_kind not in _CONTROL_KINDS:
        raise ValueError(f"{location}.control_kind must be null or one of {', '.join(_CONTROL_KINDS)}")
    control_accepted = _required_bool(sample, "control_accepted", location)
    control_accepted_at = _optional_non_negative_number(sample, "control_accepted_at_seconds", location)
    backend_stopped_at = _required_non_negative_number(sample, "backend_stopped_at_seconds", location)
    gpu_seconds = _required_non_negative_number(sample, "gpu_seconds", location)
    useful_result_weight = _required_non_negative_number(sample, "useful_result_weight", location)
    late_result_count = _required_non_negative_int(sample, "late_result_count", location)

    if control_kind is None:
        if control_accepted or control_accepted_at is not None or late_result_count:
            raise ValueError(f"{location} non-stale samples must not contain stale-work control evidence")
        if terminal_outcome == "failed" and useful_result_weight != 0:
            raise ValueError(f"{location}.useful_result_weight must be zero for failed work")
        if terminal_outcome in _STALE_OUTCOMES:
            raise ValueError(f"{location}.control_kind is required for stale terminal outcomes")
    else:
        if useful_result_weight != 0:
            raise ValueError(f"{location}.useful_result_weight must be zero for stale work")
        if control_accepted:
            if control_accepted_at is None:
                raise ValueError(f"{location}.control_accepted_at_seconds is required when control_accepted is true")
            if backend_stopped_at < control_accepted_at:
                raise ValueError(f"{location}.backend_stopped_at_seconds precedes accepted control")
        elif control_accepted_at is not None or late_result_count:
            raise ValueError(
                f"{location} cannot report accepted-control timing or late results when control_accepted is false"
            )

    return {
        "sample_id": _required_string(sample, "sample_id", location),
        "terminal_outcome": terminal_outcome,
        "control_kind": control_kind,
        "control_accepted": control_accepted,
        "control_accepted_at_seconds": control_accepted_at,
        "backend_stopped_at_seconds": backend_stopped_at,
        "gpu_seconds": gpu_seconds,
        "useful_result_weight": useful_result_weight,
        "late_result_count": late_result_count,
    }


def _validate_schedule_identity(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    expected: tuple[tuple[str, object], ...] | None = None
    expected_baseline: str | None = None
    for baseline in _BASELINES:
        samples = cast(Sequence[Mapping[str, object]], summaries[baseline]["samples"])
        signature = tuple((str(sample["sample_id"]), sample["control_kind"]) for sample in samples)
        if expected is None:
            expected = signature
            expected_baseline = baseline
        elif signature != expected:
            raise ValueError(
                "evidence.runs must use the same ordered sample IDs and control schedule across baselines; "
                f"{baseline!r} differs from {expected_baseline!r}"
            )


def _is_semantic_defect(sample: Mapping[str, object]) -> int:
    control_kind = sample["control_kind"]
    terminal_outcome = sample["terminal_outcome"]
    if control_kind is None:
        return int(terminal_outcome == "failed")
    if sample["control_accepted"] is not True:
        return 0
    return int(
        terminal_outcome != _OUTCOME_BY_CONTROL[cast(str, control_kind)]
        or cast(int, sample["late_result_count"]) > 0
    )


def _acceptance(workload: Mapping[str, object], summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not math.isclose(cast(float, workload["stale_work_ratio"]), 0.3, abs_tol=1e-12):
        return {
            "evaluated": False,
            "reason": "Preview4 stale-work acceptance thresholds apply to the 30% workload",
        }
    gpu_accounting = cast(Mapping[str, object], workload["gpu_accounting"])
    if gpu_accounting.get("acceptance_eligible") is not True:
        return {
            "evaluated": False,
            "reason": "GPU accounting is a request interval proxy and cannot evaluate GPU-second thresholds",
        }

    raw_wasted = float(summaries["raw_openai_http_sse"]["wasted_gpu_seconds"])
    orchestrated_wasted = float(summaries["orchestrated_http_sse"]["wasted_gpu_seconds"])
    direct_wasted = float(summaries["direct_nnrp"]["wasted_gpu_seconds"])
    direct_late_rate = summaries["direct_nnrp"]["late_result_rate"]

    reduction_vs_raw = None if raw_wasted == 0 else (raw_wasted - direct_wasted) / raw_wasted
    regression_vs_orchestrated = (
        None if orchestrated_wasted == 0 else (direct_wasted - orchestrated_wasted) / orchestrated_wasted
    )
    return {
        "evaluated": True,
        "wasted_gpu_reduction_vs_raw": reduction_vs_raw,
        "wasted_gpu_regression_vs_orchestrated": regression_vs_orchestrated,
        "direct_late_result_rate": direct_late_rate,
        "hypotheses": {
            "wasted_gpu_reduction_at_least_40_percent": _threshold_status(reduction_vs_raw, minimum=0.40),
            "wasted_gpu_non_inferiority_within_3_percent": _threshold_status(
                regression_vs_orchestrated,
                maximum=0.03,
            ),
            "late_result_rate_below_0_1_percent": _threshold_status(direct_late_rate, strict_maximum=0.001),
        },
    }


def _threshold_status(
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_maximum: float | None = None,
) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "not_evaluable"
    numeric = float(value)
    passed = True
    if minimum is not None:
        passed = passed and numeric >= minimum
    if maximum is not None:
        passed = passed and numeric <= maximum
    if strict_maximum is not None:
        passed = passed and numeric < strict_maximum
    return "pass" if passed else "fail"


def _distribution(samples: list[float]) -> dict[str, Any] | None:
    if not samples:
        return None
    ordered = sorted(samples)
    return {
        "sample_count": len(ordered),
        "p50_seconds": _percentile(ordered, 0.50),
        "p95_seconds": _percentile(ordered, 0.95),
        "p99_seconds": _percentile(ordered, 0.99),
        "mean_90ci_seconds": _mean_confidence_interval(ordered),
    }


def _mean_confidence_interval(samples: list[float]) -> dict[str, float | int] | None:
    if not samples:
        return None
    mean = statistics.fmean(samples)
    margin = 0.0 if len(samples) == 1 else 1.6448536269514722 * statistics.stdev(samples) / math.sqrt(len(samples))
    return {
        "sample_count": len(samples),
        "mean_seconds": mean,
        "lower_seconds": max(0.0, mean - margin),
        "upper_seconds": mean + margin,
    }


def _percentile(sorted_samples: list[float], percentile: float) -> float:
    index = min(len(sorted_samples) - 1, max(0, int(round((len(sorted_samples) - 1) * percentile))))
    return sorted_samples[index]


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


def _required_bool(value: Mapping[str, object], key: str, location: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ValueError(f"{location}.{key} must be a boolean")
    return item


def _required_number(value: Mapping[str, object], key: str, location: str) -> float:
    item = value.get(key)
    if not isinstance(item, int | float) or isinstance(item, bool) or not math.isfinite(float(item)):
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


def _optional_non_negative_number(value: Mapping[str, object], key: str, location: str) -> float | None:
    if value.get(key) is None:
        return None
    return _required_non_negative_number(value, key, location)


def _required_non_negative_int(value: Mapping[str, object], key: str, location: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise ValueError(f"{location}.{key} must be a non-negative integer")
    return item


def _required_positive_int(value: Mapping[str, object], key: str, location: str) -> int:
    item = _required_non_negative_int(value, key, location)
    if item == 0:
        raise ValueError(f"{location}.{key} must be positive")
    return item
