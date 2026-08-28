from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from vllm_nnrp_adapter.adoption_evidence import (
    aggregate_stale_work_evidence,
    aggregate_stale_work_evidence_file,
)
from vllm_nnrp_adapter.cli import main


def _source(*, ratio: float = 0.3) -> dict[str, Any]:
    sample_count = 10
    stale_count = int(sample_count * ratio)

    def run(baseline: str, stale_gpu_seconds: float, *, accepted: bool) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []
        for index in range(sample_count - stale_count):
            samples.append(
                {
                    "sample_id": f"sample-{index:06d}",
                    "scheduled_offset_seconds": index * 0.01,
                    "operation_started_at_seconds": index * 0.01,
                    "terminal_outcome": "completed",
                    "control_kind": None,
                    "control_scheduled_at_seconds": None,
                    "control_issued_at_seconds": None,
                    "control_dispatched": False,
                    "control_accepted": False,
                    "control_accepted_at_seconds": None,
                    "backend_stopped_at_seconds": 1.0 + index,
                    "gpu_seconds": 0.8,
                    "useful_result_weight": 1.0,
                    "late_result_count": 0,
                }
            )
        outcomes = ("cancelled", "aborted", "expired", "superseded")
        controls = {"cancelled": "cancel", "aborted": "abort", "expired": "deadline", "superseded": "supersede"}
        for index in range(stale_count):
            outcome = outcomes[index % len(outcomes)]
            ordinal = sample_count - stale_count + index
            operation_started_at = ordinal * 0.01
            control_scheduled_at = operation_started_at + 0.001
            accepted_at = 7.0 + index if accepted else None
            samples.append(
                {
                    "sample_id": f"sample-{ordinal:06d}",
                    "scheduled_offset_seconds": operation_started_at,
                    "operation_started_at_seconds": operation_started_at,
                    "terminal_outcome": outcome if accepted else "completed",
                    "control_kind": controls[outcome],
                    "control_scheduled_at_seconds": control_scheduled_at,
                    "control_issued_at_seconds": control_scheduled_at,
                    "control_dispatched": accepted,
                    "control_accepted": accepted,
                    "control_accepted_at_seconds": accepted_at,
                    "backend_stopped_at_seconds": (accepted_at + 0.1) if accepted_at is not None else 9.5 + index,
                    "gpu_seconds": stale_gpu_seconds,
                    "useful_result_weight": 0.0,
                    "late_result_count": 0,
                }
            )
        return {"baseline": baseline, "wall_clock_seconds": 20.0, "samples": samples}

    return {
        "schema_version": "nnrp-adoption-evidence/v1",
        "workload": {
            "scenario": "stale_work",
            "stale_work_ratio": ratio,
            "adapter_version": "0.1.0",
            "adapter_revision": "abcdef0",
            "model": "public-test-model",
            "engine": "vllm-0.26",
            "gpu": "test-gpu-class",
            "arrival_schedule": "seeded-fixed-interval",
            "arrival_interval_seconds": 0.01,
            "cancellation_schedule": "seeded-stale-selection",
            "control_delay_seconds": 0.001,
            "prompt_tokens": 4096,
            "max_completion_tokens": 128,
            "warmup": 2,
            "random_seed": 7,
            "sample_count": sample_count,
            "max_in_flight": 1,
            "gpu_accounting": {
                "method": "device_active_time",
                "scope": "dedicated_device",
                "source": "test-device-counter",
            },
        },
        "provenance": {
            "adapter_distribution": "vllm-nnrp-adapter",
            "adapter_version": "0.1.0",
            "adapter_revision": "abcdef0",
            "nnrp_sdk_version": "1.0.0rc4.post20",
        },
        "runs": [
            run("raw_openai_http_sse", 1.0, accepted=False),
            run("orchestrated_http_sse", 0.45, accepted=True),
            run("direct_nnrp", 0.4, accepted=True),
        ],
    }


def test_aggregate_stale_work_evidence_reports_value_metrics_and_thresholds() -> None:
    report = aggregate_stale_work_evidence(_source())

    assert report["benchmark_kind"] == "stale_work_adoption_evidence"
    assert report["provenance"]["adapter_revision"] == "abcdef0"
    assert [run["baseline"] for run in report["runs"]] == [
        "raw_openai_http_sse",
        "orchestrated_http_sse",
        "direct_nnrp",
    ]
    assert report["baseline_execution_order"] == [
        "raw_openai_http_sse",
        "orchestrated_http_sse",
        "direct_nnrp",
    ]
    direct = report["runs"][2]
    assert direct["sample_count"] == 10
    assert direct["stale_sample_count"] == 3
    assert direct["accepted_control_count"] == 3
    assert direct["wasted_gpu_seconds"] == pytest.approx(1.2)
    assert direct["useful_gpu_seconds"] == pytest.approx(5.6)
    assert direct["deadline_weighted_useful_goodput"] == pytest.approx(0.35)
    assert direct["cancellation_effect_latency_seconds"]["p95_seconds"] == pytest.approx(0.1)
    assert direct["late_result_rate"] == 0.0
    assert direct["semantic_defect_rate"] == 0.0
    assert direct["gpu_seconds_90ci"]["wasted"]["sample_count"] == 3
    assert report["acceptance"]["wasted_gpu_reduction_vs_raw"] == pytest.approx(0.6)
    assert report["acceptance"]["hypotheses"] == {
        "wasted_gpu_reduction_at_least_40_percent": "pass",
        "wasted_gpu_non_inferiority_within_3_percent": "pass",
        "late_result_rate_below_0_1_percent": "pass",
    }


def test_non_30_percent_workload_preserves_evidence_without_claiming_acceptance() -> None:
    report = aggregate_stale_work_evidence(_source(ratio=0.1))

    assert report["acceptance"] == {
        "evaluated": False,
        "reason": "Preview4 stale-work acceptance thresholds apply to the 30% workload",
    }


def test_request_interval_proxy_cannot_evaluate_gpu_acceptance() -> None:
    source = _source()
    source["workload"]["gpu_accounting"] = {
        "method": "request_inference_interval_proxy",
        "scope": "request",
        "source": "vllm-request-stats",
    }

    report = aggregate_stale_work_evidence(source)

    assert report["workload"]["gpu_accounting"]["acceptance_eligible"] is False
    assert report["acceptance"] == {
        "evaluated": False,
        "reason": "GPU accounting is a request interval proxy and cannot evaluate GPU-second thresholds",
    }


def test_overlapping_device_active_time_cannot_be_summed_as_request_gpu_seconds() -> None:
    source = _source()
    source["workload"]["max_in_flight"] = 2

    report = aggregate_stale_work_evidence(source)

    assert report["workload"]["gpu_accounting"]["acceptance_eligible"] is False
    assert report["acceptance"]["evaluated"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda source: source["runs"].pop(), "three comparison baselines"),
        (lambda source: source["runs"][0]["samples"].pop(), "sample_count"),
        (
            lambda source: source["runs"][1]["samples"][-1].update({"control_accepted_at_seconds": None}),
            "control_accepted_at_seconds is required",
        ),
        (
            lambda source: source["runs"][1]["samples"][-1].update({"control_dispatched": False}),
            "control_accepted requires control_dispatched",
        ),
        (
            lambda source: source["runs"][2]["samples"][-1].update({"useful_result_weight": 1.0}),
            "must be zero for stale work",
        ),
        (
            lambda source: source["provenance"].update({"adapter_revision": "1234567"}),
            "adapter_revision must match",
        ),
        (
            lambda source: source["workload"].update({"adapter_revision": "not-a-revision"}),
            "lowercase 7-40 character Git revision",
        ),
        (
            lambda source: source["runs"][0]["samples"][0].update({"sample_id": "sample-999999"}),
            "sample_id must be",
        ),
        (
            lambda source: source["runs"][0]["samples"][1].update({"scheduled_offset_seconds": 0.5}),
            "differs from workload schedule",
        ),
        (
            lambda source: source["runs"][0]["samples"][-1].update({"control_issued_at_seconds": 0.0}),
            "precedes scheduled control",
        ),
    ],
)
def test_aggregate_stale_work_evidence_rejects_incomplete_or_inconsistent_runs(
    mutate: Any,
    message: str,
) -> None:
    source = _source()
    mutate(source)

    with pytest.raises(ValueError, match=message):
        aggregate_stale_work_evidence(source)


def test_aggregate_stale_work_evidence_rejects_sensitive_manifest_metadata() -> None:
    source = _source()
    source["workload"]["gpu"] = r"C:\Users\operator\gpu-node"

    with pytest.raises(ValueError, match="Windows user path"):
        aggregate_stale_work_evidence(source)


def test_aggregate_stale_work_evidence_file_and_cli_write_machine_readable_report(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    direct_output = tmp_path / "direct.json"
    cli_output = tmp_path / "cli.json"
    input_path.write_text(json.dumps(_source()), encoding="utf-8")

    direct_report = aggregate_stale_work_evidence_file(input_path, direct_output)
    exit_code = main(
        [
            "aggregate-stale-work-evidence",
            "--input",
            str(input_path),
            "--output",
            str(cli_output),
        ]
    )

    assert exit_code == 0
    assert json.loads(direct_output.read_text(encoding="utf-8")) == direct_report
    assert json.loads(cli_output.read_text(encoding="utf-8")) == direct_report


def test_late_results_are_counted_per_accepted_operation() -> None:
    source = deepcopy(_source())
    source["runs"][2]["samples"][-1]["late_result_count"] = 2

    report = aggregate_stale_work_evidence(source)
    direct = report["runs"][2]

    assert direct["late_result_operation_count"] == 1
    assert direct["late_result_rate"] == pytest.approx(1 / 3)
    assert direct["semantic_defect_count"] == 1
    assert report["acceptance"]["hypotheses"]["late_result_rate_below_0_1_percent"] == "fail"


def test_post_dispatch_results_are_not_late_until_control_acceptance_is_observed() -> None:
    source = deepcopy(_source())
    source["runs"][0]["samples"][-1]["late_result_count"] = 2

    report = aggregate_stale_work_evidence(source)
    raw = report["runs"][0]

    assert raw["samples"][-1]["late_result_count"] == 2
    assert raw["accepted_control_count"] == 0
    assert raw["late_result_operation_count"] == 0
    assert raw["late_result_rate"] is None
    assert raw["semantic_defect_count"] == 0


def test_accepted_control_with_wrong_terminal_outcome_is_a_semantic_defect() -> None:
    source = deepcopy(_source())
    source["runs"][2]["samples"][-1]["terminal_outcome"] = "completed"

    report = aggregate_stale_work_evidence(source)
    direct = report["runs"][2]

    assert direct["stale_sample_count"] == 3
    assert direct["semantic_defect_count"] == 1
    assert direct["semantic_defect_rate"] == pytest.approx(0.1)


def test_baselines_must_share_the_same_sample_and_control_schedule() -> None:
    source = deepcopy(_source())
    source["runs"][2]["samples"][-1]["control_kind"] = "abort"

    with pytest.raises(ValueError, match="same ordered sample IDs and control schedule"):
        aggregate_stale_work_evidence(source)
